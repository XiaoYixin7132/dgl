import argparse
import time
from math import ceil

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.profiler
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
from torch.profiler import record_function

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl import AddSelfLoop
from dgl.data import RedditDataset
from dgl.data import CiteseerGraphDataset


class SAGEConvWithCV(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super().__init__()
        self.W = nn.Linear(in_feats * 2, out_feats)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, block, H, HBar=None):
        if self.training:
            with block.local_scope():
                H_src, H_dst = H
                # P*H^{\bar}: agg_HBar_dst
                HBar_src, agg_HBar_dst = HBar
                block.dstdata["agg_hbar"] = agg_HBar_dst
                block.srcdata["hdelta"] = H_src - HBar_src
                # P^{\hat} * (H - H^{\bar}): hdelta_new
                block.update_all(
                    fn.copy_u("hdelta", "m"), fn.mean("m", "hdelta_new")
                )
                # Calculation result in the outermost bracket
                h_neigh = (
                    block.dstdata["agg_hbar"] + block.dstdata["hdelta_new"]
                )
                h = self.W(th.cat([H_dst, h_neigh], 1))
                if self.activation is not None:
                    h = self.activation(h)
                return h
        else:
            with block.local_scope():
                H_src, H_dst = H
                block.srcdata["h"] = H_src
                block.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_new"))
                h_neigh = block.dstdata["h_new"]
                h = self.W(th.cat([H_dst, h_neigh], 1))
                if self.activation is not None:
                    h = self.activation(h)
                return h


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConvWithCV(in_feats, n_hidden, activation))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConvWithCV(n_hidden, n_hidden, activation))
        self.layers.append(SAGEConvWithCV(n_hidden, n_classes, None))

    def forward(self, blocks):
        h = blocks[0].srcdata["features"]
        updates = []
        for layer, block in zip(self.layers, blocks):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[: block.number_of_dst_nodes()]
            hbar_src = block.srcdata["hist"]
            agg_hbar_dst = block.dstdata["agg_hist"]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst), (hbar_src, agg_hbar_dst))
            block.dstdata["h_new"] = h
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = th.arange(g.number_of_nodes())
        ys = []
        for l, layer in enumerate(self.layers):
            y = th.zeros(
                g.number_of_nodes(),
                self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
            )

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(
                    dgl.in_subgraph(g, batch_nodes), batch_nodes
                )
                block = block.int().to(device)
                induced_nodes = block.srcdata[dgl.NID]
                x = x.to(device)

                h = x[induced_nodes]
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))

                y[start:end] = h.cpu()

            ys.append(y)
            x = y
        return y, ys


class NeighborSampler(object):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(seeds)
        blocks = []
        hist_blocks = []
        with record_function('I: Sample Blocks'):
            for fanout in self.fanouts:
                # For each seed node, sample ``fanout`` neighbors.
                frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout)
                # Include all the neighbors of the seeds into hist_frontier
                hist_frontier = dgl.in_subgraph(self.g, seeds)
                # Then we compact the frontier into a bipartite graph for message passing.
                block = dgl.to_block(frontier, seeds)
                hist_block = dgl.to_block(hist_frontier, seeds)
                # Obtain the seed nodes for next layer.
                seeds = block.srcdata[dgl.NID]

                # Insert block to the very first position
                blocks.insert(0, block)
                hist_blocks.insert(0, hist_block)
        return blocks, hist_blocks


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, labels, val_mask, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        inputs = g.ndata["features"]
        pred, _ = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_mask], labels[val_mask])


def load_subtensor(
    g, labels, blocks, hist_blocks, dev_id, aggregation_on_device=False
):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    # Prepare input features
    blocks[0].srcdata["features"] = g.ndata["features"][
        blocks[0].srcdata[dgl.NID]
    ]
    blocks[-1].dstdata["label"] = labels[blocks[-1].dstdata[dgl.NID]]
    ret_blocks = []
    ret_hist_blocks = []
    for i, (block, hist_block) in enumerate(zip(blocks, hist_blocks)):
        hist_col = "features" if i == 0 else "hist_%d" % i
        block.srcdata["hist"] = g.ndata[hist_col][block.srcdata[dgl.NID]]

        # Aggregate history
        # with record_function("Aggregate History"):
        hist_block.srcdata["hist"] = g.ndata[hist_col][
            hist_block.srcdata[dgl.NID]
        ]
        if aggregation_on_device:
            hist_block = hist_block.to(dev_id)
        # This is how message passing is presented in DGL
        hist_block.update_all(fn.copy_u("hist", "m"), fn.mean("m", "agg_hist"))

        block = block.int().to(dev_id)
        if not aggregation_on_device:
            hist_block = hist_block.to(dev_id)
        block.dstdata["agg_hist"] = hist_block.dstdata["agg_hist"]
        ret_blocks.append(block)
        ret_hist_blocks.append(hist_block)
    return ret_blocks, ret_hist_blocks


def init_history(g, model, dev_id):
    with th.no_grad():
        history = model.inference(g, g.ndata["features"], 1000, dev_id)[1]
        for layer in range(args.num_layers + 1):
            if layer > 0:
                hist_col = "hist_%d" % layer
                g.ndata[hist_col] = history[layer - 1]


def update_history(g, blocks):
    with th.no_grad():
        for i, block in enumerate(blocks):
            ids = block.dstdata[dgl.NID].cpu()
            hist_col = "hist_%d" % (i + 1)

            h_new = block.dstdata["h_new"].cpu()
            g.ndata[hist_col][ids] = h_new


def run(args, dev_id, data):
    dropout = 0.2

    th.cuda.set_device(dev_id)

    # Unpack data
    train_mask, val_mask, in_feats, labels, n_classes, g = data
    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()

    # Create sampler
    sampler = NeighborSampler(g, [int(_) for _ in args.fan_out.split(",")])

    # Create PyTorch DataLoader for constructing blocks
    dataloader = DataLoader(
        dataset=train_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers_per_gpu,
    )

    # Define model
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu)

    # Move the model to GPU and define optimizer
    model = model.to(dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(dev_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Compute history tensor and their aggregation before training on CPU

    model.eval()
    # with record_function("Init History"):
    init_history(g, model, dev_id)
    model.train()

    # Training loop
    avg = 0
    # Throughput
    iter_tput = []
    steps_per_epoch = ceil(train_nid.size(0) / args.batch_size)
    print("Steps per epoch: {}".format(steps_per_epoch))
    # Set pytorch profiler
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        # schedule=torch.profiler.schedule(
        #     wait=steps_per_epoch, warmup=steps_per_epoch,
        #     active=10*steps_per_epoch, repeat=1),
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/VR-GCN-Single-GPU'),
        record_shapes=True,
        with_stack=True
    )
    prof.schedule = torch.profiler.schedule(
            wait=steps_per_epoch, warmup=steps_per_epoch,
            active=10*steps_per_epoch, repeat=1)
    prof.start()
    for epoch in range(args.num_epochs):
        tic = time.time()
        model.train()
        tic_step = time.time()
        # step = 0
        # dataloader_iter = iter(dataloader)
        # (blocks, hist_blocks) = next(dataloader_iter, (None, None))
        # while (blocks, hist_blocks) != (None, None):
        #     with record_function("1: Mini-batch"):
        #         # The nodes for input lies at the LHS side of the first block.
        #         # The nodes for output lies at the RHS side of the last block.
        #         input_nodes = blocks[0].srcdata[dgl.NID]
        #         seeds = blocks[-1].dstdata[dgl.NID]
        #         # The blocks only contain node list now. We need to prepare the corresponding input node features and labels
        #         # of the seed nodes. Besides, the aggregation result of the historical embeddings is computed.
        #         # The blocks and hist_blocks are then sent to GPU.
        #         with record_function("2: Construct Block & Aggregate Hist & Send to GPU"):
        #             blocks, hist_blocks = load_subtensor(
        #                 g, labels, blocks, hist_blocks, dev_id, True
        #             )
        #
        #         # forward
        #         with record_function("3: Computation"):
        #             with record_function("3.1: Forward"):
        #                 batch_pred = model(blocks)
        #             # update history
        #             with record_function("3.2: Update History"):
        #                 update_history(g, blocks)
        #             # compute loss
        #             with record_function("3.3: Compute Loss"):
        #                 batch_labels = blocks[-1].dstdata["label"]
        #                 loss = loss_fcn(batch_pred, batch_labels)
        #             # backward
        #             with record_function('3.4: Backward'):
        #                 optimizer.zero_grad()
        #                 loss.backward()
        #         with record_function('4: Update weights'):
        #             optimizer.step()
        #         iter_tput.append(len(seeds) / (time.time() - tic_step))
        #         if step % args.log_every == 0:
        #             acc = compute_acc(batch_pred, batch_labels)
        #             # writer.add_scalar('Accuracy/train', acc, epoch)
        #             print(
        #                 "Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f}".format(
        #                     epoch,
        #                     step,
        #                     loss.item(),
        #                     acc.item(),
        #                     np.mean(iter_tput[3:]),
        #                 )
        #             )
        #         tic_step = time.time()
        #         prof.step()
        #         step += 1
        for step, (blocks, hist_blocks) in enumerate(dataloader):
            with record_function("II: Mini-batch"):
                # The nodes for input lies at the LHS side of the first block.
                # The nodes for output lies at the RHS side of the last block.
                input_nodes = blocks[0].srcdata[dgl.NID]
                seeds = blocks[-1].dstdata[dgl.NID]
                # The blocks only contain node list now. We need to prepare the corresponding input node features and labels
                # of the seed nodes. Besides, the aggregation result of the historical embeddings is computed.
                # The blocks and hist_blocks are then sent to GPU.
                with record_function("1: Construct Block & Aggregate Hist & Send to GPU"):
                    blocks, hist_blocks = load_subtensor(
                        g, labels, blocks, hist_blocks, dev_id, True
                    )

                # forward
                with record_function("2: Computation"):
                    with record_function("2.1: Forward"):
                        batch_pred = model(blocks)
                    # update history
                    with record_function("2.2: Update History"):
                        update_history(g, blocks)
                    # compute loss
                    with record_function("2.3: Compute Loss"):
                        batch_labels = blocks[-1].dstdata["label"]
                        loss = loss_fcn(batch_pred, batch_labels)
                    # backward
                    with record_function('2.4: Backward'):
                        optimizer.zero_grad()
                        loss.backward()
                with record_function('3: Update Weights'):
                    optimizer.step()
                iter_tput.append(len(seeds) / (time.time() - tic_step))
                if step % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    # writer.add_scalar('Accuracy/train', acc, epoch)
                    print(
                        "Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f}".format(
                            epoch,
                            step,
                            loss.item(),
                            acc.item(),
                            np.mean(iter_tput[3:]),
                        )
                    )
                tic_step = time.time()
                prof.step()
        toc = time.time()
        print("Epoch Time(s): {:.4f}".format(toc - tic))
        # writer.add_scalar('Loss/train', loss, epoch)
        if epoch >= 5:
            avg += toc - tic
        model.eval()
        eval_acc = evaluate(
            model, g, labels, val_nid, args.val_batch_size, dev_id
        )
        # writer.add_scalar('Accuracy/test', eval_acc, epoch)
        if epoch % args.eval_every == 0 and epoch != 0:
            print("Eval Acc {:.4f}".format(eval_acc))
    prof.stop()
    print("Avg epoch time: {}".format(avg / (epoch - 4)))
    print(prof.key_averages().table(sort_by="cpu_time_total"))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("single-gpu training")
    argparser.add_argument("--gpu", type=str, default="0")
    argparser.add_argument("--num-epochs", type=int, default=20)
    argparser.add_argument("--num-hidden", type=int, default=16)
    argparser.add_argument("--num-layers", type=int, default=2)
    argparser.add_argument("--fan-out", type=str, default="2,2")
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--val-batch-size", type=int, default=1000)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--eval-every", type=int, default=5)
    argparser.add_argument("--lr", type=float, default=0.003)
    argparser.add_argument("--num-workers-per-gpu", type=int, default=0)
    args = argparser.parse_args()

    # load reddit data
    # data = RedditDataset(self_loop=True)
    # load Citeseer
    data = CiteseerGraphDataset(transform=AddSelfLoop())
    n_classes = data.num_classes
    g = data[0]
    print("number of nodes: {}".format(g.number_of_nodes()))
    features = g.ndata["feat"]
    in_feats = features.shape[1]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    g.ndata["features"] = features
    g.create_formats_()
    # Pack data
    data = train_mask, val_mask, in_feats, labels, n_classes, g

    run(args, int(args.gpu), data)
    # writer.flush()
    # writer.close()
    #TODO: Add final inference and accuracy. 
