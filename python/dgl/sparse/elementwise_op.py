# pylint: disable=anomalous-backslash-in-string
"""DGL elementwise operator module."""
from typing import Union

from .diag_matrix import DiagMatrix
from .sparse_matrix import SparseMatrix
from .utils import Scalar

__all__ = ["add", "sub", "mul", "div", "power"]


def add(
    A: Union[DiagMatrix, SparseMatrix], B: Union[DiagMatrix, SparseMatrix]
) -> Union[DiagMatrix, SparseMatrix]:
    r"""Elementwise additions for ``DiagMatrix`` and ``SparseMatrix``,
    equivalent to ``A + B``.

    The supported combinations are shown as follows.

    +--------------+------------+--------------+--------+
    |    A \\ B    | DiagMatrix | SparseMatrix | scalar |
    +--------------+------------+--------------+--------+
    |  DiagMatrix  |     ✅     |      ✅      |   🚫   |
    +--------------+------------+--------------+--------+
    | SparseMatrix |     ✅     |      ✅      |   🚫   |
    +--------------+------------+--------------+--------+
    |    scalar    |     🚫     |      🚫      |   🚫   |
    +--------------+------------+--------------+--------+

    Parameters
    ----------
    A : DiagMatrix or SparseMatrix
        Diagonal matrix or sparse matrix
    B : DiagMatrix or SparseMatrix
        Diagonal matrix or sparse matrix

    Returns
    -------
    DiagMatrix or SparseMatrix
        Diagonal matrix if both :attr:`A` and :attr:`B` are diagonal matrices,
        sparse matrix otherwise

    Examples
    --------
    >>> row = torch.tensor([1, 0, 2])
    >>> col = torch.tensor([0, 1, 2])
    >>> val = torch.tensor([10, 20, 30])
    >>> A = from_coo(row, col, val)
    >>> B = diag(torch.arange(1, 4))
    >>> add(A, B)
    SparseMatrix(indices=tensor([[0, 0, 1, 1, 2],
                                 [0, 1, 0, 1, 2]]),
                 values=tensor([ 1, 20, 10,  2, 33]),
                 shape=(3, 3), nnz=5)
    """
    return A + B


def sub(A: Union[DiagMatrix], B: Union[DiagMatrix]) -> Union[DiagMatrix]:
    r"""Elementwise subtraction for ``DiagMatrix`` and ``SparseMatrix``,
    equivalent to ``A - B``.

    The supported combinations are shown as follows.

    +--------------+------------+--------------+--------+
    |    A \\ B    | DiagMatrix | SparseMatrix | scalar |
    +--------------+------------+--------------+--------+
    |  DiagMatrix  |     ✅     |      🚫      |   🚫   |
    +--------------+------------+--------------+--------+
    | SparseMatrix |     🚫     |      🚫      |   🚫   |
    +--------------+------------+--------------+--------+
    |    scalar    |     🚫     |      🚫      |   🚫   |
    +--------------+------------+--------------+--------+

    Parameters
    ----------
    A : DiagMatrix
        Diagonal matrix
    B : DiagMatrix
        Diagonal matrix

    Returns
    -------
    DiagMatrix
        Diagonal matrix

    Examples
    --------
    >>> A = diag(torch.arange(1, 4))
    >>> B = diag(torch.arange(10, 13))
    >>> sub(A, B)
    DiagMatrix(val=tensor([-9, -9, -9]),
               shape=(3, 3))
    """
    return A - B


def mul(
    A: Union[SparseMatrix, DiagMatrix, Scalar],
    B: Union[SparseMatrix, DiagMatrix, Scalar],
) -> Union[SparseMatrix, DiagMatrix]:
    r"""Elementwise multiplication for ``DiagMatrix`` and ``SparseMatrix``,
    equivalent to ``A * B``.

    The supported combinations are shown as follows.

    +--------------+------------+--------------+--------+
    |    A \\ B    | DiagMatrix | SparseMatrix | scalar |
    +--------------+------------+--------------+--------+
    |  DiagMatrix  |     ✅     |      🚫      |   ✅   |
    +--------------+------------+--------------+--------+
    | SparseMatrix |     🚫     |      🚫      |   ✅   |
    +--------------+------------+--------------+--------+
    |    scalar    |     ✅     |      ✅      |   🚫   |
    +--------------+------------+--------------+--------+

    Parameters
    ----------
    A : SparseMatrix or DiagMatrix or Scalar
        Sparse matrix or diagonal matrix or scalar value
    B : SparseMatrix or DiagMatrix or Scalar
        Sparse matrix or diagonal matrix or scalar value

    Returns
    -------
    SparseMatrix or DiagMatrix
        Either sparse matrix or diagonal matrix

    Examples
    --------
    >>> row = torch.tensor([1, 0, 2])
    >>> col = torch.tensor([0, 3, 2])
    >>> val = torch.tensor([10, 20, 30])
    >>> A = from_coo(row, col, val)
    >>> mul(A, 2)
    SparseMatrix(indices=tensor([[1, 0, 2],
                                 [0, 3, 2]]),
                 values=tensor([20, 40, 60]),
                 shape=(3, 4), nnz=3)

    >>> D = diag(torch.arange(1, 4))
    >>> mul(D, 2)
    DiagMatrix(val=tensor([2, 4, 6]),
               shape=(3, 3))

    >>> D = diag(torch.arange(1, 4))
    >>> mul(D, D)
    DiagMatrix(val=tensor([1, 4, 9]),
               shape=(3, 3))
    """
    return A * B


def div(
    A: Union[SparseMatrix, DiagMatrix], B: Union[DiagMatrix, Scalar]
) -> Union[SparseMatrix, DiagMatrix]:
    r"""Elementwise division for ``DiagMatrix`` and ``SparseMatrix``, equivalent
    to ``A / B``.

    The supported combinations are shown as follows.

    +--------------+------------+--------------+--------+
    |    A \\ B    | DiagMatrix | SparseMatrix | scalar |
    +--------------+------------+--------------+--------+
    |  DiagMatrix  |     ✅     |      🚫      |   ✅   |
    +--------------+------------+--------------+--------+
    | SparseMatrix |     🚫     |      🚫      |   ✅   |
    +--------------+------------+--------------+--------+
    |    scalar    |     🚫     |      🚫      |   🚫   |
    +--------------+------------+--------------+--------+

    Parameters
    ----------
    A : SparseMatrix or DiagMatrix
        Sparse or diagonal matrix
    B : DiagMatrix or Scalar
        Diagonal matrix or scalar value

    Returns
    -------
    DiagMatrix
        Diagonal matrix

    Examples
    --------
    >>> A = diag(torch.arange(1, 4))
    >>> B = diag(torch.arange(10, 13))
    >>> div(A, B)
    DiagMatrix(val=tensor([0.1000, 0.1818, 0.2500]),
               shape=(3, 3))

    >>> A = diag(torch.arange(1, 4))
    >>> div(A, 2)
    DiagMatrix(val=tensor([0.5000, 1.0000, 1.5000]),
               shape=(3, 3))

    >>> row = torch.tensor([1, 0, 2])
    >>> col = torch.tensor([0, 3, 2])
    >>> val = torch.tensor([1, 2, 3])
    >>> A = from_coo(row, col, val, shape=(3, 4))
    >>> A / 2
    SparseMatrix(indices=tensor([[1, 0, 2],
                                 [0, 3, 2]]),
                 values=tensor([0.5000, 1.0000, 1.5000]),
                 shape=(3, 4), nnz=3)
    """
    return A / B


def power(
    A: Union[SparseMatrix, DiagMatrix], scalar: Scalar
) -> Union[SparseMatrix, DiagMatrix]:
    r"""Elementwise exponentiation for ``DiagMatrix`` and ``SparseMatrix``,
    equivalent to ``A ** scalar``.

    The supported combinations are shown as follows.

    +--------------+------------+--------------+--------+
    |    A \\ B    | DiagMatrix | SparseMatrix | scalar |
    +--------------+------------+--------------+--------+
    |  DiagMatrix  |     🚫     |      🚫      |   ✅   |
    +--------------+------------+--------------+--------+
    | SparseMatrix |     🚫     |      🚫      |   ✅   |
    +--------------+------------+--------------+--------+
    |    scalar    |     🚫     |      🚫      |   🚫   |
    +--------------+------------+--------------+--------+

    Parameters
    ----------
    A : SparseMatrix or DiagMatrix
        Sparse matrix or diagonal matrix
    scalar : Scalar
        Exponent

    Returns
    -------
    SparseMatrix or DiagMatrix
        Sparse matrix or diagonal matrix, same type as A

    Examples
    --------
    >>> row = torch.tensor([1, 0, 2])
    >>> col = torch.tensor([0, 3, 2])
    >>> val = torch.tensor([10, 20, 30])
    >>> A = from_coo(row, col, val)
    >>> power(A, 2)
    SparseMatrix(indices=tensor([[1, 0, 2],
                                 [0, 3, 2]]),
                 values=tensor([100, 400, 900]),
                 shape=(3, 4), nnz=3)

    >>> D = diag(torch.arange(1, 4))
    >>> power(D, 2)
    DiagMatrix(val=tensor([1, 4, 9]),
               shape=(3, 3))
    """
    return A**scalar
