import time
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

# Constants for default values
DEFAULT_FROBENIUS_NORM = 'fro'
DEFAULT_CHOLESKY_UPPER = False
DEFAULT_QR_MODE = 'reduced'
DEFAULT_LU_PIVOT = True
DEFAULT_EIGH_UPLO = 'L'
DEFAULT_SVD_FULL_MATRICES = True
DEFAULT_SOLVE_LEFT = True
DEFAULT_SOLVE_CHECK_ERRORS = False
DEFAULT_SOLVE_TRIANGULAR_UPPER = True
DEFAULT_SOLVE_TRIANGULAR_UNITRIANGULAR = False
DEFAULT_LU_SOLVE_ADJOINT = False
DEFAULT_INV_CHECK_ERRORS = False
DEFAULT_PINV_HERMITIAN = False
DEFAULT_CROSS_DIM = -1
DEFAULT_TENSORINV_IND = 2
DEFAULT_VANDER_INCREASING = False
DEFAULT_CHOLESKY_EX_CHECK_ERRORS = False
DEFAULT_LU_FACTOR_EX_CHECK_ERRORS = False
DEFAULT_LDL_FACTOR_HERMITIAN = False
DEFAULT_LDL_FACTOR_EX_CHECK_ERRORS = False
DEFAULT_LDL_SOLVE_HERMITIAN = False
DEFAULT_SOLVE_EX_CHECK_ERRORS = False
DEFAULT_INV_EX_CHECK_ERRORS = False
DEFAULT_IS_HERMITIAN_ATOL = 1e-8
DEFAULT_IS_HERMITIAN_RTOL = 1e-5
DEFAULT_ORTHOGONALIZE_EPS = 1e-8
DEFAULT_IS_INVERTIBLE_ATOL = 1e-8


class LinearAlgebraOperations:
    """A class for common linear algebra operations using PyTorch.

    This class provides a set of static methods for performing various linear
    algebra operations using PyTorch. Each method includes error handling to
    ensure robustness.
    Example:
    
    linalg = LinearAlgebraOperations()
    matrix = torch.randn(10,10)
    
    try:
        norm = linalg.matrix_norm(matrix)
        print(linalg.norm(matrix))
        print(f"Matrix norm: {norm}")
        
        # Compute determinant
        det = linalg.det(matrix)
        print(f"Determinant: {det}")
        
        # Compute inverse
        inv = linalg.inv(matrix)
        print(f"Inverse matrix: {inv}")
        
        # Check if the matrix is invertible
        is_inv = linalg.is_invertible(matrix)
        print(f"Is invertible: {is_inv}")
        # # Compute eigenvalues
        eigenvals = linalg.eigvals(matrix)
        print(f"Eigenvalues: {eigenvals}")
        # Compute SVD
        U, S, V = linalg.svd(matrix)
        print(f"SVD - U: {U}, S: {S}, V: {V}") 
    except ValueError as e:
        print(f"An error occurred: {str(e)}")
    """

    @staticmethod
    def norm(input: Tensor,
             ord: Optional[Union[str, float]] = None,
             dim: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdim: bool = False,
             dtype: Optional[torch.dtype] = None) -> Tensor:
        """Computes a vector or matrix norm.

        This function calculates the norm of a tensor. It provides flexibility
        for various norm orders and dimensions.

        Args:
            input: The input tensor.
            ord: The order of the norm. For more details, refer to:
                https://pytorch.org/docs/stable/generated/torch.linalg.norm.html
            dim: The dimension(s) along which to compute the norm.
            keepdim: If True, the output tensor keeps the dimensions of the input
                tensor with the reduced dimensions having size 1.
            dtype: The desired data type of the returned tensor.

        Returns:
            The computed norm of the input tensor.

        Raises:
            ValueError: If an error occurs during the norm computation.
        """
        try:
            return torch.linalg.norm(input, ord=ord, dim=dim,
                                      keepdim=keepdim, dtype=dtype)
        except RuntimeError as e:
            raise ValueError(f"Error computing norm: {str(e)}")

    @staticmethod
    def vector_norm(input: Tensor,
                    ord: Optional[Union[float, str]] = None,
                    dim: Optional[int] = None,
                    keepdim: bool = False,
                    dtype: Optional[torch.dtype] = None) -> Tensor:
        """Computes a vector norm.

        This function is specifically designed to compute vector norms, ensuring
        that the input tensor represents a vector.

        Args:
            input: The input tensor representing a vector.
            ord: The order of the norm.
            dim: The dimension along which to compute the norm.
            keepdim: If True, the output tensor keeps the dimensions of the input
                tensor with the reduced dimensions having size 1.
            dtype: The desired data type of the returned tensor.

        Returns:
            The computed vector norm of the input tensor.

        Raises:
            ValueError: If an error occurs during the vector norm computation.
        """
        try:
            return torch.linalg.vector_norm(input, ord=ord, dim=dim,
                                            keepdim=keepdim, dtype=dtype)
        except RuntimeError as e:
            raise ValueError(f"Error computing vector norm: {str(e)}")

    @staticmethod
    def matrix_norm(input: Tensor,
                    ord: Optional[Union[str, float]] = None,
                    dim: Tuple[int, int] = (-2, -1),
                    keepdim: bool = False,
                    dtype: Optional[torch.dtype] = None) -> Tensor:
        """Computes a matrix norm.

        This function calculates the norm of a matrix. It defaults to the
        Frobenius norm if no specific order is provided.

        Args:
            input: The input tensor representing a matrix.
            ord: The order of the norm. Defaults to 'fro' (Frobenius norm).
            dim: The dimensions along which to compute the norm.
            keepdim: If True, the output tensor keeps the dimensions of the input
                tensor with the reduced dimensions having size 1.
            dtype: The desired data type of the returned tensor.

        Returns:
            The computed matrix norm of the input tensor.

        Raises:
            ValueError: If an error occurs during the matrix norm computation.
        """
        try:
            if ord is None:
                ord = DEFAULT_FROBENIUS_NORM
            return torch.linalg.matrix_norm(input, ord=ord, dim=dim,
                                            keepdim=keepdim, dtype=dtype)
        except RuntimeError as e:
            raise ValueError(f"Error computing matrix norm: {str(e)}")

    @staticmethod
    def diagonal(input: Tensor, offset: int = 0, dim1: int = -2,
                dim2: int = -1) -> Tensor:
        """Returns the diagonal of a matrix.

        Args:
            input: The input tensor.
            offset: The diagonal offset. Defaults to 0 (main diagonal).
            dim1: The first dimension defining the diagonal.
            dim2: The second dimension defining the diagonal.

        Returns:
            The diagonal of the input tensor.

        Raises:
            ValueError: If an error occurs while computing the diagonal.
        """
        try:
            return torch.diagonal(input, offset=offset, dim1=dim1, dim2=dim2)
        except RuntimeError as e:
            raise ValueError(f"Error computing diagonal: {str(e)}")

    @staticmethod
    def det(input: Tensor) -> Tensor:
        """Computes the determinant of a square matrix.

        Args:
            input: The input tensor representing a square matrix.

        Returns:
            The determinant of the input matrix.

        Raises:
            ValueError: If an error occurs while computing the determinant.
        """
        try:
            return torch.linalg.det(input)
        except RuntimeError as e:
            raise ValueError(f"Error computing determinant: {str(e)}")

    @staticmethod
    def slogdet(input: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the sign and log of the absolute value of the
        determinant of a square matrix.

        Args:
            input: The input tensor representing a square matrix.

        Returns:
            A tuple containing the sign and the log of the absolute value of the
            determinant.

        Raises:
            ValueError: If an error occurs while computing the sign and log
            determinant.
        """
        try:
            return torch.linalg.slogdet(input)
        except RuntimeError as e:
            raise ValueError(f"Error computing sign and log determinant: "
                             f"{str(e)}")

    @staticmethod
    def cond(input: Tensor, p: Optional[Union[str, float]] = None) -> Tensor:
        """Computes the condition number of a matrix with respect to a
        matrix norm.

        Args:
            input: The input tensor representing a matrix.
            p: The order of the norm. For more details, refer to:
                https://pytorch.org/docs/stable/generated/torch.linalg.cond.html

        Returns:
            The condition number of the input matrix.

        Raises:
            ValueError: If an error occurs while computing the condition
                number.
        """
        try:
            return torch.linalg.cond(input, p=p)
        except RuntimeError as e:
            raise ValueError(f"Error computing condition number: {str(e)}")

    @staticmethod
    def matrix_rank(input: Tensor, tol: Optional[float] = None,
                    hermitian: bool = False) -> Tensor:
        """Computes the numerical rank of a matrix.

        Args:
            input: The input tensor representing a matrix.
            tol: The tolerance value for singular values.
            hermitian: If True, the input matrix is assumed to be Hermitian (or
                symmetric for real matrices).

        Returns:
            The numerical rank of the input matrix.

        Raises:
            ValueError: If an error occurs while computing the matrix rank.
        """
        try:
            return torch.linalg.matrix_rank(input, tol=tol,
                                            hermitian=hermitian)
        except RuntimeError as e:
            raise ValueError(f"Error computing matrix rank: {str(e)}")

    @staticmethod
    def cholesky(input: Tensor,
                 upper: bool = DEFAULT_CHOLESKY_UPPER) -> Tensor:
        """Computes the Cholesky decomposition of a complex Hermitian or
        real symmetric positive-definite matrix.

        Args:
            input: The input tensor representing a matrix.
            upper: If True, returns the upper triangular Cholesky factor.

        Returns:
            The Cholesky factor of the input matrix.

        Raises:
            ValueError: If an error occurs while computing the Cholesky
                decomposition.
        """
        try:
            return torch.linalg.cholesky(input, upper=upper)
        except RuntimeError as e:
            raise ValueError(f"Error computing Cholesky decomposition: "
                             f"{str(e)}")

    @staticmethod
    def qr(input: Tensor,
           mode: str = DEFAULT_QR_MODE) -> Tuple[Tensor, Tensor]:
        """Computes the QR decomposition of a matrix.

        Args:
            input: The input tensor representing a matrix.
            mode: The mode of the QR decomposition. For more details, refer
                to: https://pytorch.org/docs/stable/generated/torch.linalg.qr.html

        Returns:
            A tuple containing the Q and R factors of the QR decomposition.

        Raises:
            ValueError: If an error occurs while computing the QR decomposition.
        """
        try:
            return torch.linalg.qr(input, mode=mode)
        except RuntimeError as e:
            raise ValueError(f"Error computing QR decomposition: {str(e)}")

    @staticmethod
    def lu(input: Tensor,
           pivot: bool = DEFAULT_LU_PIVOT) -> Tuple[Tensor, Tensor, Tensor]:
        """Computes the LU decomposition with partial pivoting of a matrix.

        Args:
            input: The input tensor representing a matrix.
            pivot: If True, performs partial pivoting.

        Returns:
            A tuple containing the L, U factors and the permutation matrix of
            the LU decomposition.

        Raises:
            ValueError: If an error occurs while computing the LU decomposition.
        """
        try:
            return torch.linalg.lu(input, pivot=pivot)
        except RuntimeError as e:
            raise ValueError(f"Error computing LU decomposition: {str(e)}")

    @staticmethod
    def lu_factor(input: Tensor,
                  pivot: bool = DEFAULT_LU_PIVOT) -> Tuple[Tensor, Tensor]:
        """Computes a compact representation of the LU factorization with
        partial pivoting of a matrix.

        Args:
            input: The input tensor representing a matrix.
            pivot: If True, performs partial pivoting.

        Returns:
            A tuple containing the combined L and U factors, and the pivot
            indices.

        Raises:
            ValueError: If an error occurs while computing the LU factorization.
        """
        try:
            return torch.linalg.lu_factor(input, pivot=pivot)
        except RuntimeError as e:
            raise ValueError(f"Error computing LU factorization: {str(e)}")

    @staticmethod
    def eig(input: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the eigenvalue decomposition of a square matrix if it
        exists.

        Args:
            input: The input tensor representing a square matrix.

        Returns:
            A tuple containing the eigenvalues and eigenvectors.

        Raises:
            ValueError: If an error occurs while computing the eigenvalue
                decomposition.
        """
        try:
            return torch.linalg.eig(input)
        except RuntimeError as e:
            raise ValueError(f"Error computing eigenvalue decomposition: "
                             f"{str(e)}")

    @staticmethod
    def eigvals(input: Tensor) -> Tensor:
        """Computes the eigenvalues of a square matrix.

        Args:
            input: The input tensor representing a square matrix.

        Returns:
            The eigenvalues of the input matrix.

        Raises:
            ValueError: If an error occurs while computing the eigenvalues.
        """
        try:
            return torch.linalg.eigvals(input)
        except RuntimeError as e:
            raise ValueError(f"Error computing eigenvalues: {str(e)}")

    @staticmethod
    def eigh(input: Tensor,
             UPLO: str = DEFAULT_EIGH_UPLO) -> Tuple[Tensor, Tensor]:
        """Computes the eigenvalue decomposition of a complex Hermitian
        or real symmetric matrix.

        Args:
            input: The input tensor representing a matrix.
            UPLO: Controls whether to use the upper or lower triangular part of
                the input matrix. 'L' for lower, 'U' for upper.

        Returns:
            A tuple containing the eigenvalues and eigenvectors.

        Raises:
            ValueError: If an error occurs while computing the eigenvalue
                decomposition.
        """
        try:
            return torch.linalg.eigh(input, UPLO=UPLO)
        except RuntimeError as e:
            raise ValueError(f"Error computing eigenvalue decomposition: "
                             f"{str(e)}")

    @staticmethod
    def eigvalsh(input: Tensor,
                 UPLO: str = DEFAULT_EIGH_UPLO) -> Tensor:
        """Computes the eigenvalues of a complex Hermitian or real
        symmetric matrix.

        Args:
            input: The input tensor representing a matrix.
            UPLO: Controls whether to use the upper or lower triangular part
                of the input matrix. 'L' for lower, 'U' for upper.

        Returns:
            The eigenvalues of the input matrix.

        Raises:
            ValueError: If an error occurs while computing the eigenvalues.
        """
        try:
            return torch.linalg.eigvalsh(input, UPLO=UPLO)
        except RuntimeError as e:
            raise ValueError(f"Error computing eigenvalues: {str(e)}")

    @staticmethod
    def svd(input: Tensor, full_matrices: bool = DEFAULT_SVD_FULL_MATRICES
            ) -> Tuple[Tensor, Tensor, Tensor]:
        """Computes the singular value decomposition (SVD) of a matrix.

        Args:
            input: The input tensor representing a matrix.
            full_matrices: If True, computes the full SVD.

        Returns:
            A tuple containing the U, S, and V factors of the SVD
            decomposition.

        Raises:
            ValueError: If an error occurs while computing the SVD.
        """
        try:
            return torch.linalg.svd(input, full_matrices=full_matrices)
        except RuntimeError as e:
            raise ValueError(f"Error computing SVD: {str(e)}")

    @staticmethod
    def svdvals(input: Tensor) -> Tensor:
        """Computes the singular values of a matrix.

        Args:
            input: The input tensor representing a matrix.

        Returns:
            The singular values of the input matrix.

        Raises:
            ValueError: If an error occurs while computing the singular values.
        """
        try:
            return torch.linalg.svdvals(input)
        except RuntimeError as e:
            raise ValueError(f"Error computing singular values: {str(e)}")

    @staticmethod
    def solve(input: Tensor, other: Tensor,
              left: bool = DEFAULT_SOLVE_LEFT,
              check_errors: bool = DEFAULT_SOLVE_CHECK_ERRORS) -> Tensor:
        """Computes the solution of a square system of linear equations
        with a unique solution.

        Args:
            input: The input tensor representing the coefficient matrix.
            other: The right-hand side tensor.
            left: If True, solves Ax = b. If False, solves xA = b.
            check_errors: If True, performs error checking.

        Returns:
            The solution tensor.

        Raises:
            ValueError: If an error occurs while solving the linear system.
        """
        try:
            return torch.linalg.solve(input, other, left=left)
        except RuntimeError as e:
            if check_errors:
                raise ValueError(f"Error solving linear system: {str(e)}")
            return torch.linalg.solve_ex(input, other, left=left,
                                        check_errors=False)[0]

    @staticmethod
    def solve_triangular(input: Tensor, other: Tensor,
                        upper: bool = DEFAULT_SOLVE_TRIANGULAR_UPPER,
                        left: bool = DEFAULT_SOLVE_LEFT,
                        unitriangular: bool = DEFAULT_SOLVE_TRIANGULAR_UNITRIANGULAR
                        ) -> Tensor:
        """Computes the solution of a triangular system of linear equations
        with a unique solution.

        Args:
            input: The input tensor representing the triangular matrix.
            other: The right-hand side tensor.
            upper: If True, the input matrix is upper triangular.
            left: If True, solves Ax = b. If False, solves xA = b.
            unitriangular: If True, the diagonal elements of the input matrix
                are assumed to be 1.

        Returns:
            The solution tensor.

        Raises:
            ValueError: If an error occurs while solving the triangular system.
        """
        try:
            return torch.linalg.solve_triangular(input, other, upper=upper,
                                                left=left,
                                                unitriangular=unitriangular)
        except RuntimeError as e:
            raise ValueError(f"Error solving triangular system: {str(e)}")

    @staticmethod
    def lu_solve(LU_data: Tensor, LU_pivots: Tensor, B: Tensor,
                left: bool = DEFAULT_SOLVE_LEFT,
                adjoint: bool = DEFAULT_LU_SOLVE_ADJOINT) -> Tensor:
        """Computes the solution of a square system of linear equations
        with a unique solution given an LU decomposition.

        Args:
            LU_data: The combined L and U factors from `torch.lu_factor`.
            LU_pivots: The pivot indices from `torch.lu_factor`.
            B: The right-hand side tensor.
            left: If True, solves Ax = b. If False, solves xA = b.
            adjoint: If True, solves A^H x = b (or xA^H = b).

        Returns:
            The solution tensor.

        Raises:
            ValueError: If an error occurs while solving the LU system.
        """
        try:
            return torch.linalg.lu_solve(LU_data, LU_pivots, B, left=left,
                                        adjoint=adjoint)
        except RuntimeError as e:
            raise ValueError(f"Error solving LU system: {str(e)}")

    @staticmethod
    def lstsq(input: Tensor, other: Tensor,
              rcond: Optional[float] = None) -> Tuple[Tensor, Tensor,
                                                     Tensor, Tensor]:
        """Computes a solution to the least squares problem of a system
        of linear equations.

        Args:
            input: The input tensor representing the coefficient matrix.
            other: The right-hand side tensor.
            rcond: The relative tolerance for singular values.

        Returns:
            A tuple containing the solution, residuals, rank, and singular
            values.

        Raises:
            ValueError: If an error occurs while computing the least squares
                solution.
        """
        try:
            return torch.linalg.lstsq(input, other, rcond=rcond)
        except RuntimeError as e:
            raise ValueError(f"Error computing least squares solution: "
                             f"{str(e)}")

    @staticmethod
    def inv(input: Tensor, check_errors: bool = DEFAULT_INV_CHECK_ERRORS
            ) -> Tensor:
        """Computes the inverse of a square matrix if it exists.

        Args:
            input: The input tensor representing a square matrix.
            check_errors: If True, performs error checking.

        Returns:
            The inverse of the input matrix.

        Raises:
            ValueError: If an error occurs while computing the matrix inverse.
        """
        try:
            return torch.linalg.inv(input)
        except RuntimeError as e:
            if check_errors:
                raise ValueError(f"Error computing matrix inverse: {str(e)}")
            return torch.linalg.inv_ex(input, check_errors=False)[0]

    @staticmethod
    def pinv(input: Tensor, rcond: Optional[float] = None,
             hermitian: bool = DEFAULT_PINV_HERMITIAN) -> Tensor:
        """Computes the pseudoinverse (Moore-Penrose inverse) of a matrix.

        Args:
            input: The input tensor representing a matrix.
            rcond: The relative tolerance for singular values.
            hermitian: If True, the input matrix is assumed to be Hermitian (or
                symmetric for real matrices).

        Returns:
            The pseudoinverse of the input matrix.

        Raises:
            ValueError: If an error occurs while computing the pseudoinverse.
        """
        try:
            return torch.linalg.pinv(input, rcond=rcond, hermitian=hermitian)
        except RuntimeError as e:
            raise ValueError(f"Error computing pseudoinverse: {str(e)}")

    @staticmethod
    def matrix_exp(input: Tensor) -> Tensor:
        """Computes the matrix exponential of a square matrix.

        Args:
            input: The input tensor representing a square matrix.

        Returns:
            The matrix exponential of the input matrix.

        Raises:
            ValueError: If an error occurs while computing the matrix
                exponential.
        """
        try:
            return torch.linalg.matrix_exp(input)
        except RuntimeError as e:
            raise ValueError(f"Error computing matrix exponential: {str(e)}")

    @staticmethod
    def matrix_power(input: Tensor, n: int) -> Tensor:
        """Computes the n-th power of a square matrix for an integer n.

        Args:
            input: The input tensor representing a square matrix.
            n: The exponent.

        Returns:
            The n-th power of the input matrix.

        Raises:
            ValueError: If an error occurs while computing the matrix power.
        """
        try:
            return torch.linalg.matrix_power(input, n)
        except RuntimeError as e:
            raise ValueError(f"Error computing matrix power: {str(e)}")

    @staticmethod
    def cross(input: Tensor, other: Tensor,
              dim: int = DEFAULT_CROSS_DIM) -> Tensor:
        """Computes the cross product of two 3-dimensional vectors.

        Args:
            input: The first input tensor.
            other: The second input tensor.
            dim: The dimension to be considered as the last dimension.

        Returns:
            The cross product of the input vectors.

        Raises:
            ValueError: If an error occurs while computing the cross product.
        """
        try:
            return torch.cross(input, other, dim=dim)
        except RuntimeError as e:
            raise ValueError(f"Error computing cross product: {str(e)}")

    @staticmethod
    def matmul(input: Tensor, other: Tensor) -> Tensor:
        """Alias for `torch.matmul()`.

        Args:
            input: The first input tensor.
            other: The second input tensor.

        Returns:
            The matrix product of the input tensors.

        Raises:
            ValueError: If an error occurs while computing the matrix
                multiplication.
        """
        try:
            return torch.matmul(input, other)
        except RuntimeError as e:
            raise ValueError(f"Error computing matrix multiplication: "
                             f"{str(e)}")

    @staticmethod
    def vecdot(x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
        """Computes the dot product of two batches of vectors along a
        dimension.

        Args:
            x: The first input tensor.
            y: The second input tensor.
            dim: The dimension along which to compute the dot product.

        Returns:
            The dot product of the input vectors.

        Raises:
            ValueError: If an error occurs while computing the vector dot
                product.
        """
        try:
            return torch.vecdot(x, y, dim=dim)
        except RuntimeError as e:
            raise ValueError(f"Error computing vector dot product: {str(e)}")

    @staticmethod
    def multi_dot(tensors: List[Tensor]) -> Tensor:
        """Efficiently multiplies two or more matrices by reordering the
        multiplications.

        Args:
            tensors: A list of tensors to be multiplied.

        Returns:
            The product of the input tensors.

        Raises:
            ValueError: If an error occurs while computing the multi-dot
                product.
        """
        try:
            return torch.linalg.multi_dot(tensors)
        except RuntimeError as e:
            raise ValueError(f"Error computing multi-dot product: {str(e)}")

    @staticmethod
    def householder_product(input: Tensor, tau: Tensor) -> Tensor:
        """Computes the first n columns of a product of Householder matrices.

        Args:
            input: The input tensor.
            tau: The Householder reflectors.

        Returns:
            The product of the Householder matrices.

        Raises:
            ValueError: If an error occurs while computing the Householder
                product.
        """
        try:
            return torch.linalg.householder_product(input, tau)
        except RuntimeError as e:
            raise ValueError(f"Error computing Householder product: {str(e)}")

    @staticmethod
    def tensorinv(input: Tensor, ind: int = DEFAULT_TENSORINV_IND) -> Tensor:
        """Computes the multiplicative inverse of `torch.tensordot()`.

        Args:
            input: The input tensor.
            ind: The number of dimensions to contract.

        Returns:
            The inverse tensor.

        Raises:
            ValueError: If an error occurs while computing the tensor inverse.
        """
        try:
            return torch.linalg.tensorinv(input, ind=ind)
        except RuntimeError as e:
            raise ValueError(f"Error computing tensor inverse: {str(e)}")

    @staticmethod
    def tensorsolve(input: Tensor, other: Tensor,
                    dims: Optional[List[int]] = None) -> Tensor:
        """Computes the solution X to the system `torch.tensordot(A, X) = B`.

        Args:
            input: The tensor A in the equation.
            other: The tensor B in the equation.
            dims: The dimensions to contract over.

        Returns:
            The solution tensor X.

        Raises:
            ValueError: If an error occurs while solving the tensor equation.
        """
        try:
            return torch.linalg.tensorsolve(input, other, dims=dims)
        except RuntimeError as e:
            raise ValueError(f"Error solving tensor equation: {str(e)}")

    @staticmethod
    def vander(x: Tensor, N: Optional[int] = None,
               increasing: bool = DEFAULT_VANDER_INCREASING) -> Tensor:
        """Generates a Vandermonde matrix.

        Args:
            x: The input tensor.
            N: The number of columns in the output matrix.
            increasing: If True, the powers of x in each column are in
                increasing order.

        Returns:
            The Vandermonde matrix.

        Raises:
            ValueError: If an error occurs while generating the Vandermonde
                matrix.
        """
        try:
            return torch.vander(x, N=N, increasing=increasing)
        except RuntimeError as e:
            raise ValueError(f"Error generating Vandermonde matrix: {str(e)}")

    @staticmethod
    def cholesky_ex(input: Tensor, upper: bool = DEFAULT_CHOLESKY_UPPER,
                    check_errors: bool = DEFAULT_CHOLESKY_EX_CHECK_ERRORS
                    ) -> Tuple[Tensor, Tensor]:
        """Computes the Cholesky decomposition of a complex Hermitian or
        real symmetric positive-definite matrix.

        This function is similar to `torch.linalg.cholesky` but returns
        an additional info tensor.

        Args:
            input: The input tensor representing a matrix.
            upper: If True, returns the upper triangular Cholesky factor.
            check_errors: If True, performs error checking.

        Returns:
            A tuple containing the Cholesky factor and an info tensor.

        Raises:
            ValueError: If an error occurs while computing the Cholesky
                decomposition.
        """
        try:
            return torch.linalg.cholesky_ex(input, upper=upper,
                                            check_errors=check_errors)
        except RuntimeError as e:
            raise ValueError(f"Error computing Cholesky decomposition: "
                             f"{str(e)}")

    @staticmethod
    def lu_factor_ex(input: Tensor, pivot: bool = DEFAULT_LU_PIVOT,
                    check_errors: bool = DEFAULT_LU_FACTOR_EX_CHECK_ERRORS
                    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Computes a compact representation of the LU factorization with
        partial pivoting of a matrix.

        This function is similar to `torch.linalg.lu_factor` but returns
        an additional info tensor.

        Args:
            input: The input tensor representing a matrix.
            pivot: If True, performs partial pivoting.
            check_errors: If True, performs error checking.

        Returns:
            A tuple containing the combined L and U factors, the pivot
            indices, and an info tensor.

        Raises:
            ValueError: If an error occurs while computing the LU factorization.
        """
        try:
            return torch.linalg.lu_factor_ex(input, pivot=pivot,
                                            check_errors=check_errors)
        except RuntimeError as e:
            raise ValueError(f"Error computing LU factorization: {str(e)}")

    @staticmethod
    def ldl_factor(input: Tensor,
                   hermitian: bool = DEFAULT_LDL_FACTOR_HERMITIAN
                   ) -> Tuple[Tensor, Tensor, Tensor]:
        """Computes a compact representation of the LDL factorization of
        a Hermitian or symmetric matrix.

        Args:
            input: The input tensor representing a matrix.
            hermitian: If True, the input matrix is assumed to be Hermitian (or
                symmetric for real matrices).

        Returns:
            A tuple containing the combined L and D factors, and the pivot
            indices.

        Raises:
            ValueError: If an error occurs while computing the LDL
                factorization.
        """
        try:
            return torch.linalg.ldl_factor(input, hermitian=hermitian)
        except RuntimeError as e:
            raise ValueError(f"Error computing LDL factorization: {str(e)}")

    @staticmethod
    def ldl_factor_ex(input: Tensor,
                      hermitian: bool = DEFAULT_LDL_FACTOR_HERMITIAN,
                      check_errors: bool = DEFAULT_LDL_FACTOR_EX_CHECK_ERRORS
                      ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Computes a compact representation of the LDL factorization of
        a Hermitian or symmetric matrix.

        This function is similar to `torch.linalg.ldl_factor` but returns
        an additional info tensor.

        Args:
            input: The input tensor representing a matrix.
            hermitian: If True, the input matrix is assumed to be Hermitian (or
                symmetric for real matrices).
            check_errors: If True, performs error checking.

        Returns:
            A tuple containing the combined L and D factors, the pivot indices,
            and an info tensor.

        Raises:
            ValueError: If an error occurs while computing the LDL
                factorization.
        """
        try:
            return torch.linalg.ldl_factor_ex(input, hermitian=hermitian,
                                              check_errors=check_errors)
        except RuntimeError as e:
            raise ValueError(f"Error computing LDL factorization: {str(e)}")

    @staticmethod
    def ldl_solve(LD: Tensor, pivots: Tensor, B: Tensor,
                  hermitian: bool = DEFAULT_LDL_SOLVE_HERMITIAN) -> Tensor:
        """Computes the solution of a system of linear equations using
        the LDL factorization.

        Args:
            LD: The combined L and D factors from `torch.linalg.ldl_factor`.
            pivots: The pivot indices from `torch.linalg.ldl_factor`.
            B: The right-hand side tensor.
            hermitian: If True, the input matrix was assumed to be Hermitian (or
                symmetric for real matrices) during factorization.

        Returns:
            The solution tensor.

        Raises:
            ValueError: If an error occurs while solving the LDL system.
        """
        try:
            return torch.linalg.ldl_solve(LD, pivots, B, hermitian=hermitian)
        except RuntimeError as e:
            raise ValueError(f"Error solving LDL system: {str(e)}")

    @staticmethod
    def solve_ex(input: Tensor, other: Tensor,
                left: bool = DEFAULT_SOLVE_LEFT,
                check_errors: bool = DEFAULT_SOLVE_EX_CHECK_ERRORS
                ) -> Tuple[Tensor, Tensor]:
        """A version of `solve()` that does not perform error checks
        unless `check_errors=True`.

        Args:
            input: The input tensor representing the coefficient matrix.
            other: The right-hand side tensor.
            left: If True, solves Ax = b. If False, solves xA = b.
            check_errors: If True, performs error checking.

        Returns:
            A tuple containing the solution tensor and an info tensor.

        Raises:
            ValueError: If an error occurs while solving the linear system.
        """
        try:
            return torch.linalg.solve_ex(input, other, left=left,
                                        check_errors=check_errors)
        except RuntimeError as e:
            raise ValueError(f"Error solving linear system: {str(e)}")
    @staticmethod
    def inv_ex(input: Tensor,
               check_errors: bool = DEFAULT_INV_EX_CHECK_ERRORS
               ) -> Tuple[Tensor, Tensor]:
        """Computes the inverse of a square matrix if it is invertible.

        This function is similar to `torch.linalg.inv` but returns an
        additional info tensor.

        Args:
            input: The input tensor representing a square matrix.
            check_errors: If True, performs error checking.

        Returns:
            A tuple containing the inverse of the input matrix (if
            invertible) and an info tensor.

        Raises:
            ValueError: If an error occurs while computing the matrix inverse.
        """
        try:
            return torch.linalg.inv_ex(input, check_errors=check_errors)
        except RuntimeError as e:
            raise ValueError(f"Error computing matrix inverse: {str(e)}")

    # Additional utility methods

    @staticmethod
    def is_hermitian(input: Tensor,
                     atol: float = DEFAULT_IS_HERMITIAN_ATOL,
                     rtol: float = DEFAULT_IS_HERMITIAN_RTOL) -> bool:
        """Checks if a matrix is Hermitian (or symmetric for real matrices).

        Args:
            input: The input tensor representing a matrix.
            atol: The absolute tolerance parameter.
            rtol: The relative tolerance parameter.

        Returns:
            True if the matrix is Hermitian (or symmetric), False otherwise.
        """
        return torch.allclose(input, input.conj().transpose(-2, -1),
                              atol=atol, rtol=rtol)

    @staticmethod
    def is_positive_definite(input: Tensor) -> bool:
        """Checks if a matrix is positive definite.

        A matrix is positive definite if it is Hermitian (or symmetric) and
        all its eigenvalues are positive.

        Args:
            input: The input tensor representing a matrix.

        Returns:
            True if the matrix is positive definite, False otherwise.
        """
        try:
            torch.linalg.cholesky(input)
            return True
        except RuntimeError:
            return False

    @staticmethod
    def condition_number(input: Tensor,
                         p: Optional[Union[str, float]] = None) -> Tensor:
        """Computes the condition number of a matrix.

        The condition number of a matrix measures the sensitivity of the
        solution of a linear system to errors in the input data.

        Args:
            input: The input tensor representing a matrix.
            p: The order of the norm. For more details, refer to:
                https://pytorch.org/docs/stable/generated/torch.linalg.cond.html

        Returns:
            The condition number of the input matrix.
        """
        return LinearAlgebraOperations.cond(input, p=p)

    @staticmethod
    def frobenius_norm(input: Tensor) -> Tensor:
        """Computes the Frobenius norm of a matrix.

        Args:
            input: The input tensor representing a matrix.

        Returns:
            The Frobenius norm of the input matrix.
        """
        return LinearAlgebraOperations.matrix_norm(input, ord='fro')

    @staticmethod
    def trace(input: Tensor) -> Tensor:
        """Computes the trace of a matrix.

        The trace of a matrix is the sum of its diagonal elements.

        Args:
            input: The input tensor representing a matrix.

        Returns:
            The trace of the input matrix.
        """
        return torch.trace(input)

    @staticmethod
    def orthogonalize(input: Tensor,
                       eps: float = DEFAULT_ORTHOGONALIZE_EPS) -> Tensor:
        """Orthogonalizes a set of vectors using the Gram-Schmidt process.

        Args:
            input: The input tensor representing a set of vectors.
            eps: A small value added to the diagonal of the matrix to improve
                numerical stability.

        Returns:
            The orthogonalized set of vectors.
        """
        q, _ = torch.linalg.qr(input)
        return q

    @staticmethod
    def is_invertible(input: Tensor,
                      atol: float = DEFAULT_IS_INVERTIBLE_ATOL) -> bool:
        """Checks if a matrix is invertible.

        A matrix is invertible if its determinant is nonzero.

        Args:
            input: The input tensor representing a matrix.
            atol: The absolute tolerance parameter for checking if the
                determinant is nonzero.

        Returns:
            True if the matrix is invertible, False otherwise.
        """
        return LinearAlgebraOperations.matrix_rank(input, tol=atol) == \
               input.shape[-1]
