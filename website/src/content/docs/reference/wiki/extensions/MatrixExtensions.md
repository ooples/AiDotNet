---
title: "MatrixExtensions"
description: "Provides extension methods for matrix operations, making it easier to work with matrices in AI applications."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Extensions`

Provides extension methods for matrix operations, making it easier to work with matrices in AI applications.

## How It Works

**For Beginners:** A matrix is a rectangular array of numbers arranged in rows and columns.
These extension methods add useful functionality to matrices, like adding columns or performing
mathematical operations that are commonly needed in AI and machine learning algorithms.

## Methods

| Method | Summary |
|:-----|:--------|
| `AddColumn(Matrix<>,Vector<>)` | Adds a new column to the right side of a matrix. |
| `AddConstantColumn(Matrix<>,)` | Adds a constant value as the first column of a matrix. |
| `AddVectorToEachRow(Matrix<>,Vector<>)` | Adds a vector to each row of a matrix. |
| `BackwardSubstitution(Matrix<>,Vector<>)` | Performs backward substitution to solve a system of linear equations represented by an upper triangular matrix. |
| `ConjugateTranspose(Matrix<Complex<>>)` | Computes the conjugate transpose (also known as Hermitian transpose) of a complex matrix. |
| `CreateComplexMatrix(Matrix<Complex<>>,Int32,Int32)` | Creates a new complex matrix with the specified dimensions. |
| `Determinant(Matrix<>)` | Calculates the determinant of a square matrix. |
| `Eigenvalues(Matrix<>)` | Calculates the eigenvalues of a matrix using the QR algorithm. |
| `Extract(Matrix<>,Int32,Int32)` | Extracts a submatrix of specified dimensions from the top-left corner of the original matrix. |
| `Flatten(Matrix<>)` | Converts a two-dimensional matrix into a one-dimensional vector by placing all elements in a single row. |
| `ForwardSubstitution(Matrix<>,Vector<>)` | Solves a system of linear equations Ax = b using forward substitution, where A is a lower triangular matrix. |
| `FrobeniusNorm(Matrix<>)` | Calculates the Frobenius norm of a matrix. |
| `GetBlock(Matrix<>,Int32,Int32,Int32,Int32)` | Extracts a block (sub-matrix) from a matrix starting at the specified position. |
| `GetColumn(Matrix<>,Int32)` | Extracts a specific column from the matrix as a vector. |
| `GetColumnVectors(Matrix<>,Int32[])` | Gets specific column vectors from a matrix based on the specified indices. |
| `GetColumns(Matrix<>,IEnumerable<Int32>)` | Creates a new matrix containing only the specified columns from the original matrix. |
| `GetDeterminant(Matrix<>)` | Calculates the determinant of a matrix. |
| `GetMatrixTypes(Matrix<>,IMatrixDecomposition<>,,Int32,Int32,,,Int32,Int32)` | Identifies the types of a matrix based on its properties. |
| `GetNullity(Matrix<>,)` | Calculates the nullity of a matrix, which is the dimension of its null space. |
| `GetRange(Matrix<>,)` | Computes the range (column space) of a matrix. |
| `GetRank(Matrix<>,)` | Calculates the rank of a matrix based on a given threshold. |
| `GetRow(Matrix<>,Int32)` | Retrieves a specific row from the matrix as an array. |
| `GetRowRange(Matrix<>,Int32,Int32)` | Extracts a range of consecutive rows from the matrix. |
| `GetSubColumn(Matrix<>,Int32,Int32,Int32)` | Extracts a portion of a column from the matrix as a vector. |
| `Inverse(Matrix<>,InverseType,Int32,)` | Inverts a matrix using the specified algorithm. |
| `InverseGaussianJordanElimination(Matrix<>)` | Inverts a matrix using the Gaussian-Jordan elimination method. |
| `InverseNewton(Matrix<>,Int32,)` | Inverts a matrix using Newton's iterative method. |
| `InverseStrassen(Matrix<>)` | Inverts a matrix using Strassen's algorithm. |
| `InvertDiagonalMatrix(Matrix<>)` | Inverts a diagonal matrix. |
| `InvertLowerTriangularMatrix(Matrix<>)` | Inverts a lower triangular matrix. |
| `InvertUnitaryMatrix(Matrix<Complex<>>)` | Inverts a unitary matrix by taking its transpose. |
| `InvertUpperTriangularMatrix(Matrix<>)` | Inverts an upper triangular matrix. |
| `IsAdjacencyMatrix(Matrix<>)` | Determines if a matrix is an adjacency matrix representing a graph. |
| `IsBandMatrix(Matrix<>,Int32,Int32)` | Determines if a matrix is a band matrix with specified sub-diagonal and super-diagonal thresholds. |
| `IsBlockMatrix(Matrix<>,Int32,Int32)` | Determines if a matrix can be divided into consistent blocks of a specified size. |
| `IsCauchyMatrix(Matrix<>)` | Determines if a matrix is a Cauchy matrix. |
| `IsCirculantMatrix(Matrix<>)` | Determines if a matrix is a circulant matrix. |
| `IsCompanionMatrix(Matrix<>)` | Determines if a matrix is a companion matrix. |
| `IsConsistentBlock(Matrix<>)` | Checks if all elements in a matrix block are identical. |
| `IsDenseMatrix(Matrix<>,)` | Determines if a matrix is dense (contains mostly non-zero elements). |
| `IsDiagonalMatrix(Matrix<>)` | Determines if a matrix is diagonal (all non-diagonal elements are zero). |
| `IsDoublyStochasticMatrix(Matrix<>)` | Determines if a matrix is a doubly stochastic matrix. |
| `IsHankelMatrix(Matrix<>)` | Determines if a matrix is a Hankel matrix. |
| `IsHermitianMatrix(Matrix<Complex<>>)` | Determines if a complex matrix is Hermitian (equal to its conjugate transpose). |
| `IsHilbertMatrix(Matrix<>)` | Determines if a matrix is a Hilbert matrix. |
| `IsIdempotentMatrix(Matrix<>)` | Determines if a matrix is idempotent (equal to its own square). |
| `IsIdentityMatrix(Matrix<>)` | Determines if a matrix is an identity matrix (diagonal elements are 1, all others are 0). |
| `IsIncidenceMatrix(Matrix<>)` | Determines if a matrix is an incidence matrix for a graph. |
| `IsInvertible(Matrix<>)` | Determines whether a matrix is invertible. |
| `IsInvolutoryMatrix(Matrix<>)` | Determines if a matrix is an involutory matrix. |
| `IsLaplacianMatrix(Matrix<>)` | Determines if a matrix is a Laplacian matrix. |
| `IsLowerBidiagonalMatrix(Matrix<>)` | Determines if a matrix is a lower bidiagonal matrix. |
| `IsLowerTriangularMatrix(Matrix<>,)` | Determines if a matrix is lower triangular (all elements above the main diagonal are zero). |
| `IsNonSingularMatrix(Matrix<>)` | Determines if a matrix is non-singular (invertible). |
| `IsOrthogonalMatrix(Matrix<>,IMatrixDecomposition<>)` | Determines if a matrix is orthogonal (its transpose equals its inverse). |
| `IsOrthogonalProjectionMatrix(Matrix<>)` | Determines if a matrix is an orthogonal projection matrix. |
| `IsPartitionedMatrix(Matrix<>)` | Determines if a matrix can be considered a partitioned matrix. |
| `IsPermutationMatrix(Matrix<>)` | Determines if a matrix is a permutation matrix. |
| `IsPositiveDefiniteMatrix(Matrix<>,)` | Determines if a matrix is positive definite. |
| `IsPositiveSemiDefiniteMatrix(Matrix<>)` | Determines if a matrix is positive semi-definite. |
| `IsRectangularMatrix(Matrix<>)` | Determines if a matrix is rectangular (has a different number of rows and columns). |
| `IsScalarMatrix(Matrix<>)` | Determines if a matrix is a scalar matrix (diagonal elements are equal, all others are 0). |
| `IsSingularMatrix(Matrix<>)` | Determines if a matrix is singular (non-invertible). |
| `IsSkewHermitianMatrix(Matrix<Complex<>>)` | Determines if a complex matrix is skew-Hermitian (equal to the negative of its conjugate transpose). |
| `IsSkewSymmetricMatrix(Matrix<>)` | Determines if a matrix is skew-symmetric (equal to the negative of its transpose). |
| `IsSparseMatrix(Matrix<>,)` | Determines if a matrix is sparse (contains mostly zero elements). |
| `IsSquareMatrix(Matrix<>)` | Determines if a matrix is square (has the same number of rows and columns). |
| `IsStochasticMatrix(Matrix<>)` | Determines if a matrix is stochastic (each row sums to 1 and all elements are non-negative). |
| `IsSymmetricMatrix(Matrix<>)` | Determines if a matrix is symmetric (equal to its transpose). |
| `IsToeplitzMatrix(Matrix<>)` | Determines if a matrix is a Toeplitz matrix. |
| `IsTridiagonalMatrix(Matrix<>)` | Determines if a matrix is a tridiagonal matrix. |
| `IsUnitaryMatrix(Matrix<Complex<>>,IMatrixDecomposition<Complex<>>)` | Determines if a matrix is unitary by checking if its conjugate transpose equals its inverse. |
| `IsUpperBidiagonalMatrix(Matrix<>)` | Determines if a matrix is upper bidiagonal (non-zero elements only on main diagonal and first superdiagonal). |
| `IsUpperTriangularMatrix(Matrix<>,)` | Determines if a matrix is upper triangular (all elements below the main diagonal are zero). |
| `IsVandermondeMatrix(Matrix<>)` | Determines if a matrix is a Vandermonde matrix. |
| `IsZeroMatrix(Matrix<>)` | Determines if a matrix contains only zero values. |
| `KroneckerProduct(Matrix<>,Matrix<>)` | Computes the Kronecker product of two matrices. |
| `LogDeterminant(Matrix<>)` | Calculates the logarithm of the determinant of a matrix. |
| `MatrixExponential(Matrix<>,Int32)` | Computes the matrix exponential e^A using a truncated Taylor series. |
| `MatrixPower(Matrix<>,Int32)` | Computes the matrix power A^k by repeated multiplication. |
| `Max(Matrix<>,Func<,>)` | Finds the maximum value in the matrix after applying a transformation function to each element. |
| `Negate(Matrix<>)` | Creates a new matrix with all elements negated. |
| `Nullspace(Matrix<>,)` | Computes the null space (kernel) of a matrix. |
| `PointwiseMultiply(Matrix<>,Matrix<>)` | Performs element-by-element multiplication of two matrices of the same dimensions. |
| `PointwiseMultiply(Matrix<>,Vector<>)` | Multiplies each row of a matrix by the corresponding element in a vector. |
| `Reshape(Matrix<>,Int32,Int32)` | Reorganizes the elements of a matrix into a new matrix with different dimensions while preserving all data. |
| `RowWiseArgmax(Matrix<>)` | For each row in the matrix, finds the index of the column containing the maximum value. |
| `SetSubmatrix(Matrix<>,Int32,Int32,Matrix<>)` | Sets a submatrix within a larger matrix. |
| `Submatrix(Matrix<>,Int32,Int32,Int32,Int32)` | Extracts a submatrix from the original matrix. |
| `Submatrix(Matrix<>,Int32[])` | Creates a submatrix from the given matrix using the specified indices. |
| `SumColumns(Matrix<>)` | Calculates the sum of each column in the matrix. |
| `SwapRows(Matrix<>,Int32,Int32)` | Swaps two rows in a matrix. |
| `ToComplexMatrix(Matrix<>)` | Converts a real-valued matrix to a complex-valued matrix. |
| `ToComplexVector(Vector<>)` | Converts a real-valued vector to a complex-valued vector. |
| `ToRealMatrix(Matrix<Complex<>>)` | Extracts the real part of a complex-valued matrix to create a real-valued matrix. |
| `ToVector(Matrix<>)` | Converts a matrix to a vector by flattening its elements in row-major order. |
| `Trace(Matrix<>)` | Computes the trace of a square matrix (sum of diagonal elements). |
| `Transpose(Matrix<>)` | Transposes a matrix by swapping its rows and columns. |

