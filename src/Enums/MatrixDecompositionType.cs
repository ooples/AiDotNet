namespace AiDotNet.Enums;

/// <summary>
/// Different types of matrix decomposition. 
/// 
/// The decomposition is used to solve linear equations and to calculate the determinant and the inverse of a matrix.
/// 
/// The Cholesky decomposition is used for symmetric positive-definite matrices.
/// The Eigenvalue decomposition is used for non-symmetric matrices.
/// The Gram-Schmidt decomposition is used for non-symmetric matrices.
/// The LU decomposition is used for non-symmetric matrices.
/// The QR decomposition is used for non-symmetric matrices.
/// The SVD decomposition is used for non-symmetric matrices.
/// </summary>
public enum MatrixDecompositionType
{
    Cramer,
    Cholesky,
    GramSchmidt,
    Lu,
    Qr,
    Svd,
    Normal,
    Lq,
    Takagi,
    Hessenberg,
    Schur,
    Eigen
}