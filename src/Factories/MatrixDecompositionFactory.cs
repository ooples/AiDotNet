namespace AiDotNet.Factories;

public static class MatrixDecompositionFactory
{
    public static IMatrixDecomposition<T> CreateDecomposition<T>(Matrix<T> matrix, MatrixDecompositionType decompositionType)
    {
        return decompositionType switch
        {
            MatrixDecompositionType.Lu => new LuDecomposition<T>(matrix),
            MatrixDecompositionType.Qr => new QrDecomposition<T>(matrix),
            MatrixDecompositionType.Cholesky => new CholeskyDecomposition<T>(matrix),
            MatrixDecompositionType.Svd => new SvdDecomposition<T>(matrix),
            MatrixDecompositionType.Cramer => new CramerDecomposition<T>(matrix),
            MatrixDecompositionType.Eigen => new EigenDecomposition<T>(matrix),
            MatrixDecompositionType.Schur => new SchurDecomposition<T>(matrix),
            MatrixDecompositionType.Takagi => new TakagiDecomposition<T>(matrix),
            MatrixDecompositionType.Polar => new PolarDecomposition<T>(matrix),
            MatrixDecompositionType.Hessenberg => new HessenbergDecomposition<T>(matrix),
            MatrixDecompositionType.Tridiagonal => new TridiagonalDecomposition<T>(matrix),
            MatrixDecompositionType.Bidiagonal => new BidiagonalDecomposition<T>(matrix),
            MatrixDecompositionType.Ldl => new LdlDecomposition<T>(matrix),
            MatrixDecompositionType.Udu => new UduDecomposition<T>(matrix),
            _ => throw new ArgumentException($"Unsupported decomposition type: {decompositionType}")
        };
    }
}