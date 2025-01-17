using AiDotNet.DecompositionMethods.MatrixDecomposition;

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

    public static MatrixDecompositionType GetDecompositionType<T>(IMatrixDecomposition<T>? decomposition)
    {
        return decomposition switch
        {
            LuDecomposition<T> => MatrixDecompositionType.Lu,
            QrDecomposition<T> => MatrixDecompositionType.Qr,
            CholeskyDecomposition<T> => MatrixDecompositionType.Cholesky,
            SvdDecomposition<T> => MatrixDecompositionType.Svd,
            CramerDecomposition<T> => MatrixDecompositionType.Cramer,
            EigenDecomposition<T> => MatrixDecompositionType.Eigen,
            SchurDecomposition<T> => MatrixDecompositionType.Schur,
            TakagiDecomposition<T> => MatrixDecompositionType.Takagi,
            PolarDecomposition<T> => MatrixDecompositionType.Polar,
            HessenbergDecomposition<T> => MatrixDecompositionType.Hessenberg,
            TridiagonalDecomposition<T> => MatrixDecompositionType.Tridiagonal,
            BidiagonalDecomposition<T> => MatrixDecompositionType.Bidiagonal,
            LdlDecomposition<T> => MatrixDecompositionType.Ldl,
            UduDecomposition<T> => MatrixDecompositionType.Udu,
            _ => throw new ArgumentException($"Unsupported decomposition type: {decomposition?.GetType().Name}")
        };
    }
}