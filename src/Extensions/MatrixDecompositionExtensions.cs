using AiDotNet.DecompositionMethods;

namespace AiDotNet.Extensions;

public static class MatrixDecompositionExtensions
{
    public static IMatrixDecomposition<Complex<T>> ToComplexDecomposition<T>(this IMatrixDecomposition<T> decomposition)
    {
        return new ComplexMatrixDecomposition<T>(decomposition);
    }
}