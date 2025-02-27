namespace AiDotNet.Normalizers;

public class NoNormalizer<T> : INormalizer<T>
{
    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        return (vector, new NormalizationParameters<T> { Method = NormalizationMethod.None });
    }

    public (Matrix<T>, List<NormalizationParameters<T>>) NormalizeMatrix(Matrix<T> matrix)
    {
        var parameters = Enumerable.Repeat(new NormalizationParameters<T> { Method = NormalizationMethod.None }, matrix.Columns).ToList();
        return (matrix, parameters);
    }

    public Vector<T> DenormalizeVector(Vector<T> vector, NormalizationParameters<T> parameters)
    {
        return vector;
    }

    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        return coefficients;
    }

    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        return MathHelper.CalculateYIntercept(xMatrix, y, coefficients);
    }
}