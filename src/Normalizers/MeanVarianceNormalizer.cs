namespace AiDotNet.Normalizers;

public class MeanVarianceNormalizer<T> : INormalizer<T>
{
    private readonly INumericOperations<T> _numOps;

    public MeanVarianceNormalizer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T mean = StatisticsHelper<T>.CalculateMean(vector);
        T variance = StatisticsHelper<T>.CalculateVariance(vector, mean);
        T stdDev = _numOps.Sqrt(variance);

        var normalizedVector = vector.Transform(x => _numOps.Divide(_numOps.Subtract(x, mean), stdDev));
        var parameters = new NormalizationParameters<T> { Mean = mean, StdDev = stdDev, Method = NormalizationMethod.MeanVariance };

        return (normalizedVector, parameters);
    }

    public (Matrix<T>, List<NormalizationParameters<T>>) NormalizeMatrix(Matrix<T> matrix)
    {
        var normalizedMatrix = Matrix<T>.CreateZeros(matrix.Rows, matrix.Columns);
        var parametersList = new List<NormalizationParameters<T>>();

        for (int i = 0; i < matrix.Columns; i++)
        {
            var column = matrix.GetColumn(i);
            var (normalizedColumn, parameters) = NormalizeVector(column);
            normalizedMatrix.SetColumn(i, normalizedColumn);
            parametersList.Add(parameters);
        }

        return (normalizedMatrix, parametersList);
    }

    public Vector<T> DenormalizeVector(Vector<T> vector, NormalizationParameters<T> parameters)
    {
        return vector.Multiply(parameters.StdDev).Add(parameters.Mean);
    }

    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        return coefficients.PointwiseMultiply(Vector<T>.FromArray(xParams.Select(p => _numOps.Divide(yParams.StdDev, p.StdDev)).ToArray()));
    }

    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients, 
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        T denormalizedIntercept = yParams.Mean;
        for (int i = 0; i < coefficients.Length; i++)
        {
            T term = _numOps.Multiply(coefficients[i], xParams[i].Mean);
            term = _numOps.Multiply(term, _numOps.Divide(yParams.StdDev, xParams[i].StdDev));
            denormalizedIntercept = _numOps.Subtract(denormalizedIntercept, term);
        }
        return denormalizedIntercept;
    }
}