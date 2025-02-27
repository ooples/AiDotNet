namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes the data by subtracting the mean from each value and dividing by the standard deviation.
/// </summary>
public class ZScoreNormalizer<T> : INormalizer<T>
{
    private readonly INumericOperations<T> _numOps;

    public ZScoreNormalizer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T mean = StatisticsHelper<T>.CalculateMean(vector);
        T variance = StatisticsHelper<T>.CalculateVariance(vector, mean);
        T stdDev = _numOps.Sqrt(variance);

        Vector<T> normalizedVector = vector.Transform(x => _numOps.Divide(_numOps.Subtract(x, mean), stdDev));

        return (normalizedVector, new NormalizationParameters<T> { Method = NormalizationMethod.ZScore, Mean = mean, StdDev = stdDev });
    }

    public (Matrix<T>, List<NormalizationParameters<T>>) NormalizeMatrix(Matrix<T> matrix)
    {
        var normalizedColumns = new List<Vector<T>>();
        var parameters = new List<NormalizationParameters<T>>();

        for (int i = 0; i < matrix.Columns; i++)
        {
            var (normalizedColumn, columnParams) = NormalizeVector(matrix.GetColumn(i));
            normalizedColumns.Add(normalizedColumn);
            parameters.Add(columnParams);
        }

        return (Matrix<T>.FromColumnVectors(normalizedColumns), parameters);
    }

    public Vector<T> DenormalizeVector(Vector<T> vector, NormalizationParameters<T> parameters)
    {
        return vector.Transform(x => _numOps.Add(_numOps.Multiply(x, parameters.StdDev), parameters.Mean));
    }

    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        return coefficients.PointwiseMultiply(Vector<T>.FromEnumerable(xParams.Select(p => _numOps.Divide(p.StdDev, yParams.StdDev))));
    }

    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients, 
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        T yMean = yParams.Mean;
        var xMeans = Vector<T>.FromEnumerable(xParams.Select(p => p.Mean));
        var denormalizedCoefficients = DenormalizeCoefficients(coefficients, xParams, yParams);

        return _numOps.Subtract(yMean, xMeans.DotProduct(denormalizedCoefficients));
    }
}