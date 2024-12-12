namespace AiDotNet.Normalizers;

public class RobustScalingNormalizer<T> : INormalizer<T>
{
    private readonly INumericOperations<T> _numOps;

    public RobustScalingNormalizer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T median = StatisticsHelper<T>.CalculateMedian(vector);
        T q1 = StatisticsHelper<T>.CalculateQuartile(vector, _numOps.FromDouble(0.25));
        T q3 = StatisticsHelper<T>.CalculateQuartile(vector, _numOps.FromDouble(0.75));
        T iqr = _numOps.Subtract(q3, q1);
        if (_numOps.Equals(iqr, _numOps.Zero)) iqr = _numOps.One;

        var normalizedVector = vector.Subtract(median).Divide(iqr);
        var parameters = new NormalizationParameters<T> { Median = median, IQR = iqr, Method = NormalizationMethod.RobustScaling };

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
        return vector.Multiply(parameters.IQR).Add(parameters.Median);
    }

    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        return coefficients.PointwiseMultiply(Vector<T>.FromArray([.. xParams.Select(p => _numOps.Divide(yParams.IQR, p.IQR))]));
    }

    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients, 
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        T denormalizedIntercept = yParams.Median;
        for (int i = 0; i < coefficients.Length; i++)
        {
            T term = _numOps.Multiply(coefficients[i], xParams[i].Median);
            term = _numOps.Multiply(term, _numOps.Divide(yParams.IQR, xParams[i].IQR));
            denormalizedIntercept = _numOps.Subtract(denormalizedIntercept, term);
        }

        return denormalizedIntercept;
    }
}