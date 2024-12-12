namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes the data by
/// 1) Subtracting the minimum value from each value
/// 2) Dividing each value from step #1 by the absolute difference between the maximum and minimum values
/// </summary>
public class MinMaxNormalizer<T> : INormalizer<T>
{
    private readonly INumericOperations<T> _numOps;

    public MinMaxNormalizer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T min = vector.Min();
        T max = vector.Max();
        var normalized = vector.Transform(x => _numOps.Divide(_numOps.Subtract(x, min), _numOps.Subtract(max, min)));

        return (normalized, new NormalizationParameters<T> { Method = NormalizationMethod.MinMax, Min = min, Max = max });
    }

    public (Matrix<T>, List<NormalizationParameters<T>>) NormalizeMatrix(Matrix<T> matrix)
    {
        var normalizedColumns = new List<Vector<T>>();
        var parameters = new List<NormalizationParameters<T>>();

        for (int i = 0; i < matrix.Columns; i++)
        {
            var column = matrix.GetColumn(i);
            var (normalizedColumn, param) = NormalizeVector(column);
            normalizedColumns.Add(normalizedColumn);
            parameters.Add(param);
        }

        return (Matrix<T>.FromColumnVectors(normalizedColumns), parameters);
    }

    public Vector<T> DenormalizeVector(Vector<T> vector, NormalizationParameters<T> parameters)
    {
        return vector.Transform(x => _numOps.Add(_numOps.Multiply(x, _numOps.Subtract(parameters.Max, parameters.Min)), parameters.Min));
    }

    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        var denormalizedCoefficients = new T[coefficients.Length];
        for (int i = 0; i < coefficients.Length; i++)
        {
            denormalizedCoefficients[i] = _numOps.Divide(
                _numOps.Multiply(coefficients[i], _numOps.Subtract(yParams.Max, yParams.Min)),
                _numOps.Subtract(xParams[i].Max, xParams[i].Min)
            );
        }

        return Vector<T>.FromArray(denormalizedCoefficients);
    }

    public T DenormalizeYIntercept(Matrix<T> x, Vector<T> y, Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        T yIntercept = yParams.Min;
        for (int i = 0; i < coefficients.Length; i++)
        {
            yIntercept = _numOps.Subtract(yIntercept, 
                _numOps.Divide(
                    _numOps.Multiply(
                        _numOps.Multiply(coefficients[i], xParams[i].Min),
                        _numOps.Subtract(yParams.Max, yParams.Min)
                    ),
                    _numOps.Subtract(xParams[i].Max, xParams[i].Min)
                )
            );
        }

        return yIntercept;
    }
}