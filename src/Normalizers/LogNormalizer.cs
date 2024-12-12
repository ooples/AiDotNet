namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes the data by taking the natural log of each value.
/// </summary>
public class LogNormalizer<T> : INormalizer<T>
{
    private readonly INumericOperations<T> _numOps;

    public LogNormalizer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T min = vector.Min();
        T max = vector.Max();
        T shift = _numOps.GreaterThan(min, _numOps.Zero) ? _numOps.Zero : _numOps.Add(_numOps.Negate(min), _numOps.One);

        var normalizedVector = vector.Transform(x =>
        {
            T shiftedValue = _numOps.Add(x, shift);
            return _numOps.GreaterThan(shiftedValue, _numOps.Zero) 
                ? _numOps.Divide(
                    _numOps.Subtract(_numOps.Log(shiftedValue), _numOps.Log(_numOps.Add(min, shift))),
                    _numOps.Subtract(_numOps.Log(_numOps.Add(max, shift)), _numOps.Log(_numOps.Add(min, shift))))
                : _numOps.Zero;
        });

        var parameters = new NormalizationParameters<T>
        {
            Method = NormalizationMethod.Log,
            Min = min,
            Max = max,
            Shift = shift
        };

        return (normalizedVector, parameters);
    }

    public (Matrix<T>, List<NormalizationParameters<T>>) NormalizeMatrix(Matrix<T> matrix)
    {
        var normalizedColumns = new List<Vector<T>>();
        var parametersList = new List<NormalizationParameters<T>>();

        for (int i = 0; i < matrix.Columns; i++)
        {
            var column = matrix.GetColumn(i);
            var (normalizedColumn, parameters) = NormalizeVector(column);
            normalizedColumns.Add(normalizedColumn);
            parametersList.Add(parameters);
        }

        var normalizedMatrix = Matrix<T>.FromColumnVectors(normalizedColumns);
        return (normalizedMatrix, parametersList);
    }

    public Vector<T> DenormalizeVector(Vector<T> normalizedVector, NormalizationParameters<T> parameters)
    {
        return normalizedVector.Transform(x =>
        {
            T expValue = _numOps.Exp(_numOps.Add(
                _numOps.Multiply(x, _numOps.Subtract(
                    _numOps.Log(_numOps.Add(parameters.Max, parameters.Shift)),
                    _numOps.Log(_numOps.Add(parameters.Min, parameters.Shift)))),
                _numOps.Log(_numOps.Add(parameters.Min, parameters.Shift))));
            return _numOps.Subtract(expValue, parameters.Shift);
        });
    }

    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        return coefficients.PointwiseMultiply(Vector<T>.FromEnumerable(
            xParams.Select(p => _numOps.Divide(
                _numOps.Subtract(_numOps.Log(_numOps.Add(yParams.Max, yParams.Shift)), _numOps.Log(_numOps.Add(yParams.Min, yParams.Shift))),
                _numOps.Subtract(_numOps.Log(_numOps.Add(p.Max, p.Shift)), _numOps.Log(_numOps.Add(p.Min, p.Shift)))))));
    }

    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients,
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        var denormalizedCoefficients = DenormalizeCoefficients(coefficients, xParams, yParams);
        var meanX = Vector<T>.FromEnumerable(xMatrix.EnumerateColumns().Select(col => col.Mean()));
        var meanY = y.Mean();

        T intercept = meanY;
        for (int i = 0; i < coefficients.Length; i++)
        {
            intercept = _numOps.Subtract(intercept, _numOps.Multiply(denormalizedCoefficients[i], meanX[i]));
        }

        return intercept;
    }
}