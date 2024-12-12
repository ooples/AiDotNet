namespace AiDotNet.Normalizers;

public class LogMeanVarianceNormalizer<T> : INormalizer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _epsilon;

    public LogMeanVarianceNormalizer(INumericOperations<T>? numOps = null)
    {
        _numOps = numOps ?? MathHelper.GetNumericOperations<T>();
        _epsilon = _numOps.FromDouble(1e-10);
    }

    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T minValue = vector.Min();
        T shift = _numOps.GreaterThan(minValue, _numOps.Zero) 
            ? _numOps.Zero
            : _numOps.Add(_numOps.Add(_numOps.Negate(minValue), _numOps.One), _epsilon);

        var logVector = vector.Transform(x => _numOps.Log(_numOps.Add(x, shift)));
        T mean = logVector.Average();
        
        T variance = logVector.Select(x => _numOps.Power(_numOps.Subtract(x, mean), _numOps.FromDouble(2))).Average();
        T stdDev = _numOps.Sqrt(_numOps.GreaterThan(variance, _epsilon) ? variance : _epsilon);

        var normalizedVector = logVector.Transform(x => _numOps.Divide(_numOps.Subtract(x, mean), stdDev));
        normalizedVector = normalizedVector.Transform(x => _numOps.IsNaN(x) ? _numOps.Zero : x);

        var parameters = new NormalizationParameters<T>(_numOps) 
        { 
            Method = NormalizationMethod.LogMeanVariance,
            Shift = shift, 
            Mean = mean, 
            StdDev = stdDev 
        };
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
        return vector
            .Multiply(parameters.StdDev)
            .Add(parameters.Mean)
            .Transform(x => _numOps.Subtract(_numOps.Exp(x), parameters.Shift));
    }

    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        return coefficients.PointwiseMultiply(
            Vector<T>.FromArray([.. xParams.Select(p => 
                _numOps.Divide(yParams.StdDev, _numOps.GreaterThan(p.StdDev, _epsilon) ? p.StdDev : _epsilon)
            )])
        );
    }

    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients, 
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        T denormalizedLogIntercept = yParams.Mean;
        for (int i = 0; i < coefficients.Length; i++)
        {
            denormalizedLogIntercept = _numOps.Subtract(denormalizedLogIntercept, 
                _numOps.Multiply(coefficients[i], 
                    _numOps.Divide(
                        _numOps.Multiply(xParams[i].Mean, yParams.StdDev), 
                        _numOps.GreaterThan(xParams[i].StdDev, _epsilon) ? xParams[i].StdDev : _epsilon
                    )
                )
            );
        }

        return _numOps.Subtract(_numOps.Exp(denormalizedLogIntercept), yParams.Shift);
    }
}