namespace AiDotNet.Normalizers;

public class GlobalContrastNormalizer<T> : INormalizer<T>
{
    private readonly INumericOperations<T> _numOps;

    public GlobalContrastNormalizer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T mean = vector.Average();
        T variance = vector.Select(x => _numOps.Multiply(_numOps.Subtract(x, mean), _numOps.Subtract(x, mean))).Average();
        T stdDev = _numOps.Sqrt(variance);

        var normalizedVector = vector.Transform(x => 
            _numOps.Add(
                _numOps.Divide(
                    _numOps.Subtract(x, mean),
                    _numOps.Multiply(_numOps.FromDouble(2), stdDev)
                ),
                _numOps.FromDouble(0.5)
            )
        );
        var parameters = new NormalizationParameters<T> { Mean = mean, StdDev = stdDev, Method = NormalizationMethod.GlobalContrast };

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
            .Transform(x => _numOps.Subtract(x, _numOps.FromDouble(0.5)))
            .Transform(x => _numOps.Multiply(x, _numOps.Multiply(_numOps.FromDouble(2), parameters.StdDev)))
            .Transform(x => _numOps.Add(x, parameters.Mean));
    }

    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        var scalingFactors = xParams.Select(p => 
            _numOps.Divide(
                _numOps.Multiply(_numOps.FromDouble(2), yParams.StdDev),
                _numOps.Multiply(_numOps.FromDouble(2), p.StdDev)
            )
        ).ToArray();
        return coefficients.PointwiseMultiply(Vector<T>.FromArray(scalingFactors));
    }

    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients, 
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        T denormalizedIntercept = _numOps.Subtract(
            yParams.Mean,
            _numOps.Multiply(
                _numOps.FromDouble(0.5),
                _numOps.Multiply(_numOps.FromDouble(2), yParams.StdDev)
            )
        );

        for (int i = 0; i < coefficients.Length; i++)
        {
            T term1 = _numOps.Multiply(
                xParams[i].Mean,
                _numOps.Divide(
                    _numOps.Multiply(_numOps.FromDouble(2), yParams.StdDev),
                    _numOps.Multiply(_numOps.FromDouble(2), xParams[i].StdDev)
                )
            );
            T term2 = _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Multiply(_numOps.FromDouble(2), yParams.StdDev));
            T difference = _numOps.Subtract(term1, term2);
            T product = _numOps.Multiply(coefficients[i], difference);
            denormalizedIntercept = _numOps.Subtract(denormalizedIntercept, product);
        }

        return denormalizedIntercept;
    }
}