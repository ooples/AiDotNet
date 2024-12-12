using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;
using System.Linq;

namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes the data by dividing each value by the smallest multiple of 10 that is greater than the largest value.
/// </summary>
public class DecimalNormalizer<T> : INormalizer<T>
{
    private readonly INumericOperations<T> _numOps;

    public DecimalNormalizer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T maxAbs = vector.AbsoluteMaximum();
        T scale = _numOps.One;
        T ten = _numOps.FromDouble(10);

        while (_numOps.GreaterThanOrEquals(maxAbs, scale))
        {
            scale = _numOps.Multiply(scale, ten);
        }

        var normalizedVector = vector.Transform(x => _numOps.Divide(x, scale));
        var parameters = new NormalizationParameters<T>
        {
            Method = NormalizationMethod.Decimal,
            Scale = scale
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
        return vector.Transform(x => _numOps.Multiply(x, parameters.Scale));
    }

    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        var scalingFactors = xParams.Select(p => _numOps.Divide(yParams.Scale, p.Scale)).ToArray();
        return coefficients.PointwiseMultiply(Vector<T>.FromArray(scalingFactors));
    }

    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients,
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        return _numOps.Zero; // The y-intercept for decimal normalization is always 0
    }
}