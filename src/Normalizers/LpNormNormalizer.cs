using AiDotNet.Helpers;
using AiDotNet.NumericOperations;
using System;
using System.Linq;

namespace AiDotNet.Normalizers;

public class LpNormNormalizer<T> : INormalizer<T>
{
    private readonly T _p;
    private readonly INumericOperations<T> _numOps;

    public LpNormNormalizer(T p)
    {
        _p = p;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        T sum = vector.Select(x => _numOps.Power(_numOps.Abs(x), _p)).Aggregate(_numOps.Zero, _numOps.Add);
        T norm = _numOps.Power(sum, _numOps.Divide(_numOps.One, _p));

        var normalizedVector = vector.Transform(x => _numOps.Divide(x, norm));
        var parameters = new NormalizationParameters<T> { Scale = norm, P = _p, Method = NormalizationMethod.LpNorm };
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
        return _numOps.Zero;
    }
}