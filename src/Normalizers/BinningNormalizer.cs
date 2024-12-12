using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;

namespace AiDotNet.Normalizers;

public class BinningNormalizer<T> : INormalizer<T>
{
    private const int DefaultBinCount = 10;
    private readonly INumericOperations<T> _numOps;

    public BinningNormalizer()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector)
    {
        var sortedVector = vector.ToArray();
        Array.Sort(sortedVector);

        var bins = new List<T>();
        for (int i = 0; i <= DefaultBinCount; i++)
        {
            int index = _numOps.ToInt32(_numOps.Multiply(_numOps.FromDouble((double)i / DefaultBinCount), _numOps.FromDouble(sortedVector.Length - 1)));
            bins.Add(sortedVector[index]);
        }

        var normalizedVector = vector.Transform(x => 
        {
            int binIndex = bins.FindIndex(b => _numOps.LessThanOrEquals(x, b));
            return _numOps.Divide(_numOps.FromDouble(binIndex == -1 ? DefaultBinCount - 1 : binIndex), _numOps.FromDouble(DefaultBinCount - 1));
        });

        var parameters = new NormalizationParameters<T> { Method = NormalizationMethod.Binning, Bins = bins };
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
        if (parameters.Bins == null || parameters.Bins.Count == 0)
        {
            throw new ArgumentException("Invalid bin parameters. Bins list is null or empty.");
        }

        return vector.Transform(x => 
        {
            // Ensure x is within [0, 1] range
            var min = _numOps.LessThan(_numOps.One, x) ? _numOps.One : x;
            x = _numOps.GreaterThan(_numOps.Zero, min) ? _numOps.Zero : min;
        
            // Calculate the bin index
            int binIndex = _numOps.ToInt32(_numOps.Multiply(x, _numOps.FromDouble(parameters.Bins.Count - 1)));
        
            // Ensure binIndex is within valid range
            binIndex = Math.Max(0, Math.Min(parameters.Bins.Count - 2, binIndex));

            // Return the average of the current bin and the next bin
            return _numOps.Divide(_numOps.Add(parameters.Bins[binIndex], parameters.Bins[binIndex + 1]), _numOps.FromDouble(2));
        });
    }

    public Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        // Denormalizing coefficients for binning is complex and may not always be meaningful.
        // This is a simplified approach that may not be suitable for all use cases.
        return coefficients.PointwiseMultiply(Vector<T>.FromArray(xParams.Select((p, i) => 
            _numOps.Divide(
                _numOps.Subtract(yParams.Bins.Last(), yParams.Bins.First()),
                _numOps.Subtract(p.Bins.Last(), p.Bins.First())
            )).ToArray()));
    }

    public T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients, 
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams)
    {
        // Denormalizing y-intercept for binning is complex and may not always be meaningful.
        // This is a simplified approach that may not be suitable for all use cases.
        T denormalizedIntercept = _numOps.Divide(_numOps.Add(yParams.Bins.First(), yParams.Bins.Last()), _numOps.FromDouble(2));
        for (int i = 0; i < coefficients.Length; i++)
        {
            denormalizedIntercept = _numOps.Subtract(denormalizedIntercept, 
                _numOps.Multiply(coefficients[i], 
                    _numOps.Divide(_numOps.Add(xParams[i].Bins.First(), xParams[i].Bins.Last()), _numOps.FromDouble(2))));
        }

        return denormalizedIntercept;
    }
}