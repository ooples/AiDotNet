namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes the data by taking the natural log of each value.
/// </summary>
public class LogNormalizer : INormalizer
{
    public (Vector<double>, NormalizationParameters) NormalizeVector(Vector<double> vector)
    {
        double min = vector.Min();
        double max = vector.Max();
        double shift = min > 0 ? 0 : -min + 1;

        var normalizedVector = vector.Transform(x =>
        {
            double shiftedValue = x + shift;
            return shiftedValue > 0 ? (Math.Log(shiftedValue) - Math.Log(min + shift)) / (Math.Log(max + shift) - Math.Log(min + shift)) : 0;
        });

        var parameters = new NormalizationParameters
        {
            Method = NormalizationMethod.Log,
            Min = min,
            Max = max,
            Shift = shift
        };

        return (normalizedVector, parameters);
    }

    public (Matrix<double>, List<NormalizationParameters>) NormalizeMatrix(Matrix<double> matrix)
    {
        var normalizedColumns = new List<Vector<double>>();
        var parametersList = new List<NormalizationParameters>();

        for (int i = 0; i < matrix.Columns; i++)
        {
            var column = matrix.GetColumn(i);
            var (normalizedColumn, parameters) = NormalizeVector(column);
            normalizedColumns.Add(normalizedColumn);
            parametersList.Add(parameters);
        }

        var normalizedMatrix = Matrix<double>.FromColumnVectors(normalizedColumns);
        return (normalizedMatrix, parametersList);
    }

    public Vector<double> DenormalizeVector(Vector<double> normalizedVector, NormalizationParameters parameters)
    {
        return normalizedVector.Transform(x =>
        {
            double expValue = Math.Exp(x * (Math.Log(parameters.Max + parameters.Shift) - Math.Log(parameters.Min + parameters.Shift))
                + Math.Log(parameters.Min + parameters.Shift));
            return expValue - parameters.Shift;
        });
    }

    public Vector<double> DenormalizeCoefficients(Vector<double> coefficients, List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        return coefficients.PointwiseMultiply(Vector<double>.FromEnumerable(
            xParams.Select(p => (Math.Log(yParams.Max + yParams.Shift) - Math.Log(yParams.Min + yParams.Shift)) /
                                (Math.Log(p.Max + p.Shift) - Math.Log(p.Min + p.Shift)))));
    }

    public double DenormalizeYIntercept(Matrix<double> xMatrix, Vector<double> y, Vector<double> coefficients,
        List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        var denormalizedCoefficients = DenormalizeCoefficients(coefficients, xParams, yParams);
        var meanX = Vector<double>.FromEnumerable(xMatrix.EnumerateColumns().Select(col => col.Mean()));
        var meanY = y.Mean();

        double intercept = meanY;
        for (int i = 0; i < coefficients.Length; i++)
        {
            intercept -= denormalizedCoefficients[i] * meanX[i];
        }

        return intercept;
    }
}