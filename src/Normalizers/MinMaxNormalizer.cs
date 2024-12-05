namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes the data by
/// 1) Subtracting the minimum value from each value
/// 2) Dividing each value from step #1 by the absolute difference between the maximum and minimum values
/// </summary>
public class MinMaxNormalizer : INormalizer
{
    public (Vector<double>, NormalizationParameters) NormalizeVector(Vector<double> vector)
    {
        double min = vector.Min();
        double max = vector.Max();
        var normalized = (vector - min) / (max - min);

        return (normalized, new NormalizationParameters { Method = NormalizationMethod.MinMax, Min = min, Max = max });
    }

    public (Matrix<double>, List<NormalizationParameters>) NormalizeMatrix(Matrix<double> matrix)
    {
        var normalizedColumns = new List<Vector<double>>();
        var parameters = new List<NormalizationParameters>();

        for (int i = 0; i < matrix.Columns; i++)
        {
            var column = matrix.GetColumn(i);
            var (normalizedColumn, param) = NormalizeVector(column);
            normalizedColumns.Add(normalizedColumn);
            parameters.Add(param);
        }

        return (Matrix<double>.FromColumnVectors(normalizedColumns), parameters);
    }

    public Vector<double> DenormalizeVector(Vector<double> vector, NormalizationParameters parameters)
    {
        return vector * (parameters.Max - parameters.Min) + parameters.Min;
    }

    public Vector<double> DenormalizeCoefficients(Vector<double> coefficients, List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        var denormalizedCoefficients = new double[coefficients.Length];
        for (int i = 0; i < coefficients.Length; i++)
        {
            denormalizedCoefficients[i] = coefficients[i] * (yParams.Max - yParams.Min) / (xParams[i].Max - xParams[i].Min);
        }

        return Vector<double>.FromArray(denormalizedCoefficients);
    }

    public double DenormalizeYIntercept(Matrix<double> x, Vector<double> y, Vector<double> coefficients, List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        double yIntercept = yParams.Min;
        for (int i = 0; i < coefficients.Length; i++)
        {
            yIntercept -= coefficients[i] * xParams[i].Min * (yParams.Max - yParams.Min) / (xParams[i].Max - xParams[i].Min);
        }

        return yIntercept;
    }
}