namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes the data by subtracting the mean from each value and dividing by the standard deviation.
/// </summary>
public class ZScoreNormalizer : INormalizer
{
    public (Vector<double>, NormalizationParameters) NormalizeVector(Vector<double> vector)
    {
        double mean = vector.Average();
        double stdDev = Math.Sqrt(vector.Select(x => Math.Pow(x - mean, 2)).Average());
    
        Vector<double> normalizedVector = vector.Transform(x => (x - mean) / stdDev);
    
        return (normalizedVector, new NormalizationParameters { Method = NormalizationMethod.ZScore, Mean = mean, StdDev = stdDev });
    }

    public (Matrix<double>, List<NormalizationParameters>) NormalizeMatrix(Matrix<double> matrix)
    {
        var normalizedColumns = new List<Vector<double>>();
        var parameters = new List<NormalizationParameters>();

        for (int i = 0; i < matrix.Columns; i++)
        {
            var (normalizedColumn, columnParams) = NormalizeVector(matrix.GetColumn(i));
            normalizedColumns.Add(normalizedColumn);
            parameters.Add(columnParams);
        }

        return (Matrix<double>.FromColumnVectors(normalizedColumns), parameters);
    }

    public Vector<double> DenormalizeVector(Vector<double> vector, NormalizationParameters parameters)
    {
        return vector.Transform(x => x * parameters.StdDev + parameters.Mean);
    }

    public Vector<double> DenormalizeCoefficients(Vector<double> coefficients, List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        return coefficients.PointwiseMultiply(Vector<double>.FromEnumerable(xParams.Select(p => p.StdDev / yParams.StdDev)));
    }

    public double DenormalizeYIntercept(Matrix<double> xMatrix, Vector<double> y, Vector<double> coefficients, 
        List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        double yMean = yParams.Mean;
        var xMeans = Vector<double>.FromEnumerable(xParams.Select(p => p.Mean));
        var denormalizedCoefficients = DenormalizeCoefficients(coefficients, xParams, yParams);

        return yMean - xMeans.DotProduct(denormalizedCoefficients);
    }
}