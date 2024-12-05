namespace AiDotNet.Normalizers;

public class MeanVarianceNormalizer : INormalizer
{
    public (Vector<double>, NormalizationParameters) NormalizeVector(Vector<double> vector)
    {
        double mean = vector.Average();
        double variance = vector.Select(x => Math.Pow(x - mean, 2)).Average();
        double stdDev = Math.Sqrt(variance);

        var normalizedVector = vector.Transform(x => (x - mean) / stdDev);
        var parameters = new NormalizationParameters { Mean = mean, StdDev = stdDev, Method = NormalizationMethod.MeanVariance };

        return (normalizedVector, parameters);
    }

    public (Matrix<double>, List<NormalizationParameters>) NormalizeMatrix(Matrix<double> matrix)
    {
        var normalizedMatrix = Matrix<double>.CreateZeros(matrix.Rows, matrix.Columns);
        var parametersList = new List<NormalizationParameters>();

        for (int i = 0; i < matrix.Columns; i++)
        {
            var column = matrix.GetColumn(i);
            var (normalizedColumn, parameters) = NormalizeVector(column);
            normalizedMatrix.SetColumn(i, normalizedColumn);
            parametersList.Add(parameters);
        }

        return (normalizedMatrix, parametersList);
    }

    public Vector<double> DenormalizeVector(Vector<double> vector, NormalizationParameters parameters)
    {
        return vector.Multiply(parameters.StdDev).Add(parameters.Mean);
    }

    public Vector<double> DenormalizeCoefficients(Vector<double> coefficients, List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        return coefficients.PointwiseMultiply(Vector<double>.FromArray(xParams.Select(p => yParams.StdDev / p.StdDev).ToArray()));
    }

    public double DenormalizeYIntercept(Matrix<double> xMatrix, Vector<double> y, Vector<double> coefficients, 
        List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        double denormalizedIntercept = yParams.Mean;
        for (int i = 0; i < coefficients.Length; i++)
        {
            denormalizedIntercept -= coefficients[i] * (xParams[i].Mean * yParams.StdDev / xParams[i].StdDev);
        }
        return denormalizedIntercept;
    }
}