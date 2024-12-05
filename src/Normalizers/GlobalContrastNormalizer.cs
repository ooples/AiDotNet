namespace AiDotNet.Normalizers;

public class GlobalContrastNormalizer : INormalizer
{
    public (Vector<double>, NormalizationParameters) NormalizeVector(Vector<double> vector)
    {
        double mean = vector.Average();
        double stdDev = Math.Sqrt(vector.Select(x => Math.Pow(x - mean, 2)).Average());

        var normalizedVector = vector.Transform(x => (x - mean) / (2 * stdDev) + 0.5);
        var parameters = new NormalizationParameters { Mean = mean, StdDev = stdDev, Method = NormalizationMethod.GlobalContrast };

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
        return vector.Subtract(0.5).Multiply(2 * parameters.StdDev).Add(parameters.Mean);
    }

    public Vector<double> DenormalizeCoefficients(Vector<double> coefficients, List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        return coefficients.PointwiseMultiply(Vector<double>.FromArray(xParams.Select(p => 2 * yParams.StdDev / (2 * p.StdDev)).ToArray()));
    }

    public double DenormalizeYIntercept(Matrix<double> xMatrix, Vector<double> y, Vector<double> coefficients, 
        List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        double denormalizedIntercept = yParams.Mean - 0.5 * (2 * yParams.StdDev);
        for (int i = 0; i < coefficients.Length; i++)
        {
            denormalizedIntercept -= coefficients[i] * (xParams[i].Mean * 2 * yParams.StdDev / (2 * xParams[i].StdDev) - 0.5 * 2 * yParams.StdDev);
        }
        return denormalizedIntercept;
    }
}