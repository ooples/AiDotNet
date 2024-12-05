namespace AiDotNet.Normalizers;

/// <summary>
/// Normalizes the data by dividing each value by the smallest multiple of 10 that is greater than the largest value.
/// </summary>
public class DecimalNormalizer : INormalizer
{
    public (Vector<double>, NormalizationParameters) NormalizeVector(Vector<double> vector)
    {
        double maxAbs = vector.AbsoluteMaximum();
        double scale = Math.Pow(10, Math.Floor(Math.Log10(maxAbs)) + 1);

        var normalizedVector = vector.Divide(scale);
        var parameters = new NormalizationParameters
        {
            Method = NormalizationMethod.Decimal,
            Scale = scale
        };
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
        return vector.Multiply(parameters.Scale);
    }

    public Vector<double> DenormalizeCoefficients(Vector<double> coefficients, List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        return coefficients.PointwiseMultiply(Vector<double>.FromArray(xParams.Select(p => yParams.Scale / p.Scale).ToArray()));
    }

    public double DenormalizeYIntercept(Matrix<double> xMatrix, Vector<double> y, Vector<double> coefficients,
        List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        return 0; // The y-intercept for decimal normalization is always 0
    }
}