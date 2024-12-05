namespace AiDotNet.Normalizers;

public class NoNormalizer : INormalizer
{
    public (Vector<double>, NormalizationParameters) NormalizeVector(Vector<double> vector)
    {
        return (vector, new NormalizationParameters { Method = NormalizationMethod.None });
    }

    public (Matrix<double>, List<NormalizationParameters>) NormalizeMatrix(Matrix<double> matrix)
    {
        var parameters = Enumerable.Repeat(new NormalizationParameters { Method = NormalizationMethod.None }, matrix.Columns).ToList();
        return (matrix, parameters);
    }

    public Vector<double> DenormalizeVector(Vector<double> vector, NormalizationParameters parameters)
    {
        return vector;
    }

    public Vector<double> DenormalizeCoefficients(Vector<double> coefficients, List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        return coefficients;
    }

    public double DenormalizeYIntercept(Matrix<double> xMatrix, Vector<double> y, Vector<double> coefficients, List<NormalizationParameters> xParams, NormalizationParameters yParams)
    {
        return MathHelper.CalculateYIntercept(xMatrix, y, coefficients);
    }
}