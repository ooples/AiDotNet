namespace AiDotNet.Interfaces;

public interface INormalizer
{
    (Vector<double>, NormalizationParameters) NormalizeVector(Vector<double> vector);
    (Matrix<double>, List<NormalizationParameters>) NormalizeMatrix(Matrix<double> matrix);
    Vector<double> DenormalizeVector(Vector<double> vector, NormalizationParameters parameters);
    Vector<double> DenormalizeCoefficients(Vector<double> coefficients, List<NormalizationParameters> xParams, NormalizationParameters yParams);
    double DenormalizeYIntercept(Matrix<double> xMatrix, Vector<double> y, Vector<double> coefficients, 
        List<NormalizationParameters> xParams, NormalizationParameters yParams);
}