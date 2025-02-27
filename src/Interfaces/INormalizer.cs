namespace AiDotNet.Interfaces;

public interface INormalizer<T>
{
    (Vector<T>, NormalizationParameters<T>) NormalizeVector(Vector<T> vector);
    (Matrix<T>, List<NormalizationParameters<T>>) NormalizeMatrix(Matrix<T> matrix);
    Vector<T> DenormalizeVector(Vector<T> vector, NormalizationParameters<T> parameters);
    Vector<T> DenormalizeCoefficients(Vector<T> coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams);
    T DenormalizeYIntercept(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients, 
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams);
}