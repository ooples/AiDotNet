namespace AiDotNet.Interfaces;

public interface IRegularization<T>
{
    Matrix<T> RegularizeMatrix(Matrix<T> featuresMatrix);
    Vector<T> RegularizeCoefficients(Vector<T> coefficients);
}