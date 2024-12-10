namespace AiDotNet.Interfaces;

public interface IRegularization
{
    Matrix<double> RegularizeMatrix(Matrix<double> featuresMatrix);
    Vector<double> RegularizeCoefficients(Vector<double> coefficients);
}