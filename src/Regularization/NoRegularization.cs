namespace AiDotNet.Regularization;

public class NoRegularization : IRegularization
{
    public Matrix<double> RegularizeMatrix(Matrix<double> featuresMatrix)
    {
        return featuresMatrix;
    }

    public Vector<double> RegularizeCoefficients(Vector<double> coefficients)
    {
        return coefficients;
    }
}