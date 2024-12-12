namespace AiDotNet.Regularization;

public class NoRegularization<T> : IRegularization<T>
{
    public Matrix<T> RegularizeMatrix(Matrix<T> featuresMatrix)
    {
        return featuresMatrix;
    }

    public Vector<T> RegularizeCoefficients(Vector<T> coefficients)
    {
        return coefficients;
    }
}