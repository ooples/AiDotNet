namespace AiDotNet.Regularization;

public class NoRegularization<T> : IRegularization<T>
{
    public NoRegularization(INumericOperations<T> numOps, RegularizationOptions options)
    {
        // No need to store these as we don't use them
    }

    public Matrix<T> RegularizeMatrix(Matrix<T> matrix)
    {
        return matrix;
    }

    public Vector<T> RegularizeCoefficients(Vector<T> coefficients)
    {
        return coefficients;
    }
}