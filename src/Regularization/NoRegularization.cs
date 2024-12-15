namespace AiDotNet.Regularization;

public class NoRegularization<T> : IRegularization<T>
{
    public NoRegularization()
    {
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