namespace AiDotNet.Regularization;

public class NoRegularization<T> : RegularizationBase<T>
{
    public NoRegularization()
    {
    }
    
    public override Matrix<T> RegularizeMatrix(Matrix<T> matrix)
    {
        return matrix;
    }

    public override Vector<T> RegularizeCoefficients(Vector<T> coefficients)
    {
        return coefficients;
    }

    public override Vector<T> RegularizeGradient(Vector<T> gradient, Vector<T> coefficients)
    {
        return gradient;
    }
}