namespace AiDotNet.Regularization;

public class L2Regularization<T> : IRegularization<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly RegularizationOptions _options;

    public L2Regularization(INumericOperations<T> numOps, RegularizationOptions options)
    {
        _numOps = numOps;
        _options = options;
    }

    public Matrix<T> RegularizeMatrix(Matrix<T> matrix)
    {
        var identity = Matrix<T>.CreateIdentity(matrix.Rows, _numOps);
        var regularizationStrength = _numOps.FromDouble(_options.Strength);
        return matrix.Add(identity.Multiply(regularizationStrength));
    }

    public Vector<T> RegularizeCoefficients(Vector<T> coefficients)
    {
        var regularizationStrength = _numOps.FromDouble(_options.Strength);
        return coefficients.Multiply(_numOps.Subtract(_numOps.FromDouble(1), regularizationStrength));
    }
}