namespace AiDotNet.Regularization;

public class L1Regularization<T> : IRegularization<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly RegularizationOptions _options;

    public L1Regularization(INumericOperations<T> numOps, RegularizationOptions options)
    {
        _numOps = numOps;
        _options = options;
    }

    public Matrix<T> RegularizeMatrix(Matrix<T> matrix)
    {
        // L1 regularization doesn't modify the matrix directly
        return matrix;
    }

    public Vector<T> RegularizeCoefficients(Vector<T> coefficients)
    {
        var regularizationStrength = _numOps.FromDouble(_options.Strength);
        return coefficients.Transform(c =>
        {
            var sub = _numOps.Subtract(_numOps.Abs(c), regularizationStrength);
            return _numOps.Multiply(
                _numOps.SignOrZero(c),
                _numOps.GreaterThan(sub, _numOps.Zero) ? sub : _numOps.Zero
            );
        });
    }
}