namespace AiDotNet.Regularization;

public class L1Regularization<T> : IRegularization<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly RegularizationOptions _options;

    public L1Regularization(RegularizationOptions? options = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new RegularizationOptions();
    }

    public Matrix<T> RegularizeMatrix(Matrix<T> matrix)
    {
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