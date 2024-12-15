namespace AiDotNet.Regularization;

public class ElasticNetRegularization<T> : IRegularization<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly RegularizationOptions _options;

    public ElasticNetRegularization(RegularizationOptions? options = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new RegularizationOptions();
    }

    public Matrix<T> RegularizeMatrix(Matrix<T> matrix)
    {
        var identity = Matrix<T>.CreateIdentity(matrix.Columns, _numOps);
        var regularizationStrength = _numOps.FromDouble(_options.Strength);
        var l2Ratio = _numOps.FromDouble(1 - _options.L1Ratio);

        return matrix.Add(identity.Multiply(_numOps.Multiply(regularizationStrength, l2Ratio)));
    }

    public Vector<T> RegularizeCoefficients(Vector<T> coefficients)
    {
        var regularizationStrength = _numOps.FromDouble(_options.Strength);
        var l1Ratio = _numOps.FromDouble(_options.L1Ratio);
        var l2Ratio = _numOps.FromDouble(1 - _options.L1Ratio);

        return coefficients.Transform(c =>
        {
            var subPart = _numOps.Subtract(_numOps.Abs(c), _numOps.Multiply(regularizationStrength, l1Ratio));
            var l1Part = _numOps.Multiply(
                _numOps.SignOrZero(c),
                _numOps.GreaterThan(subPart, _numOps.Zero) ? subPart : _numOps.Zero
            );
            var l2Part = _numOps.Multiply(
                c,
                _numOps.Subtract(_numOps.One, _numOps.Multiply(regularizationStrength, l2Ratio))
            );
            return _numOps.Add(l1Part, l2Part);
        });
    }
}