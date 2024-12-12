namespace AiDotNet.Regularization;

public class ElasticNetRegularization<T> : IRegularization<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly RegularizationOptions _options;

    public ElasticNetRegularization(INumericOperations<T> numOps, RegularizationOptions options)
    {
        _numOps = numOps;
        _options = options;
    }

    public Matrix<T> RegularizeMatrix(Matrix<T> matrix)
    {
        var identity = Matrix<T>.CreateIdentity(matrix.Columns, _numOps);
        var regularizationStrength = _numOps.FromDouble(_options.Strength);
        var l2Ratio = _numOps.FromDouble(1 - _options.L1Ratio);
        return matrix.Add(identity.Multiply(regularizationStrength.Multiply(l2Ratio)));
    }

    public Vector<T> RegularizeCoefficients(Vector<T> coefficients)
    {
        var regularizationStrength = _numOps.FromDouble(_options.Strength);
        var l1Ratio = _numOps.FromDouble(_options.L1Ratio);
        var l2Ratio = _numOps.FromDouble(1 - _options.L1Ratio);

        return coefficients.Transform(c =>
        {
            var l1Part = _numOps.Sign(c).Multiply(_numOps.Max(_numOps.Abs(c).Subtract(regularizationStrength.Multiply(l1Ratio)), _numOps.Zero));
            var l2Part = c.Multiply(_numOps.Subtract(_numOps.One, regularizationStrength.Multiply(l2Ratio)));
            return _numOps.Add(l1Part, l2Part);
        });
    }
}