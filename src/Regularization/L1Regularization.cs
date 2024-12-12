namespace AiDotNet.Regularization;

public class L1Regularization<T> : IRegularization<T>
{
    private readonly T _regularizationStrength;
    private readonly INumericOperations<T> _numOps;

    public L1Regularization(T regularizationStrength)
    {
        _regularizationStrength = regularizationStrength;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Matrix<T> RegularizeMatrix(Matrix<T> featuresMatrix)
    {
        return featuresMatrix;
    }

    public Vector<T> RegularizeCoefficients(Vector<T> coefficients)
    {
        var count = coefficients.Length;
        var regularizedCoefficients = new Vector<T>(count);

        for (int i = 0; i < count; i++)
        {
            T coeff = coefficients[i];
            if (_numOps.GreaterThan(coeff, _regularizationStrength))
                regularizedCoefficients[i] = _numOps.Subtract(coeff, _regularizationStrength);
            else if (_numOps.LessThan(coeff, _numOps.Negate(_regularizationStrength)))
                regularizedCoefficients[i] = _numOps.Add(coeff, _regularizationStrength);
            else
                regularizedCoefficients[i] = _numOps.Zero;
        }

        return regularizedCoefficients;
    }
}