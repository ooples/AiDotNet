namespace AiDotNet.Regularization;
public class ElasticNetRegularization<T> : IRegularization<T>
{
    private readonly T _l1Ratio;
    private readonly T _regularizationStrength;
    private readonly INumericOperations<T> _numOps;

    public ElasticNetRegularization(T regularizationStrength, T l1Ratio)
    {
        _regularizationStrength = regularizationStrength;
        _l1Ratio = l1Ratio;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Matrix<T> RegularizeMatrix(Matrix<T> featuresMatrix)
    {
        int featureCount = featuresMatrix.Columns;
        var regularizationMatrix = new Matrix<T>(featureCount, featureCount);
        T l2Strength = _numOps.Multiply(_regularizationStrength, _numOps.Subtract(_numOps.One, _l1Ratio));

        for (int i = 0; i < featureCount; i++)
        {
            regularizationMatrix[i, i] = l2Strength;
        }

        return featuresMatrix.Add(regularizationMatrix);
    }

    public Vector<T> RegularizeCoefficients(Vector<T> coefficients)
    {
        var count = coefficients.Length;
        var regularizedCoefficients = new Vector<T>(count);
        T l1Strength = _numOps.Multiply(_regularizationStrength, _l1Ratio);

        for (int i = 0; i < count; i++)
        {
            T coeff = coefficients[i];
            if (_numOps.GreaterThan(coeff, l1Strength))
                regularizedCoefficients[i] = _numOps.Subtract(coeff, l1Strength);
            else if (_numOps.LessThan(coeff, _numOps.Negate(l1Strength)))
                regularizedCoefficients[i] = _numOps.Add(coeff, l1Strength);
            else
                regularizedCoefficients[i] = _numOps.Zero;
        }

        return regularizedCoefficients;
    }
}