namespace AiDotNet.Regularization;

public class L1Regularization : IRegularization
{
    private readonly double _regularizationStrength;

    public L1Regularization(double regularizationStrength)
    {
        _regularizationStrength = regularizationStrength;
    }

    public Matrix<double> RegularizeMatrix(Matrix<double> featuresMatrix)
    {
        return featuresMatrix;
    }

    public Vector<double> RegularizeCoefficients(Vector<double> coefficients)
    {
        var count = coefficients.Length;
        var regularizedCoefficients = new Vector<double>(count);

        for (int i = 0; i < count; i++)
        {
            double coeff = coefficients[i];
            if (coeff > _regularizationStrength)
                regularizedCoefficients[i] = coeff - _regularizationStrength;
            else if (coeff < -_regularizationStrength)
                regularizedCoefficients[i] = coeff + _regularizationStrength;
            else
                regularizedCoefficients[i] = 0;
        }

        return regularizedCoefficients;
    }
}