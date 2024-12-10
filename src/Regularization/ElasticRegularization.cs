namespace AiDotNet.Regularization;

public class ElasticNetRegularization : IRegularization
{
    private readonly double _l1Ratio;
    private readonly double _regularizationStrength;

    public ElasticNetRegularization(double regularizationStrength, double l1Ratio)
    {
        _regularizationStrength = regularizationStrength;
        _l1Ratio = l1Ratio;
    }

    public Matrix<double> RegularizeMatrix(Matrix<double> featuresMatrix)
    {
        int featureCount = featuresMatrix.Columns;
        var regularizationMatrix = new Matrix<double>(featureCount, featureCount);
        double l2Strength = _regularizationStrength * (1 - _l1Ratio);

        for (int i = 0; i < featureCount; i++)
        {
            regularizationMatrix[i, i] = l2Strength;
        }

        return featuresMatrix.Add(regularizationMatrix);
    }

    public Vector<double> RegularizeCoefficients(Vector<double> coefficients)
    {
        var count = coefficients.Length;
        var regularizedCoefficients = new Vector<double>(count);
        double l1Strength = _regularizationStrength * _l1Ratio;

        for (int i = 0; i < count; i++)
        {
            double coeff = coefficients[i];
            if (coeff > l1Strength)
                regularizedCoefficients[i] = coeff - l1Strength;
            else if (coeff < -l1Strength)
                regularizedCoefficients[i] = coeff + l1Strength;
            else
                regularizedCoefficients[i] = 0;
        }

        return regularizedCoefficients;
    }
}