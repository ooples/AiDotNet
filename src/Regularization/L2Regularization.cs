namespace AiDotNet.Regularization;

public class L2Regularization : IRegularization
{
    private readonly double _regularizationStrength;

    public L2Regularization(double regularizationStrength)
    {
        _regularizationStrength = regularizationStrength;
    }

    public Matrix<double> RegularizeMatrix(Matrix<double> featuresMatrix)
    {
        int featureCount = featuresMatrix.Columns;
        var regularizationMatrix = new Matrix<double>(featureCount, featureCount);

        for (int i = 0; i < featureCount; i++)
        {
            regularizationMatrix[i, i] = _regularizationStrength;
        }

        return featuresMatrix.Add(regularizationMatrix);
    }

    public Vector<double> RegularizeCoefficients(Vector<double> coefficients)
    {
        return coefficients;
    }
}