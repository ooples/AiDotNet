namespace AiDotNet.Regularization;

public class L2Regularization<T> : IRegularization<T>
{
    private readonly T _regularizationStrength;

    public L2Regularization(T regularizationStrength)
    {
        _regularizationStrength = regularizationStrength;
    }

    public Matrix<T> RegularizeMatrix(Matrix<T> featuresMatrix)
    {
        int featureCount = featuresMatrix.Columns;
        var regularizationMatrix = new Matrix<T>(featureCount, featureCount);

        for (int i = 0; i < featureCount; i++)
        {
            regularizationMatrix[i, i] = _regularizationStrength;
        }

        return featuresMatrix.Add(regularizationMatrix);
    }

    public Vector<T> RegularizeCoefficients(Vector<T> coefficients)
    {
        return coefficients;
    }
}