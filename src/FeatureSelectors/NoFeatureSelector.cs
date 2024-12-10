namespace AiDotNet.FeatureSelectors;

public class NoFeatureSelector : IFeatureSelector
{
    public Matrix<double> SelectFeatures(Matrix<double> allFeaturesMatrix)
    {
        return allFeaturesMatrix;
    }
}