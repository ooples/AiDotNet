namespace AiDotNet.FeatureSelectors;

public class NoFeatureSelector<T> : IFeatureSelector<T>
{
    public Matrix<T> SelectFeatures(Matrix<T> allFeaturesMatrix)
    {
        return allFeaturesMatrix;
    }
}