namespace AiDotNet.Interfaces;

public interface IFeatureSelector<T>
{
    Matrix<T> SelectFeatures(Matrix<T> allFeaturesMatrix);
}