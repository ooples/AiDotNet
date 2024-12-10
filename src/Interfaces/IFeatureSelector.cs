namespace AiDotNet.Interfaces;

public interface IFeatureSelector
{
    Matrix<double> SelectFeatures(Matrix<double> allFeaturesMatrix);
}