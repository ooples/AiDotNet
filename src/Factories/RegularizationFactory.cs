namespace AiDotNet.Factories;

public static class RegularizationFactory
{
    public static IRegularization<T> CreateRegularization<T>(RegularizationType regularizationType)
    {
        return regularizationType switch
        {
            RegularizationType.None => new NoRegularization<T>(),
            RegularizationType.L1 => new L1Regularization<T>(),
            RegularizationType.L2 => new L2Regularization<T>(),
            RegularizationType.ElasticNet => new ElasticNetRegularization<T>(),
            _ => throw new ArgumentException($"Unknown regularization type: {regularizationType}", nameof(regularizationType))
        };
    }
}