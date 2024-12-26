
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

    public static RegularizationType GetRegularizationType<T>(IRegularization<T> regularization)
    {
        return regularization switch
        {
            NoRegularization<T> => RegularizationType.None,
            L1Regularization<T> => RegularizationType.L1,
            L2Regularization<T> => RegularizationType.L2,
            ElasticNetRegularization<T> => RegularizationType.ElasticNet,
            _ => throw new ArgumentException($"Unsupported regularization type: {regularization.GetType().Name}")
        };
    }
}