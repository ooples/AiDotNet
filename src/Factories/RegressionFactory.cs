global using AiDotNet.Regression;

namespace AiDotNet.Factories;

public static class RegressionFactory
{
    public static MultivariateRegression<T> CreateRidgeRegression<T>(INumericOperations<T> numOps, RegressionOptions<T> options)
    {
        return new MultivariateRegression<T>(numOps, options);
    }

    public static MultivariateRegression<T> CreateLassoRegression<T>(INumericOperations<T> numOps, RegressionOptions<T> options)
    {
        return new MultivariateRegression<T>(numOps, options);
    }

    public static MultivariateRegression<T> CreateElasticNetRegression<T>(INumericOperations<T> numOps, RegressionOptions<T> options)
    {
        return new MultivariateRegression<T>(numOps, options);
    }
}