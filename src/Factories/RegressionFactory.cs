global using AiDotNet.Regression;

namespace AiDotNet.Factories;

public static class RegressionFactory
{
    public static MultivariateRegression<T> CreateRidgeRegression<T>(RegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
    {
        return new MultivariateRegression<T>(options, regularization);
    }

    public static MultivariateRegression<T> CreateLassoRegression<T>(RegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
    {
        return new MultivariateRegression<T>(options, regularization);
    }

    public static MultivariateRegression<T> CreateElasticNetRegression<T>(RegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
    {
        return new MultivariateRegression<T>(options, regularization);
    }
}