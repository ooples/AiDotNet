global using AiDotNet.Regression;
global using AiDotNet.Models.Options;

namespace AiDotNet.Factories;

/// <summary>
/// A factory class that creates different types of regularized regression models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Regression is a statistical method used to find relationships between 
/// variables and make predictions. Regularized regression adds constraints to prevent overfitting 
/// (when a model performs well on training data but poorly on new data).
/// </para>
/// <para>
/// This factory helps you create different types of regularized regression models without needing 
/// to know their internal implementation details. Think of it like ordering a specific tool from 
/// a catalog - you just specify what you need, and the factory provides it.
/// </para>
/// </remarks>
public static class RegressionFactory
{
    /// <summary>
    /// Creates a Ridge regression model with optional custom options and regularization.
    /// </summary>
    /// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
    /// <param name="options">Optional configuration options for the regression model.</param>
    /// <param name="regularization">Optional custom regularization implementation.</param>
    /// <returns>A multivariate regression model configured for Ridge regression.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ridge regression is a technique used when your data suffers from multicollinearity 
    /// (when predictor variables are highly correlated). It adds a penalty term to the sum of squared residuals 
    /// that is proportional to the square of the magnitude of coefficients.
    /// </para>
    /// <para>
    /// Ridge regression shrinks the coefficients of correlated variables toward each other, reducing their 
    /// variance and helping to prevent overfitting. Unlike Lasso, Ridge regression keeps all variables in the model 
    /// but reduces their impact.
    /// </para>
    /// </remarks>
    public static MultivariateRegression<T> CreateRidgeRegression<T>(RegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
    {
        return new MultivariateRegression<T>(options, regularization);
    }

    /// <summary>
    /// Creates a Lasso regression model with optional custom options and regularization.
    /// </summary>
    /// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
    /// <param name="options">Optional configuration options for the regression model.</param>
    /// <param name="regularization">Optional custom regularization implementation.</param>
    /// <returns>A multivariate regression model configured for Lasso regression.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Lasso (Least Absolute Shrinkage and Selection Operator) regression is a technique 
    /// that performs both variable selection and regularization. It adds a penalty term to the sum of squared 
    /// residuals that is proportional to the absolute value of the coefficients.
    /// </para>
    /// <para>
    /// Lasso regression can completely eliminate the influence of some variables by setting their coefficients 
    /// to zero. This makes it useful for feature selection in datasets with many variables, as it automatically 
    /// identifies and keeps only the most important predictors.
    /// </para>
    /// </remarks>
    public static MultivariateRegression<T> CreateLassoRegression<T>(RegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
    {
        return new MultivariateRegression<T>(options, regularization);
    }

    /// <summary>
    /// Creates an Elastic Net regression model with optional custom options and regularization.
    /// </summary>
    /// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
    /// <param name="options">Optional configuration options for the regression model.</param>
    /// <param name="regularization">Optional custom regularization implementation.</param>
    /// <returns>A multivariate regression model configured for Elastic Net regression.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Elastic Net regression combines the penalties of both Ridge and Lasso regression. 
    /// It adds two penalty terms to the sum of squared residuals: one proportional to the square of the magnitude 
    /// of coefficients (like Ridge) and another proportional to the absolute value of the coefficients (like Lasso).
    /// </para>
    /// <para>
    /// Elastic Net is particularly useful when you have many correlated variables. It can perform variable selection 
    /// like Lasso (setting some coefficients to zero) while still maintaining the regularization benefits of Ridge. 
    /// This makes it a good middle ground when you're not sure whether to use Ridge or Lasso.
    /// </para>
    /// </remarks>
    public static MultivariateRegression<T> CreateElasticNetRegression<T>(RegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
    {
        return new MultivariateRegression<T>(options, regularization);
    }
}