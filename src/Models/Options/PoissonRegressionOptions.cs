namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Poisson Regression, a specialized form of regression analysis used for modeling
/// count data and contingency tables where the dependent variable consists of non-negative integers.
/// </summary>
/// <remarks>
/// <para>
/// Poisson Regression is particularly suited for analyzing count data where the response variable represents
/// the number of occurrences of an event within a fixed period or space. Unlike linear regression, which assumes
/// a normal distribution of errors, Poisson regression assumes the response variable follows a Poisson distribution.
/// This makes it appropriate for cases where data is discrete, non-negative, and often skewed. The model uses a 
/// logarithmic link function to ensure predictions are always positive. Poisson regression is widely used in fields
/// such as epidemiology, insurance (claim counts), ecology (species counts), and marketing (number of purchases).
/// </para>
/// <para><b>For Beginners:</b> Poisson Regression is a technique specially designed for predicting counts of things.
/// 
/// Imagine you want to predict:
/// - How many customers will visit a store each hour
/// - How many calls a support center will receive each day
/// - How many defects will appear in manufactured products
/// 
/// When working with counts:
/// - You never have negative numbers (you can't have -3 customers)
/// - You often have small whole numbers (0, 1, 2, 3, etc.)
/// - The data often clusters near zero with a "long tail" of occasional higher values
/// 
/// Regular regression methods can give nonsensical results for this kind of data (like predicting -2.5 customers).
/// Poisson regression specifically handles count data by:
/// - Always predicting non-negative values
/// - Understanding the special patterns common in count data
/// - Properly handling the case when counts are zero
/// 
/// This class lets you configure how the algorithm searches for the best model to fit your count data.
/// </para>
/// </remarks>
public class PoissonRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the maximum number of iterations allowed in the model fitting process.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how many times the algorithm will attempt to refine the model coefficients during
    /// the fitting process. Poisson regression typically uses iterative methods like Iteratively Reweighted Least
    /// Squares (IRLS) to maximize the likelihood function. The algorithm iteratively updates the model parameters
    /// until convergence is reached or the maximum number of iterations is exceeded. Setting this value too low
    /// might prevent the model from converging to an optimal solution, while setting it too high might waste
    /// computational resources if the model has already converged.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many attempts the algorithm makes to find the best model.
    /// 
    /// The default value of 100 means:
    /// - The algorithm will try up to 100 times to improve its predictions
    /// - It stops earlier if it finds a solution that's good enough (based on the Tolerance setting)
    /// - It stops after 100 attempts even if it hasn't found an ideal solution
    /// 
    /// Think of it like finding the lowest point in a valley while blindfolded:
    /// - You take a step in what seems like the downward direction
    /// - You check if you're lower than before
    /// - You repeat this process, gradually getting closer to the bottom
    /// - MaxIterations is like saying "I'll take at most 100 steps, then stop wherever I am"
    /// 
    /// You might want more iterations (like 500 or 1000):
    /// - For complex datasets where finding the best model is difficult
    /// - When high precision is critical for your application
    /// - If you notice the model isn't converging with the default setting
    /// 
    /// You might want fewer iterations (like 50 or 20):
    /// - For simpler datasets where solutions are found quickly
    /// - When you need faster training times
    /// - In exploratory analysis where approximate solutions are acceptable
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the convergence tolerance threshold for the model fitting process.
    /// </summary>
    /// <value>The convergence tolerance, defaulting to 1e-6 (0.000001).</value>
    /// <remarks>
    /// <para>
    /// This parameter determines how precise the solution needs to be before the algorithm considers the model
    /// to have converged. Specifically, it defines the threshold for the change in model parameters or log-likelihood
    /// between consecutive iterations. When the change falls below this tolerance level, the iterative process
    /// terminates, as further improvements would be negligible. A smaller tolerance value requires more precision
    /// and potentially more iterations, while a larger value allows for earlier termination with a less precise solution.
    /// The optimal setting depends on the specific application's requirements for precision versus computational efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how precise the model solution needs to be before the algorithm stops refining it.
    /// 
    /// The default value of 0.000001 (1e-6) means:
    /// - The algorithm will stop when changes between iterations become very small (less than one millionth)
    /// - It's considered "close enough" to the perfect solution at this point
    /// - Further refinements would make negligible difference to predictions
    /// 
    /// Think of it like adjusting the focus on a camera:
    /// - At first, turning the focus ring makes a big difference to clarity
    /// - As you get closer to perfect focus, small adjustments make less noticeable improvements
    /// - Eventually, you reach a point where further adjustments don't visibly improve the image
    /// - Tolerance is like deciding "this is sharp enough" and stopping
    /// 
    /// You might want a smaller tolerance (like 1e-8 or 1e-10):
    /// - For applications requiring extremely precise results
    /// - When using the model for scientific research or critical decisions
    /// - If you have the computational resources to support more iterations
    /// 
    /// You might want a larger tolerance (like 1e-4 or 1e-3):
    /// - When approximate solutions are sufficient
    /// - To speed up model training, especially with large datasets
    /// - For preliminary analysis or when building many models
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-6;
}
