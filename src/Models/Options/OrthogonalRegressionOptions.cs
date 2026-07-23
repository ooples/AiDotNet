namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Orthogonal Regression (also known as Total Least Squares), which minimizes 
/// the perpendicular distances from data points to the fitted model, accounting for errors in both 
/// dependent and independent variables.
/// </summary>
/// <remarks>
/// <para>
/// Orthogonal Regression differs from standard regression techniques by considering measurement errors 
/// in both the predictor (independent) and response (dependent) variables. While ordinary least squares 
/// regression minimizes vertical distances from points to the regression line, orthogonal regression 
/// minimizes perpendicular distances, making it more appropriate when both variables contain measurement 
/// error or uncertainty. This approach is particularly valuable in fields like physics, chemistry, and 
/// engineering where measurement instruments may introduce errors in all variables. The algorithm typically 
/// employs singular value decomposition or iterative methods to find the optimal solution.
/// </para>
/// <para><b>For Beginners:</b> Orthogonal Regression is a special type of regression that treats all variables fairly when finding patterns.
/// 
/// In standard regression:
/// - We assume that only the y-variable (what we're predicting) contains errors
/// - We minimize the vertical distances from points to the line
/// 
/// Imagine measuring the heights and weights of people:
/// - Standard regression assumes heights are measured perfectly, only weights have errors
/// - Orthogonal regression recognizes that both height AND weight measurements have errors
/// 
/// This matters because:
/// - When both variables have measurement errors, standard regression can give biased results
/// - Orthogonal regression fits a line that's "fair" to both variables
/// - The line minimizes the perpendicular distance from points to the line, not just vertical distance
/// 
/// This technique is especially useful in scientific applications where:
/// - All measurements come from instruments with known error rates
/// - We're looking for true physical relationships rather than just predictions
/// - The variables play symmetrical roles rather than strictly "input" and "output"
/// 
/// This class lets you configure how the orthogonal regression algorithm works, controlling
/// its precision, computational limits, and data preprocessing.
/// </para>
/// </remarks>
public class OrthogonalRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the convergence tolerance that determines when the iterative optimization algorithm should stop.
    /// </summary>
    /// <value>The convergence tolerance, defaulting to 0.000001 (1e-6).</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the threshold for determining when the optimization has converged. The algorithm
    /// will stop when the improvement between consecutive iterations falls below this tolerance value. A smaller
    /// tolerance requires more precision in the parameter estimates, potentially leading to better model fit but
    /// requiring more iterations. This is particularly important in orthogonal regression where finding the optimal
    /// solution often requires iterative approaches. The appropriate value depends on the scale of your variables
    /// and the required precision of your model.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how precise the algorithm should be
    /// before deciding it has found the best solution.
    /// 
    /// The default value of 0.000001 (one millionth) means:
    /// - If consecutive iterations of the algorithm improve the solution by less than one millionth
    /// - The algorithm decides it's "close enough" and stops
    /// 
    /// Think of it like measuring ingredients for a recipe:
    /// - A small tolerance is like measuring to the nearest milligram (very precise)
    /// - A larger tolerance is like measuring to the nearest gram (less precise)
    /// 
    /// You might want a smaller value (like 1e-8) if:
    /// - Your application requires extremely high precision
    /// - You have well-conditioned data with minimal noise
    /// - You're using the results for sensitive scientific calculations
    /// 
    /// You might want a larger value (like 1e-4) if:
    /// - You need faster computations
    /// - Your data contains substantial noise anyway
    /// - You're doing exploratory analysis rather than final modeling
    /// 
    /// Finding the right tolerance balances precision with computational efficiency.
    /// Too strict (too small) and the algorithm might take unnecessarily long;
    /// too loose (too large) and you might get suboptimal results.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the maximum number of iterations allowed for the optimization algorithm.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// This parameter sets an upper limit on how many iterations the optimization algorithm will perform
    /// when fitting the orthogonal regression model. Each iteration refines the model parameters to better
    /// minimize the sum of squared perpendicular distances. The algorithm may terminate earlier if
    /// convergence is achieved based on the tolerance value. Orthogonal regression often requires an
    /// iterative approach, especially for nonlinear models. The appropriate number of iterations depends
    /// on the complexity of the relationship, the number of data points, and the initial parameter values.
    /// </para>
    /// <para><b>For Beginners:</b> This setting limits how many attempts the algorithm
    /// makes to improve its solution before stopping.
    /// 
    /// The default value of 100 means:
    /// - The algorithm will make at most 100 attempts to refine the solution
    /// - It might stop earlier if it reaches the desired precision (set by Tolerance)
    /// 
    /// Imagine polishing a surface:
    /// - Each iteration is like one pass with the polishing cloth
    /// - You want enough passes to get a good finish, but not waste time after it's already smooth
    /// 
    /// You might want more iterations (like 500) if:
    /// - You're working with complex relationships
    /// - You notice the algorithm is still improving significantly at 100 iterations
    /// - You have a very strict tolerance setting
    /// 
    /// You might want fewer iterations (like 50) if:
    /// - You need faster results
    /// - Your data is well-behaved and converges quickly
    /// - You're doing preliminary analysis
    /// 
    /// This setting works together with Tolerance - the algorithm stops when either:
    /// - It reaches the maximum number of iterations, OR
    /// - The improvement between iterations becomes smaller than the tolerance
    /// 
    /// For most applications, the default of 100 iterations provides a good balance
    /// between thorough optimization and reasonable computation time.
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to standardize variables before fitting the model.
    /// </summary>
    /// <value>Whether to scale variables, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines whether the input variables should be standardized (scaled to have
    /// zero mean and unit variance) before fitting the orthogonal regression model. Standardization
    /// is particularly important for orthogonal regression because the method minimizes perpendicular
    /// distances, which are directly affected by the scale of each variable. Without standardization,
    /// variables with larger scales would dominate the optimization. By default, this is set to true,
    /// which ensures that variables with different units or scales contribute equally to the model fit.
    /// The standardization is reversed when making predictions, so outputs are returned in the original scale.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether the algorithm should
    /// adjust all variables to a similar scale before finding the best fit.
    /// 
    /// The default value of true means:
    /// - Before fitting, all variables are rescaled to have similar ranges
    /// - This ensures that no variable dominates just because it uses larger numbers
    /// 
    /// For example, if you're relating:
    /// - Age (typically 0-100 years) and
    /// - Income (typically thousands or tens of thousands of dollars)
    /// 
    /// Without scaling:
    /// - Income would dominate the calculations because its numbers are much larger
    /// - The resulting line might fit the income well but ignore patterns in age
    /// 
    /// With scaling (the default):
    /// - Both variables are adjusted to similar ranges (typically mean 0, variance 1)
    /// - The resulting line treats both variables fairly
    /// 
    /// You might want to set this to false if:
    /// - Your variables are already on the same scale
    /// - You specifically want variables with larger values to have more influence
    /// - You have domain-specific reasons to preserve the original scales
    /// 
    /// In most cases, leaving this set to true is recommended, especially when
    /// variables have different units or widely different numerical ranges.
    /// </para>
    /// </remarks>
    public bool ScaleVariables { get; set; } = true;

    /// <summary>
    /// Gets or sets the matrix decomposition type to use when solving the linear system.
    /// </summary>
    /// <value>The matrix decomposition type, defaulting to SVD decomposition.</value>
    /// <remarks>
    /// <para>
    /// The decomposition type determines how the system of linear equations is solved during optimization.
    /// SVD (Singular Value Decomposition) is particularly well-suited for orthogonal regression as it
    /// naturally handles the total least squares formulation.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls the mathematical method used to solve equations
    /// during model fitting. The default SVD method is ideal for orthogonal regression as it handles
    /// measurement errors in all variables properly.
    /// </para>
    /// </remarks>
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Svd;
}
