namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for robust regression models, which are designed to be less sensitive
/// to outliers and violations of standard regression assumptions.
/// </summary>
/// <typeparam name="T">The data type used in matrix operations for the regression model.</typeparam>
/// <remarks>
/// <para>
/// Robust regression methods provide an alternative to standard least squares regression when data contains 
/// outliers or exhibits heteroscedasticity (non-constant variance in errors). Unlike ordinary least squares 
/// regression, which can be heavily influenced by outliers, robust regression methods use specialized techniques 
/// to reduce the impact of outlying observations. This class extends the standard RegressionOptions class to 
/// include additional parameters specific to robust regression algorithms, such as the tuning constant, 
/// maximum iterations for iterative reweighting procedures, convergence tolerance, weight function selection, 
/// and an optional initial regression model for initialization. These options allow fine-tuning of the robust 
/// regression algorithm to best handle the specific characteristics of the dataset being analyzed.
/// </para>
/// <para><b>For Beginners:</b> Robust regression helps your model handle outliers and unusual data points.
/// 
/// Standard regression can be easily thrown off by outliers (extreme values):
/// - Imagine predicting house prices where most homes are $200,000-$400,000
/// - But there's one luxury mansion worth $10 million in your data
/// - Standard regression might shift significantly to accommodate this outlier
/// 
/// Robust regression solves this problem by:
/// - Giving less weight to data points that are far from the pattern
/// - Focusing more on the typical cases that represent the true relationship
/// - Producing more reliable predictions when your data has unusual values
/// 
/// Think of it like taking a poll:
/// - Standard regression counts everyone's vote equally
/// - Robust regression gives more consideration to mainstream opinions and less to extreme views
/// 
/// This class lets you configure exactly how the robust regression algorithm handles outliers
/// and unusual data points.
/// </para>
/// </remarks>
public class RobustRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the tuning constant that controls the sensitivity to outliers.
    /// </summary>
    /// <value>A positive double value, defaulting to 1.345.</value>
    /// <remarks>
    /// <para>
    /// The tuning constant determines how strongly outliers are downweighted in the robust regression procedure. 
    /// It controls the trade-off between efficiency (statistical optimality under ideal conditions) and robustness 
    /// (resistance to outliers). A smaller tuning constant provides more resistance to outliers but may be less 
    /// efficient when the data actually follows the assumed distribution. A larger tuning constant provides less 
    /// resistance to outliers but better efficiency under ideal conditions. The default value of 1.345 for the 
    /// Huber weight function provides approximately 95% efficiency for normally distributed errors while still 
    /// offering protection against outliers. Different weight functions may have different optimal tuning constants, 
    /// and the appropriate value may depend on the specific characteristics of the dataset and the chosen weight function.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how aggressively the model ignores outliers.
    /// 
    /// The tuning constant determines the balance between:
    /// - Treating all data points equally (like standard regression)
    /// - Completely ignoring unusual data points
    /// 
    /// The default value of 1.345:
    /// - Is specifically chosen for the Huber method (the default weight function)
    /// - Provides a good balance for most applications
    /// - Gives approximately 95% statistical efficiency while still protecting against outliers
    /// 
    /// When to adjust this value:
    /// - Lower values (like 0.8 or 1.0): More aggressive outlier rejection, use when outliers are severe
    /// - Higher values (like 2.0 or 2.5): More inclusive, use when you want to give outliers more influence
    /// 
    /// For example, in financial data with occasional extreme market events, you might use a lower
    /// tuning constant to prevent these rare events from overly influencing your model.
    /// 
    /// Note that the optimal value depends on which weight function you're using, so if you change
    /// the WeightFunction property, you may need to adjust this value accordingly.
    /// </para>
    /// </remarks>
    public double TuningConstant { get; set; } = 1.345;

    /// <summary>
    /// Gets or sets the maximum number of iterations for the iterative reweighting procedure.
    /// </summary>
    /// <value>A positive integer, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// Robust regression methods typically use an iterative reweighting procedure to compute the regression 
    /// coefficients. This property specifies the maximum number of iterations allowed before the algorithm 
    /// terminates, even if convergence (as determined by the Tolerance property) has not been achieved. 
    /// The default value of 100 is sufficient for most applications to achieve convergence. However, for 
    /// particularly complex datasets or when using very small tolerance values, more iterations may be required. 
    /// Conversely, for simpler datasets or when approximate solutions are acceptable, fewer iterations may be 
    /// sufficient. Setting a reasonable maximum prevents the algorithm from running indefinitely in cases where 
    /// convergence is difficult to achieve.
    /// </para>
    /// <para><b>For Beginners:</b> This setting limits how long the algorithm will try to find the best solution.
    /// 
    /// Robust regression works through an iterative process:
    /// - It starts with an initial guess (often from standard regression)
    /// - Then it repeatedly refines this solution, adjusting weights for outliers
    /// - Each iteration should bring it closer to the optimal solution
    /// 
    /// The MaxIterations value (default 100) sets a limit on this process:
    /// - It prevents the algorithm from running forever if it can't find a perfect solution
    /// - Most problems converge (reach a stable solution) well before 100 iterations
    /// - If the algorithm reaches this limit, it returns the best solution found so far
    /// 
    /// When to adjust this value:
    /// - Increase it (e.g., to 200 or 500) for complex datasets where convergence is difficult
    /// - Decrease it (e.g., to 50 or 20) when you need faster results and approximate solutions are acceptable
    /// 
    /// This is similar to giving someone a time limit to solve a puzzle - they'll return the best
    /// solution they've found when time runs out, even if it's not perfect.
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the convergence tolerance for the iterative reweighting procedure.
    /// </summary>
    /// <value>A positive double value, defaulting to 1e-6 (0.000001).</value>
    /// <remarks>
    /// <para>
    /// This property specifies the convergence criterion for the iterative reweighting procedure used in robust 
    /// regression. The algorithm terminates when the change in parameter estimates between consecutive iterations 
    /// is less than this tolerance value. A smaller tolerance requires more precise convergence, potentially 
    /// leading to more accurate results but requiring more iterations. A larger tolerance allows for earlier 
    /// termination, potentially saving computational resources at the cost of less precise parameter estimates. 
    /// The default value of 1e-6 (0.000001) provides a good balance between precision and computational efficiency 
    /// for most applications. For high-precision requirements, a smaller value might be appropriate, while for 
    /// exploratory analysis or when computational resources are limited, a larger value might be used.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how precise the final solution needs to be.
    /// 
    /// During the iterative process, the algorithm keeps refining its solution:
    /// - It compares each new solution to the previous one
    /// - When the difference becomes smaller than the tolerance value, it stops
    /// - This indicates the solution has stabilized and further iterations won't improve much
    /// 
    /// The Tolerance value (default 0.000001 or 1e-6) means:
    /// - The algorithm will stop when changes between iterations are very small
    /// - A smaller value requires more precision (and usually more iterations)
    /// - A larger value allows the algorithm to stop earlier with an approximate solution
    /// 
    /// When to adjust this value:
    /// - Decrease it (e.g., to 1e-8) when you need extremely precise results
    /// - Increase it (e.g., to 1e-4) when you need faster results and can accept slight imprecision
    /// 
    /// This is like telling someone to keep refining their work until the improvements are so small
    /// they're not worth the extra effort - the tolerance defines what "small" means in this context.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the weight function used to reduce the influence of outliers.
    /// </summary>
    /// <value>A value from the WeightFunction enumeration, defaulting to WeightFunction.Huber.</value>
    /// <remarks>
    /// <para>
    /// This property specifies which weight function is used to downweight outliers in the robust regression 
    /// procedure. Different weight functions have different characteristics in terms of efficiency, breakdown 
    /// point (the proportion of outliers that can be handled), and computational complexity. The Huber function 
    /// (the default) provides a good balance between efficiency and robustness for many applications. Other 
    /// common options include Bisquare (which completely downweights extreme outliers), Andrews (similar to 
    /// Bisquare but with different characteristics), and Fair (which is less aggressive in downweighting outliers). 
    /// The choice of weight function should be based on the specific characteristics of the dataset, particularly 
    /// the expected frequency and magnitude of outliers, and the desired trade-off between robustness and efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This setting selects the method used to handle outliers.
    /// 
    /// Different weight functions have different approaches to dealing with unusual data points:
    /// - Huber (default): Reduces the influence of outliers gradually
    /// - Bisquare: Completely ignores extreme outliers
    /// - Andrews: Similar to Bisquare but with different mathematical properties
    /// - Fair: More gentle reduction of outlier influence
    /// 
    /// The default Huber function:
    /// - Treats normal data points using standard least squares
    /// - Gradually reduces the influence of points as they become more unusual
    /// - Provides a good balance for most applications
    /// 
    /// When to choose a different function:
    /// - Bisquare: When you have extreme outliers that should be completely ignored
    /// - Fair: When you want outliers to still have some influence, just reduced
    /// - Andrews: Similar to Bisquare but sometimes more stable numerically
    /// 
    /// For example, in a retail sales model, you might use Bisquare to completely ignore
    /// holiday sales spikes if you're trying to model normal day-to-day business.
    /// 
    /// Each function has its own ideal tuning constant, so if you change this setting,
    /// consider also adjusting the TuningConstant property.
    /// </para>
    /// </remarks>
    public WeightFunction WeightFunction { get; set; } = WeightFunction.Huber;

    /// <summary>
    /// Gets or sets the initial regression model used to start the iterative procedure.
    /// </summary>
    /// <value>An implementation of IRegression&lt;T&gt; or null to use the default initialization.</value>
    /// <remarks>
    /// <para>
    /// Robust regression methods typically use an iterative procedure that requires an initial estimate of the 
    /// regression coefficients. This property allows specifying a custom regression model to provide these initial 
    /// estimates. When set to null (the default), the algorithm will use its own initialization method, typically 
    /// ordinary least squares regression. Providing a custom initial regression can be useful when there is prior 
    /// knowledge about the data that suggests a better starting point than ordinary least squares, or when 
    /// continuing a previous robust regression analysis with updated data. The initial regression should implement 
    /// the IRegression&lt;T&gt; interface and should be compatible with the data type T used in the robust regression.
    /// </para>
    /// <para><b>For Beginners:</b> This setting lets you provide a starting point for the algorithm.
    /// 
    /// Robust regression works iteratively, starting from an initial guess:
    /// - By default (null), it starts with a standard least squares regression
    /// - This property lets you provide a different starting point
    /// 
    /// When you might want to set this:
    /// - When you already have a regression model that's close to what you expect
    /// - When you're updating an existing model with new data
    /// - When standard regression performs poorly as a starting point
    /// 
    /// For example, if you're analyzing seasonal data and have a previous model from
    /// similar seasons, you might use that as the starting point rather than starting fresh.
    /// 
    /// Most users can leave this as null and let the algorithm use its default initialization.
    /// This is an advanced option that's primarily useful in specific scenarios where you have
    /// good prior information about what the solution should look like.
    /// </para>
    /// </remarks>
    public IRegression<T>? InitialRegression { get; set; }
}
