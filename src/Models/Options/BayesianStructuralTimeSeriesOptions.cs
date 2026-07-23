namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Bayesian Structural Time Series models.
/// </summary>
/// <typeparam name="T">The data type used by the time series model.</typeparam>
/// <remarks>
/// <para>
/// Bayesian Structural Time Series (BSTS) models are flexible time series models that decompose a time series
/// into trend, seasonal, and regression components. They use Bayesian methods to estimate model parameters
/// and can handle missing data, incorporate prior knowledge, and provide uncertainty estimates.
/// </para>
/// <para><b>For Beginners:</b> A time series is simply data collected over time (like daily temperatures, 
/// monthly sales, or yearly population). Bayesian Structural Time Series is a powerful way to analyze this 
/// kind of data by breaking it down into different parts: the overall direction (trend), repeating patterns 
/// (seasonality), and the influence of other factors (regression). The "Bayesian" part means it can express 
/// uncertainty about its predictions and incorporate what you already know about the data. Think of it like 
/// weather forecasting that not only predicts tomorrow's temperature but also tells you how confident it is 
/// in that prediction.</para>
/// </remarks>
public class BayesianStructuralTimeSeriesOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the initial value for the level component of the time series.
    /// </summary>
    /// <value>The initial level value, defaulting to 0.0.</value>
    /// <remarks>
    /// <para>
    /// The level component represents the baseline value of the time series at the start of the modeling period.
    /// Setting an appropriate initial value can help the model converge faster.
    /// </para>
    /// <para><b>For Beginners:</b> The "level" is like the starting point or baseline of your data. 
    /// For example, if you're tracking monthly sales that typically hover around $10,000, you might set this 
    /// to 10000 instead of the default 0.0. If you're unsure, you can leave it at 0.0, and the algorithm will 
    /// figure it out from your data, though it might take a few more iterations to do so.</para>
    /// </remarks>
    public double InitialLevelValue { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the initial value for the trend component of the time series.
    /// </summary>
    /// <value>The initial trend value, defaulting to 0.0.</value>
    /// <remarks>
    /// <para>
    /// The trend component represents the rate of change of the level component over time.
    /// A positive value indicates an upward trend, while a negative value indicates a downward trend.
    /// </para>
    /// <para><b>For Beginners:</b> The "trend" represents how much your data is generally increasing or 
    /// decreasing over time. If you know your data is growing by about 5% each period, you might set this 
    /// to 0.05. A value of 0.0 (the default) means you're starting with no assumption about whether the data 
    /// is trending up or down. The model will learn the actual trend from your data.</para>
    /// </remarks>
    public double InitialTrendValue { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the list of seasonal periods to model in the time series.
    /// </summary>
    /// <value>A list of integers representing seasonal periods, defaulting to an empty list.</value>
    /// <remarks>
    /// <para>
    /// Each integer in this list represents a seasonal pattern to model, where the value is the number of
    /// time steps in one complete seasonal cycle. Multiple seasonal patterns can be modeled simultaneously.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you tell the model about any repeating patterns in your data. 
    /// For example:
    /// <list type="bullet">
    ///   <item>For weekly patterns in daily data, add 7 to the list</item>
    ///   <item>For monthly patterns in daily data, add 30 or 31</item>
    ///   <item>For yearly patterns in monthly data, add 12</item>
    /// </list>
    /// You can include multiple patterns (like both weekly and yearly cycles) by adding multiple numbers to the list. 
    /// If your data doesn't have any repeating patterns, you can leave this empty (the default).</para>
    /// </remarks>
    public List<int> SeasonalPeriods { get; set; } = new List<int> { };

    /// <summary>
    /// Gets or sets the initial variance of the observation noise.
    /// </summary>
    /// <value>The initial observation variance, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter represents the expected amount of random noise or unexplained variation in the observed data.
    /// It affects how closely the model will try to fit the observed data points.
    /// </para>
    /// <para><b>For Beginners:</b> This setting represents how "noisy" or variable you expect your data to be. 
    /// A higher value (greater than 1.0) tells the model to expect more randomness in the data, resulting in a 
    /// smoother fit that doesn't chase every up and down. A lower value tells the model that most variations in 
    /// the data are meaningful and should be captured. The default value of 1.0 is a good starting point for most 
    /// datasets.</para>
    /// </remarks>
    public double InitialObservationVariance { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to perform backward smoothing after forward filtering.
    /// </summary>
    /// <value>True to perform backward smoothing (default), false otherwise.</value>
    /// <remarks>
    /// <para>
    /// Backward smoothing (also known as the Kalman smoother) improves estimates by incorporating
    /// information from the entire time series, not just the past observations. This typically
    /// produces more accurate results but requires additional computation.
    /// </para>
    /// <para><b>For Beginners:</b> When set to true (the default), the model will make two passes through your data: 
    /// first forward in time, then backward. This is like being able to see the future when interpreting the past, 
    /// which usually gives more accurate results. The downside is it takes a bit longer to compute. Set this to false 
    /// if you need faster processing and are willing to accept slightly less accurate results, or if you're doing 
    /// real-time predictions where future data isn't available yet.</para>
    /// </remarks>
    public bool PerformBackwardSmoothing { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of iterations for parameter estimation.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// This parameter limits how many iterations the algorithm will perform when trying to find
    /// the optimal model parameters. The algorithm will stop either when it reaches this limit
    /// or when the convergence tolerance is met, whichever comes first.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how long the algorithm will keep trying to improve its estimates. 
    /// The default value of 100 iterations is sufficient for most problems. If you notice the model isn't fitting 
    /// well, you might increase this to give it more chances to find a good solution. However, more iterations mean 
    /// longer computation time, so there's a trade-off.</para>
    /// </remarks>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the convergence tolerance for parameter estimation.
    /// </summary>
    /// <value>The convergence tolerance, defaulting to 1e-6 (0.000001).</value>
    /// <remarks>
    /// <para>
    /// The algorithm will stop iterating when the change in parameter estimates between iterations
    /// is less than this tolerance value, indicating that it has converged to a solution.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how precise the algorithm needs to be before it decides it's 
    /// "good enough" and stops calculating. The default value (0.000001) is very small, meaning high precision. 
    /// You rarely need to change this unless you're dealing with extremely precise data (smaller value) or want 
    /// faster but less precise results (larger value).</para>
    /// </remarks>
    public double ConvergenceTolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets whether to automatically determine prior distributions.
    /// </summary>
    /// <value>True to use automatic priors (default), false to use manually specified priors.</value>
    /// <remarks>
    /// <para>
    /// When set to true, the algorithm will automatically determine appropriate prior distributions
    /// based on the data. When false, it will use the manually specified prior values for level,
    /// trend, and seasonal components.
    /// </para>
    /// <para><b>For Beginners:</b> In Bayesian statistics, "priors" represent your initial beliefs about the data 
    /// before you analyze it. When this is set to true (the default), the algorithm will automatically choose 
    /// reasonable priors based on your data. Set this to false if you have specific knowledge about how your data 
    /// behaves and want to manually set the smoothing priors below. For most users, especially beginners, keeping 
    /// this as true is recommended.</para>
    /// </remarks>
    public bool UseAutomaticPriors { get; set; } = true;

    /// <summary>
    /// Gets or sets the prior for the level component's smoothing parameter.
    /// </summary>
    /// <value>The level smoothing prior, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how quickly the level component can change over time. A smaller value
    /// results in a smoother level component, while a larger value allows for more rapid changes.
    /// Only used when UseAutomaticPriors is false.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the baseline of your data can change. A small value 
    /// like the default (0.01) means the baseline changes slowly and smoothly. A larger value (closer to 1) would 
    /// allow the baseline to jump up and down more rapidly. This setting only matters if you've set UseAutomaticPriors 
    /// to false. For most cases, the default value works well.</para>
    /// </remarks>
    public double LevelSmoothingPrior { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the prior for the trend component's smoothing parameter.
    /// </summary>
    /// <value>The trend smoothing prior, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how quickly the trend component can change over time. A smaller value
    /// results in a more stable trend, while a larger value allows the trend to change more rapidly.
    /// Only used when UseAutomaticPriors is false.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the trend (direction) of your data can change. 
    /// A small value like the default (0.01) means the trend changes slowly - if your data is growing, it will 
    /// continue growing at a similar rate. A larger value would allow the growth rate to change more dramatically 
    /// between time periods. This setting only matters if you've set UseAutomaticPriors to false.</para>
    /// </remarks>
    public double TrendSmoothingPrior { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the prior for the seasonal component's smoothing parameter.
    /// </summary>
    /// <value>The seasonal smoothing prior, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how quickly the seasonal patterns can change over time. A smaller value
    /// results in more consistent seasonal patterns, while a larger value allows the seasonal patterns
    /// to evolve more rapidly. Only used when UseAutomaticPriors is false.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much the repeating patterns in your data can change over time. 
    /// For example, if you're tracking retail sales with a holiday season spike, this determines whether that spike 
    /// should be similar each year (use the default small value of 0.01) or whether it can change significantly 
    /// from year to year (use a larger value). This setting only matters if you've set UseAutomaticPriors to false.</para>
    /// </remarks>
    public double SeasonalSmoothingPrior { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets whether to include regression components in the model.
    /// </summary>
    /// <value>True to include regression, false otherwise (default).</value>
    /// <remarks>
    /// <para>
    /// When set to true, the model will include regression components that can capture the relationship
    /// between the time series and external variables (covariates). This allows the model to account for
    /// the influence of known factors on the time series.
    /// </para>
    /// <para><b>For Beginners:</b> Setting this to true lets you include other factors that might influence your data. 
    /// For example, if you're predicting ice cream sales over time, you might want to include temperature as a factor 
    /// because hot days typically lead to more sales. By default, this is set to false, meaning the model will only 
    /// look at patterns in your time series data itself without considering external factors.</para>
    /// </remarks>
    public bool IncludeRegression { get; set; } = false;

    /// <summary>
    /// Gets or sets the matrix decomposition method used for regression calculations.
    /// </summary>
    /// <value>The matrix decomposition type, defaulting to LU decomposition.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines the mathematical method used to solve the regression equations.
    /// Different decomposition methods have different numerical properties and performance characteristics.
    /// This setting only has an effect when IncludeRegression is set to true.
    /// </para>
    /// <para><b>For Beginners:</b> This is a technical setting that determines how the math is done behind the scenes 
    /// when including regression components. The default (LU decomposition) works well for most cases. You typically 
    /// don't need to change this unless you're experiencing numerical issues with large datasets or have specific 
    /// performance requirements. This setting only matters if IncludeRegression is set to true.</para>
    /// </remarks>
    public MatrixDecompositionType RegressionDecompositionType { get; set; } = MatrixDecompositionType.Lu;

    /// <summary>
    /// Gets or sets the ridge parameter for regression regularization.
    /// </summary>
    /// <value>The ridge parameter, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// The ridge parameter controls the amount of regularization applied to the regression coefficients.
    /// Regularization helps prevent overfitting by penalizing large coefficient values. A larger ridge
    /// parameter results in more regularization (smaller coefficients), while a smaller value allows
    /// for larger coefficients. This setting only has an effect when IncludeRegression is set to true.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps prevent the model from becoming too complex and "overfitting" 
    /// the data (meaning it fits the training data too closely and performs poorly on new data). The default value 
    /// of 0.1 provides a moderate amount of regularization. If your model seems to be capturing too much noise in 
    /// the data, try increasing this value. If it's not capturing important patterns, try decreasing it. This setting 
    /// only matters if IncludeRegression is set to true.</para>
    /// </remarks>
    public double RidgeParameter { get; set; } = 0.1;
}
