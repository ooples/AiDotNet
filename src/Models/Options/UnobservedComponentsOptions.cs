namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Unobserved Components Models (UCM), which decompose time series into
/// trend, seasonal, cycle, and irregular components.
/// </summary>
/// <remarks>
/// <para>
/// Unobserved Components Models (UCM), also known as Structural Time Series Models, provide a flexible 
/// framework for decomposing time series data into distinct components that are not directly observable. 
/// These components typically include trend (long-term movement), seasonal (regular patterns at fixed 
/// intervals), cycle (irregular fluctuations of varying length), and irregular (random noise) components. 
/// UCM is particularly useful for understanding the underlying structure of time series data, forecasting, 
/// and detecting structural changes. This approach is based on state space models and is often estimated 
/// using the Kalman filter. This class provides configuration options for controlling the components 
/// included in the model and the estimation process.
/// </para>
/// <para><b>For Beginners:</b> Unobserved Components Models help you break down time series data into meaningful parts.
/// 
/// When analyzing time series data:
/// - It's often useful to separate the data into different components
/// - These components aren't directly observable but can be estimated
/// 
/// UCM decomposes time series into:
/// - Trend: The long-term direction (upward, downward, or stable)
/// - Seasonal: Regular patterns that repeat at fixed intervals (daily, weekly, monthly, etc.)
/// - Cycle: Irregular fluctuations that don't have a fixed period
/// - Irregular: Random noise or unexplained variation
/// 
/// This approach offers several benefits:
/// - Better understanding of what drives the time series
/// - Improved forecasting by modeling each component separately
/// - Ability to detect structural changes over time
/// - Flexibility to include or exclude components based on domain knowledge
/// 
/// This class lets you configure which components to include and how to estimate them.
/// </para>
/// </remarks>
public class UnobservedComponentsOptions<T, TInput, TOutput> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the maximum number of iterations for the estimation algorithm.
    /// </summary>
    /// <value>A positive integer, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum number of iterations for the estimation algorithm used to fit the 
    /// Unobserved Components Model. The estimation process iteratively refines the parameter estimates until 
    /// convergence is reached or the maximum number of iterations is exceeded. A larger value allows for more 
    /// iterations and potentially better parameter estimates but increases the computation time. The default 
    /// value of 100 provides a reasonable upper limit for many applications, allowing sufficient iterations 
    /// for convergence while preventing excessive computation. The optimal value depends on the complexity of 
    /// the data and the desired trade-off between accuracy and computation time.
    /// </para>
    /// <para><b>For Beginners:</b> This setting limits how long the model will spend trying to find the best parameters.
    /// 
    /// The maximum iterations:
    /// - Sets an upper limit on the number of optimization steps
    /// - Prevents the algorithm from running indefinitely
    /// - Balances computation time with parameter quality
    /// 
    /// The default value of 100 means:
    /// - The algorithm will perform at most 100 iterations to find optimal parameters
    /// - This is sufficient for many time series problems
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 200): More opportunity to find better parameters, but longer runtime
    /// - Lower values (e.g., 50): Faster results, but potentially suboptimal parameters
    /// 
    /// When to adjust this value:
    /// - Increase it for complex time series with multiple components
    /// - Decrease it when computational resources are limited or for simpler time series
    /// - If the algorithm frequently hits this limit, consider increasing it
    /// 
    /// For example, for a complex time series with strong seasonal and cyclical patterns,
    /// you might increase this to 200 to ensure the model has enough iterations to converge.
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets a value indicating whether to optimize the model parameters.
    /// </summary>
    /// <value>A boolean value, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// This property specifies whether the model parameters should be optimized during the estimation process. 
    /// When set to true, the algorithm will search for the parameter values that maximize the likelihood of the 
    /// observed data. When set to false, the algorithm will use the initial parameter values without optimization, 
    /// which can be useful when the parameters are known or when performing sensitivity analysis. The default 
    /// value of true enables parameter optimization, which is appropriate for most applications where the optimal 
    /// parameter values are unknown. Disabling optimization might be preferred when the parameters have been 
    /// pre-determined based on domain knowledge or when computational efficiency is a priority.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether the model searches for the best parameters or uses the ones you provide.
    /// 
    /// Parameter optimization:
    /// - When enabled, the model finds the best parameters to fit your data
    /// - When disabled, the model uses the initial parameters without changing them
    /// - Affects both model quality and computation time
    /// 
    /// The default value of true means:
    /// - The model will automatically search for the optimal parameters
    /// - This is appropriate for most situations where you don't know the best parameters
    /// 
    /// Think of it like this:
    /// - Enabled (true): Let the model find the best parameters (recommended for most cases)
    /// - Disabled (false): Use fixed parameters that you specify
    /// 
    /// When to adjust this value:
    /// - Keep enabled (true) for most applications
    /// - Disable (false) when you have specific parameters you want to use
    /// - Disable for sensitivity analysis or when parameters are known from domain knowledge
    /// 
    /// For example, if you're an expert who knows the exact variance parameters for each component,
    /// you might set this to false and provide those specific parameters.
    /// </para>
    /// </remarks>
    public bool OptimizeParameters { get; set; } = true;

    /// <summary>
    /// Gets or sets the optimizer used for parameter estimation.
    /// </summary>
    /// <value>An instance of a class implementing IOptimizer&lt;T&gt;, or null to use the default optimizer.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the optimization algorithm used to estimate the parameters of the Unobserved 
    /// Components Model. Different optimizers have different strengths and weaknesses in terms of convergence 
    /// speed, ability to escape local optima, and sensitivity to initial conditions. The default value of null 
    /// indicates that a default optimizer appropriate for the specific model implementation should be used. 
    /// Common optimization algorithms for UCM include maximum likelihood estimation via BFGS or L-BFGS, EM 
    /// (Expectation-Maximization), and Bayesian methods. The optimal choice depends on the specific characteristics 
    /// of the data and the desired trade-off between estimation accuracy and computational efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the algorithm used to find the best model parameters.
    /// 
    /// The optimizer:
    /// - Controls how the model finds the optimal parameter values
    /// - Different optimizers have different strengths and weaknesses
    /// - Can significantly affect both the quality of results and computation time
    /// 
    /// The default value of null means:
    /// - The system will choose an appropriate default optimizer
    /// - This is suitable for most applications
    /// 
    /// Common optimizer types include:
    /// - BFGS: Efficient for smooth likelihood functions
    /// - EM: Often used for state space models like UCM
    /// - Bayesian: Provides uncertainty estimates for parameters
    /// 
    /// When to adjust this value:
    /// - Specify a different optimizer when the default doesn't converge well
    /// - Choose based on whether you prioritize speed or solution quality
    /// - This is an advanced setting that most users can leave as default
    /// 
    /// For example, if the default optimizer struggles with your complex model,
    /// you might specify a more robust optimizer that's less likely to get
    /// stuck in local optima.
    /// </para>
    /// </remarks>
    public IOptimizer<T, Matrix<T>, Vector<T>>? Optimizer { get; set; }

    /// <summary>
    /// Gets or sets the seasonal period for the model.
    /// </summary>
    /// <value>A positive integer, defaulting to 1 (no seasonality).</value>
    /// <remarks>
    /// <para>
    /// This property specifies the length of the seasonal cycle in the Unobserved Components Model. A value 
    /// greater than 1 indicates that a seasonal component should be included in the model, with the specified 
    /// period. For example, a value of 4 for quarterly data would model annual seasonality, while a value of 12 
    /// for monthly data would also model annual seasonality. A value of 1 (the default) indicates that no 
    /// seasonal component should be included. This property overrides the SeasonalPeriod property inherited 
    /// from the base class to provide a more appropriate default value for UCM. The optimal value depends on 
    /// the known or expected seasonal patterns in the data and the frequency of the observations.
    /// </para>
    /// <para><b>For Beginners:</b> This setting specifies the length of seasonal patterns in your data.
    /// 
    /// The seasonal period:
    /// - Defines the length of repeating patterns in your data
    /// - Determines whether and how seasonality is modeled
    /// - Should match the known or expected seasonal cycle in your data
    /// 
    /// The default value of 1 means:
    /// - No seasonality is included in the model
    /// - This is appropriate for data without seasonal patterns
    /// 
    /// Common values for different data frequencies:
    /// - Monthly data: 12 (annual seasonality)
    /// - Quarterly data: 4 (annual seasonality)
    /// - Daily data: 7 (weekly seasonality) or 365 (annual seasonality)
    /// - Hourly data: 24 (daily seasonality)
    /// 
    /// When to adjust this value:
    /// - Set based on your knowledge of the data and its natural cycles
    /// - Set to 1 to exclude seasonality from the model
    /// - This setting overrides the value inherited from the base class
    /// 
    /// For example, for monthly sales data with annual patterns,
    /// you would set this to 12 to capture the yearly seasonality.
    /// </para>
    /// </remarks>
    public new int SeasonalPeriod { get; set; } = 1;

    /// <summary>
    /// Gets or sets a value indicating whether to include a cycle component in the model.
    /// </summary>
    /// <value>A boolean value, defaulting to false.</value>
    /// <remarks>
    /// <para>
    /// This property specifies whether a cycle component should be included in the Unobserved Components Model. 
    /// Unlike the seasonal component, which has a fixed period, the cycle component models irregular fluctuations 
    /// with a stochastic period that can vary over time. Cycles are often used to model business cycles or other 
    /// medium to long-term fluctuations that don't have a fixed periodicity. The default value of false excludes 
    /// the cycle component, which is appropriate for many applications where such irregular cycles are not present 
    /// or not of interest. Including a cycle component increases the flexibility of the model but also the number 
    /// of parameters to estimate and the risk of overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether the model includes irregular cyclical patterns.
    /// 
    /// The cycle component:
    /// - Models irregular fluctuations that don't have a fixed period
    /// - Different from seasonality, which has a constant period
    /// - Useful for business cycles, economic cycles, or other varying-length patterns
    /// 
    /// The default value of false means:
    /// - No cycle component is included in the model
    /// - This is appropriate for many time series without irregular cycles
    /// 
    /// Think of it like this:
    /// - Enabled (true): Include a component for irregular cycles with varying length
    /// - Disabled (false): Don't model irregular cyclical patterns
    /// 
    /// When to adjust this value:
    /// - Enable (true) when your data shows irregular cycles beyond seasonality
    /// - Keep disabled (false) for simpler models or when cycles aren't present
    /// - Enable for economic data, which often exhibits business cycles
    /// 
    /// For example, for GDP or unemployment data that shows business cycle fluctuations,
    /// you would set this to true to capture these irregular cycles.
    /// </para>
    /// </remarks>
    public bool IncludeCycle { get; set; } = false;

    /// <summary>
    /// Gets or sets the smoothing parameter for the cycle component.
    /// </summary>
    /// <value>A positive double value, defaulting to 1600.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the smoothing parameter (lambda) for the cycle component in the Unobserved 
    /// Components Model. This parameter controls the smoothness of the estimated cycle, with larger values 
    /// resulting in smoother cycles. The default value of 1600 is a common choice for quarterly data, based on 
    /// the Hodrick-Prescott filter literature. For monthly data, a value of 14400 is often used, while for annual 
    /// data, a value of 100 is common. This parameter is only relevant when IncludeCycle is set to true. The 
    /// optimal value depends on the frequency of the data and the expected smoothness of the cycle component.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how smooth the cycle component should be.
    /// 
    /// The cycle lambda parameter:
    /// - Controls the smoothness of the estimated cycle component
    /// - Higher values produce smoother cycles with less variation
    /// - Only relevant when IncludeCycle is true
    /// 
    /// The default value of 1600 means:
    /// - The cycle will be moderately smooth
    /// - This is a standard value for quarterly economic data
    /// 
    /// Common values for different data frequencies:
    /// - Annual data: 100
    /// - Quarterly data: 1600
    /// - Monthly data: 14400
    /// - Daily data: 107000
    /// 
    /// When to adjust this value:
    /// - Increase it when you want smoother, more gradual cycles
    /// - Decrease it when you want to allow more variation in the cycle
    /// - Adjust based on the frequency of your data
    /// 
    /// For example, for monthly data with business cycles,
    /// you would typically increase this to 14400 to get appropriate smoothing.
    /// </para>
    /// </remarks>
    public double CycleLambda { get; set; } = 1600;

    /// <summary>
    /// Gets or sets the minimum period for the cycle component.
    /// </summary>
    /// <value>A positive integer, defaulting to 2.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the minimum period (in terms of the time unit of the data) for the cycle component 
    /// in the Unobserved Components Model. It sets a lower bound on how short the stochastic cycle can be. For 
    /// example, with quarterly data, a minimum period of 2 means the cycle cannot be shorter than 2 quarters 
    /// (6 months). This parameter is only relevant when IncludeCycle is set to true. The default value of 2 
    /// provides a reasonable lower bound for many applications, preventing very high-frequency fluctuations from 
    /// being modeled as cycles. The optimal value depends on the frequency of the data and the minimum cycle 
    /// length that is meaningful in the specific application.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets a lower limit on how short the cycles can be.
    /// 
    /// The cycle minimum period:
    /// - Sets the shortest possible length for a cycle
    /// - Prevents very short fluctuations from being considered cycles
    /// - Only relevant when IncludeCycle is true
    /// 
    /// The default value of 2 means:
    /// - Cycles must be at least 2 time units long
    /// - This prevents very short-term fluctuations from being modeled as cycles
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 4): Only longer cycles will be modeled
    /// - Lower values (e.g., 2): Allows shorter cycles to be captured
    /// 
    /// When to adjust this value:
    /// - Increase it when you only want to model medium to long-term cycles
    /// - Keep at the default for most applications
    /// - Should be set based on domain knowledge about meaningful cycle lengths
    /// 
    /// For example, for quarterly economic data where business cycles are typically at least 2 years,
    /// you might increase this to 8 (8 quarters = 2 years) to focus on business cycles.
    /// </para>
    /// </remarks>
    public int CycleMinPeriod { get; set; } = 2;

    /// <summary>
    /// Gets or sets the maximum period for the cycle component.
    /// </summary>
    /// <value>A positive integer, defaulting to 40.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum period (in terms of the time unit of the data) for the cycle component 
    /// in the Unobserved Components Model. It sets an upper bound on how long the stochastic cycle can be. For 
    /// example, with quarterly data, a maximum period of 40 means the cycle cannot be longer than 40 quarters 
    /// (10 years). This parameter is only relevant when IncludeCycle is true. The default value of 40 provides a 
    /// reasonable upper bound for many applications, particularly for quarterly economic data where business cycles 
    /// typically don't exceed 10 years. The optimal value depends on the frequency of the data and the maximum cycle 
    /// length that is meaningful in the specific application.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets an upper limit on how long the cycles can be.
    /// 
    /// The cycle maximum period:
    /// - Sets the longest possible length for a cycle
    /// - Prevents very long fluctuations from being considered cycles
    /// - Only relevant when IncludeCycle is true
    /// 
    /// The default value of 40 means:
    /// - Cycles must be no longer than 40 time units
    /// - For quarterly data, this corresponds to 10 years
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 60): Allows longer cycles to be modeled
    /// - Lower values (e.g., 20): Only shorter cycles will be captured
    /// 
    /// When to adjust this value:
    /// - Increase it when you want to model very long cycles
    /// - Decrease it when you only want to focus on shorter cycles
    /// - Should be set based on domain knowledge about meaningful cycle lengths
    /// 
    /// For example, for monthly data where you want to capture cycles up to 8 years,
    /// you would set this to 96 (96 months = 8 years).
    /// </para>
    /// </remarks>
    public int CycleMaxPeriod { get; set; } = 40;

    /// <summary>
    /// Gets or sets the matrix decomposition method used in the estimation algorithm.
    /// </summary>
    /// <value>An instance of a class implementing IMatrixDecomposition&lt;T&gt;, or null to use the default decomposition.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the matrix decomposition method used in the estimation algorithm for the Unobserved 
    /// Components Model. Matrix decomposition is used to solve linear systems that arise during the Kalman filter 
    /// and smoother steps of the estimation process. Different decomposition methods have different numerical 
    /// properties and computational requirements. The default value of null indicates that a default decomposition 
    /// method appropriate for the specific model implementation should be used. Common decomposition methods include 
    /// Cholesky decomposition (efficient for positive definite matrices), QR decomposition (more stable for 
    /// ill-conditioned matrices), and SVD (Singular Value Decomposition, the most stable but also the most 
    /// computationally expensive). The optimal choice depends on the numerical properties of the specific problem 
    /// and the desired trade-off between numerical stability and computational efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the mathematical method used to solve equations during model estimation.
    /// 
    /// The matrix decomposition:
    /// - Affects how certain mathematical operations are performed during estimation
    /// - Different methods have different trade-offs between speed and numerical stability
    /// - Most users don't need to change this setting
    /// 
    /// The default value of null means:
    /// - The system will choose an appropriate default decomposition method
    /// - This is suitable for most applications
    /// 
    /// Common decomposition methods include:
    /// - Cholesky: Efficient for well-behaved problems
    /// - QR: More stable for difficult numerical problems
    /// - SVD: Most numerically stable, but significantly slower
    /// 
    /// When to adjust this value:
    /// - Specify a different decomposition when the default causes numerical issues
    /// - Choose based on whether you prioritize speed or numerical stability
    /// - This is an advanced setting that rarely needs adjustment
    /// 
    /// For example, if your model estimation process fails with numerical errors,
    /// you might specify a more stable decomposition method like SVD.
    /// </para>
    /// </remarks>
    public IMatrixDecomposition<T>? Decomposition { get; set; }
}
