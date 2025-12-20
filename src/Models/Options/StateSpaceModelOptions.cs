namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for State Space Models, which represent time series data through
/// hidden states and observable outputs for forecasting and analysis.
/// </summary>
/// <typeparam name="T">The data type used in matrix operations for the state space model.</typeparam>
/// <remarks>
/// <para>
/// State Space Models (SSMs) are a flexible class of time series models that represent the dynamics of a system 
/// through hidden states and observable outputs. They provide a unified framework for modeling various time series 
/// patterns, including trends, seasonality, and cycles. The core of a state space model consists of two equations: 
/// a state equation that describes how the hidden state evolves over time, and an observation equation that relates 
/// the hidden state to the observed data. Common examples of state space models include the Kalman Filter, 
/// Hidden Markov Models, and structural time series models. This class provides configuration options for state 
/// space models, including the dimensions of the state and observation vectors, learning parameters for model 
/// estimation, and convergence criteria. These options allow customization of the model complexity and fitting 
/// process to match the specific characteristics of the time series being analyzed.
/// </para>
/// <para><b>For Beginners:</b> State Space Models help analyze time series data by tracking hidden variables that influence observable measurements.
/// 
/// In many real-world systems:
/// - We can only measure certain outputs (like temperature, price, or position)
/// - But these measurements are influenced by hidden internal states
/// - State Space Models help us track these hidden states over time
/// 
/// For example, in tracking a moving object:
/// - We might only observe its position at certain times (observations)
/// - But its velocity and acceleration are hidden states that affect future positions
/// - A state space model can estimate these hidden states from the observations
/// 
/// These models are powerful because they:
/// - Handle noisy measurements
/// - Can incorporate multiple influencing factors
/// - Provide a framework for forecasting future values
/// - Work well with missing data
/// 
/// Common applications include:
/// - Economic forecasting
/// - Object tracking in computer vision
/// - Signal processing
/// - Financial time series analysis
/// 
/// This class lets you configure the structure and training process for state space models.
/// </para>
/// </remarks>
public class StateSpaceModelOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the dimension of the state vector in the state space model.
    /// </summary>
    /// <value>A positive integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the dimension of the state vector, which represents the hidden variables in the 
    /// state space model. The state vector captures the internal dynamics of the system that are not directly 
    /// observable but influence the observed data. A larger state size allows the model to capture more complex 
    /// dynamics but increases the number of parameters to estimate and the risk of overfitting. The default value 
    /// of 1 is appropriate for simple time series with a single underlying trend or level. For more complex time 
    /// series with multiple components (e.g., trend, seasonality, and cycle), a larger state size might be needed. 
    /// For example, a basic local level model might use a state size of 1, while a local linear trend model might 
    /// use a state size of 2 (for level and slope), and a model with additional seasonal components would require 
    /// an even larger state size.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many hidden variables the model will track.
    /// 
    /// The state size defines:
    /// - How many hidden variables (states) the model uses internally
    /// - How complex the internal representation of the system can be
    /// 
    /// The default value of 1 means:
    /// - The model tracks a single hidden state (like a basic level or trend)
    /// - This is sufficient for simple time series
    /// 
    /// Think of it like this:
    /// - StateSize=1: Tracks just the level (like current temperature)
    /// - StateSize=2: Might track level and trend (temperature and how fast it's changing)
    /// - StateSize=4: Could track level, trend, and seasonal components (temperature, trend, and daily/weekly patterns)
    /// 
    /// When to adjust this value:
    /// - Increase it when your data has multiple components (trend, seasonality, cycles)
    /// - Keep it small when you have limited data or want a simpler model
    /// 
    /// For example, in economic forecasting, a StateSize of 3 might represent the current level,
    /// growth rate, and seasonal component of an economic indicator.
    /// </para>
    /// </remarks>
    public int StateSize { get; set; } = 1;

    /// <summary>
    /// Gets or sets the dimension of the observation vector in the state space model.
    /// </summary>
    /// <value>A positive integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the dimension of the observation vector, which represents the variables that are 
    /// directly measured or observed. In many time series applications, there is only one observed variable at 
    /// each time point, corresponding to the default value of 1. However, in multivariate time series analysis, 
    /// multiple variables might be observed simultaneously, requiring a larger observation size. For example, in 
    /// economic forecasting, one might simultaneously observe GDP, inflation, and unemployment rate, requiring an 
    /// observation size of 3. The observation size should match the number of distinct time series being modeled 
    /// jointly. A larger observation size allows the model to capture relationships between multiple observed 
    /// variables but increases the complexity of the model and the amount of data required for reliable estimation.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many different measurements or outputs the model works with at each time point.
    /// 
    /// The observation size defines:
    /// - How many different variables you're measuring at each time point
    /// - Whether you're working with a single time series or multiple related series
    /// 
    /// The default value of 1 means:
    /// - The model works with a single measurement at each time point
    /// - This is appropriate for most basic time series analysis
    /// 
    /// Think of it like this:
    /// - ObservationSize=1: Tracking just temperature over time
    /// - ObservationSize=2: Tracking both temperature and humidity together
    /// - ObservationSize=3: Tracking temperature, humidity, and pressure as related measurements
    /// 
    /// When to adjust this value:
    /// - Keep it at 1 for single time series analysis
    /// - Increase it when analyzing multiple related time series together
    /// 
    /// For example, in financial analysis, an ObservationSize of 3 might represent tracking
    /// stock price, trading volume, and volatility as related observations at each time point.
    /// </para>
    /// </remarks>
    public int ObservationSize { get; set; } = 1;

    /// <summary>
    /// Gets or sets the learning rate for gradient-based parameter estimation.
    /// </summary>
    /// <value>A positive double value, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the learning rate or step size used in gradient-based optimization algorithms for 
    /// estimating the parameters of the state space model. The learning rate controls how much the parameter 
    /// estimates are updated in each iteration based on the computed gradients. A larger learning rate can lead 
    /// to faster convergence but risks overshooting the optimal values or causing instability. A smaller learning 
    /// rate provides more stable updates but may require more iterations to converge. The default value of 0.01 
    /// provides a moderate learning rate suitable for many applications. Adaptive learning rate schedules or 
    /// optimization algorithms like Adam or RMSProp might adjust this base learning rate during the optimization 
    /// process. The optimal learning rate can depend on the scale and characteristics of the specific time series 
    /// being analyzed.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the model updates its parameters during training.
    /// 
    /// The learning rate determines:
    /// - How large the steps are when updating model parameters
    /// - The balance between speed of learning and stability
    /// 
    /// The default value of 0.01 means:
    /// - The model takes moderate-sized steps when updating parameters
    /// - This provides a good balance between learning speed and stability
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.1): Larger steps, faster learning but might overshoot
    /// - Lower values (e.g., 0.001): Smaller steps, more stable but slower learning
    /// 
    /// When to adjust this value:
    /// - Decrease it if training is unstable or parameters oscillate
    /// - Increase it if training is too slow to converge
    /// 
    /// For example, if your model's error is decreasing very slowly during training,
    /// you might increase the learning rate to 0.05 to speed up convergence.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the maximum number of iterations for the parameter estimation algorithm.
    /// </summary>
    /// <value>A positive integer, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum number of iterations allowed for the numerical optimization algorithm 
    /// used to estimate the parameters of the state space model. It serves as a stopping criterion to prevent the 
    /// algorithm from running indefinitely in cases where convergence is difficult to achieve. The default value 
    /// of 1000 is sufficient for many applications to achieve convergence. However, for complex models with many 
    /// parameters or difficult optimization landscapes, more iterations may be required. Conversely, for simpler 
    /// models or when approximate solutions are acceptable, fewer iterations may be sufficient. If the algorithm 
    /// reaches this maximum without converging according to the Tolerance criterion, it will return the best 
    /// solution found so far, possibly with a warning.
    /// </para>
    /// <para><b>For Beginners:</b> This setting limits how many attempts the algorithm makes to improve the model parameters.
    /// 
    /// During training:
    /// - The algorithm iteratively updates the model parameters
    /// - Each iteration aims to improve the model's fit to the data
    /// - This setting caps the total number of iterations
    /// 
    /// The default value of 1000 means:
    /// - The algorithm will make at most 1000 attempts to improve the parameters
    /// - This prevents the training process from running indefinitely
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 5000): More thorough optimization, potentially better results but longer training time
    /// - Lower values (e.g., 500): Faster training but might not find the optimal parameters
    /// 
    /// When to adjust this value:
    /// - Increase it for complex models or when you need very precise parameter estimates
    /// - Decrease it when you need faster results or have simpler models
    /// 
    /// For example, if your model is still improving significantly when it reaches 1000 iterations,
    /// you might increase this value to 2000 to allow for further optimization.
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the convergence tolerance for the parameter estimation algorithm.
    /// </summary>
    /// <value>A positive double value, defaulting to 1e-6 (0.000001).</value>
    /// <remarks>
    /// <para>
    /// This property specifies the convergence criterion for the numerical optimization algorithm used to estimate 
    /// the parameters of the state space model. The algorithm terminates when the relative change in the log-likelihood 
    /// function or parameter estimates between consecutive iterations is less than this tolerance value. A smaller 
    /// tolerance requires more precise convergence, potentially leading to more accurate parameter estimates but 
    /// requiring more iterations. A larger tolerance allows for earlier termination, potentially saving computational 
    /// resources at the cost of less precise parameter estimates. The default value of 1e-6 (0.000001) provides a 
    /// relatively strict convergence criterion suitable for many applications. For high-precision requirements, a 
    /// smaller value might be appropriate, while for exploratory analysis or when computational resources are limited, 
    /// a larger value might be used.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how precise the parameter estimates need to be before training stops.
    /// 
    /// During training:
    /// - The algorithm keeps refining the model parameters
    /// - This setting defines when the improvements are small enough to stop
    /// 
    /// The default value of 1e-6 (0.000001) means:
    /// - Training stops when parameter changes between iterations become very small
    /// - This provides precise parameter estimates for most applications
    /// 
    /// Think of it like this:
    /// - Smaller values (e.g., 1e-8): More precise parameter estimates, but might take more iterations
    /// - Larger values (e.g., 1e-4): Less precise estimates, but faster training
    /// 
    /// When to adjust this value:
    /// - Decrease it when you need extremely precise parameter estimates
    /// - Increase it when you need faster results and can accept slight imprecision
    /// 
    /// For example, in scientific applications requiring high precision, you might decrease
    /// this value to 1e-8, while for quick exploratory analysis, you might increase it to 1e-4.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-6;
}
