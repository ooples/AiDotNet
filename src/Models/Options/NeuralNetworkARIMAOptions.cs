namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Neural Network ARIMA (AutoRegressive Integrated Moving Average) models, 
/// which combine traditional statistical time series methods with neural networks for improved forecasting.
/// </summary>
/// <remarks>
/// <para>
/// Neural Network ARIMA is a hybrid approach that extends traditional ARIMA models by incorporating
/// neural networks to capture complex nonlinear patterns in time series data. This approach leverages
/// both the statistical foundation of ARIMA for handling linear dependencies, seasonality, and trends,
/// while using neural networks to model nonlinear relationships that traditional ARIMA cannot capture.
/// The resulting model can provide more accurate forecasts for time series with complex patterns,
/// regime shifts, or other nonlinear dynamics that are common in real-world data.
/// </para>
/// <para><b>For Beginners:</b> Neural Network ARIMA combines two powerful approaches for predicting future values in a time series:
/// 
/// Imagine you're trying to predict tomorrow's temperature:
/// - ARIMA is like using mathematical formulas that look at recent temperatures and patterns
/// - Neural Networks are like having a smart system that can learn complex relationships from data
/// - Neural Network ARIMA combines both approaches to get better predictions
/// 
/// Traditional ARIMA is good at capturing:
/// - How today's temperature relates to yesterday's (AR - AutoRegressive)
/// - How random fluctuations from previous days affect today (MA - Moving Average)
/// 
/// But it struggles with complex patterns like:
/// - When a cold front arrives, temperatures might drop suddenly but then recover differently than normal
/// - How humidity and cloud cover might interact in non-obvious ways to affect temperature
/// 
/// The neural network part helps capture these complex relationships, while the ARIMA part
/// ensures the basic time patterns are properly handled. This combination often produces
/// better forecasts than either approach alone.
/// 
/// This class lets you configure both the ARIMA components and the neural network that will
/// work together to make predictions.
/// </para>
/// </remarks>
public class NeuralNetworkARIMAOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets a value indicating whether the model should attempt to optimize its parameters during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, the model may use the configured <see cref="Optimizer"/> (or a default optimizer) to refine
    /// AR/MA coefficients and neural network parameters beyond their initial estimates.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Turning this on tells the model to spend extra time trying to find better parameter values. This can
    /// improve accuracy, but it can also make training take longer.
    /// </para>
    /// </remarks>
    public bool OptimizeParameters { get; set; } = false;

    /// <summary>
    /// Gets or sets the AutoRegressive (AR) order, which determines how many previous time steps
    /// are used as inputs to predict the current value.
    /// </summary>
    /// <value>The AR order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The AR order specifies the number of lagged observations of the time series itself that are
    /// included in the model as predictors. This parameter is crucial for capturing direct dependencies
    /// between current values and past values. Higher AR orders allow the model to capture longer-term
    /// dependencies but increase model complexity and computational requirements. The optimal AR order
    /// often depends on the inherent memory of the process generating the time series.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many previous time points
    /// the model looks at to make predictions.
    /// 
    /// Imagine predicting today's temperature:
    /// - AR order = 1: Only yesterday's temperature is considered
    /// - AR order = 7: An entire week of previous temperatures is considered
    /// 
    /// The default value of 1 means:
    /// - The model only looks at the most recent value to predict the next one
    /// - This is often a good starting point for many time series
    /// 
    /// You might want a higher value if:
    /// - Your data shows patterns that depend on values from multiple time periods ago
    /// - The series has weekly, monthly, or other cyclic patterns
    /// - You have enough data to support learning these longer-term relationships
    /// 
    /// You might want to keep it at 1 if:
    /// - You have limited data
    /// - The most recent value is the strongest predictor
    /// - You're dealing with a fast-changing time series where older values become irrelevant quickly
    /// 
    /// AR stands for "AutoRegressive" - using the series' own past values to predict its future.
    /// </para>
    /// </remarks>
    public int AROrder { get; set; } = 1;

    /// <summary>
    /// The order of differencing applied to the time series data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Specifies how many times the differencing operation is applied to the time series to achieve stationarity.
    /// Differencing helps remove trends and seasonality by computing differences between consecutive observations.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how the model handles trends in your data.
    /// 
    /// Imagine you're tracking daily temperatures:
    /// - If temperatures are steadily rising over time (a trend), it's harder to predict exact values
    /// - Differencing transforms the data to focus on changes rather than absolute values
    /// - For example, instead of predicting "it will be 75°F tomorrow," the model might work with 
    ///   "it will be 2°F warmer than today"
    /// 
    /// The differencing order tells the model how many times to apply this transformation:
    /// - Order 0: Use the original values (no differencing)
    /// - Order 1: Use the differences between consecutive values
    /// - Order 2: Use the differences of the differences
    /// 
    /// Higher orders help handle more complex trends, but too high may introduce unnecessary complexity.
    /// </para>
    /// </remarks>
    public int DifferencingOrder { get; set; }

    /// <summary>
    /// Gets or sets the Moving Average (MA) order, which determines how many previous error terms
    /// are used in the prediction model.
    /// </summary>
    /// <value>The MA order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The MA order specifies the number of lagged forecast errors that are included in the model.
    /// This component captures the relationship between the current value and previous random shocks
    /// to the system. MA terms help model the short-term effects of unexpected events or noise in
    /// the time series. Higher MA orders allow the model to account for longer-lasting effects of
    /// past disturbances but can make the model more difficult to estimate reliably.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many previous prediction errors
    /// the model considers.
    /// 
    /// Continuing with our temperature example:
    /// - Suppose yesterday the model predicted 75°F but it was actually 78°F (an error of +3°F)
    /// - The MA part lets the model learn from this error
    /// 
    /// The default value of 1 means:
    /// - The model only considers the most recent prediction error
    /// - This helps adjust for recent unexpected changes
    /// 
    /// You might want a higher value if:
    /// - Your data shows that effects from unexpected events tend to linger
    /// - The time series has persistent errors in one direction
    /// - You notice that after a big miss, the model continues to miss in following periods
    /// 
    /// You might want to keep it at 1 if:
    /// - Errors tend to be random and don't show patterns
    /// - You want to keep the model simpler
    /// - You have limited data
    /// 
    /// MA stands for "Moving Average" - though it's not actually taking averages of the data,
    /// but rather modeling how unexpected shocks continue to affect future values.
    /// </para>
    /// </remarks>
    public int MAOrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of lagged predictions to use as inputs to the neural network.
    /// </summary>
    /// <value>The number of lagged predictions, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines how many of the model's own previous predictions are fed back into
    /// the neural network as inputs for making the next prediction. This creates a recurrent structure
    /// even in feedforward networks, allowing the model to capture temporal dependencies that span
    /// multiple time steps. Using lagged predictions can improve performance in cases where future
    /// values depend not just on past observations but on how those observations were predicted.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many of the model's own
    /// previous predictions it uses to make new predictions.
    /// 
    /// This is different from AR order:
    /// - AR order uses actual historical values from your data
    /// - LaggedPredictions uses the model's own previous predictions
    /// 
    /// The default value of 1 means:
    /// - The model considers its most recent prediction when making a new one
    /// - This helps create continuity in forecasts
    /// 
    /// You might want a higher value if:
    /// - You're making longer-term forecasts
    /// - The way values change over time is as important as the values themselves
    /// - Your time series has momentum that should persist in forecasts
    /// 
    /// You might want to keep it at 1 if:
    /// - You're primarily interested in short-term forecasts
    /// - You want to minimize error accumulation
    /// - You prefer simpler model behavior
    /// 
    /// This feature helps the neural network portion develop a "memory" of its own predictions,
    /// which can lead to more coherent forecasting sequences.
    /// </para>
    /// </remarks>
    public int LaggedPredictions { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of exogenous (external) variables to include in the model.
    /// </summary>
    /// <value>The number of exogenous variables, defaulting to 0.</value>
    /// <remarks>
    /// <para>
    /// Exogenous variables are external factors not part of the time series itself but that may
    /// influence its behavior. This parameter specifies how many such variables are incorporated
    /// into the model. Including relevant exogenous variables can significantly improve forecast
    /// accuracy by accounting for external drivers of the time series. When this value is set to
    /// a positive number, the model expects these additional variables to be provided during training
    /// and forecasting.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many external factors
    /// (beyond the time series itself) the model will consider.
    /// 
    /// Going back to our temperature example:
    /// - Time series data: Historical daily temperatures
    /// - Exogenous variables might include: humidity, cloud cover, wind speed
    /// 
    /// The default value of 0 means:
    /// - The model only looks at the time series itself
    /// - No external factors are considered
    /// 
    /// You might want to increase this value if:
    /// - You know there are specific external factors that influence your data
    /// - You have reliable data for these external factors
    /// - These external factors help explain sudden changes or patterns in your data
    /// 
    /// For example, if you're forecasting:
    /// - Retail sales, exogenous variables might include: holidays, promotions, weather
    /// - Energy demand, exogenous variables might include: temperature, day of week, special events
    /// - Stock prices, exogenous variables might include: interest rates, sector indices, news sentiment
    /// 
    /// Adding exogenous variables can make your model much more accurate, but requires having
    /// this additional data available both for training and when making future predictions.
    /// </para>
    /// </remarks>
    public int ExogenousVariables { get; set; } = 0;

    /// <summary>
    /// Gets or sets the neural network to use in the hybrid model.
    /// </summary>
    /// <value>The neural network instance, defaulting to null (in which case a default network will be created).</value>
    /// <remarks>
    /// <para>
    /// This property allows specification of a custom neural network architecture to be used within
    /// the hybrid ARIMA model. When provided, this neural network will be used to model the nonlinear
    /// components of the time series after the linear ARIMA components have been applied. The network
    /// should accept inputs that conform to the AR order, MA order, and exogenous variables specified,
    /// and produce outputs suitable for the forecasting task. If left null, a default architecture
    /// appropriate for the specified parameters will be created automatically.
    /// </para>
    /// <para><b>For Beginners:</b> This setting lets you provide your own custom neural network
    /// instead of using the default one.
    /// 
    /// The neural network is the "smart" part of the model that learns complex patterns:
    /// - It takes inputs (past values, errors, external factors)
    /// - It processes them through layers of interconnected nodes
    /// - It produces predictions based on patterns it learned during training
    /// 
    /// The default value of null means:
    /// - The system will automatically create a suitable neural network for you
    /// - This is convenient and works well in many cases
    /// 
    /// You might want to provide your own network if:
    /// - You have expertise in neural network design
    /// - Your time series has very specific characteristics that need a specialized architecture
    /// - You want to experiment with different network structures
    /// 
    /// For example, you might create:
    /// - A deeper network with more layers for very complex data
    /// - A network with specific activation functions suited to your data characteristics
    /// - A network with dropout or regularization to prevent overfitting
    /// 
    /// If you're new to neural networks, it's recommended to start with the default (null)
    /// and let the system create an appropriate network automatically.
    /// </para>
    /// </remarks>
    public INeuralNetwork<T>? NeuralNetwork { get; set; }

    /// <summary>
    /// Gets or sets the optimizer to use for training the neural network component.
    /// </summary>
    /// <value>The optimizer instance, defaulting to null (in which case a default optimizer will be used).</value>
    /// <remarks>
    /// <para>
    /// This property allows specification of the optimization algorithm to be used when training the
    /// neural network portion of the model. The optimizer controls how the network's weights are updated
    /// based on the calculated gradients during backpropagation. Different optimizers have various
    /// properties regarding convergence speed, likelihood of finding global optima, and behavior in the
    /// presence of noisy gradients. If left null, a default optimizer (typically a variant of stochastic
    /// gradient descent with momentum) will be used.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the method used to train
    /// the neural network part of your model.
    /// 
    /// Think of training a neural network like descending a mountain to find the lowest point:
    /// - The optimizer is the strategy you use to navigate downhill
    /// - Different strategies have different tradeoffs in speed, accuracy, and reliability
    /// 
    /// The default value of null means:
    /// - The system will choose a standard optimizer for you
    /// - This is usually Adam or SGD with momentum, which work well for many problems
    /// 
    /// You might want to provide your own optimizer if:
    /// - Your model is getting stuck in training
    /// - You need faster convergence
    /// - You're dealing with a particularly challenging dataset
    /// 
    /// Common optimizer options include:
    /// - Adam: Good all-around performer that adapts learning rates
    /// - SGD with momentum: Simple but effective, especially with proper tuning
    /// - RMSProp: Good for non-stationary objectives
    /// - Nesterov Accelerated Gradient: Helps avoid overshooting minima
    /// 
    /// If you're new to time series forecasting, start with the default (null) optimizer
    /// and only change it if you encounter specific training issues.
    /// </para>
    /// </remarks>
    public IOptimizer<T, Matrix<T>, Vector<T>>? Optimizer { get; set; }
}
