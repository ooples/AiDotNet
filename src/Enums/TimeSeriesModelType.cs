namespace AiDotNet.Enums;

/// <summary>
/// Represents different types of time series forecasting models used for analyzing and predicting sequential data over time.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Time series models help us understand patterns in data that change over time and make predictions about future values.
/// 
/// Think of time series data as any measurement collected regularly over time - like daily temperature readings, 
/// monthly sales figures, or hourly website traffic. These models help us answer questions like:
/// 
/// - "What will our sales be next month?"
/// - "How many visitors will our website get tomorrow?"
/// - "What will the temperature be next week?"
/// 
/// Different models are designed to capture different patterns in time data:
/// - Some are good at finding seasonal patterns (like holiday shopping spikes)
/// - Others excel at detecting long-term trends (like gradual population growth)
/// - Some can handle sudden changes or outliers (like a viral social media post)
/// 
/// The right model depends on your specific data and what patterns you expect to find in it.
/// </remarks>
public enum TimeSeriesModelType
{
    /// <summary>
    /// Auto-Regressive Integrated Moving Average model - a standard statistical method for time series forecasting
    /// that combines autoregression, differencing, and moving average components.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> ARIMA is like a Swiss Army knife for time series forecasting - it's versatile and widely used.
    /// 
    /// ARIMA looks at three aspects of your data:
    /// - AR (Auto-Regressive): How much the current value depends on previous values
    /// - I (Integrated): How many times we need to subtract consecutive values to make the data stable
    /// - MA (Moving Average): How much random "noise" from previous time points affects the current value
    /// 
    /// ARIMA is typically written as ARIMA(p,d,q) where:
    /// - p = number of lag observations (AR component)
    /// - d = degree of differencing (I component)
    /// - q = size of the moving average window (MA component)
    /// 
    /// When to use it: When your data doesn't have strong seasonal patterns and you need a reliable, 
    /// well-established forecasting method.
    /// 
    /// Example: Forecasting monthly product sales that show a general trend but no clear seasonal pattern.
    /// </remarks>
    ARIMA,

    /// <summary>
    /// Seasonal Auto-Regressive Integrated Moving Average model - extends ARIMA to handle data with seasonal patterns.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> SARIMA is like ARIMA but with added capabilities to handle predictable seasonal patterns.
    /// 
    /// If your data shows regular patterns that repeat (like higher ice cream sales every summer), 
    /// SARIMA can capture both the overall trend and these seasonal fluctuations.
    /// 
    /// SARIMA adds seasonal components to the standard ARIMA model and is typically written as 
    /// SARIMA(p,d,q)(P,D,Q)m where:
    /// - (p,d,q) are the non-seasonal parameters (same as ARIMA)
    /// - (P,D,Q) are the seasonal parameters
    /// - m is the number of time periods in each season (e.g., 12 for monthly data with yearly seasonality)
    /// 
    /// When to use it: When your data shows clear seasonal patterns that repeat at regular intervals.
    /// 
    /// Example: Retail sales data that shows holiday shopping spikes every December.
    /// </remarks>
    SARIMA,

    /// <summary>
    /// Auto-Regressive Moving Average model - combines autoregressive and moving average components
    /// without the differencing (integration) step.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> ARMA is a simpler version of ARIMA used when your data is already stable.
    /// 
    /// ARMA combines:
    /// - AR (Auto-Regressive): How past values influence the current value
    /// - MA (Moving Average): How past random fluctuations influence the current value
    /// 
    /// The key difference from ARIMA is that ARMA assumes your data is already "stationary" 
    /// (meaning its statistical properties don't change over time).
    /// 
    /// When to use it: When your data doesn't show strong trends that change over time.
    /// 
    /// Example: Analyzing stable temperature fluctuations around a consistent average.
    /// </remarks>
    ARMA,

    /// <summary>
    /// Auto-Regressive model - predicts future values based solely on past values of the same variable.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> AutoRegressive models predict the future based only on the past values of the data itself.
    ///
    /// Think of AutoRegressive like predicting tomorrow's temperature based only on today's and yesterday's temperatures.
    /// The model assumes that recent past values have the strongest influence on the next value.
    ///
    /// AR(p) means we're using the previous p values to make our prediction.
    ///
    /// When to use it: When you believe recent past values strongly influence future values.
    ///
    /// Example: Predicting stock prices where recent price movements tend to influence short-term future prices.
    /// </remarks>
    AutoRegressive,

    /// <summary>
    /// Moving Average model - predicts future values based on past forecast errors rather than past values.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> MA models focus on the errors or surprises in previous predictions.
    /// 
    /// Instead of using past values directly, MA uses past prediction errors (the difference between 
    /// what we predicted and what actually happened). It's like saying "I was off by X yesterday, 
    /// so I should adjust my prediction today."
    /// 
    /// MA(q) means we're using the previous q prediction errors to make our forecast.
    /// 
    /// When to use it: When random shocks or events have lingering effects on your data.
    /// 
    /// Example: Call center volume forecasting where unexpected spikes (like after a product issue) 
    /// affect call volumes for several days.
    /// </remarks>
    MA,

    /// <summary>
    /// A general class of forecasting methods that give more weight to recent observations and less weight to older observations.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Exponential Smoothing is like having a weighted average where newer data points matter more than older ones.
    /// 
    /// Imagine you're trying to predict tomorrow's temperature. You might care more about today's temperature 
    /// than what happened a week ago. Exponential smoothing does exactly this - it gives more importance to 
    /// recent observations and less to older ones.
    /// 
    /// This is a general category that includes several specific methods (simple, double, and triple).
    /// 
    /// When to use it: When recent observations are more relevant to your prediction than older ones.
    /// 
    /// Example: Forecasting customer demand where recent purchasing patterns are more relevant than historical ones.
    /// </remarks>
    ExponentialSmoothing,

    /// <summary>
    /// The most basic form of exponential smoothing that handles data with no clear trend or seasonality.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Simple Exponential Smoothing is the most basic version that works well for stable data without trends.
    /// 
    /// This method calculates a weighted average of all past observations, with weights decreasing exponentially 
    /// as observations get older. It's controlled by a single parameter (alpha) that determines how quickly 
    /// the influence of past observations decays.
    /// 
    /// When to use it: When your data fluctuates around a stable average with no clear upward/downward trend 
    /// or seasonal patterns.
    /// 
    /// Example: Forecasting stable inventory levels for products with consistent demand.
    /// </remarks>
    SimpleExponentialSmoothing,

    /// <summary>
    /// An extension of simple exponential smoothing that can handle data with a trend component.
    /// Also known as Holt's method.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Double Exponential Smoothing adds the ability to handle data that shows an upward or downward trend.
    /// 
    /// While Simple Exponential Smoothing works for data that fluctuates around a stable average, 
    /// Double Exponential Smoothing (also called Holt's method) can handle data that's consistently 
    /// increasing or decreasing over time.
    /// 
    /// It uses two smoothing parameters:
    /// - Alpha: Controls how much recent levels affect the forecast
    /// - Beta: Controls how much recent trends affect the forecast
    /// 
    /// When to use it: When your data shows a consistent upward or downward trend.
    /// 
    /// Example: Forecasting a company's growing monthly revenue that shows a steady upward trend.
    /// </remarks>
    DoubleExponentialSmoothing,

    /// <summary>
    /// An extension of double exponential smoothing that can handle data with both trend and seasonal components.
    /// Also known as Holt-Winters' method.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Triple Exponential Smoothing handles data with both trends and seasonal patterns.
    /// 
    /// This method (also called Holt-Winters) is the most comprehensive of the exponential smoothing family. 
    /// It can model:
    /// - The overall level of the data
    /// - Upward or downward trends
    /// - Seasonal patterns that repeat at regular intervals
    /// 
    /// It uses three smoothing parameters:
    /// - Alpha: Controls how much recent levels affect the forecast
    /// - Beta: Controls how much recent trends affect the forecast
    /// - Gamma: Controls how much recent seasonal patterns affect the forecast
    /// 
    /// When to use it: When your data shows both a trend and regular seasonal patterns.
    /// 
    /// Example: Forecasting ice cream sales that show both an overall increasing trend and higher sales every summer.
    /// </remarks>
    TripleExponentialSmoothing,

    /// <summary>
    /// A flexible framework for time series modeling that represents a system's behavior using state variables.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> State Space models track the "hidden state" of a system that we can't directly observe.
    /// 
    /// Imagine you're trying to track a person's location based only on their cell phone signal strength. 
    /// You can't directly see where they are (the hidden state), but you can make educated guesses based 
    /// on the signal strength you observe.
    /// 
    /// State Space models work similarly for time series - they try to uncover the underlying state of the 
    /// system that's generating your observable data.
    /// 
    /// When to use it: When you believe there are underlying factors driving your data that aren't directly observable.
    /// 
    /// Example: Tracking the true economic health of a country based on various economic indicators.
    /// </remarks>
    StateSpace,

    /// <summary>
    /// A flexible time series model that handles complex seasonal patterns using trigonometric components.
    /// TBATS stands for Trigonometric, Box-Cox transform, ARMA errors, Trend, and Seasonal components.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> TBATS is a specialized model for handling complex or multiple seasonal patterns.
    /// 
    /// While models like SARIMA work well for simple seasonal patterns, TBATS can handle more complex situations like:
    /// - Multiple seasonal patterns (e.g., daily, weekly, and yearly patterns all at once)
    /// - Changing seasonal patterns
    /// - Non-integer seasonality (e.g., 365.25 days in a year)
    /// 
    /// TBATS uses mathematical techniques (trigonometric functions) to represent these complex patterns.
    /// 
    /// When to use it: When your data has multiple or complex seasonal patterns.
    /// 
    /// Example: Hourly electricity demand data that shows daily patterns, weekly patterns, and yearly seasonal patterns all at once.
    /// </remarks>
    TBATS,

    /// <summary>
    /// A model that combines regression with ARIMA modeling of the error terms to account for both external factors and time dependencies.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This model combines traditional regression with time series techniques.
    /// 
    /// Imagine you're forecasting ice cream sales. You know temperature affects sales (higher temperature = higher sales), 
    /// but there are also time-based patterns. This model lets you:
    /// 1. Use regression to account for the temperature effect
    /// 2. Use ARIMA to model the remaining patterns in your data
    /// 
    /// When to use it: When external factors influence your time series and you want to account for both these factors 
    /// and time-based patterns.
    /// 
    /// Example: Forecasting product sales while accounting for price changes, marketing spend, and seasonal patterns.
    /// </remarks>
    DynamicRegressionWithARIMAErrors,

    /// <summary>
    /// ARIMA model with additional explanatory variables (exogenous variables) that can influence the forecast.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> ARIMAX is like ARIMA but allows you to include external factors that might affect your forecast.
    /// 
    /// Standard ARIMA only looks at past values of the thing you're trying to predict. ARIMAX lets you add other 
    /// relevant variables. For example, when forecasting ice cream sales, you could include temperature as an 
    /// external factor.
    /// 
    /// The "X" in ARIMAX stands for "exogenous variables" - factors outside your main time series that influence it.
    /// 
    /// When to use it: When you know external factors influence what you're trying to predict.
    /// 
    /// Example: Forecasting energy consumption while accounting for temperature, day of week, and holidays.
    /// </remarks>
    ARIMAX,

    /// <summary>
    /// Generalized Autoregressive Conditional Heteroskedasticity model - specialized for forecasting volatility in time series.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> GARCH models are designed specifically for predicting how much a value will fluctuate or vary over time.
    /// 
    /// While most time series models focus on predicting the actual values, GARCH focuses on predicting the volatility 
    /// (how much the values jump around). This is especially useful in finance where understanding risk (volatility) 
    /// is often as important as predicting prices.
    /// 
    /// GARCH models recognize that periods of high volatility tend to cluster together - if today is highly volatile, 
    /// tomorrow is likely to be volatile too.
    /// 
    /// When to use it: When you care about forecasting the variability or uncertainty in your data, not just the values themselves.
    /// 
    /// Example: Forecasting stock market volatility to assess investment risk.
    /// </remarks>
    GARCH,

    /// <summary>
    /// Vector Autoregression model - extends autoregressive models to multiple related time series that influence each other.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> VAR models handle multiple related time series that affect each other.
    /// 
    /// While models like ARIMA work with a single time series, VAR handles multiple related series simultaneously. 
    /// For example, prices of related products might influence each other - if beef prices rise, chicken demand 
    /// (and then prices) might also increase as consumers switch.
    /// 
    /// VAR captures these interactions and lets each variable be influenced by its own past values AND the past values 
    /// of other variables in the system.
    /// 
    /// When to use it: When you have multiple time series that influence each other.
    /// 
    /// Example: Analyzing how changes in interest rates, inflation, and unemployment affect each other over time.
    /// </remarks>
    VAR,

    /// <summary>
    /// Vector Autoregression Moving-Average model - combines VAR and moving average components for multiple related time series.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> VARMA extends the VAR model by adding moving average components.
    /// 
    /// VARMA combines two approaches:
    /// - Vector Autoregression (VAR): How past values of multiple related variables affect current values
    /// - Moving Average (MA): How past random shocks or surprises affect current values
    /// 
    /// This gives VARMA more flexibility to capture complex relationships between multiple time series.
    /// 
    /// When to use it: When you have multiple related time series with complex interactions that aren't 
    /// fully captured by simpler models.
    /// 
    /// Example: Analyzing interactions between economic indicators like GDP, unemployment, and inflation 
    /// where both past values and unexpected events influence each other.
    /// </remarks>
    VARMA,

    // Machine Learning based models

    /// <summary>
    /// A forecasting model developed by Facebook that handles multiple seasonality patterns and is robust to missing data and outliers.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Prophet is a user-friendly forecasting tool designed by Facebook to be easy to use while still being powerful.
    /// 
    /// Prophet works like an advanced version of decomposition methods - it breaks down your time series into:
    /// - Trend: The overall direction (increasing or decreasing)
    /// - Seasonality: Regular patterns at different time scales (daily, weekly, yearly)
    /// - Holiday effects: Irregular but predictable events
    /// 
    /// What makes Prophet special:
    /// - It's robust to missing data and outliers
    /// - It can automatically detect changepoints (where trends shift)
    /// - It handles multiple seasonal patterns well
    /// - It allows you to incorporate domain knowledge easily
    /// 
    /// When to use it: When you need reliable forecasts without extensive time series expertise, 
    /// especially for data with strong seasonal patterns or irregular events.
    /// 
    /// Example: Forecasting business metrics like website traffic or product demand that have 
    /// weekly patterns, yearly seasonality, and holiday effects.
    /// </remarks>
    ProphetModel,

    /// <summary>
    /// A hybrid model that combines neural networks with traditional ARIMA models to leverage the strengths of both approaches.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This hybrid model combines traditional statistical methods with modern machine learning.
    /// 
    /// Think of this as a "best of both worlds" approach:
    /// - ARIMA provides a solid statistical foundation that works well for linear patterns
    /// - Neural Networks add the ability to capture complex, non-linear patterns
    /// 
    /// The hybrid approach typically works by:
    /// 1. Using ARIMA to model the linear components of your data
    /// 2. Using neural networks to capture the remaining non-linear patterns
    /// 
    /// When to use it: When your data contains both simple linear patterns and complex non-linear relationships 
    /// that a single model type might miss.
    /// 
    /// Example: Forecasting energy consumption where there are both predictable patterns (like daily cycles) 
    /// and complex relationships with multiple factors (weather, events, etc.).
    /// </remarks>
    NeuralNetworkARIMA,

    // Bayesian models

    /// <summary>
    /// A flexible Bayesian approach to time series modeling that incorporates prior knowledge and uncertainty.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This model uses Bayesian statistics to incorporate uncertainty and prior knowledge into forecasts.
    /// 
    /// Bayesian models are different because they:
    /// - Start with "prior beliefs" about what might happen
    /// - Update these beliefs as new data comes in
    /// - Express results as probability distributions rather than single-point forecasts
    /// 
    /// The "structural" part means the model breaks down your time series into components like trend, 
    /// seasonality, and cycle components.
    /// 
    /// When to use it: When you have prior knowledge about your data that you want to incorporate, 
    /// or when understanding the uncertainty in your forecast is important.
    /// 
    /// Example: Economic forecasting where you want to incorporate expert knowledge and clearly 
    /// communicate the range of possible outcomes and their probabilities.
    /// </remarks>
    BayesianStructuralTimeSeriesModel,

    // Spectral Analysis models

    /// <summary>
    /// Analyzes time series data by decomposing it into different frequency components to identify cyclical patterns.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Spectral Analysis is like breaking down a song into individual notes to understand its composition.
    /// 
    /// This approach transforms time series data from the time domain to the frequency domain. 
    /// Instead of asking "what happens next?", it asks "what cycles or rhythms exist in this data?"
    /// 
    /// Imagine your data as a complex musical chord - spectral analysis breaks it down into the 
    /// individual notes (frequencies) that make up that chord.
    /// 
    /// When to use it: When you want to identify hidden cycles or periodic patterns in your data, 
    /// especially when multiple cycles might be overlapping.
    /// 
    /// Example: Analyzing sunspot activity to identify various solar cycles, or analyzing economic 
    /// data to identify business cycles of different lengths.
    /// </remarks>
    SpectralAnalysis,

    // Decomposition models

    /// <summary>
    /// Seasonal and Trend decomposition using Loess - breaks down time series into trend, seasonal, and remainder components.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> STL Decomposition is like separating a smoothie back into its original ingredients.
    /// 
    /// This method breaks your time series into three components:
    /// - Trend: The long-term progression (increasing or decreasing)
    /// - Seasonality: Regular patterns that repeat at fixed intervals
    /// - Remainder: What's left after removing trend and seasonality (irregular fluctuations)
    /// 
    /// The "Loess" part refers to a statistical method used to estimate the trend component 
    /// through local regression.
    /// 
    /// When to use it: When you want to understand the different components driving your time series 
    /// or remove seasonality before further analysis.
    /// 
    /// Example: Breaking down retail sales data to separate the overall growth trend from 
    /// seasonal holiday patterns and random fluctuations.
    /// </remarks>
    STLDecomposition,

    // Other specialized models

    /// <summary>
    /// Analyzes how specific events or interventions affect a time series and quantifies their impact.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Intervention Analysis helps measure how specific events changed your time series.
    /// 
    /// This approach focuses on quantifying the impact of known events or "interventions" on your data. 
    /// For example:
    /// - How did a marketing campaign affect sales?
    /// - What was the impact of a policy change on crime rates?
    /// - How did a website redesign affect user engagement?
    /// 
    /// The analysis typically compares what actually happened after the intervention with what 
    /// would have happened without it (the "counterfactual").
    /// 
    /// When to use it: When you want to measure the impact of specific events or actions on your time series.
    /// 
    /// Example: Analyzing how a price change affected product demand by comparing actual sales 
    /// after the change with the predicted sales if no change had occurred.
    /// </remarks>
    InterventionAnalysis,

    /// <summary>
    /// Models how one time series affects another with potential time delays between cause and effect.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Transfer Function Models help understand how one variable affects another over time.
    /// 
    /// These models are designed to capture how changes in an input variable (X) affect an output 
    /// variable (Y) over time, including:
    /// - How strong the effect is
    /// - How long it takes for the effect to appear (delay)
    /// - How long the effect lasts
    /// 
    /// Unlike simple correlation, transfer functions can model complex relationships where effects 
    /// are spread out over time.
    /// 
    /// When to use it: When you want to model how one time series influences another, especially 
    /// when there are time delays between cause and effect.
    /// 
    /// Example: Modeling how advertising expenditure affects sales over time, where spending today 
    /// might influence sales for several weeks or months.
    /// </remarks>
    TransferFunctionModel,

    /// <summary>
    /// Models time series by representing them as combinations of unobserved components like trend, cycle, and seasonality.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This model breaks down your time series into hidden components that can't be directly observed.
    /// 
    /// Unobserved Components Models (UCM) assume your time series is made up of several underlying 
    /// components that you can't directly measure, such as:
    /// - Trend: The long-term direction
    /// - Cycle: Medium-term fluctuations
    /// - Seasonality: Regular patterns that repeat
    /// - Irregular: Random fluctuations
    /// 
    /// The model uses statistical techniques to estimate these hidden components from your observable data.
    /// 
    /// When to use it: When you want a flexible framework for decomposing your time series into 
    /// meaningful components for analysis or forecasting.
    /// 
    /// Example: Analyzing economic indicators by separating long-term growth trends from business 
    /// cycles and seasonal patterns.
    /// </remarks>
    UnobservedComponentsModel,

    /// <summary>
    /// Represents a custom or user-defined time series model not covered by the standard types.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This option allows you to implement your own specialized time series model.
    /// 
    /// The Custom option gives you flexibility to:
    /// - Implement models not included in the standard list
    /// - Create hybrid approaches combining multiple techniques
    /// - Develop domain-specific models tailored to your particular data
    /// 
    /// When to use it: When standard models don't meet your specific needs or when you want 
    /// to implement a novel approach to time series analysis.
    /// 
    /// Example: Implementing a specialized forecasting model designed specifically for your 
    /// industry or unique data characteristics.
    /// </remarks>
    Custom
}
