namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the N-BEATS (Neural Basis Expansion Analysis for Time Series) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// N-BEATS is a deep learning architecture specifically designed for time series forecasting.
/// It uses a hierarchical doubly residual architecture with basis expansion to decompose
/// time series into trend and seasonality components, providing both accurate forecasts
/// and interpretability.
/// </para>
/// <para><b>For Beginners:</b> N-BEATS is a modern neural network approach to time series forecasting
/// that can automatically learn patterns from your data without requiring manual feature engineering.
///
/// Key concepts:
/// - Stacks: Groups of blocks that process the data hierarchically
/// - Blocks: Individual processing units within each stack
/// - Lookback Window: How many past time steps to consider for predictions
/// - Forecast Horizon: How many future time steps to predict
/// - Hidden Size: The capacity of the network (larger values can learn more complex patterns)
///
/// The model automatically decomposes your time series into interpretable components like
/// trend (long-term direction) and seasonality (repeating patterns).
/// </para>
/// </remarks>
public class NBEATSModelOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the number of stacks in the N-BEATS architecture.
    /// </summary>
    /// <value>The number of stacks, defaulting to 30.</value>
    /// <remarks>
    /// <para>
    /// Each stack consists of multiple blocks that process the time series hierarchically.
    /// The original N-BEATS paper uses 30 stacks for generic architecture.
    /// </para>
    /// <para><b>For Beginners:</b> Stacks are layers of processing in the network. More stacks
    /// allow the model to learn deeper patterns but require more computation and training data.
    /// The default of 30 works well for most applications.
    /// </para>
    /// </remarks>
    public int NumStacks { get; set; } = 30;

    /// <summary>
    /// Gets or sets the number of blocks per stack.
    /// </summary>
    /// <value>The number of blocks per stack, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// Each block performs basis expansion and produces backcast (past) and forecast (future) predictions.
    /// Multiple blocks per stack allow for more complex representations.
    /// </para>
    /// <para><b>For Beginners:</b> Blocks are the basic building units within each stack.
    /// Each block learns to extract specific patterns from the data. More blocks per stack
    /// can improve accuracy but increase training time.
    /// </para>
    /// </remarks>
    public int NumBlocksPerStack { get; set; } = 1;

    /// <summary>
    /// Gets or sets the polynomial degree for trend basis expansion.
    /// </summary>
    /// <value>The polynomial degree, defaulting to 3.</value>
    /// <remarks>
    /// <para>
    /// Controls the complexity of polynomial trends the model can represent. Higher degrees
    /// allow for more complex trend shapes but may lead to overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how complex the trend patterns can be.
    /// A degree of 3 means the model can represent trends that curve up and down.
    /// Higher values allow for more wiggly trends, while lower values are more restricted.
    /// </para>
    /// </remarks>
    public int PolynomialDegree { get; set; } = 3;

    /// <summary>
    /// Gets or sets the lookback window size (number of historical time steps used as input).
    /// </summary>
    /// <value>The lookback window size, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// Determines how many past observations are used to predict future values.
    /// Should be set based on the expected temporal dependencies in your data.
    /// </para>
    /// <para><b>For Beginners:</b> This is how far back in time the model looks when making
    /// predictions. If predicting daily sales, a lookback of 7 means it considers the past week.
    /// Larger values allow the model to see more history but require more computation.
    /// </para>
    /// </remarks>
    public int LookbackWindow { get; set; } = 10;

    /// <summary>
    /// Gets or sets the forecast horizon (number of future time steps to predict).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 5.</value>
    /// <remarks>
    /// <para>
    /// Specifies how many time steps ahead the model should forecast. The model is
    /// trained to produce forecasts for all steps from 1 to ForecastHorizon simultaneously.
    /// </para>
    /// <para><b>For Beginners:</b> This is how far into the future you want to predict.
    /// If forecasting daily sales with a horizon of 5, the model will predict the next 5 days.
    /// Longer horizons are generally harder to predict accurately.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 5;

    /// <summary>
    /// Gets or sets the hidden layer size for the fully connected layers within each block.
    /// </summary>
    /// <value>The hidden layer size, defaulting to 256.</value>
    /// <remarks>
    /// <para>
    /// Controls the capacity of the neural network within each block. Larger values
    /// allow the model to learn more complex patterns but increase computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how many neurons are in the hidden layers
    /// of the network. More neurons mean the model can learn more complex relationships,
    /// but require more training data and computation. 256 is a good starting point for
    /// most time series problems.
    /// </para>
    /// </remarks>
    public int HiddenLayerSize { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of hidden layers within each block.
    /// </summary>
    /// <value>The number of hidden layers, defaulting to 4.</value>
    /// <remarks>
    /// <para>
    /// Each block uses multiple fully connected layers. More layers allow for deeper
    /// feature learning but increase model complexity.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many layers of processing each block has.
    /// More layers allow the model to learn more abstract patterns, similar to how
    /// deep neural networks can recognize complex features. The default of 4 works well
    /// for most applications.
    /// </para>
    /// </remarks>
    public int NumHiddenLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the learning rate for training the model.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.001.</value>
    /// <remarks>
    /// <para>
    /// Controls how quickly the model updates its parameters during training.
    /// Lower values lead to more stable but slower training, while higher values
    /// can speed up training but risk instability.
    /// </para>
    /// <para><b>For Beginners:</b> The learning rate controls how big the steps are when
    /// the model is learning. Think of it like learning to ride a bike - small steps (low
    /// learning rate) are safer but slower, while big steps (high learning rate) are faster
    /// but riskier. 0.001 is a conservative, safe choice.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>The number of epochs, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// An epoch is one complete pass through the entire training dataset.
    /// More epochs allow the model to learn better but increase training time.
    /// </para>
    /// <para><b>For Beginners:</b> An epoch is like one complete study session through all
    /// your training data. More epochs mean more practice, which usually leads to better
    /// learning, but too many can cause the model to memorize the training data instead
    /// of learning general patterns (overfitting).
    /// </para>
    /// </remarks>
    public int Epochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the batch size for training.
    /// </summary>
    /// <value>The batch size, defaulting to 32.</value>
    /// <remarks>
    /// <para>
    /// Determines how many samples are processed together before updating model parameters.
    /// Larger batches provide more stable gradients but require more memory.
    /// </para>
    /// <para><b>For Beginners:</b> Instead of updating the model after every single example,
    /// we group examples into batches. A batch size of 32 means the model looks at 32 examples
    /// before adjusting its parameters. This makes training more efficient and stable.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets whether to share weights across blocks within a stack.
    /// </summary>
    /// <value>True to share weights, defaulting to false.</value>
    /// <remarks>
    /// <para>
    /// When enabled, all blocks within a stack share the same weights, reducing the
    /// total number of parameters and acting as a regularization technique.
    /// </para>
    /// <para><b>For Beginners:</b> Weight sharing means that all blocks in a stack use
    /// the same learned patterns, which reduces the model size and can help prevent
    /// overfitting. However, it also limits the model's flexibility. The default is false,
    /// allowing each block to learn independently.
    /// </para>
    /// </remarks>
    public bool ShareWeightsInStack { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use interpretable basis functions (trend and seasonality).
    /// </summary>
    /// <value>True to use interpretable basis, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// When enabled, the model uses polynomial and Fourier basis functions for interpretability.
    /// When disabled, uses generic basis functions that may be more flexible but less interpretable.
    /// </para>
    /// <para><b>For Beginners:</b> Interpretable basis means the model explicitly separates
    /// the forecast into trend (long-term direction) and seasonal (repeating patterns) components
    /// that you can visualize and understand. This is helpful for explaining the model's predictions.
    /// The default is true to provide interpretability.
    /// </para>
    /// </remarks>
    public bool UseInterpretableBasis { get; set; } = true;
}
