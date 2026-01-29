using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the LSTNet (Long Short-Term Time-series Network) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// LSTNet is a neural network architecture specifically designed for multivariate time series forecasting.
/// It combines multiple components to capture patterns at different temporal scales:
/// - Convolutional layers for short-term local patterns
/// - Recurrent layers (GRU) for long-term dependencies
/// - Skip-RNN for very long periodic patterns
/// - Autoregressive component for local linear trends
/// </para>
/// <para><b>For Beginners:</b> LSTNet is like having multiple specialists working together to predict the future:
///
/// 1. The convolutional part is like scanning for local patterns - like noticing "sales always spike on weekends"
/// 2. The recurrent part remembers long-term trends - like "sales grow 10% each month"
/// 3. The skip-RNN looks for seasonal patterns - like "Christmas sales are always highest"
/// 4. The autoregressive part handles simple linear trends - like "each day is slightly higher than yesterday"
///
/// By combining all these, LSTNet can capture complex patterns in data where multiple time scales matter,
/// such as electricity consumption (hourly, daily, weekly patterns), stock prices, or traffic flow.
/// </para>
/// </remarks>
public class LSTNetOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LSTNetOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default configuration that works well for most
    /// time series forecasting tasks. You can adjust individual properties after creation.
    /// </para>
    /// </remarks>
    public LSTNetOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a copy of existing options, useful when you want
    /// to try variations of a working configuration without modifying the original.
    /// </para>
    /// </remarks>
    public LSTNetOptions(LSTNetOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        HiddenRecurrentSize = other.HiddenRecurrentSize;
        HiddenSkipSize = other.HiddenSkipSize;
        ConvolutionKernelSize = other.ConvolutionKernelSize;
        ConvolutionFilters = other.ConvolutionFilters;
        SkipPeriod = other.SkipPeriod;
        SkipRecurrentLayers = other.SkipRecurrentLayers;
        AutoregressiveWindow = other.AutoregressiveWindow;
        UseHighway = other.UseHighway;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
    }

    /// <summary>
    /// Gets or sets the lookback window size (context length).
    /// </summary>
    /// <value>The lookback window, defaulting to 168 (one week of hourly data).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model looks at before
    /// making a prediction. For hourly data, 168 would mean looking at the past week.
    /// This should be at least as long as the longest seasonal pattern you expect.
    /// </para>
    /// </remarks>
    public int LookbackWindow { get; set; } = 168;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 24.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict. For hourly data,
    /// 24 would mean predicting the next day.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the hidden state size of the main recurrent layers (GRU/LSTM).
    /// </summary>
    /// <value>The hidden recurrent size, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The capacity of the main memory component. Larger values
    /// allow the model to remember more complex long-term patterns but require more training data.
    /// </para>
    /// </remarks>
    public int HiddenRecurrentSize { get; set; } = 100;

    /// <summary>
    /// Gets or sets the hidden state size of the skip recurrent layers.
    /// </summary>
    /// <value>The hidden skip size, defaulting to 5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The capacity of the skip-connection memory. This component
    /// specializes in capturing periodic patterns (like daily or weekly cycles). Usually smaller
    /// than the main recurrent size.
    /// </para>
    /// </remarks>
    public int HiddenSkipSize { get; set; } = 5;

    /// <summary>
    /// Gets or sets the kernel size for convolutional layers.
    /// </summary>
    /// <value>The convolution kernel size, defaulting to 6.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many consecutive time steps the convolution looks at
    /// when scanning for local patterns. A size of 6 means it looks at 6 time steps at once.
    /// Larger values can capture longer local patterns but may miss shorter ones.
    /// </para>
    /// </remarks>
    public int ConvolutionKernelSize { get; set; } = 6;

    /// <summary>
    /// Gets or sets the number of convolutional filters.
    /// </summary>
    /// <value>The number of convolution filters, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many different local patterns the convolution layer
    /// can learn to detect. More filters mean more diverse patterns but more computation.
    /// </para>
    /// </remarks>
    public int ConvolutionFilters { get; set; } = 100;

    /// <summary>
    /// Gets or sets the skip period for the Skip-RNN component.
    /// </summary>
    /// <value>The skip period, defaulting to 24 (daily for hourly data).</value>
    /// <remarks>
    /// <para>
    /// This should match the main seasonal period in your data. For hourly data:
    /// - 24 = daily patterns
    /// - 168 = weekly patterns
    /// For daily data:
    /// - 7 = weekly patterns
    /// - 30 = monthly patterns
    /// </para>
    /// <para><b>For Beginners:</b> The Skip-RNN skips ahead by this many time steps,
    /// allowing it to directly compare today's 3 PM with yesterday's 3 PM (if skip=24 for hourly data).
    /// This helps capture seasonal patterns more efficiently.
    /// </para>
    /// </remarks>
    public int SkipPeriod { get; set; } = 24;

    /// <summary>
    /// Gets or sets the number of skip recurrent layers.
    /// </summary>
    /// <value>The number of skip recurrent layers, defaulting to 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many layers of skip-RNN to use. One layer is usually
    /// sufficient for most applications.
    /// </para>
    /// </remarks>
    public int SkipRecurrentLayers { get; set; } = 1;

    /// <summary>
    /// Gets or sets the window size for the autoregressive component.
    /// </summary>
    /// <value>The autoregressive window, defaulting to 24.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The autoregressive component models simple linear relationships
    /// between recent values and future values. This sets how many recent values it considers.
    /// Set to 0 to disable the autoregressive component.
    /// </para>
    /// </remarks>
    public int AutoregressiveWindow { get; set; } = 24;

    /// <summary>
    /// Gets or sets whether to use highway connections.
    /// </summary>
    /// <value>True to use highway connections, defaulting to true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Highway connections are like shortcuts that allow information
    /// to flow directly from earlier layers to later ones. They help the model train faster
    /// and capture both simple and complex patterns. Generally, leave this enabled.
    /// </para>
    /// </remarks>
    public bool UseHighway { get; set; } = true;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent overfitting by randomly
    /// ignoring some neurons during training. A value of 0.2 means 20% are ignored.
    /// LSTNet typically uses higher dropout than simpler models.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the learning rate for training.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How big of a step to take when updating the model.
    /// Smaller values learn slower but more precisely. 0.001 is a good starting point.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>The number of epochs, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many times to go through the entire training dataset.
    /// More epochs generally improve accuracy up to a point, after which overfitting may occur.
    /// </para>
    /// </remarks>
    public int Epochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the batch size for training.
    /// </summary>
    /// <value>The batch size, defaulting to 128.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many training examples to process at once.
    /// Larger batches train faster but use more memory. LSTNet typically uses larger
    /// batches than some other models.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 128;
}
