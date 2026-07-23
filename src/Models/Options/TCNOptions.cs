using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the TCN (Temporal Convolutional Network) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// TCN uses dilated causal convolutions to model temporal sequences. Unlike recurrent networks,
/// TCN processes sequences in parallel, making it faster to train while still capturing
/// long-range dependencies through its exponentially increasing dilation factors.
/// </para>
/// <para><b>For Beginners:</b> TCN is a powerful alternative to LSTM/GRU for sequence modeling:
///
/// <b>Key Concepts:</b>
/// - <b>Causal Convolutions:</b> Each prediction only depends on past values, never future ones
///   (important for real-time prediction)
/// - <b>Dilated Convolutions:</b> Instead of looking at consecutive time steps, TCN skips steps
///   with increasing gaps (dilation). With dilations [1, 2, 4, 8], the network can "see" far
///   into the past without needing huge filters.
///
/// <b>Example:</b>
/// - Layer 1 (dilation=1): Looks at times [t-2, t-1, t]
/// - Layer 2 (dilation=2): Looks at times [t-4, t-2, t]
/// - Layer 3 (dilation=4): Looks at times [t-8, t-4, t]
/// - Together: Can see 14 time steps back with only 3 layers!
///
/// <b>Benefits:</b>
/// - Parallelizable (faster training than RNNs)
/// - Flexible receptive field (controls how far back to look)
/// - No vanishing gradient problem
/// - Good for long sequences
/// </para>
/// <para>
/// <b>Reference:</b> Bai et al., "An Empirical Evaluation of Generic Convolutional and Recurrent
/// Networks for Sequence Modeling", 2018. https://arxiv.org/abs/1803.01271
/// </para>
/// </remarks>
public class TCNOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TCNOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default TCN configuration. The defaults are designed
    /// to work well for common time series forecasting tasks.
    /// </para>
    /// </remarks>
    public TCNOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a copy of existing options, useful when you want
    /// to try variations of a working configuration.
    /// </para>
    /// </remarks>
    public TCNOptions(TCNOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        NumChannels = other.NumChannels;
        KernelSize = other.KernelSize;
        NumLayers = other.NumLayers;
        DropoutRate = other.DropoutRate;
        UseResidualConnections = other.UseResidualConnections;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
    }

    /// <summary>
    /// Gets or sets the lookback window size (input sequence length).
    /// </summary>
    /// <value>The lookback window, defaulting to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model looks at before
    /// making a prediction. For hourly data, 96 would mean looking at the past 4 days.
    /// </para>
    /// </remarks>
    public int LookbackWindow { get; set; } = 96;

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
    /// Gets or sets the number of channels (filters) in each convolutional layer.
    /// </summary>
    /// <value>The number of channels, defaulting to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each channel learns to detect a different pattern.
    /// More channels mean the network can recognize more diverse patterns but requires
    /// more computation. 64 is a good balance for most tasks.
    /// </para>
    /// </remarks>
    public int NumChannels { get; set; } = 64;

    /// <summary>
    /// Gets or sets the kernel size for convolutional layers.
    /// </summary>
    /// <value>The kernel size, defaulting to 3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The size of the sliding window used to scan the time series.
    /// A kernel size of 3 means each convolution looks at 3 consecutive time steps.
    /// Combined with dilation, even small kernels can have a large receptive field.
    /// </para>
    /// </remarks>
    public int KernelSize { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of TCN layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 8.</value>
    /// <remarks>
    /// <para>
    /// The number of layers determines the receptive field of the network:
    /// Receptive field = 1 + 2 * (KernelSize - 1) * (2^NumLayers - 1)
    ///
    /// With KernelSize=3 and NumLayers=8:
    /// Receptive field = 1 + 2 * 2 * (256 - 1) = 1021 time steps
    /// </para>
    /// <para><b>For Beginners:</b> More layers allow the network to look further into the past.
    /// With 8 layers, TCN can effectively consider over 1000 past time steps.
    /// Each additional layer roughly doubles the receptive field.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 8;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent overfitting by randomly
    /// ignoring some neurons during training. A value of 0.2 means 20% are ignored.
    /// TCN typically uses dropout between convolutional layers.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets whether to use residual connections.
    /// </summary>
    /// <value>True to use residual connections, defaulting to true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Residual connections add the input of each block
    /// directly to its output: output = block(input) + input. This helps with:
    /// - Gradient flow during training
    /// - Allowing the network to learn identity mappings
    /// - Training deeper networks more effectively
    ///
    /// Keep this enabled for best results.
    /// </para>
    /// </remarks>
    public bool UseResidualConnections { get; set; } = true;

    /// <summary>
    /// Gets or sets the learning rate for training.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How big of a step to take when updating the model.
    /// Smaller values learn slower but more precisely. 0.001 is a standard starting point.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>The number of epochs, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many times to go through the entire training dataset.
    /// TCN often trains faster than RNNs, so you may need fewer epochs.
    /// </para>
    /// </remarks>
    public int Epochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the batch size for training.
    /// </summary>
    /// <value>The batch size, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many training examples to process at once.
    /// TCN can often handle larger batches than RNNs because it processes
    /// sequences in parallel.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;
}
