using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the MQCNN (Multi-Quantile Convolutional Neural Network) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// MQCNN is a probabilistic forecasting model that predicts multiple quantiles simultaneously,
/// providing uncertainty estimates along with point forecasts. It uses dilated causal convolutions
/// to model temporal dependencies and outputs predictions at multiple quantile levels.
/// </para>
/// <para><b>For Beginners:</b> MQCNN combines CNNs with quantile regression for forecasting:
///
/// <b>What is Quantile Forecasting?</b>
/// Instead of predicting a single value, MQCNN predicts a range:
/// - The 10th percentile (P10): "90% of actual values will be above this"
/// - The 50th percentile (P50): The median prediction
/// - The 90th percentile (P90): "90% of actual values will be below this"
///
/// <b>Why Multiple Quantiles?</b>
/// - Captures uncertainty in predictions
/// - Provides prediction intervals, not just point estimates
/// - Useful for risk management (worst-case/best-case scenarios)
/// - Better decision making with confidence bounds
///
/// <b>Example:</b>
/// For tomorrow's stock price, instead of "100.50", you get:
/// - P10: 98.20 (likely floor)
/// - P50: 100.50 (median)
/// - P90: 102.80 (likely ceiling)
///
/// <b>Architecture:</b>
/// 1. <b>Encoder:</b> Dilated causal convolutions process the input sequence
/// 2. <b>Context:</b> Extracted features represent temporal patterns
/// 3. <b>Decoder:</b> Separate output heads for each quantile
/// 4. <b>Loss:</b> Quantile loss (pinball loss) for each quantile level
/// </para>
/// <para>
/// <b>Reference:</b> Wen et al., "A Multi-Horizon Quantile Recurrent Forecaster", 2017.
/// https://arxiv.org/abs/1711.11053
/// </para>
/// </remarks>
public class MQCNNOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MQCNNOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default MQCNN configuration. The defaults are designed
    /// to work well for typical probabilistic forecasting tasks.
    /// </para>
    /// </remarks>
    public MQCNNOptions()
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
    public MQCNNOptions(MQCNNOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        Quantiles = other.Quantiles;
        EncoderChannels = other.EncoderChannels;
        DecoderChannels = other.DecoderChannels;
        KernelSize = other.KernelSize;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
    }

    /// <summary>
    /// Gets or sets the lookback window size (input sequence length).
    /// </summary>
    /// <value>The lookback window, defaulting to 168.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model looks at before
    /// making predictions. For hourly data, 168 would mean looking at the past week.
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
    /// Gets or sets the quantile levels to predict.
    /// </summary>
    /// <value>Array of quantile levels, defaulting to [0.1, 0.5, 0.9].</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The percentiles to predict. Default is:
    /// - 0.1 (10th percentile): Lower bound of 80% prediction interval
    /// - 0.5 (50th percentile): Median prediction
    /// - 0.9 (90th percentile): Upper bound of 80% prediction interval
    ///
    /// You can add more quantiles like [0.05, 0.25, 0.5, 0.75, 0.95] for finer
    /// uncertainty estimation, but this increases computation.
    /// </para>
    /// </remarks>
    public double[] Quantiles { get; set; } = [0.1, 0.5, 0.9];

    /// <summary>
    /// Gets or sets the number of channels in the encoder network.
    /// </summary>
    /// <value>The encoder channels, defaulting to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The encoder processes the input sequence through
    /// dilated convolutions. More channels mean more capacity to learn patterns
    /// but require more computation.
    /// </para>
    /// </remarks>
    public int EncoderChannels { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of channels in the decoder network.
    /// </summary>
    /// <value>The decoder channels, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The decoder takes the encoder's context and produces
    /// quantile predictions. Usually smaller than encoder since it's doing simpler work.
    /// </para>
    /// </remarks>
    public int DecoderChannels { get; set; } = 32;

    /// <summary>
    /// Gets or sets the kernel size for convolutional layers.
    /// </summary>
    /// <value>The kernel size, defaulting to 3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The size of the sliding window in convolutions.
    /// Combined with dilation, small kernels can cover large receptive fields.
    /// </para>
    /// </remarks>
    public int KernelSize { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of encoder layers.
    /// </summary>
    /// <value>The number of encoder layers, defaulting to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> More encoder layers mean deeper processing of
    /// the input sequence and larger receptive field, but also more computation.
    /// </para>
    /// </remarks>
    public int NumEncoderLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of decoder layers.
    /// </summary>
    /// <value>The number of decoder layers, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Decoder layers process the context for each quantile.
    /// Usually fewer than encoder layers since the heavy lifting is done by the encoder.
    /// </para>
    /// </remarks>
    public int NumDecoderLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent overfitting by randomly
    /// ignoring some neurons during training. 0.2 means 20% are ignored.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the learning rate for training.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How big of a step to take when updating the model.
    /// Smaller values learn slower but more precisely.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>The number of epochs, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many times to go through the entire training dataset.
    /// </para>
    /// </remarks>
    public int Epochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the batch size for training.
    /// </summary>
    /// <value>The batch size, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many training examples to process at once.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;
}
