using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the WaveNet model adapted for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// WaveNet was originally developed by DeepMind for audio generation but has proven highly effective
/// for time series forecasting. It uses dilated causal convolutions with gated activations and
/// residual/skip connections.
/// </para>
/// <para><b>For Beginners:</b> WaveNet is similar to TCN but with some key differences:
///
/// <b>Gated Activation Units:</b>
/// Instead of simple ReLU activations, WaveNet uses gates:
/// - tanh(Wf * x) * sigmoid(Wg * x)
/// - The sigmoid acts as a "gate" controlling information flow
/// - This helps model complex patterns more effectively
///
/// <b>Skip Connections:</b>
/// WaveNet has TWO types of connections:
/// 1. <b>Residual:</b> Connect input to output of each block (like TCN)
/// 2. <b>Skip:</b> Each block also sends output directly to the final layers
///    - This allows the network to combine features from different time scales
///
/// <b>Stacked Dilations:</b>
/// WaveNet often repeats the dilation pattern multiple times:
/// - [1, 2, 4, 8, 16, 1, 2, 4, 8, 16, ...]
/// - This creates very deep networks with huge receptive fields
///
/// Originally designed for generating audio one sample at a time, WaveNet's architecture
/// is now used for many sequence prediction tasks including financial forecasting.
/// </para>
/// <para>
/// <b>Reference:</b> van den Oord et al., "WaveNet: A Generative Model for Raw Audio", 2016.
/// https://arxiv.org/abs/1609.03499
/// </para>
/// </remarks>
public class WaveNetOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="WaveNetOptions{T}"/> class with default values.
    /// </summary>
    public WaveNetOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public WaveNetOptions(WaveNetOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        ResidualChannels = other.ResidualChannels;
        SkipChannels = other.SkipChannels;
        DilationDepth = other.DilationDepth;
        NumStacks = other.NumStacks;
        KernelSize = other.KernelSize;
        UseGatedActivations = other.UseGatedActivations;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
    }

    /// <summary>
    /// Gets or sets the lookback window size (input sequence length).
    /// </summary>
    /// <value>The lookback window, defaulting to 128.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model looks at.
    /// WaveNet can handle longer sequences efficiently due to its dilated structure.
    /// </para>
    /// </remarks>
    public int LookbackWindow { get; set; } = 128;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 24.</value>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the number of residual channels.
    /// </summary>
    /// <value>The number of residual channels, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The dimensionality of the main processing pathway.
    /// This determines the capacity of each layer to represent features.
    /// </para>
    /// </remarks>
    public int ResidualChannels { get; set; } = 32;

    /// <summary>
    /// Gets or sets the number of skip channels.
    /// </summary>
    /// <value>The number of skip channels, defaulting to 256.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Skip connections collect information from all layers
    /// and combine them before the output. More skip channels mean richer combinations.
    /// Often larger than residual channels.
    /// </para>
    /// </remarks>
    public int SkipChannels { get; set; } = 256;

    /// <summary>
    /// Gets or sets the dilation depth (number of dilation doublings per stack).
    /// </summary>
    /// <value>The dilation depth, defaulting to 8.</value>
    /// <remarks>
    /// <para>
    /// With dilation depth d, each stack has layers with dilations:
    /// 1, 2, 4, 8, ..., 2^(d-1)
    ///
    /// The receptive field per stack is approximately 2^d.
    /// </para>
    /// <para><b>For Beginners:</b> Controls how far back each "stack" can see.
    /// With depth 8, dilations are [1, 2, 4, 8, 16, 32, 64, 128], giving receptive
    /// field of 255 time steps per stack.
    /// </para>
    /// </remarks>
    public int DilationDepth { get; set; } = 8;

    /// <summary>
    /// Gets or sets the number of stacks (repetitions of the dilation pattern).
    /// </summary>
    /// <value>The number of stacks, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> WaveNet repeats its dilation pattern multiple times.
    /// With 2 stacks and depth 8, you get:
    /// [1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128]
    /// This creates a very deep network with huge receptive field.
    /// </para>
    /// </remarks>
    public int NumStacks { get; set; } = 2;

    /// <summary>
    /// Gets or sets the kernel size for dilated convolutions.
    /// </summary>
    /// <value>The kernel size, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> WaveNet typically uses kernel size 2, which means
    /// each convolution looks at the current and one previous (dilated) position.
    /// Larger kernels increase receptive field but add complexity.
    /// </para>
    /// </remarks>
    public int KernelSize { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether to use gated activation units.
    /// </summary>
    /// <value>True to use gated activations, defaulting to true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gated activations are WaveNet's secret sauce:
    /// output = tanh(Wf * x) * sigmoid(Wg * x)
    ///
    /// The tanh provides the feature transformation, while the sigmoid "gate"
    /// controls how much of each feature passes through. This is similar to
    /// how LSTM gates work and helps model complex patterns.
    /// </para>
    /// </remarks>
    public bool UseGatedActivations { get; set; } = true;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout helps prevent overfitting.
    /// WaveNet often uses lower dropout than other architectures.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the learning rate for training.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.001.</value>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>The number of epochs, defaulting to 100.</value>
    public int Epochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the batch size for training.
    /// </summary>
    /// <value>The batch size, defaulting to 32.</value>
    public int BatchSize { get; set; } = 32;
}
