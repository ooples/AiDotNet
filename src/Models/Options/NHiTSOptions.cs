using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the N-HiTS (Neural Hierarchical Interpolation for Time Series) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// N-HiTS is an evolution of N-BEATS that incorporates hierarchical interpolation and multi-rate signal sampling.
/// It achieves better accuracy on long-horizon forecasting tasks while being more parameter-efficient.
/// Key improvements over N-BEATS include:
/// - Hierarchical multi-rate data pooling for capturing patterns at different frequencies
/// - Interpolation-based basis functions for smoother forecasts
/// - More efficient parameter usage through stack-specific pooling
/// </para>
/// <para><b>For Beginners:</b> N-HiTS is an advanced neural network for time series forecasting that
/// works by looking at your data at multiple resolutions simultaneously - similar to how you might
/// zoom in and out when analyzing a chart. This multi-scale approach helps it capture both
/// short-term patterns (like daily fluctuations) and long-term trends (like seasonal cycles).
/// </para>
/// </remarks>
public class NHiTSOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="NHiTSOptions{T}"/> class.
    /// </summary>
    public NHiTSOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public NHiTSOptions(NHiTSOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumStacks = other.NumStacks;
        NumBlocksPerStack = other.NumBlocksPerStack;
        PoolingModes = other.PoolingModes != null ? (string[])other.PoolingModes.Clone() : null;
        InterpolationModes = other.InterpolationModes != null ? (string[])other.InterpolationModes.Clone() : null;
        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        HiddenLayerSize = other.HiddenLayerSize;
        NumHiddenLayers = other.NumHiddenLayers;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
        PoolingKernelSizes = other.PoolingKernelSizes != null ? (int[])other.PoolingKernelSizes.Clone() : null;
        DropoutRate = other.DropoutRate;
    }

    /// <summary>
    /// Gets or sets the number of stacks in the N-HiTS architecture.
    /// </summary>
    /// <value>The number of stacks, defaulting to 3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> N-HiTS typically uses 3 stacks, each operating at a different
    /// time resolution. The first stack captures high-frequency patterns, the second captures
    /// medium-frequency patterns, and the third captures low-frequency trends.
    /// </para>
    /// </remarks>
    public int NumStacks { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of blocks per stack.
    /// </summary>
    /// <value>The number of blocks per stack, defaulting to 1.</value>
    public int NumBlocksPerStack { get; set; } = 1;

    /// <summary>
    /// Gets or sets the pooling modes for each stack.
    /// </summary>
    /// <value>Array of pooling modes, defaulting to ["MaxPool", "AvgPool", "AvgPool"].</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Pooling is a downsampling technique that reduces the resolution
    /// of the input data. Different stacks use different pooling strategies to capture patterns
    /// at different time scales. "MaxPool" keeps the maximum values, while "AvgPool" averages values.
    /// </para>
    /// </remarks>
    public string[] PoolingModes { get; set; } = new string[] { "MaxPool", "AvgPool", "AvgPool" };

    /// <summary>
    /// Gets or sets the interpolation modes for each stack.
    /// </summary>
    /// <value>Array of interpolation modes, defaulting to ["Linear", "Linear", "Linear"].</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Interpolation determines how the model fills in values between
    /// known data points. "Linear" interpolation draws straight lines between points, while other
    /// methods like "Cubic" use curves for smoother results.
    /// </para>
    /// </remarks>
    public string[] InterpolationModes { get; set; } = new string[] { "Linear", "Linear", "Linear" };

    /// <summary>
    /// Gets or sets the lookback window size (number of historical time steps used as input).
    /// </summary>
    /// <value>The lookback window size, defaulting to 48.</value>
    public int LookbackWindow { get; set; } = 48;

    /// <summary>
    /// Gets or sets the forecast horizon (number of future time steps to predict).
    /// </summary>
    /// <value>The forecast horizon, defaulting to 24.</value>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the hidden layer size for fully connected layers within each block.
    /// </summary>
    /// <value>The hidden layer size, defaulting to 512.</value>
    public int HiddenLayerSize { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of hidden layers within each block.
    /// </summary>
    /// <value>The number of hidden layers, defaulting to 2.</value>
    public int NumHiddenLayers { get; set; } = 2;

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

    /// <summary>
    /// Gets or sets the pooling kernel sizes for each stack.
    /// </summary>
    /// <value>Array of kernel sizes, defaulting to [8, 4, 1].</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Kernel size controls how much downsampling occurs in each stack.
    /// Larger kernel sizes (like 8) create coarser representations that capture long-term trends,
    /// while smaller sizes (like 1) preserve fine-grained details.
    /// </para>
    /// </remarks>
    public int[] PoolingKernelSizes { get; set; } = new int[] { 8, 4, 1 };

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout randomly ignores some neurons during training to prevent
    /// overfitting (memorizing the training data instead of learning general patterns).
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;
}
