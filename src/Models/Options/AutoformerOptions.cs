namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Autoformer model (Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Autoformer (Wu et al., NeurIPS 2021) introduces a novel decomposition-based transformer architecture
/// for long-term time series forecasting. Key innovations include:
/// - Series Decomposition Block: Progressive trend-seasonal separation at each layer
/// - Auto-Correlation Mechanism: Efficient O(L log L) sub-series aggregation replacing self-attention
/// - Moving Average Kernel: Learnable trend extraction from time series
/// </para>
/// <para><b>For Beginners:</b> Autoformer is designed to capture both the trend (long-term direction)
/// and seasonality (repeating patterns) in time series data. Unlike Informer which focuses on attention
/// efficiency, Autoformer focuses on decomposing the signal into meaningful components.
///
/// Think of it like separating a song into vocals and instrumentals - by processing these separately,
/// the model can better understand and predict each component.
///
/// Key features:
/// - **Auto-Correlation**: Instead of attending to individual time points, looks at how sub-sequences
///   correlate with each other (like finding repeating patterns)
/// - **Series Decomposition**: Separates trend from seasonal patterns at every layer
/// - **Progressive Refinement**: Each layer further refines the decomposition
///
/// Best suited for:
/// - Long-horizon forecasting (weeks/months ahead)
/// - Data with clear seasonal patterns (energy consumption, retail sales)
/// - Complex trend patterns (economic indicators)
/// </para>
/// </remarks>
public class AutoformerOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Creates a new instance with default values.
    /// </summary>
    public AutoformerOptions() { }

    /// <summary>
    /// Creates a copy of the specified options.
    /// </summary>
    public AutoformerOptions(AutoformerOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        EmbeddingDim = other.EmbeddingDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        MovingAverageKernel = other.MovingAverageKernel;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
        AutoCorrelationFactor = other.AutoCorrelationFactor;
    }

    /// <summary>
    /// Gets or sets the lookback window (encoder input length).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far back in time the model looks to make predictions.
    /// A value of 96 with hourly data means looking at the past 4 days.
    /// </para>
    /// </remarks>
    public int LookbackWindow { get; set; } = 96;

    /// <summary>
    /// Gets or sets the forecast horizon (decoder output length).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far ahead the model predicts.
    /// A value of 24 with hourly data means predicting the next day.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 24;

    /// <summary>
    /// Gets or sets the embedding dimension (model width).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size of the model.
    /// Larger values can capture more complex patterns but require more computation.
    /// 512 is a good balance for most time series.
    /// </para>
    /// </remarks>
    public int EmbeddingDim { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of encoder layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many times the model processes the input.
    /// More layers can capture more complex patterns but risk overfitting.
    /// </para>
    /// </remarks>
    public int NumEncoderLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of decoder layers.
    /// </summary>
    public int NumDecoderLayers { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of attention heads in auto-correlation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Number of parallel pattern-detection mechanisms.
    /// More heads can capture different types of patterns simultaneously.
    /// </para>
    /// </remarks>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the kernel size for moving average in series decomposition.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The window size for separating trend from seasonality.
    /// Larger values capture longer-term trends. Should be odd for symmetric smoothing.
    /// A value of 25 works well for daily patterns in hourly data.
    /// </para>
    /// </remarks>
    public int MovingAverageKernel { get; set; } = 25;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    public double DropoutRate { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the learning rate for optimization.
    /// </summary>
    public double LearningRate { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    public int Epochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the batch size for training.
    /// </summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the auto-correlation aggregation factor (c in the paper).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how many top correlations to consider.
    /// The formula is: top_k = c * log(L) where L is sequence length.
    /// A factor of 3 provides good accuracy/efficiency tradeoff.
    /// </para>
    /// </remarks>
    public int AutoCorrelationFactor { get; set; } = 3;
}
