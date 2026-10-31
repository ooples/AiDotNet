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
/// - Moving Average Kernel: the window of the moving average that extracts the trend
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
    /// Creates a new instance with the reference's defaults.
    /// </summary>
    public AutoformerOptions() { }

    /// <summary>
    /// Creates a copy of the specified options.
    /// </summary>
    public AutoformerOptions(AutoformerOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
        LookbackWindow = other.LookbackWindow;
        _labelLength = other._labelLength;
        ForecastHorizon = other.ForecastHorizon;
        EmbeddingDim = other.EmbeddingDim;
        FeedForwardDim = other.FeedForwardDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        MovingAverageKernel = other.MovingAverageKernel;
        AutoCorrelationFactor = other.AutoCorrelationFactor;
        UseGelu = other.UseGelu;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
    }

    /// <summary>
    /// Gets or sets the lookback window, the encoder input length (seq_len; the reference's default 96).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far back in time the model looks to make predictions.
    /// A value of 96 with hourly data means looking at the past 4 days.
    /// </para>
    /// </remarks>
    public int LookbackWindow { get; set; } = 96;

    /// <summary>
    /// Gets or sets the start-token length: the last steps of the input's seasonal and trend parts that open the
    /// decoder (label_len). Unless set, half of <see cref="LookbackWindow"/>: the reference's 48 for its 96-step
    /// window, and a valid start for any shorter window.
    /// </summary>
    public int LabelLength
    {
        get => _labelLength ?? LookbackWindow / 2;
        set => _labelLength = value;
    }

    private int? _labelLength;

    /// <summary>
    /// Gets or sets the forecast horizon (pred_len, 96).
    /// </summary>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the model width (d_model, 512).
    /// </summary>
    public int EmbeddingDim { get; set; } = 512;

    /// <summary>
    /// Gets or sets the feed-forward width (d_ff, 2048).
    /// </summary>
    public int FeedForwardDim { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the number of encoder layers (e_layers, 2).
    /// </summary>
    public int NumEncoderLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of decoder layers (d_layers, 1).
    /// </summary>
    public int NumDecoderLayers { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of heads of the Auto-Correlation projections (n_heads, 8).
    /// </summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the moving-average window of the series decomposition, an odd number (moving_avg, 25).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The trend at each step is the average of this many neighbouring steps; what is
    /// left over is the seasonal part.</para>
    /// </remarks>
    public int MovingAverageKernel { get; set; } = 25;

    /// <summary>
    /// Gets or sets the Auto-Correlation factor c: the ⌊c·ln L⌋ strongest lags are aggregated (factor, 1).
    /// </summary>
    public int AutoCorrelationFactor { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether the feed-forward activation is GELU (activation 'gelu'; false: ReLU).
    /// </summary>
    public bool UseGelu { get; set; } = true;

    /// <summary>
    /// Gets or sets the dropout rate (dropout, 0.05).
    /// </summary>
    public double DropoutRate { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the initial learning rate of Adam (learning_rate, 1e-4), halved after every epoch (lradj 'type1').
    /// </summary>
    public double LearningRate { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the number of training epochs (train_epochs, 10).
    /// </summary>
    public int Epochs { get; set; } = 10;

    /// <summary>
    /// Gets or sets the batch size (batch_size, 32).
    /// </summary>
    public int BatchSize { get; set; } = 32;
}
