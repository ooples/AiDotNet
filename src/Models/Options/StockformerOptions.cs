namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for <see cref="AiDotNet.Finance.Trading.Factors.Stockformer{T}"/>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Defaults are taken from the reference implementation's <c>config/Multitask_Stock.conf</c>
/// (github.com/Eric991005/Multitask-Stockformer), not invented: <c>layers = 2</c>, <c>heads = 1</c>,
/// <c>dims = 128</c>, <c>samples = 1</c>, <c>wave = sym2</c>, <c>level = 1</c>, <c>T1 = 20</c>,
/// <c>T2 = 2</c>, <c>learning_rate = 0.001</c>, <c>batch_size = 12</c>.
/// </para>
/// <para>
/// Two of those are easy to get wrong by reading the paper alone. The wavelet is a SINGLE-level sym2
/// split, not a deep multi-resolution pyramid; and attention is SINGLE-headed, despite "multi-task
/// self-attention networks" in the title suggesting otherwise.
/// </para>
/// <para><b>For Beginners:</b> Stockformer splits each stock's return series into a slow-moving part
/// and a fast-moving part, studies them with different machinery, then predicts both the size of the
/// next move and its direction at the same time. These settings control the sizes involved.</para>
/// </remarks>
public class StockformerOptions<T> : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public StockformerOptions() { }

    /// <summary>
    /// Initializes a new instance by copying every property from another instance.
    /// </summary>
    /// <param name="other">The instance to copy from.</param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="other"/> is null.
    /// </exception>
    /// <remarks>
    /// EVERY property is copied, deliberately and exhaustively. A copy constructor that misses one
    /// is silent data loss: the clone keeps the default while the original keeps the configured
    /// value, and nothing reports the divergence -- the bug class behind the Tacotron2 and
    /// TimeBridge clone failures. When a property is added to this class it must be added here too.
    /// </remarks>
    public StockformerOptions(StockformerOptions<T> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        NumAssets = other.NumAssets;
        NumFeatures = other.NumFeatures;
        HiddenDimension = other.HiddenDimension;
        NumHeads = other.NumHeads;
        NumLayers = other.NumLayers;
        SpatialSamples = other.SpatialSamples;
        SequenceLength = other.SequenceLength;
        PredictionHorizon = other.PredictionHorizon;
        WaveletOrder = other.WaveletOrder;
        WaveletLevels = other.WaveletLevels;
        NumDirectionClasses = other.NumDirectionClasses;
        TaskLossWeight = other.TaskLossWeight;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        MissingValueSentinel = other.MissingValueSentinel;
    }

    /// <summary>
    /// Gets or sets the number of stocks (graph nodes) in the cross-section. Default 500.
    /// </summary>
    public int NumAssets { get; set; } = 500;

    /// <summary>
    /// Gets or sets the number of price-volume factors per stock per timestep. Default 360, the count
    /// the reference implementation constructs in its factor pipeline.
    /// </summary>
    public int NumFeatures { get; set; } = 360;

    /// <summary>
    /// Gets or sets the model width (<c>dims</c>). Default 128.
    /// </summary>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the attention head count (<c>heads</c>). Default 1.
    /// </summary>
    /// <remarks>
    /// One head, per the reference config. The paper's title mentions multi-task self-attention; the
    /// "multi" refers to the TASKS (return and direction), not the heads.
    /// </remarks>
    public int NumHeads { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of dual-encoder layers (<c>layers</c>). Default 2.
    /// </summary>
    public int NumLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the sparsity parameter of the spatial attention (<c>samples</c>). Default 1.
    /// </summary>
    public int SpatialSamples { get; set; } = 1;

    /// <summary>
    /// Gets or sets the input window length in timesteps (<c>T1</c>). Default 20.
    /// </summary>
    public int SequenceLength { get; set; } = 20;

    /// <summary>
    /// Gets or sets the forecast horizon in timesteps (<c>T2</c>). Default 2.
    /// </summary>
    public int PredictionHorizon { get; set; } = 2;

    /// <summary>
    /// Gets or sets the Symlet order for the discrete wavelet transform (<c>wave = sym2</c>).
    /// Default 2.
    /// </summary>
    /// <remarks>
    /// sym2 is numerically identical to db2 (Daubechies-2) — same four filter taps. Reusing the
    /// library's <c>SymletWavelet</c> at order 2 rather than hand-rolling the transform.
    /// </remarks>
    public int WaveletOrder { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of DWT decomposition levels (<c>level</c>). Default 1.
    /// </summary>
    /// <remarks>
    /// ONE level. The model consumes exactly two bands — one low-frequency and one high-frequency —
    /// and its dual encoder has exactly two branches, so additional levels would have nowhere to go.
    /// </remarks>
    public int WaveletLevels { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of direction classes for the classification task. Default 2
    /// (down / up).
    /// </summary>
    /// <remarks>
    /// The reference implementation drives this from its <c>trend_indicator</c> dataset, so the class
    /// count is data-defined rather than fixed by the paper. Two is the binary up/down reading; set
    /// three if the target encodes a flat band.
    /// </remarks>
    public int NumDirectionClasses { get; set; } = 2;

    /// <summary>
    /// Gets or sets lambda, the weight on the classification task in <c>L = L_reg + lambda*L_cla</c>.
    /// Default 1.0.
    /// </summary>
    /// <remarks>
    /// The PAPER specifies a weighting (Eq. 12); the reference implementation sums the two tasks 1:1
    /// with the weighted form commented out. So 1.0 is the reference's choice, not the paper's only
    /// option, and an earlier revision of this code wrongly asserted the paper forbids weighting.
    /// </remarks>
    public double TaskLossWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the learning rate. Default 0.001.
    /// </summary>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the dropout rate. Default 0.1.
    /// </summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the value treated as "missing" by the masked regression loss. Default 0.0.
    /// </summary>
    /// <remarks>
    /// The reference loss is <c>masked_mae(preds, labels, 0.0)</c> — mean absolute error over entries
    /// whose label is NOT this value, with the mask renormalized by its own mean so the result is a
    /// mean over VALID entries rather than over all entries. It is MAE, not MSE.
    /// </remarks>
    public double MissingValueSentinel { get; set; } = 0.0;

    // No Seed property here: ModelOptions already declares one, and shadowing it with `new` would
    // give callers two seeds that silently disagree depending on the static type they hold. The
    // reference config's seed = 1 is applied by the model when the inherited Seed is unset.
}
