namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for <see cref="AiDotNet.Finance.Trading.Factors.Stockformer{T}"/>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>Reference:</b> Bohan Ma, Yiheng Wang, Yuchao Lu, Tianzixuan Hu, Jinling Xu, Patrick Houlihan
/// and Xin Liu, "Stockformer: A Price-Volume Factor Stock Selection Model Using Wavelet Transform and
/// Multi-Task Self-Attention Networks", 2024 (arXiv:2401.06139).</para>
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

        // INHERITED PROPERTIES COUNT TOO. Seed is declared on ModelOptions rather than here, so an
        // audit that reads only this file's declarations misses it -- which is exactly how it was
        // missed. Losing it on a clone changes deterministic initialization and training behaviour
        // silently, which is the same failure mode as any other dropped property.
        Seed = other.Seed;

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

    /// <summary>Gets or sets the number of stocks (graph nodes) in the cross-section.</summary>
    /// <value>The stock count. Default 500.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many stocks the model considers at once. Stockformer looks at
    /// them jointly, as a graph, so that what one stock does can inform its reading of another.</para>
    /// </remarks>
    public int NumAssets { get; set; } = 500;

    /// <summary>Gets or sets the number of price-volume factors per stock per timestep.</summary>
    /// <value>The feature count. Default 360, the count the reference implementation constructs in its factor pipeline.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many numbers describe each stock at each point in time — things
    /// derived from price and trading volume. This has to match the data you supply.</para>
    /// </remarks>
    public int NumFeatures { get; set; } = 360;

    /// <summary>Gets or sets the model width (<c>dims</c>).</summary>
    /// <value>The internal width. Default 128.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model's internal capacity. Larger can represent more, at the
    /// cost of memory, training time, and a greater tendency to memorize rather than generalize.</para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>Gets or sets the attention head count (<c>heads</c>).</summary>
    /// <value>The number of attention heads. Default 1.</value>
    /// <remarks>
    /// <para>
    /// One head, per the reference config. The paper's title mentions multi-task self-attention; the
    /// "multi" refers to the TASKS (return and direction), not the heads.
    /// </para>
    /// <para><b>For Beginners:</b> Despite what the paper's title suggests, this model uses a single
    /// attention head. Leave it at 1 to reproduce the published results.</para>
    /// </remarks>
    public int NumHeads { get; set; } = 1;

    /// <summary>Gets or sets the number of dual-encoder layers (<c>layers</c>).</summary>
    /// <value>The encoder depth. Default 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many times the model refines its reading. "Dual" means each
    /// layer processes the slow-moving and fast-moving parts of the signal on separate branches.</para>
    /// </remarks>
    public int NumLayers { get; set; } = 2;

    /// <summary>Gets or sets the sparsity parameter of the spatial attention (<c>samples</c>).</summary>
    /// <value>The sampling factor. Default 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how many stock-to-stock comparisons the attention actually
    /// computes. Sparser attention is cheaper on a 500-stock cross-section.</para>
    /// </remarks>
    public int SpatialSamples { get; set; } = 1;

    /// <summary>Gets or sets the input window length in timesteps (<c>T1</c>).</summary>
    /// <value>The lookback length. Default 20.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past days the model reads before predicting. Twenty
    /// trading days is about a month.</para>
    /// </remarks>
    public int SequenceLength { get; set; } = 20;

    /// <summary>Gets or sets the forecast horizon in timesteps (<c>T2</c>).</summary>
    /// <value>The number of steps predicted ahead. Default 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far ahead the model predicts.</para>
    /// </remarks>
    public int PredictionHorizon { get; set; } = 2;

    /// <summary>Gets or sets the Symlet order for the discrete wavelet transform (<c>wave = sym2</c>).</summary>
    /// <value>The Symlet order. Default 2.</value>
    /// <remarks>
    /// <para>
    /// sym2 is numerically identical to db2 (Daubechies-2) — same four filter taps. Reusing the
    /// library's <c>SymletWavelet</c> at order 2 rather than hand-rolling the transform.
    /// </para>
    /// <para><b>For Beginners:</b> A wavelet is the shape used to split a signal into slow and fast
    /// parts. This picks which one; the reference uses sym2 and there is no reason to change it.</para>
    /// </remarks>
    public int WaveletOrder { get; set; } = 2;

    /// <summary>Gets or sets the number of DWT decomposition levels (<c>level</c>).</summary>
    /// <value>The decomposition depth. Default 1.</value>
    /// <remarks>
    /// <para>
    /// ONE level. The model consumes exactly two bands — one low-frequency and one high-frequency —
    /// and its dual encoder has exactly two branches, so additional levels would have nowhere to go.
    /// </para>
    /// <para><b>For Beginners:</b> How many times to split the signal into slower and faster halves.
    /// The architecture has exactly two branches, so one split is all it can use.</para>
    /// </remarks>
    public int WaveletLevels { get; set; } = 1;

    /// <summary>Gets or sets the number of direction classes for the classification task.</summary>
    /// <value>The class count. Default 2 (down / up).</value>
    /// <remarks>
    /// <para>
    /// The reference implementation drives this from its <c>trend_indicator</c> dataset, so the class
    /// count is data-defined rather than fixed by the paper. Two is the binary up/down reading; set
    /// three if the target encodes a flat band.
    /// </para>
    /// <para><b>For Beginners:</b> The model predicts both how much a stock moves and which way. This
    /// is how many "ways" there are — two for up/down, three if your labels also have a flat case.</para>
    /// </remarks>
    public int NumDirectionClasses { get; set; } = 2;

    /// <summary>
    /// Gets or sets lambda, the weight on the classification task in <c>L = L_reg + lambda*L_cla</c>.
    /// </summary>
    /// <value>The classification weight. Default 1.0.</value>
    /// <remarks>
    /// <para>
    /// The PAPER specifies a weighting (Eq. 12); the reference implementation sums the two tasks 1:1
    /// with the weighted form commented out. So 1.0 is the reference's choice, not the paper's only
    /// option, and an earlier revision of this code wrongly asserted the paper forbids weighting.
    /// </para>
    /// <para><b>For Beginners:</b> The model learns two things at once: how big the next move is, and
    /// which direction. This sets how much the direction half matters. Above 1.0 favours getting the
    /// direction right; below 1.0 favours getting the size right.</para>
    /// </remarks>
    public double TaskLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the learning rate.</summary>
    /// <value>The optimizer learning rate. Default 0.001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How big a step the model takes each time it learns. Too large and
    /// training becomes unstable; too small and it barely improves.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>Gets or sets the dropout rate.</summary>
    /// <value>The dropout probability, in [0, 1). Default 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout randomly ignores part of the model during training so it
    /// cannot lean too hard on any one signal. Raise it if the model does well on training data and
    /// badly on new data.</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Gets or sets the value treated as "missing" by the masked regression loss.</summary>
    /// <value>The sentinel value. Default 0.0.</value>
    /// <remarks>
    /// <para>
    /// The reference loss is <c>masked_mae(preds, labels, 0.0)</c> — mean absolute error over entries
    /// whose label is NOT this value, with the mask renormalized by its own mean so the result is a
    /// mean over VALID entries rather than over all entries. It is MAE, not MSE.
    /// </para>
    /// <para><b>For Beginners:</b> Real market data has gaps. Labels equal to this value are treated
    /// as "no data here" and skipped when scoring, instead of being learned as a real zero return.</para>
    /// </remarks>
    public double MissingValueSentinel { get; set; } = 0.0;

    // No Seed property here: ModelOptions already declares one, and shadowing it with `new` would
    // give callers two seeds that silently disagree depending on the static type they hold. The
    // reference config's seed = 1 is applied by the model when the inherited Seed is unset.

    /// <summary>Validates the configuration.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A value cannot describe a usable model.</exception>
    /// <remarks>
    /// Called from the <c>Stockformer&lt;T&gt;</c> constructor before it builds any layer, because the
    /// failures these catch are otherwise mute: a zero dimension reaches layer construction and throws
    /// somewhere with no mention of the option that set it, and a NaN passes every ordinary range
    /// comparison (NaN &lt;= 0.0 is false) to surface later as a model that trains to NaN.
    /// </remarks>
    public void Validate()
    {
        RequirePositive(NumAssets, nameof(NumAssets));
        RequirePositive(NumFeatures, nameof(NumFeatures));
        RequirePositive(HiddenDimension, nameof(HiddenDimension));
        RequirePositive(NumHeads, nameof(NumHeads));
        RequirePositive(NumLayers, nameof(NumLayers));
        RequirePositive(SpatialSamples, nameof(SpatialSamples));
        RequirePositive(PredictionHorizon, nameof(PredictionHorizon));

        // Two, not one: a single-level wavelet split needs at least two samples to separate into a
        // low- and a high-frequency band. BuildArchitecture used to clamp this with Math.Max(2, ...),
        // which silently gave a caller asking for a length of 1 a model of length 2 instead.
        if (SequenceLength < 2)
            throw new ArgumentOutOfRangeException(nameof(SequenceLength), SequenceLength,
                "SequenceLength must be at least 2; the wavelet split needs two samples to separate.");
        RequirePositive(WaveletOrder, nameof(WaveletOrder));
        RequirePositive(WaveletLevels, nameof(WaveletLevels));

        if (NumDirectionClasses < 2)
            throw new ArgumentOutOfRangeException(nameof(NumDirectionClasses), NumDirectionClasses,
                "NumDirectionClasses must be at least 2; a classifier with one class predicts nothing.");

        RequireFinite(TaskLossWeight, nameof(TaskLossWeight));
        RequireFinite(LearningRate, nameof(LearningRate));
        RequireFinite(DropoutRate, nameof(DropoutRate));
        RequireFinite(MissingValueSentinel, nameof(MissingValueSentinel));

        if (TaskLossWeight < 0.0)
            throw new ArgumentOutOfRangeException(nameof(TaskLossWeight), TaskLossWeight,
                "TaskLossWeight cannot be negative; a negative weight rewards getting the direction wrong.");
        if (LearningRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(LearningRate), LearningRate,
                "LearningRate must be positive.");
        if (DropoutRate is < 0.0 or >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(DropoutRate), DropoutRate,
                "DropoutRate must be in [0, 1); at 1.0 every unit is dropped and nothing can train.");

        // The width has to split evenly across heads, or the per-head width is ill-defined and the
        // reshape into heads silently drops or duplicates features.
        if (HiddenDimension % NumHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(NumHeads), NumHeads,
                $"NumHeads must divide HiddenDimension ({HiddenDimension}); got {HiddenDimension} % {NumHeads} != 0.");
    }

    private static void RequirePositive(int value, string name)
    {
        if (value <= 0)
            throw new ArgumentOutOfRangeException(name, value, $"{name} must be positive.");
    }

    /// <summary>Rejects NaN and both infinities.</summary>
    /// <remarks>
    /// Spelled as <c>IsNaN || IsInfinity</c> rather than <c>!double.IsFinite</c>: this project also
    /// targets net471, where <c>double.IsFinite</c> does not exist.
    /// </remarks>
    private static void RequireFinite(double value, string name)
    {
        if (double.IsNaN(value) || double.IsInfinity(value))
            throw new ArgumentOutOfRangeException(name, value, $"{name} must be a finite number.");
    }
}
