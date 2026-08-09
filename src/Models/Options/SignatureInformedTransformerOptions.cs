using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for <see cref="AiDotNet.Finance.Portfolio.SignatureInformedTransformer{T}"/>.
/// </summary>
/// <remarks>
/// <para><b>Reference:</b> Yoontae Hwang and Stefan Zohren, "Signature-Informed Transformer for
/// Asset Allocation", 2025 (arXiv:2510.03129).</para>
/// <para>
/// Defaults follow that paper and its reference implementation: lookback 60, horizon 20, signature
/// level 2, 8 heads, 1 layer, feed-forward 64, temperature 1.3, and Adam at 1e-3 with batch 64 and
/// dropout 0.1.
/// </para>
/// <para>
/// <c>ModelDimension</c> is the one default NOT taken directly from the reference. The paper selects
/// d_model per dataset by validation over {8, 16, 32, 64}, so there is no single paper value; the
/// reference README's example of 8 would leave one dimension per head against the default 8 heads.
/// This class defaults to 64 — see the property for the full reasoning. Stating it here too, because
/// a class summary that lists a different default than the property is the version a reader trusts.
/// </para>
/// <para><b>For Beginners:</b> This model reads a window of price history for a set of assets and
/// outputs how much of the portfolio to put in each. It is trained to keep bad outcomes small rather
/// than to forecast prices accurately.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class SignatureInformedTransformerOptions<T> : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public SignatureInformedTransformerOptions() { }

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
    public SignatureInformedTransformerOptions(SignatureInformedTransformerOptions<T> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        // INHERITED PROPERTIES COUNT TOO. Seed comes from ModelOptions and EncoderLayerCount from
        // NeuralNetworkOptions, so neither appears in this file's declarations -- which is how both
        // were missed. Losing Seed on a clone silently changes deterministic initialization.
        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;

        NumAssets = other.NumAssets;
        LookbackWindow = other.LookbackWindow;
        Horizon = other.Horizon;
        SignatureLevel = other.SignatureLevel;
        ModelDimension = other.ModelDimension;
        FeedForwardDimension = other.FeedForwardDimension;
        NumHeads = other.NumHeads;
        NumLayers = other.NumLayers;
        RelationalHiddenDimension = other.RelationalHiddenDimension;
        Temperature = other.Temperature;
        CVaRAlpha = other.CVaRAlpha;
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        BatchSize = other.BatchSize;
        MaxEpochs = other.MaxEpochs;
        EarlyStoppingPatience = other.EarlyStoppingPatience;
        TransactionCostBasisPoints = other.TransactionCostBasisPoints;
    }

    /// <summary>Gets or sets the number of assets in the investment universe.</summary>
    /// <value>The asset count. Default 30, the paper's smallest pool.</value>
    /// <remarks>
    /// <para>The paper evaluates 30/40/50-asset subsets of the S&amp;P 100, plus DOW30 and CSI300.</para>
    /// <para><b>For Beginners:</b> How many different stocks the model is choosing between. This must
    /// match the number of assets in the data you feed it.</para>
    /// </remarks>
    public int NumAssets { get; set; } = 30;

    /// <summary>Gets or sets the lookback window H, in time steps.</summary>
    /// <value>The number of past time steps the model reads. Default 60, the reference default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much price history the model looks at before deciding. Sixty
    /// trading days is roughly three months.</para>
    /// </remarks>
    public int LookbackWindow { get; set; } = 60;

    /// <summary>Gets or sets the forecast/holding horizon K, in time steps.</summary>
    /// <value>
    /// The number of steps a chosen allocation is held. Default 20, the reference default -- about one
    /// month of trading days, matching the paper's monthly rebalancing.
    /// </value>
    /// <remarks>
    /// <para>
    /// This is also the number of per-step losses the CVaR is taken over, so shortening it narrows the
    /// loss sample the tail is estimated from.
    /// </para>
    /// <para><b>For Beginners:</b> How long the model holds a portfolio before choosing again. Shorter
    /// horizons trade more often, which costs more and gives the risk measure less to work with.</para>
    /// </remarks>
    public int Horizon { get; set; } = 20;

    /// <summary>Gets or sets the signature truncation level M.</summary>
    /// <value>The truncation level, 1 or 2. Default 2.</value>
    /// <remarks>
    /// <para>
    /// The paper fixes this at 2 rather than searching it, because the SECOND-order cross terms are
    /// what encode the signed area that measures lead-lag. Level 1 keeps only the total increment and
    /// removes the model's entire inductive bias.
    /// </para>
    /// <para><b>For Beginners:</b> A "signature" is a compact summary of the shape of a price path.
    /// Level 2 keeps the information about which asset tends to move first, which is the whole point
    /// of this model — leave it at 2 unless you are experimenting.</para>
    /// </remarks>
    public int SignatureLevel { get; set; } = 2;

    /// <summary>Gets or sets the model width d_model.</summary>
    /// <value>The transformer's internal width. Default 64, the top of the paper's searched range {8, 16, 32, 64}.</value>
    /// <remarks>
    /// <para>
    /// The paper selects this per dataset by validation, so there is no single "paper value" to copy.
    /// 64 is chosen over the reference README's example value of 8 because 8 combined with the default
    /// 8 heads leaves ONE dimension per head — a degenerate corner of the grid where attention has
    /// almost no capacity to work with. It also lines up with the reference defaults for the
    /// feed-forward and relational widths, which are both 64.
    /// </para>
    /// <para><b>For Beginners:</b> How much internal capacity the model has. Bigger can fit more, and
    /// also overfits more easily on the short histories finance data usually provides. It must divide
    /// evenly by <see cref="NumHeads"/>.</para>
    /// </remarks>
    public int ModelDimension { get; set; } = 64;

    /// <summary>Gets or sets the feed-forward width.</summary>
    /// <value>The width of the per-position feed-forward block. Default 64; the paper searches {8, 16, 32, 64}.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The size of the small network applied at each time step after
    /// attention. Leaving it equal to <see cref="ModelDimension"/> is a safe starting point.</para>
    /// </remarks>
    public int FeedForwardDimension { get; set; } = 64;

    /// <summary>Gets or sets the attention head count.</summary>
    /// <value>The number of attention heads. Default 8; the paper searches {2, 4, 8}.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Attention heads let the model look for several different patterns
    /// at once. This must divide <see cref="ModelDimension"/> evenly, or each head would get a
    /// fractional slice of the width.</para>
    /// </remarks>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the encoder depth.</summary>
    /// <value>The number of stacked encoder layers. Default 1; the paper searches {1, 2}.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many times the model refines its reading of the history. The
    /// paper finds one or two is enough here — this is a much shallower model than a language
    /// transformer, because financial histories are short.</para>
    /// </remarks>
    public int NumLayers { get; set; } = 1;

    /// <summary>
    /// Gets or sets the hidden width of the MLPs producing the dynamic queries and relational
    /// embeddings.
    /// </summary>
    /// <value>The hidden width. Default 64, the reference default (the paper searches {8, 16, 32}).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Capacity for the part of the model that learns how assets relate to
    /// each other, as opposed to how each one behaves on its own.</para>
    /// </remarks>
    public int RelationalHiddenDimension { get; set; } = 64;

    /// <summary>Gets or sets tau, the softmax temperature controlling allocation concentration.</summary>
    /// <value>The temperature. Default 1.3.</value>
    /// <remarks>
    /// <para>
    /// The paper's sensitivity analysis finds an INTERIOR optimum near 1.3: more concentrated settings
    /// (0.8-0.9) and very diffuse ones both do worse, so this is a genuine trade-off between
    /// diversification and conviction rather than a value to push to an extreme.
    /// </para>
    /// <para><b>For Beginners:</b> Lower values make the model bet heavily on a few assets; higher
    /// values spread money more evenly. The paper found the middle is genuinely best, so this is not a
    /// knob to turn all the way in either direction.</para>
    /// </remarks>
    public double Temperature { get; set; } = 1.3;

    /// <summary>Gets or sets the CVaR confidence level alpha.</summary>
    /// <value>The confidence level, in (0, 1). Default 0.95.</value>
    /// <remarks>
    /// <para>
    /// The paper states this level symbolically and never gives a number, and the reference
    /// implementation does not document one, so 0.95 here is the standard risk-management convention
    /// (the average of the worst 5%) rather than a value taken from the paper.
    /// </para>
    /// <para><b>For Beginners:</b> The model is trained to make its BAD days less bad. 0.95 means "pay
    /// attention to the worst 5% of outcomes". Raising it to 0.99 focuses on rarer, more extreme
    /// losses, which also means fewer examples to learn that tail from.</para>
    /// </remarks>
    public double CVaRAlpha { get; set; } = 0.95;

    /// <summary>Gets or sets the dropout rate.</summary>
    /// <value>The dropout probability, in [0, 1). Default 0.1, the paper's value.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout randomly ignores part of the model during training so it
    /// cannot lean on any one signal. Raise it if the model does well on training data and badly on
    /// new data.</para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Gets or sets the learning rate.</summary>
    /// <value>The Adam learning rate. Default 1e-3, the paper's value.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How big a step the model takes each time it learns. Too large and
    /// training becomes unstable; too small and it barely moves.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the batch size.</summary>
    /// <value>The number of samples per training step. Default 64, the paper's value.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many historical windows the model looks at before each update.
    /// Larger batches are steadier but need more memory.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 64;

    /// <summary>Gets or sets the maximum training epochs.</summary>
    /// <value>The epoch ceiling. Default 100, the paper's value.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The most passes over the data training is allowed. Early stopping
    /// usually ends training before this is reached.</para>
    /// </remarks>
    public int MaxEpochs { get; set; } = 100;

    /// <summary>Gets or sets the early-stopping patience, in epochs.</summary>
    /// <value>Epochs without improvement before training stops. Default 10, the paper's value.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How long to keep going when results stop improving, before giving
    /// up and keeping the best model found.</para>
    /// </remarks>
    public int EarlyStoppingPatience { get; set; } = 10;

    /// <summary>
    /// Gets or sets the round-trip transaction cost in basis points used when EVALUATING a strategy.
    /// </summary>
    /// <value>The cost in basis points. Default 0, matching the reference implementation.</value>
    /// <remarks>
    /// <para>
    /// Not part of the training objective. The paper treats costs as a post-hoc sensitivity analysis,
    /// and folding them into the loss would change the objective the method is defined by.
    /// </para>
    /// <para><b>For Beginners:</b> What it costs to trade, charged during evaluation only. One basis
    /// point is 0.01%. Setting this does NOT teach the model to trade less — it only makes the
    /// reported results honest about fees.</para>
    /// </remarks>
    public double TransactionCostBasisPoints { get; set; } = 0.0;

    /// <summary>Validates the configuration.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A value cannot describe a usable model.</exception>
    public void Validate()
    {
        RequirePositive(NumAssets, nameof(NumAssets));
        RequirePositive(LookbackWindow, nameof(LookbackWindow));
        RequirePositive(Horizon, nameof(Horizon));
        RequirePositive(ModelDimension, nameof(ModelDimension));
        RequirePositive(FeedForwardDimension, nameof(FeedForwardDimension));
        RequirePositive(NumHeads, nameof(NumHeads));
        RequirePositive(NumLayers, nameof(NumLayers));
        RequirePositive(RelationalHiddenDimension, nameof(RelationalHiddenDimension));
        RequirePositive(BatchSize, nameof(BatchSize));
        RequirePositive(MaxEpochs, nameof(MaxEpochs));

        if (SignatureLevel is < 1 or > 2)
            throw new ArgumentOutOfRangeException(nameof(SignatureLevel), SignatureLevel,
                "SignatureLevel must be 1 or 2; the paper uses 2.");
        if (EarlyStoppingPatience < 0)
            throw new ArgumentOutOfRangeException(nameof(EarlyStoppingPatience), EarlyStoppingPatience,
                "EarlyStoppingPatience cannot be negative.");
        // FINITENESS IS CHECKED ON EVERY UNBOUNDED DOUBLE, not just Temperature. An ordinary range
        // comparison lets PositiveInfinity through -- Infinity > 0.0 is true -- so LearningRate and
        // TransactionCostBasisPoints could reach training and turn every weight into NaN on the first
        // step, with the validator having passed them. CVaRAlpha and DropoutRate have upper bounds
        // that already exclude +Infinity, but not -Infinity, so they get the same treatment for one
        // rule rather than two.
        RequireFinite(Temperature, nameof(Temperature));
        RequireFinite(CVaRAlpha, nameof(CVaRAlpha));
        RequireFinite(DropoutRate, nameof(DropoutRate));
        RequireFinite(LearningRate, nameof(LearningRate));
        RequireFinite(TransactionCostBasisPoints, nameof(TransactionCostBasisPoints));

        if (Temperature <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(Temperature), Temperature,
                "Temperature must be finite and positive.");
        if (CVaRAlpha is <= 0.0 or >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(CVaRAlpha), CVaRAlpha,
                "CVaRAlpha must be in (0, 1).");
        if (DropoutRate is < 0.0 or >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(DropoutRate), DropoutRate,
                "DropoutRate must be in [0, 1).");
        if (LearningRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(LearningRate), LearningRate,
                "LearningRate must be positive.");
        if (TransactionCostBasisPoints < 0.0)
            throw new ArgumentOutOfRangeException(nameof(TransactionCostBasisPoints),
                TransactionCostBasisPoints, "TransactionCostBasisPoints cannot be negative.");

        // d_model must split evenly across heads, or the per-head width is ill-defined and the
        // reshape into heads silently drops or duplicates features.
        if (ModelDimension % NumHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(NumHeads), NumHeads,
                $"NumHeads must divide ModelDimension ({ModelDimension}); got {ModelDimension} % {NumHeads} != 0.");
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
