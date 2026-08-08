using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for <see cref="AiDotNet.Finance.Portfolio.SignatureInformedTransformer{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Yoontae Hwang and Stefan Zohren, "Signature-Informed Transformer for Asset
/// Allocation" (arXiv:2510.03129) and its reference implementation: lookback 60, horizon 20,
/// signature level 2, d_model 8, 8 heads, 1 layer, feed-forward 64, temperature 1.3, and Adam at
/// 1e-3 with batch 64 and dropout 0.1.
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

    /// <summary>Gets or sets the number of assets in the universe. Default 30, the paper's smallest pool.</summary>
    /// <remarks>The paper evaluates 30/40/50-asset subsets of the S&amp;P 100, plus DOW30 and CSI300.</remarks>
    public int NumAssets { get; set; } = 30;

    /// <summary>Gets or sets the lookback window H in time steps. Default 60, the reference default.</summary>
    public int LookbackWindow { get; set; } = 60;

    /// <summary>
    /// Gets or sets the forecast/holding horizon K in time steps. Default 20, the reference default
    /// (about one month of trading days, matching the paper's monthly rebalancing).
    /// </summary>
    /// <remarks>
    /// This is also the number of per-step losses the CVaR is taken over, so shortening it narrows the
    /// loss sample the tail is estimated from.
    /// </remarks>
    public int Horizon { get; set; } = 20;

    /// <summary>
    /// Gets or sets the signature truncation level M. Default 2.
    /// </summary>
    /// <remarks>
    /// The paper fixes this at 2 rather than searching it, because the SECOND-order cross terms are
    /// what encode the signed area that measures lead-lag. Level 1 keeps only the total increment and
    /// removes the model's entire inductive bias.
    /// </remarks>
    public int SignatureLevel { get; set; } = 2;

    /// <summary>
    /// Gets or sets the model width d_model. Default 64, the top of the paper's searched range
    /// {8, 16, 32, 64}.
    /// </summary>
    /// <remarks>
    /// The paper selects this per dataset by validation, so there is no single "paper value" to copy.
    /// 64 is chosen over the reference README's example value of 8 because 8 combined with the default
    /// 8 heads leaves ONE dimension per head — a degenerate corner of the grid where attention has
    /// almost no capacity to work with. It also lines up with the reference defaults for the
    /// feed-forward and relational widths, which are both 64.
    /// </remarks>
    public int ModelDimension { get; set; } = 64;

    /// <summary>Gets or sets the feed-forward width. Default 64; the paper searches {8, 16, 32, 64}.</summary>
    public int FeedForwardDimension { get; set; } = 64;

    /// <summary>Gets or sets the attention head count. Default 8; the paper searches {2, 4, 8}.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the encoder depth. Default 1; the paper searches {1, 2}.</summary>
    public int NumLayers { get; set; } = 1;

    /// <summary>
    /// Gets or sets the hidden width of the MLPs producing the dynamic queries and relational
    /// embeddings. Default 64, the reference default (the paper searches {8, 16, 32}).
    /// </summary>
    public int RelationalHiddenDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets tau, the softmax temperature controlling allocation concentration. Default 1.3.
    /// </summary>
    /// <remarks>
    /// The paper's sensitivity analysis finds an INTERIOR optimum near 1.3: more concentrated settings
    /// (0.8-0.9) and very diffuse ones both do worse, so this is a genuine trade-off between
    /// diversification and conviction rather than a value to push to an extreme.
    /// </remarks>
    public double Temperature { get; set; } = 1.3;

    /// <summary>
    /// Gets or sets the CVaR confidence level alpha. Default 0.95.
    /// </summary>
    /// <remarks>
    /// The paper states this level symbolically and never gives a number, and the reference
    /// implementation does not document one, so 0.95 here is the standard risk-management convention
    /// (the average of the worst 5%) rather than a value taken from the paper.
    /// </remarks>
    public double CVaRAlpha { get; set; } = 0.95;

    /// <summary>Gets or sets the dropout rate. Default 0.1, the paper's value.</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Gets or sets the learning rate. Default 1e-3, the paper's value (Adam).</summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the batch size. Default 64, the paper's value.</summary>
    public int BatchSize { get; set; } = 64;

    /// <summary>Gets or sets the maximum training epochs. Default 100, the paper's value.</summary>
    public int MaxEpochs { get; set; } = 100;

    /// <summary>Gets or sets the early-stopping patience in epochs. Default 10, the paper's value.</summary>
    public int EarlyStoppingPatience { get; set; } = 10;

    /// <summary>
    /// Gets or sets the round-trip transaction cost in basis points used when EVALUATING a strategy.
    /// Default 0, matching the reference implementation.
    /// </summary>
    /// <remarks>
    /// Not part of the training objective. The paper treats costs as a post-hoc sensitivity analysis,
    /// and folding them into the loss would change the objective the method is defined by.
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
        if (Temperature <= 0.0 || double.IsNaN(Temperature) || double.IsInfinity(Temperature))
            throw new ArgumentOutOfRangeException(nameof(Temperature), Temperature,
                "Temperature must be finite and positive.");
        if (CVaRAlpha is <= 0.0 or >= 1.0 || double.IsNaN(CVaRAlpha))
            throw new ArgumentOutOfRangeException(nameof(CVaRAlpha), CVaRAlpha,
                "CVaRAlpha must be in (0, 1).");
        if (DropoutRate is < 0.0 or >= 1.0 || double.IsNaN(DropoutRate))
            throw new ArgumentOutOfRangeException(nameof(DropoutRate), DropoutRate,
                "DropoutRate must be in [0, 1).");
        if (LearningRate <= 0.0 || double.IsNaN(LearningRate))
            throw new ArgumentOutOfRangeException(nameof(LearningRate), LearningRate,
                "LearningRate must be positive.");
        if (TransactionCostBasisPoints < 0.0 || double.IsNaN(TransactionCostBasisPoints))
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
}
