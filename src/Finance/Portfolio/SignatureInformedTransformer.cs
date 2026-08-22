using System;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Finance.Base;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Finance.Portfolio;

/// <summary>
/// SIT: a decision-focused asset allocator that reads path signatures and outputs portfolio weights
/// directly, trained on portfolio CVaR (Hwang and Zohren, arXiv:2510.03129).
/// </summary>
/// <remarks>
/// <para>
/// The paper's central claim is that predict-then-optimize is the wrong decomposition. Minimizing a
/// forecasting error is not a proxy for allocation quality: a downstream optimizer amplifies small
/// prediction inaccuracies into fragile portfolios, which is why the paper's forecasting baselines
/// show high variance across runs and often fail to beat an equally weighted portfolio. SIT collapses
/// feature extraction and decision-making into ONE policy trained on the financial objective itself.
/// </para>
/// <para>
/// Three pieces, each independently testable:
/// <see cref="Signatures"/> (truncated path signatures and the pairwise signed area that measures
/// lead-lag), <see cref="Attention"/> (a query-conditioned, softplus-gated bias that injects that
/// lead-lag evidence INSIDE attention rather than as an input feature), and
/// <see cref="Objective"/> (CVaR over realized portfolio losses, with long-only fully-invested
/// weights from a tempered softmax).
/// </para>
/// <para>
/// <b>CVaR is the only training signal.</b> The paper states explicitly that no auxiliary prediction
/// losses are used. Adding an MSE or expected-return term would reintroduce precisely the objective
/// mismatch this architecture exists to remove, so none is present and none should be added.
/// </para>
/// <para><b>For Beginners:</b> Give it recent price history for a basket of assets and it returns the
/// fraction of your money to put in each. It is trained to keep the worst outcomes shallow, rather
/// than to predict prices well and hope good predictions imply a good portfolio.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
/// <example>
/// <code>
/// var model = new SignatureInformedTransformer&lt;double&gt;(
///     new SignatureInformedTransformerOptions&lt;double&gt; { NumAssets = 30 });
/// Vector&lt;double&gt; weights = model.OptimizePortfolio(priceWindow);   // long-only, sums to 1
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Signature-Informed Transformer for Asset Allocation",
    "https://arxiv.org/abs/2510.03129",
    Year = 2025,
    Authors = "Yoontae Hwang, Stefan Zohren")]
public partial class SignatureInformedTransformer<T> : PortfolioOptimizerBase<T>
{
    private readonly SignatureInformedTransformerOptions<T> _options;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Gets the path-signature transform, including the pairwise signed-area (lead-lag) measure.</summary>
    public PathSignatureTransform<T> Signatures { get; }

    /// <summary>Gets the signature-augmented attention bias.</summary>
    public SignatureAugmentedAttention<T> Attention { get; }

    /// <summary>Gets the CVaR objective and the weight parameterization.</summary>
    public CVaRPortfolioObjective<T> Objective { get; }

    /// <summary>Creates a SIT allocator with the paper's defaults.</summary>
    public SignatureInformedTransformer()
        : this(new SignatureInformedTransformerOptions<T>())
    {
    }

    /// <summary>Creates a SIT allocator.</summary>
    /// <param name="options">Configuration; defaults are the paper's.</param>
    /// <param name="architecture">Optional custom architecture.</param>
    /// <param name="lossFunction">
    /// Optional loss for the inherited training surface. Note that SIT's actual objective is
    /// <see cref="Objective"/>'s CVaR, not a pointwise loss — see the class remarks.
    /// </param>
    public SignatureInformedTransformer(
        SignatureInformedTransformerOptions<T> options,
        NeuralNetworkArchitecture<T>? architecture = null,
        ILossFunction<T>? lossFunction = null)
        : this(ResolveArchitecture(options, architecture), options, lossFunction)
    {
    }

    // Private chaining constructor so the resolved architecture can be passed to the base TWICE — once
    // as the architecture and once for its CalculatedInputSize — without building it twice.
    private SignatureInformedTransformer(
        NeuralNetworkArchitecture<T> resolvedArchitecture,
        SignatureInformedTransformerOptions<T> options,
        ILossFunction<T>? lossFunction)
        : base(resolvedArchitecture, options.NumAssets, resolvedArchitecture.CalculatedInputSize, lossFunction)
    {
        _options = options;

        Signatures = new PathSignatureTransform<T>(options.SignatureLevel);
        Attention = new SignatureAugmentedAttention<T>();
        Objective = new CVaRPortfolioObjective<T>(options.CVaRAlpha);

        InitializeLayers();
    }

    private static NeuralNetworkArchitecture<T> ResolveArchitecture(
        SignatureInformedTransformerOptions<T> options, NeuralNetworkArchitecture<T>? architecture)
    {
        Guard.NotNull(options);
        options.Validate();
        return architecture ?? CreateDefaultArchitecture(options);
    }

    private static NeuralNetworkArchitecture<T> CreateDefaultArchitecture(
        SignatureInformedTransformerOptions<T> options)
    {
        Guard.NotNull(options);

        // Validate() is NOT repeated here. ResolveArchitecture, the only caller, has already run it,
        // and GraphAttentionPortfolio validates once at the same point. One convention across both.

        // The network reads a [lookback, assets] price window and scores each asset, so the input is
        // two-dimensional and the output is one score per asset. The softmax in Objective.Weights then
        // turns those scores into a long-only, fully-invested allocation.
        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: options.LookbackWindow,
            inputWidth: options.NumAssets,
            outputSize: options.NumAssets);
    }

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        Layers.AddRange(LayerHelper<T>.CreateDefaultSignatureInformedTransformerLayers(
            Architecture,
            NumFeatures,
            _options.ModelDimension,
            _options.NumHeads,
            _options.NumAssets,
            _options.LookbackWindow,
            _options.DropoutRate));
    }

    /// <summary>
    /// Produces long-only, fully invested portfolio weights for a price window.
    /// </summary>
    /// <param name="marketData">Price history, shaped <c>[lookback, assets]</c>.</param>
    /// <returns>Weights over assets: strictly positive and summing to one.</returns>
    /// <remarks>
    /// The network emits unconstrained SCORES; the tempered softmax in
    /// <see cref="CVaRPortfolioObjective{T}.Weights"/> is what turns them into a valid allocation. That
    /// separation is deliberate — the constraint is enforced by the parameterization rather than by a
    /// penalty, so it holds exactly at every point in training rather than approximately at
    /// convergence.
    /// </remarks>
    public override Vector<T> OptimizePortfolio(Tensor<T> marketData)
    {
        Guard.NotNull(marketData);

        var scores = Predict(marketData).ToVector();

        // Predict can emit more values than there are assets when a custom architecture widens the
        // head; score the first NumAssets so the allocation length always matches the universe rather
        // than silently taking whatever shape came back.
        if (scores.Length > _options.NumAssets)
            scores = scores.Slice(0, _options.NumAssets);

        // A SHORT vector was previously passed through unchanged, producing weights for fewer assets
        // than the universe holds. Callers index weights by asset, so every entry from the first
        // missing one onward referred to the wrong asset.
        if (scores.Length < _options.NumAssets)
        {
            throw new InvalidOperationException(
                $"The network produced {scores.Length} scores for a universe of "
                + $"{_options.NumAssets} assets. Weights cannot be built from fewer scores than "
                + "assets; check that the architecture's output head matches NumAssets.");
        }

        return Objective.Weights(scores, _options.Temperature);
    }

    /// <summary>
    /// Pairwise signed areas over the price window: entry (j, l) is positive when asset j tends to
    /// lead asset l.
    /// </summary>
    /// <remarks>
    /// Exposed because this is the quantity the paper's inductive bias rests on, and it is meaningful
    /// on its own as a lead-lag diagnostic — unlike a correlation matrix it is antisymmetric, so it
    /// says which asset moves first rather than merely how strongly they co-move.
    /// </remarks>
    public Tensor<T> LeadLagMatrix(Tensor<T> marketData)
    {
        Guard.NotNull(marketData);
        return Signatures.CrossSignedAreas(marketData);
    }

    /// <summary>
    /// The CVaR of realized per-step losses for a fixed allocation over a horizon: the quantity
    /// training minimizes.
    /// </summary>
    /// <param name="weights">The allocation held over the horizon.</param>
    /// <param name="realizedReturns">Realized returns, shaped <c>[steps, assets]</c>.</param>
    public double PortfolioCVaR(Vector<T> weights, Tensor<T> realizedReturns)
    {
        Guard.NotNull(weights);
        Guard.NotNull(realizedReturns);

        if (realizedReturns.Shape.Length != 2)
            throw new ArgumentException(
                $"realizedReturns must be [steps, assets]; got rank {realizedReturns.Shape.Length}.",
                nameof(realizedReturns));

        int steps = realizedReturns.Shape[0];
        int assets = realizedReturns.Shape[1];
        if (assets != weights.Length)
            throw new ArgumentException(
                $"realizedReturns has {assets} assets but weights has {weights.Length}.",
                nameof(realizedReturns));

        var losses = new double[steps];
        for (int k = 0; k < steps; k++)
        {
            var stepReturns = new Vector<T>(assets);
            for (int a = 0; a < assets; a++) stepReturns[a] = realizedReturns[(k * assets) + a];
            losses[k] = Objective.Loss(weights, stepReturns);
        }

        return Objective.ConditionalValueAtRisk(losses);
    }
}
