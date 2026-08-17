using System;
using System.Collections.Generic;
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
/// A time-varying portfolio optimiser over a filtered asset graph, trained directly on the Sharpe
/// ratio (Korangi, Mues and Bravo, arXiv:2407.15532).
/// </summary>
/// <remarks>
/// <para>
/// The pipeline is: return-volatility series → distance correlation → TMFG planar filtering → graph
/// attention → two feed-forward blocks → sum-normalising allocation layer, trained on
/// <c>-ln(mu_p) + ln(sigma_p)</c>. Like SIT it is decision-focused — it emits weights rather than
/// return forecasts — but the two differ in every mechanism: this one uses a filtered graph and a
/// Sharpe objective where SIT uses path signatures and CVaR.
/// </para>
/// <para>
/// The paper's own framing of its contribution is that it keeps every firm, including those that later
/// default or leave the mid-cap universe. Studies that drop such firms introduce selection bias, and
/// distance correlation is what makes keeping them practical: it tolerates the short, irregular and
/// partly missing histories those firms have.
/// </para>
/// <para>
/// Three components, each independently testable: <see cref="Graph"/> (volatility series, distance
/// correlation, TMFG), <see cref="Attention"/> (masked graph attention with concatenated heads), and
/// <see cref="Objective"/> (the Sharpe loss and the sum-normalising allocation layer that replaces a
/// softmax).
/// </para>
/// <para><b>For Beginners:</b> It works out which companies genuinely move together, keeps only the
/// most informative connections, learns over that network which companies to hold, and is trained to
/// maximize return per unit of risk.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
/// <example>
/// <code>
/// var model = new GraphAttentionPortfolio&lt;double&gt;(
///     new GraphAttentionPortfolioOptions&lt;double&gt; { NumAssets = 30 });
/// Vector&lt;double&gt; weights = model.OptimizePortfolio(returnPanel);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GraphNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Large-scale Time-Varying Portfolio Optimisation using Graph Attention Networks",
    "https://arxiv.org/abs/2407.15532",
    Year = 2025,
    Authors = "Kamesh Korangi, Christophe Mues, Cristian Bravo")]
public partial class GraphAttentionPortfolio<T> : PortfolioOptimizerBase<T>
{
    private readonly GraphAttentionPortfolioOptions<T> _options;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Gets the graph builder: volatility series, distance correlation and TMFG filtering.</summary>
    public AssetGraphBuilder<T> Graph { get; }

    /// <summary>Gets the masked graph-attention core.</summary>
    public GraphAttentionLayerCore<T> Attention { get; }

    /// <summary>Gets the Sharpe objective and the sum-normalising allocation layer.</summary>
    public SharpeRatioPortfolioObjective<T> Objective { get; }

    /// <summary>Creates the model with the paper's defaults.</summary>
    public GraphAttentionPortfolio()
        : this(new GraphAttentionPortfolioOptions<T>())
    {
    }

    /// <summary>Creates the model.</summary>
    /// <param name="options">Configuration; defaults are the paper's where it states them.</param>
    /// <param name="architecture">Optional custom architecture.</param>
    /// <param name="lossFunction">
    /// Optional loss for the inherited training surface. The model's actual objective is
    /// <see cref="Objective"/>'s negative log Sharpe ratio, which is a property of the whole return
    /// series rather than a pointwise loss.
    /// </param>
    public GraphAttentionPortfolio(
        GraphAttentionPortfolioOptions<T> options,
        NeuralNetworkArchitecture<T>? architecture = null,
        ILossFunction<T>? lossFunction = null)
        : this(ResolveArchitecture(options, architecture), options, lossFunction)
    {
    }

    // Private chaining constructor so the resolved architecture reaches the base both as the
    // architecture and for its CalculatedInputSize without being constructed twice.
    private GraphAttentionPortfolio(
        NeuralNetworkArchitecture<T> resolvedArchitecture,
        GraphAttentionPortfolioOptions<T> options,
        ILossFunction<T>? lossFunction)
        : base(resolvedArchitecture, options.NumAssets, resolvedArchitecture.CalculatedInputSize, lossFunction)
    {
        _options = options;

        Graph = new AssetGraphBuilder<T>(options.VolatilityLookback);
        Attention = new GraphAttentionLayerCore<T>(options.LeakyReLUSlope);
        Objective = new SharpeRatioPortfolioObjective<T>();

        InitializeLayers();
    }

    private static NeuralNetworkArchitecture<T> ResolveArchitecture(
        GraphAttentionPortfolioOptions<T> options, NeuralNetworkArchitecture<T>? architecture)
    {
        Guard.NotNull(options);
        options.Validate();
        return architecture ?? CreateDefaultArchitecture(options);
    }

    private static NeuralNetworkArchitecture<T> CreateDefaultArchitecture(
        GraphAttentionPortfolioOptions<T> options)
    {
        // Input is a [window, assets] return panel; output is one score per asset, which the
        // allocation layer turns into weights.
        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: options.CorrelationWindow,
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

        Layers.AddRange(LayerHelper<T>.CreateDefaultGraphAttentionPortfolioLayers(
            Architecture,
            NumFeatures,
            _options.AttentionFeatureDimension * _options.NumHeads,
            _options.NumHeads,
            _options.NumAssets,
            _options.CorrelationWindow,
            _options.DropoutRate));
    }

    /// <summary>
    /// Builds the filtered asset graph for a return panel.
    /// </summary>
    /// <param name="returnPanel">Daily returns, shaped <c>[steps, assets]</c>.</param>
    /// <returns>The retained TMFG edges — at most <c>3(assets - 2)</c> of them.</returns>
    /// <remarks>
    /// Rebuilt per window rather than once, which is what makes the model time-varying: the set of
    /// active firms and their relationships both drift, and a graph fixed at the start of a 30-year
    /// sample would describe a market that no longer exists.
    /// </remarks>
    public IReadOnlyList<AssetGraphBuilder<T>.GraphEdge> BuildGraph(Tensor<T> returnPanel)
    {
        Guard.NotNull(returnPanel);

        var volatility = Graph.VolatilitySeries(returnPanel);
        var dependencies = Graph.DistanceCorrelationMatrix(volatility);
        return Graph.FilterTmfg(dependencies);
    }

    /// <summary>
    /// The graph adjacency (with self-loops) that the attention is masked by.
    /// </summary>
    public Tensor<T> BuildAdjacency(Tensor<T> returnPanel)
    {
        Guard.NotNull(returnPanel);

        // Rank is validated once, here. The old form fell back to _options.NumAssets for a
        // non-rank-2 panel, but BuildGraph -> Graph.VolatilitySeries throws ArgumentException for
        // exactly that input, so the fallback could never be reached and documented a contract the
        // method does not support.
        if (returnPanel.Shape.Length != 2)
        {
            throw new ArgumentException(
                $"Expected a rank-2 [steps, assets] return panel; got rank {returnPanel.Shape.Length}.",
                nameof(returnPanel));
        }

        int assets = returnPanel.Shape[1];
        return Graph.AdjacencyMask(BuildGraph(returnPanel), assets);
    }

    /// <summary>
    /// Produces long-only, fully invested portfolio weights, sparse by construction.
    /// </summary>
    /// <param name="marketData">Daily returns, shaped <c>[steps, assets]</c>.</param>
    /// <remarks>
    /// The network emits SCORES; <see cref="SharpeRatioPortfolioObjective{T}.Allocate"/> sum-normalises
    /// them. That is the paper's "Importance Layer" and it is deliberately not a softmax — a score of
    /// zero yields a weight of exactly zero, so the asset leaves the portfolio rather than receiving a
    /// sliver of capital that costs more to trade than it contributes.
    /// </remarks>
    public override Vector<T> OptimizePortfolio(Tensor<T> marketData)
    {
        Guard.NotNull(marketData);

        var scores = Predict(marketData).ToVector();

        // A custom architecture may widen the head; score the first NumAssets so the allocation length
        // always matches the universe rather than whatever shape came back.
        if (scores.Length > _options.NumAssets)
            scores = scores.Slice(0, _options.NumAssets);

        // A SHORT vector was previously passed through unchanged, producing an allocation with fewer
        // entries than the asset universe. Callers index weights by asset, so those weights silently
        // referred to the wrong assets from the first missing entry onward.
        if (scores.Length < _options.NumAssets)
        {
            throw new InvalidOperationException(
                $"The network produced {scores.Length} scores for a universe of "
                + $"{_options.NumAssets} assets. An allocation cannot be built from fewer scores than "
                + "assets; check that the architecture's output head matches NumAssets.");
        }

        return Objective.Allocate(scores);
    }

    /// <summary>
    /// The Sharpe ratio realized by an allocation over a return panel.
    /// </summary>
    public double RealizedSharpe(Vector<T> weights, Tensor<T> assetReturns)
    {
        Guard.NotNull(weights);
        Guard.NotNull(assetReturns);

        return Objective.SharpeRatio(Objective.PortfolioReturns(weights, assetReturns));
    }

    /// <summary>
    /// The training loss <c>-ln(mu_p) + ln(sigma_p)</c> for an allocation over a return panel.
    /// </summary>
    public double PortfolioLoss(Vector<T> weights, Tensor<T> assetReturns)
    {
        Guard.NotNull(weights);
        Guard.NotNull(assetReturns);

        return Objective.Loss(Objective.PortfolioReturns(weights, assetReturns));
    }

    // UpdateParameters restated the base verbatim; ModelBase routes it to SetParameters.
    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var copy = new GraphAttentionPortfolioOptions<T>
        {
            NumAssets = _options.NumAssets,
            VolatilityLookback = _options.VolatilityLookback,
            CorrelationWindow = _options.CorrelationWindow,
            AttentionFeatureDimension = _options.AttentionFeatureDimension,
            NumHeads = _options.NumHeads,
            LeakyReLUSlope = _options.LeakyReLUSlope,
            DropoutRate = _options.DropoutRate,
            L1Regularization = _options.L1Regularization,
            LearningRate = _options.LearningRate,
            BatchSize = _options.BatchSize,
            MaxEpochs = _options.MaxEpochs,
        };

        // LossFunction always carries across; calling the single-argument constructor took the
        // implicit default and a model built with a custom loss cloned into a different one.
        //
        // Architecture carries across ONLY when it holds no layers. InitializeLayers adds
        // Architecture.Layers into Layers BY REFERENCE when that collection is non-empty, and
        // ILayer<T> has no Clone, so handing a layer-carrying architecture to the clone would give
        // both models the SAME layer objects -- training or UpdateParameters on either would mutate
        // both. A clone that silently shares state is a worse defect than one that rebuilds default
        // layers, so the layer-carrying case falls back to the default build until layers can be
        // deep-copied.
        bool architectureCarriesLayers = Architecture.Layers is not null && Architecture.Layers.Count > 0;

        return architectureCarriesLayers
            ? new GraphAttentionPortfolio<T>(copy, null, LossFunction)
            : new GraphAttentionPortfolio<T>(copy, Architecture, LossFunction);
    }
}
