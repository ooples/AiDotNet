using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Finance.Base;

/// <summary>
/// Base class for financial models that operate on a CROSS-SECTION of assets connected by a graph,
/// and that predict more than one quantity per asset.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Two things separate these models from the rest of <see cref="FinancialModelBase{T}"/>, and neither
/// fits the single-tensor <c>Predict</c> contract:
/// </para>
/// <list type="number">
/// <item><description><b>A graph is an input.</b> Assets influence one another through a similarity or
/// correlation structure supplied alongside the features — typically precomputed (struc2vec,
/// correlation thresholding) rather than learned. Passing it through the feature tensor would require
/// packing a <c>[assets, assets]</c> matrix into a sequence tensor, which no caller could reasonably
/// be expected to get right.</description></item>
/// <item><description><b>There is more than one prediction head.</b> Multi-task financial models
/// predict, say, a return AND a direction, and the tasks are trained jointly because that joint
/// training is the contribution. Exposing only one head through <c>Predict</c> would present a
/// partial view of the model as if it were the whole thing.</description></item>
/// </list>
/// <para>
/// <b>Why a base class rather than folding this into FinancialModelBase.</b> Most financial models
/// here have neither property, and widening the common base for a minority would put an unused
/// <c>Adjacency</c> on every regression and forecasting model in the library. More than one model
/// needs this shape — <see cref="AiDotNet.Finance.Trading.Factors.Stockformer{T}"/> today, and
/// graph-attention portfolio construction next — which is what justifies the abstraction rather than
/// a one-off.
/// </para>
/// <para>
/// <b>What this deliberately does NOT do:</b> implement <c>IFactorModel</c>. That interface requires
/// <c>ExtractFactors</c>, <c>GetFactorLoadings</c>, <c>GetFactorCovariance</c> and <c>ComputeAlpha</c>
/// — latent-factor semantics. A cross-sectional graph forecaster has no factor decomposition, and
/// supplying degenerate values so the interface can be claimed would misrepresent the model. Models
/// that genuinely are factor models keep implementing it directly.
/// </para>
/// <para><b>For Beginners:</b> Some financial models look at many stocks at once and let related
/// stocks inform each other, instead of treating each in isolation. This is the shared foundation for
/// those, including the part where they answer more than one question at a time.</para>
/// </remarks>
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Length, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Length, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public abstract partial class CrossSectionalGraphModelBase<T> : FinancialModelBase<T>
{
    /// <summary>
    /// Initializes the shared cross-sectional state.
    /// </summary>
    /// <param name="architecture">The network architecture.</param>
    /// <param name="sequenceLength">Input window length in timesteps.</param>
    /// <param name="predictionHorizon">Forecast horizon in timesteps.</param>
    /// <param name="numFeatures">Features per asset per timestep.</param>
    /// <param name="lossFunction">
    /// Optional loss. Multi-task models generally compute their own combined objective rather than
    /// using a single scalar loss, so this is passed through unchanged for the base's benefit.
    /// </param>
    protected CrossSectionalGraphModelBase(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength,
        int predictionHorizon,
        int numFeatures,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, sequenceLength, predictionHorizon, numFeatures, lossFunction)
    {
    }

    /// <summary>
    /// Gets or sets the asset relationship TOPOLOGY, <c>[assets, assets]</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Who is connected to whom. This is what message-passing models need — GCN and GAT attend over a
    /// node's NEIGHBOURS, so they require the edge structure itself.
    /// </para>
    /// <para>
    /// Deliberately separate from <see cref="AssetEmbedding"/>. Topology and features are different
    /// things, and the split follows the standard convention (PyTorch Geometric keeps <c>edge_index</c>
    /// apart from <c>x</c>). Collapsing them into one member forces a model that wants one to
    /// misinterpret the other.
    /// </para>
    /// <para>
    /// Data, not a learned parameter: supplied by the caller, absent from the parameter vector, and so
    /// not part of training, serialization or cloning.
    /// </para>
    /// </remarks>
    public Matrix<T>? AssetGraph { get; set; }

    /// <summary>
    /// Gets or sets a precomputed per-asset structural EMBEDDING, <c>[assets, embeddingWidth]</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The output of struc2vec, node2vec, DeepWalk or similar: the graph's structure already encoded
    /// as a feature vector per asset. Models consume it by ADDING it to their inputs, not by treating
    /// it as an adjacency — Stockformer's Eq. 10 is <c>X~ = X + rho^spa + rho^tem</c>.
    /// </para>
    /// <para>
    /// A model needing topology should read <see cref="AssetGraph"/>; one needing structural features
    /// should read this. Supplying an embedding where topology is expected (or the reverse) is a
    /// modelling error, not something to silently coerce.
    /// </para>
    /// </remarks>
    public Matrix<T>? AssetEmbedding { get; set; }

    /// <summary>
    /// Whether asset TOPOLOGY has been supplied.
    /// </summary>
    /// <remarks>
    /// Exposed so callers and tests can detect the degenerate case. Without a graph, a message-passing
    /// model falls back to the identity — every asset seeing only itself — which is legitimate but
    /// disables the cross-sectional mechanism these models exist for.
    /// </remarks>
    public bool HasGraph => AssetGraph is not null;

    /// <summary>
    /// Whether a precomputed structural EMBEDDING has been supplied.
    /// </summary>
    /// <remarks>
    /// Separate from <see cref="HasGraph"/> because the two inputs are separate. A model can
    /// legitimately have one and not the other, and reporting a single "has graph" flag would hide
    /// which of them is actually present.
    /// </remarks>
    public bool HasEmbedding => AssetEmbedding is not null;

    /// <summary>
    /// Gets the number of prediction heads this model exposes.
    /// </summary>
    /// <remarks>
    /// At least one. A model reporting a single head still belongs here if it consumes a graph.
    /// </remarks>
    public abstract int TaskCount { get; }

    /// <summary>
    /// Gets the name of each prediction head, in the order
    /// <see cref="PredictAllTasks"/> returns them.
    /// </summary>
    /// <remarks>
    /// Named rather than positional so a caller reading results does not have to know the model's
    /// internal ordering — getting "return" and "direction" the wrong way round is silent and the
    /// numbers look plausible either way.
    /// </remarks>
    public abstract IReadOnlyList<string> TaskNames { get; }

    /// <summary>
    /// Runs every prediction head over the cross-section.
    /// </summary>
    /// <param name="input">Cross-sectional input; shape is model-specific.</param>
    /// <returns>
    /// One tensor per head, ordered to match <see cref="TaskNames"/>, with
    /// <see cref="TaskCount"/> entries.
    /// </returns>
    /// <remarks>
    /// This is the honest full-output contract. <c>Predict</c> remains available and returns the
    /// PRIMARY head only, so single-output callers still work — but they are getting one head, not the
    /// model's complete answer.
    /// </remarks>
    public abstract IReadOnlyList<Tensor<T>> PredictAllTasks(Tensor<T> input);

    /// <summary>
    /// Resolves the graph to use, substituting the identity when none was supplied.
    /// </summary>
    /// <param name="assetCount">Number of assets in the current cross-section.</param>
    /// <exception cref="InvalidOperationException">
    /// Thrown when a graph was supplied whose size disagrees with the cross-section. Silently
    /// resizing or ignoring it would let a stale graph from a different universe of assets be applied
    /// to the wrong ones, which produces plausible numbers and no error.
    /// </exception>
    protected Matrix<T> ResolveGraph(int assetCount)
    {
        if (assetCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(assetCount), assetCount, "Asset count must be positive.");

        var graph = AssetGraph;
        if (graph is null)
        {
            var identity = new Matrix<T>(assetCount, assetCount);
            var one = MathHelper.GetNumericOperations<T>().One;
            for (int i = 0; i < assetCount; i++) identity[i, i] = one;
            return identity;
        }

        if (graph.Rows != assetCount || graph.Columns != assetCount)
        {
            throw new InvalidOperationException(
                $"{nameof(AssetGraph)} is [{graph.Rows}, {graph.Columns}] but this cross-section has " +
                $"{assetCount} assets. A graph built for a different asset universe would be applied to " +
                "the wrong assets and still produce plausible-looking output, so this fails instead.");
        }

        return graph;
    }

    /// <summary>
    /// Resolves the structural embedding, substituting zeros when none was supplied.
    /// </summary>
    /// <param name="assetCount">Number of assets in the current cross-section.</param>
    /// <param name="width">Embedding width the model expects.</param>
    /// <exception cref="InvalidOperationException">
    /// Thrown when a supplied embedding disagrees with the cross-section or the expected width. An
    /// embedding built for a different asset universe would attach the wrong structural prior to each
    /// asset and still produce plausible output.
    /// </exception>
    protected Matrix<T> ResolveEmbedding(int assetCount, int width)
    {
        if (assetCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(assetCount), assetCount, "Asset count must be positive.");
        if (width <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), width, "Embedding width must be positive.");

        var embedding = AssetEmbedding;
        if (embedding is null)
        {
            // Zeros: an additive embedding of zero is the identity, so the model degrades to
            // "no structural prior" rather than to something arbitrary.
            return new Matrix<T>(assetCount, width);
        }

        if (embedding.Rows != assetCount || embedding.Columns != width)
        {
            throw new InvalidOperationException(
                $"{nameof(AssetEmbedding)} is [{embedding.Rows}, {embedding.Columns}] but this model " +
                $"expects [{assetCount}, {width}]. An embedding from a different asset universe or a " +
                "different encoder width would attach the wrong prior to each asset and still look " +
                "plausible, so this fails instead.");
        }

        return embedding;
    }
}
