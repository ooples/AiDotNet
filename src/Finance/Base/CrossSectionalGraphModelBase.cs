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
public abstract class CrossSectionalGraphModelBase<T> : FinancialModelBase<T>
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
    /// Gets or sets the asset relationship graph, <c>[assets, assets]</c>.
    /// </summary>
    /// <value>
    /// The adjacency or embedding-derived affinity between assets, or <c>null</c> to treat assets as
    /// unconnected.
    /// </value>
    /// <remarks>
    /// <para>
    /// Data, not a learned parameter: it is supplied by the caller and is not part of the model's
    /// parameter vector, so it does not participate in training, serialization or cloning. Setting it
    /// changes predictions without changing any weight.
    /// </para>
    /// <para>
    /// <c>null</c> means the identity graph — every asset sees only itself. That is a legitimate
    /// baseline, but it disables the cross-sectional mechanism these models exist for, so
    /// <see cref="HasGraph"/> is exposed to let callers and tests detect the degenerate case rather
    /// than silently getting isolated-asset behaviour.
    /// </para>
    /// </remarks>
    public Matrix<T>? Adjacency { get; set; }

    /// <summary>
    /// Whether a non-trivial asset graph has been supplied.
    /// </summary>
    public bool HasGraph => Adjacency is not null;

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

        var graph = Adjacency;
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
                $"{nameof(Adjacency)} is [{graph.Rows}, {graph.Columns}] but this cross-section has " +
                $"{assetCount} assets. A graph built for a different asset universe would be applied to " +
                "the wrong assets and still produce plausible-looking output, so this fails instead.");
        }

        return graph;
    }
}
