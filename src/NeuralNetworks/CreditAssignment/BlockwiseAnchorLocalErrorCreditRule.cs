using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// <b>Blockwise anchor local error</b>: decoupled block-local sequence credit assignment that partitions a deep
/// network into contiguous blocks of at most <see cref="AnchorStride"/> trainable pre-readout layers, places a shared
/// (full or candidate-sampled) tied-readout next-token loss at the end of every block, and detaches activations
/// (<c>StopGradient</c>) strictly at block boundaries so every layer inside a block receives multi-layer local credit
/// while zero gradient crosses between blocks.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// Pure layer-by-layer detachment (<c>LayerStride &gt; 1</c> on <see cref="TiedReadoutLocalErrorCreditRule{T}"/>) gives
/// unselected intermediate layers zero gradient, while detaching every single layer (<c>LayerStride = 1</c>) prevents
/// stacked linear-attention / GatedDeltaNet layers from cooperating with a periodic global attention / MLA anchor.
/// Partitioning at anchor boundaries (for example <c>AnchorStride = 4</c> matching a 3:1 GDN:MLA hybrid block) gives
/// 100% of layers constructive block-local credit with zero end-to-end backpropagation.
/// </para>
/// <para>
/// When <see cref="StreamBlocksSequentially"/> is <c>true</c> (the default), <c>LocalObjectiveGradientComputer</c>
/// executes one isolated <c>GradientTape</c> and <c>TensorScope</c> per decoupled block sequentially, bounding peak
/// activation memory to <c>O(AnchorStride)</c> layers with 0% forward recomputation overhead while producing the exact
/// same block-decoupled gradients as single-tape execution.
/// </para>
/// </remarks>
public sealed class BlockwiseAnchorLocalErrorCreditRule<T>
    : ILocalObjectiveCreditRule<T>,
      IOutputPredictionOptionalRule,
      IBatchedLocalReadoutRule,
      ISequentialBlockLocalCreditRule
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly double _localWeight;
    private readonly int _seed;
    private Tensor<T>? _lastStepInput;
    private int _stepCounter;

    /// <summary>Creates a blockwise anchor local-error credit rule.</summary>
    /// <param name="readoutLayerCount">Trailing layers forming the shared readout (language model: norm + head = 2).</param>
    /// <param name="anchorStride">Maximum number of trainable hidden layers per decoupled block (e.g. 4 for 3 GDN + 1 MLA).</param>
    /// <param name="maxCandidates">Candidate vocabulary subset size <c>U</c> for intermediate block anchors (<see cref="int.MaxValue"/> uses full vocabulary).</param>
    /// <param name="localWeight">Base weight λ applied to each intermediate block anchor's local loss.</param>
    /// <param name="detachReadoutOnHiddenBlocks">When <c>true</c> (default), intermediate block anchors detach the shared readout's own parameters so only the terminal block updates the readout.</param>
    /// <param name="useDepthProgressiveWeight">When <c>true</c>, scales intermediate anchor <c>k</c>'s weight by <c>((k + 1) / H)²</c>.</param>
    /// <param name="streamBlocksSequentially">When <c>true</c> (default), runs each decoupled block on its own sequential tape/scope for <c>O(AnchorStride)</c> activation memory with zero recomputation.</param>
    /// <param name="seed">Optional deterministic seed when <paramref name="maxCandidates"/> samples negative vocabulary candidates.</param>
    public BlockwiseAnchorLocalErrorCreditRule(
        int readoutLayerCount = 2,
        int anchorStride = 4,
        int maxCandidates = 1024,
        double localWeight = 1.0,
        bool detachReadoutOnHiddenBlocks = true,
        bool useDepthProgressiveWeight = true,
        bool streamBlocksSequentially = true,
        int? seed = null)
    {
        if (readoutLayerCount < 1)
            throw new ArgumentOutOfRangeException(nameof(readoutLayerCount), "Must be at least 1.");
        if (anchorStride < 1)
            throw new ArgumentOutOfRangeException(nameof(anchorStride), "Must be at least 1.");
        if (maxCandidates < 2)
            throw new ArgumentOutOfRangeException(nameof(maxCandidates), "Must be at least 2.");
        if (double.IsNaN(localWeight) || double.IsInfinity(localWeight) || localWeight <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(localWeight), "Must be positive and finite.");

        ReadoutLayerCount = readoutLayerCount;
        AnchorStride = anchorStride;
        MaxCandidates = maxCandidates;
        _localWeight = localWeight;
        DetachReadoutOnHiddenBlocks = detachReadoutOnHiddenBlocks;
        UseDepthProgressiveWeight = useDepthProgressiveWeight;
        StreamBlocksSequentially = streamBlocksSequentially;
        _seed = seed ?? 20261111;
    }

    /// <inheritdoc />
    public string Name => "BlockwiseAnchorLocalError";

    /// <inheritdoc />
    public bool IsExactBackprop => false;

    /// <inheritdoc />
    public int ReadoutLayerCount { get; }

    /// <summary>Gets the number of trainable hidden layers per decoupled anchor block.</summary>
    public int AnchorStride { get; }

    /// <summary>Gets the candidate vocabulary subset size <c>U</c> (<see cref="int.MaxValue"/> for full vocabulary).</summary>
    public int MaxCandidates { get; }

    /// <summary>Gets whether intermediate block anchors detach the shared readout's own parameters.</summary>
    public bool DetachReadoutOnHiddenBlocks { get; }

    /// <summary>Gets whether intermediate block anchor weights scale quadratically with normalized depth.</summary>
    public bool UseDepthProgressiveWeight { get; }

    /// <summary>Gets whether decoupled blocks execute sequentially with <c>O(AnchorStride)</c> activation memory.</summary>
    public bool StreamBlocksSequentially { get; }

    bool IOutputPredictionOptionalRule.UsesOutputPrediction => false;

    BatchedLocalReadoutMode IBatchedLocalReadoutRule.BatchedReadoutMode =>
        MaxCandidates == int.MaxValue ? BatchedLocalReadoutMode.FullVocabulary : BatchedLocalReadoutMode.SampledVocabulary;
    bool IBatchedLocalReadoutRule.DetachReadoutOnHiddenLayers => DetachReadoutOnHiddenBlocks;
    int IBatchedLocalReadoutRule.LayerStride => AnchorStride;
    bool IBatchedLocalReadoutRule.UseDepthProgressiveWeight => UseDepthProgressiveWeight;
    double IBatchedLocalReadoutRule.BaseLocalWeight => _localWeight;
    int IBatchedLocalReadoutRule.SampledMaxCandidates => MaxCandidates;
    int IBatchedLocalReadoutRule.AdvanceAndGetStepSeed(object stepToken) => AdvanceStepSeedCore(stepToken);
    bool IBatchedLocalReadoutRule.IsSelectedIntermediateHidden(int hiddenIndex, int hiddenLayerCount)
        => IsIntermediateAnchor(hiddenIndex, hiddenLayerCount);

    bool ILocalBoundaryCreditRule.ShouldDetachBeforeHiddenLayer(int hiddenIndex, int hiddenLayerCount)
        => hiddenIndex > 0 && IsIntermediateAnchor(hiddenIndex - 1, hiddenLayerCount);

    /// <summary>
    /// Returns <c>true</c> when hidden trainable layer <paramref name="hiddenIndex"/> (0-based out of
    /// <paramref name="hiddenLayerCount"/>) is the anchor terminating an intermediate decoupled block.
    /// </summary>
    public bool IsIntermediateAnchor(int hiddenIndex, int hiddenLayerCount)
    {
        int lastHidden = hiddenLayerCount - 1;
        if (hiddenIndex < 0 || hiddenIndex >= lastHidden)
            return false;
        return ((lastHidden - hiddenIndex) % AnchorStride) == 0;
    }

    private int AdvanceStepSeedCore(object stepToken)
    {
        if (!ReferenceEquals(stepToken, _lastStepInput))
        {
            _lastStepInput = stepToken as Tensor<T>;
            _stepCounter = unchecked(_stepCounter + 1);
        }
        return unchecked(_seed + _stepCounter * 7919);
    }

    /// <inheritdoc />
    public IReadOnlyList<Tensor<T>> AuxiliaryParameters => Array.Empty<Tensor<T>>();

    /// <inheritdoc />
    public void Initialize(ICreditAssignmentContext<T> context) { }

    /// <inheritdoc />
    public void ComputeTeachingSignals(ICreditAssignmentContext<T> context) =>
        throw new NotSupportedException($"{Name} is a local-objective rule; it is driven through {nameof(ILocalObjectiveCreditRule<T>)}.");

    /// <inheritdoc />
    public Tensor<T>? BuildLocalObjective(LocalObjectiveContext<T> context)
    {
        if (!IsIntermediateAnchor(context.HiddenIndex, context.Step.HiddenLayerCount))
            return null;

        var s = context.Step;
        Tensor<T>? local = null;
        if (MaxCandidates < int.MaxValue && s.TrySampledReadoutLoss is not null)
        {
            int stepSeed = AdvanceStepSeedCore(s.Input);
            local = s.TrySampledReadoutLoss(
                context.Output, s.Target, MaxCandidates, DetachReadoutOnHiddenBlocks, stepSeed);
        }

        local ??= DetachReadoutOnHiddenBlocks
            ? s.ReadoutLossDetached(context.Output, s.Target)
            : s.ReadoutLoss(context.Output, s.Target);

        double weight = _localWeight;
        if (UseDepthProgressiveWeight)
        {
            double depthFrac = (context.HiddenIndex + 1.0) / s.HiddenLayerCount;
            weight *= depthFrac * depthFrac;
        }

        return BitConverter.DoubleToInt64Bits(weight) == BitConverter.DoubleToInt64Bits(1.0)
            ? local
            : s.Engine.TensorMultiplyScalar(local, Ops.FromDouble(weight));
    }

    /// <inheritdoc />
    public Tensor<T>? BuildOutputObjective(LocalObjectiveStep<T> step, Tensor<T> prediction, Tensor<T> defaultLoss) => null;

    /// <inheritdoc />
    public void ApplyAuxiliaryGradients(IReadOnlyDictionary<Tensor<T>, Tensor<T>> gradients)
    {
        _lastStepInput = null;
    }
}
