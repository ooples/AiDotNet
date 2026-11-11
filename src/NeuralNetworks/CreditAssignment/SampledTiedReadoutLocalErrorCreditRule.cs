using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// <b>Sampled tied-readout local error</b>: zero-auxiliary-parameter sequence local credit assignment that evaluates
/// each hidden layer's next-token cross-entropy over a compact candidate subset <c>U ≪ V</c> of the network's shared
/// readout (all unique batch target tokens plus uniform deterministic negative vocabulary tokens) with readout
/// parameters detached on intermediate layers by default.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// Evaluating a full <c>V = 49,152</c> vocabulary projection on every intermediate hidden layer costs <c>O(L · N · D · V)</c>
/// FLOPs, dominating deep zero-backprop language-model steps. Gathering the <c>U</c> batch-positive + negative columns
/// of the shared readout once per step via <c>IEngine.TensorEmbeddingLookup</c> reduces intermediate readout GEMMs from
/// <c>[N, D] × [D, V]</c> to <c>[N, D] × [D, U]</c> (e.g. <c>48×–96×</c> smaller at <c>U = 512..1024</c>) while
/// preserving exact target-token coverage and zero auxiliary model parameters.
/// </para>
/// <para>
/// When <see cref="MaxCandidates"/> is at least the vocabulary size or the active loss is not class-index
/// <see cref="AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss{T}"/>, the rule falls back seamlessly to exact
/// full-readout evaluation (<see cref="LocalObjectiveStep{T}.ReadoutLossDetached"/> / <see cref="LocalObjectiveStep{T}.ReadoutLoss"/>).
/// </para>
/// </remarks>
public sealed class SampledTiedReadoutLocalErrorCreditRule<T>
    : ILocalObjectiveCreditRule<T>, IOutputPredictionOptionalRule, IBatchedLocalReadoutRule
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly double _localWeight;
    private readonly int _seed;
    private Tensor<T>? _lastStepInput;
    private int _stepCounter;

    /// <summary>Creates a sampled tied-readout local credit rule.</summary>
    /// <param name="readoutLayerCount">Trailing layers forming the shared readout (language model: norm + head = 2).</param>
    /// <param name="maxCandidates">Target candidate vocabulary subset size <c>U</c> (all unique batch targets are always retained even if their count exceeds <paramref name="maxCandidates"/>).</param>
    /// <param name="localWeight">Base weight λ of each hidden layer's local loss.</param>
    /// <param name="detachReadoutOnHiddenLayers">When <c>true</c> (default), intermediate hidden layers do not accumulate gradients into the shared readout parameters.</param>
    /// <param name="layerStride">Stride across hidden layers receiving local readout losses (1 = every hidden layer).</param>
    /// <param name="useDepthProgressiveWeight">When <c>true</c>, scales hidden layer <c>k</c>'s local weight by <c>((k + 1) / H)²</c>.</param>
    /// <param name="seed">Optional deterministic seed for negative vocabulary sampling.</param>
    public SampledTiedReadoutLocalErrorCreditRule(
        int readoutLayerCount = 2,
        int maxCandidates = 1024,
        double localWeight = 1.0,
        bool detachReadoutOnHiddenLayers = true,
        int layerStride = 1,
        bool useDepthProgressiveWeight = false,
        int? seed = null)
    {
        if (readoutLayerCount < 1) throw new ArgumentOutOfRangeException(nameof(readoutLayerCount), "Must be at least 1.");
        if (maxCandidates < 2) throw new ArgumentOutOfRangeException(nameof(maxCandidates), "Must be at least 2.");
        if (double.IsNaN(localWeight) || double.IsInfinity(localWeight) || localWeight <= 0) throw new ArgumentOutOfRangeException(nameof(localWeight), "Must be positive and finite.");
        if (layerStride < 1) throw new ArgumentOutOfRangeException(nameof(layerStride), "Must be at least 1.");

        ReadoutLayerCount = readoutLayerCount;
        MaxCandidates = maxCandidates;
        _localWeight = localWeight;
        DetachReadoutOnHiddenLayers = detachReadoutOnHiddenLayers;
        LayerStride = layerStride;
        UseDepthProgressiveWeight = useDepthProgressiveWeight;
        _seed = seed ?? 20261111;
    }

    /// <inheritdoc />
    public string Name => "SampledTiedReadoutLocalError";

    /// <inheritdoc />
    public bool IsExactBackprop => false;

    /// <inheritdoc />
    public int ReadoutLayerCount { get; }

    /// <summary>Gets the target candidate vocabulary subset size <c>U</c>.</summary>
    public int MaxCandidates { get; }

    /// <summary>Gets whether intermediate hidden layers detach the shared readout's own parameters.</summary>
    public bool DetachReadoutOnHiddenLayers { get; }

    /// <summary>Gets the stride between hidden layers receiving local readout objectives.</summary>
    public int LayerStride { get; }

    /// <summary>Gets whether local loss weights scale quadratically with normalized layer depth.</summary>
    public bool UseDepthProgressiveWeight { get; }

    bool IOutputPredictionOptionalRule.UsesOutputPrediction => false;

    BatchedLocalReadoutMode IBatchedLocalReadoutRule.BatchedReadoutMode => BatchedLocalReadoutMode.SampledVocabulary;
    double IBatchedLocalReadoutRule.BaseLocalWeight => _localWeight;
    int IBatchedLocalReadoutRule.SampledMaxCandidates => MaxCandidates;

    int IBatchedLocalReadoutRule.AdvanceAndGetStepSeed(object stepToken) => AdvanceStepSeedCore(stepToken);

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
        int lastHidden = context.Step.HiddenLayerCount - 1;
        if (context.HiddenIndex == lastHidden) return null;
        if (LayerStride > 1 && ((lastHidden - 1 - context.HiddenIndex) % LayerStride) != 0)
            return null;

        var s = context.Step;
        int stepSeed = AdvanceStepSeedCore(s.Input);

        var local = s.TrySampledReadoutLoss?.Invoke(
                context.Output, s.Target, MaxCandidates, DetachReadoutOnHiddenLayers, stepSeed)
            ?? (DetachReadoutOnHiddenLayers
                ? s.ReadoutLossDetached(context.Output, s.Target)
                : s.ReadoutLoss(context.Output, s.Target));

        double weight = _localWeight;
        if (UseDepthProgressiveWeight)
        {
            double depthFrac = (context.HiddenIndex + 1.0) / s.HiddenLayerCount;
            weight *= depthFrac * depthFrac;
        }

        return weight == 1.0 ? local : s.Engine.TensorMultiplyScalar(local, Ops.FromDouble(weight));
    }

    /// <inheritdoc />
    public Tensor<T>? BuildOutputObjective(LocalObjectiveStep<T> step, Tensor<T> prediction, Tensor<T> defaultLoss) => null;

    /// <inheritdoc />
    public void ApplyAuxiliaryGradients(IReadOnlyDictionary<Tensor<T>, Tensor<T>> gradients)
    {
        _lastStepInput = null;
    }
}
