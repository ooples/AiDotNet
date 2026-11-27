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
    private readonly int _auxRank;
    private readonly int _hiddenDim;
    private readonly double _auxLr;
    private readonly Tensor<T>[] _auxU;
    private readonly Tensor<T>[] _auxV;
    private readonly Tensor<T>[] _auxParams;
    private readonly Tensor<T>[] _auxM;
    private readonly Tensor<T>[] _auxV2;
    private int _auxAdamSteps;
    private Tensor<T>? _lastStepInput;
    private int _stepCounter;

    /// <summary>Creates a blockwise anchor local-error credit rule.</summary>
    public BlockwiseAnchorLocalErrorCreditRule(
        int readoutLayerCount = 2,
        int anchorStride = 4,
        int maxCandidates = 1024,
        double localWeight = 1.0,
        bool detachReadoutOnHiddenBlocks = true,
        bool useDepthProgressiveWeight = true,
        bool streamBlocksSequentially = true,
        int? seed = null)
        : this(readoutLayerCount, anchorStride, maxCandidates, localWeight, detachReadoutOnHiddenBlocks,
               useDepthProgressiveWeight, streamBlocksSequentially, seed, 0, 0, 8, 1.5e-3)
    {
    }

    /// <summary>Creates a blockwise anchor local-error credit rule with training-only residual frame-alignment heads.</summary>
    /// <remarks>
    /// Each intermediate anchor block <c>b</c> evaluates its local readout loss on <c>h + (h U_b) V_bᵀ</c> instead of <c>h</c>,
    /// where <c>U_b</c> and <c>V_b</c> are <c>[hiddenDimension, auxiliaryProjectionRank]</c> matrices owned by the rule
    /// (<see cref="AuxiliaryParameters"/>) and trained by its own AdamW step in <see cref="ApplyAuxiliaryGradients"/>. The
    /// heads exist only in the training objective: the model's forward pass and its parameters are unchanged. <c>V_b</c> starts
    /// small, so the first steps see almost the plain anchor loss.
    /// </remarks>
    /// <param name="readoutLayerCount">Trailing layers forming the shared readout (language model: norm + head = 2).</param>
    /// <param name="anchorStride">Maximum number of trainable hidden layers per decoupled block (e.g. 4 for 3 GDN + 1 MLA).</param>
    /// <param name="maxCandidates">Candidate vocabulary subset size <c>U</c> for intermediate block anchors (<see cref="int.MaxValue"/> uses full vocabulary).</param>
    /// <param name="localWeight">Base weight λ applied to each intermediate block anchor's local loss.</param>
    /// <param name="detachReadoutOnHiddenBlocks">When <c>true</c> (default), intermediate block anchors detach the shared readout's own parameters so only the terminal block updates the readout.</param>
    /// <param name="useDepthProgressiveWeight">When <c>true</c>, scales intermediate anchor <c>k</c>'s weight by <c>((k + 1) / H)²</c>.</param>
    /// <param name="streamBlocksSequentially">When <c>true</c> (default), runs each decoupled block on its own sequential tape/scope for <c>O(AnchorStride)</c> activation memory with zero recomputation.</param>
    /// <param name="seed">Optional deterministic seed when <paramref name="maxCandidates"/> samples negative vocabulary candidates.</param>
    /// <param name="auxiliaryProjectionRank">Rank <c>r</c> of each block's alignment head; 0 disables the heads.</param>
    /// <param name="hiddenDimension">Width of the hidden states the anchors see; required when <paramref name="auxiliaryProjectionRank"/> is positive.</param>
    /// <param name="maxAnchorBlocks">Number of intermediate anchor blocks that get a head; a network with more intermediate anchors throws.</param>
    /// <param name="auxiliaryLearningRate">AdamW learning rate of the heads; must be positive and finite.</param>
    public static BlockwiseAnchorLocalErrorCreditRule<T> CreateWithFrameAlignment(
        int readoutLayerCount,
        int anchorStride,
        int maxCandidates,
        double localWeight,
        bool detachReadoutOnHiddenBlocks,
        bool useDepthProgressiveWeight,
        bool streamBlocksSequentially,
        int? seed,
        int auxiliaryProjectionRank,
        int hiddenDimension,
        int maxAnchorBlocks = 8,
        double auxiliaryLearningRate = 1.5e-3)
        => new(readoutLayerCount, anchorStride, maxCandidates, localWeight, detachReadoutOnHiddenBlocks,
               useDepthProgressiveWeight, streamBlocksSequentially, seed,
               auxiliaryProjectionRank, hiddenDimension, maxAnchorBlocks, auxiliaryLearningRate);

    private BlockwiseAnchorLocalErrorCreditRule(
        int readoutLayerCount,
        int anchorStride,
        int maxCandidates,
        double localWeight,
        bool detachReadoutOnHiddenBlocks,
        bool useDepthProgressiveWeight,
        bool streamBlocksSequentially,
        int? seed,
        int auxiliaryProjectionRank,
        int hiddenDimension,
        int maxAnchorBlocks,
        double auxiliaryLearningRate)
    {
        if (readoutLayerCount < 1)
            throw new ArgumentOutOfRangeException(nameof(readoutLayerCount), "Must be at least 1.");
        if (anchorStride < 1)
            throw new ArgumentOutOfRangeException(nameof(anchorStride), "Must be at least 1.");
        if (maxCandidates < 2)
            throw new ArgumentOutOfRangeException(nameof(maxCandidates), "Must be at least 2.");
        if (double.IsNaN(localWeight) || double.IsInfinity(localWeight) || localWeight <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(localWeight), "Must be positive and finite.");
        if (auxiliaryProjectionRank < 0)
            throw new ArgumentOutOfRangeException(nameof(auxiliaryProjectionRank), "Must be non-negative.");
        if (auxiliaryProjectionRank > 0 && hiddenDimension < 1)
            throw new ArgumentOutOfRangeException(nameof(hiddenDimension), "Must be positive when auxiliaryProjectionRank > 0.");
        if (auxiliaryProjectionRank > 0 && maxAnchorBlocks < 1)
            throw new ArgumentOutOfRangeException(nameof(maxAnchorBlocks), "Must be positive when auxiliaryProjectionRank > 0.");
        if (double.IsNaN(auxiliaryLearningRate) || double.IsInfinity(auxiliaryLearningRate) || auxiliaryLearningRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(auxiliaryLearningRate), "Must be positive and finite.");

        ReadoutLayerCount = readoutLayerCount;
        AnchorStride = anchorStride;
        MaxCandidates = maxCandidates;
        _localWeight = localWeight;
        DetachReadoutOnHiddenBlocks = detachReadoutOnHiddenBlocks;
        UseDepthProgressiveWeight = useDepthProgressiveWeight;
        StreamBlocksSequentially = streamBlocksSequentially;
        _seed = seed ?? 20261111;
        _auxRank = auxiliaryProjectionRank;
        _hiddenDim = hiddenDimension;
        _auxLr = auxiliaryLearningRate;

        if (_auxRank > 0)
        {
            using var suspend = AiDotNet.Tensors.Engines.TensorScope.Suspend();
            using var noArena = AiDotNet.Tensors.Helpers.TensorArena.Suspend();
            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();

            var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
            var rng = new Random(_seed ^ 0x5A17C3);
            double scaleU = Math.Sqrt(3.0 / _hiddenDim);
            double scaleV = 0.02 * Math.Sqrt(3.0 / _auxRank);
            T posU = Ops.FromDouble(scaleU), negU = Ops.FromDouble(-scaleU);
            T posV = Ops.FromDouble(scaleV), negV = Ops.FromDouble(-scaleV);
            _auxU = new Tensor<T>[maxAnchorBlocks];
            _auxV = new Tensor<T>[maxAnchorBlocks];
            _auxParams = new Tensor<T>[maxAnchorBlocks * 2];
            _auxM = new Tensor<T>[maxAnchorBlocks * 2];
            _auxV2 = new Tensor<T>[maxAnchorBlocks * 2];
            int[] projShape = [_hiddenDim, _auxRank];
            for (int b = 0; b < maxAnchorBlocks; b++)
            {
                _auxU[b] = AiDotNet.Tensors.Engines.TensorScope.Detach(engine.TensorRandomUniformRange<T>(projShape, negU, posU, rng.Next()));
                _auxV[b] = AiDotNet.Tensors.Engines.TensorScope.Detach(engine.TensorRandomUniformRange<T>(projShape, negV, posV, rng.Next()));
                _auxParams[2 * b] = _auxU[b];
                _auxParams[2 * b + 1] = _auxV[b];
                _auxM[2 * b] = AiDotNet.Tensors.Engines.TensorScope.Detach(new Tensor<T>(projShape));
                _auxM[2 * b + 1] = AiDotNet.Tensors.Engines.TensorScope.Detach(new Tensor<T>(projShape));
                _auxV2[2 * b] = AiDotNet.Tensors.Engines.TensorScope.Detach(new Tensor<T>(projShape));
                _auxV2[2 * b + 1] = AiDotNet.Tensors.Engines.TensorScope.Detach(new Tensor<T>(projShape));
            }
        }
        else
        {
            _auxU = Array.Empty<Tensor<T>>();
            _auxV = Array.Empty<Tensor<T>>();
            _auxParams = Array.Empty<Tensor<T>>();
            _auxM = Array.Empty<Tensor<T>>();
            _auxV2 = Array.Empty<Tensor<T>>();
        }
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
    public IReadOnlyList<Tensor<T>> AuxiliaryParameters => _auxParams;

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
        var hEval = context.Output;
        if (_auxU.Length > 0)
        {
            if (hEval.Shape.Length < 2 || hEval.Shape[hEval.Shape.Length - 1] != _hiddenDim)
                throw new InvalidOperationException(
                    $"Frame-alignment heads expect hidden states of width {_hiddenDim}; hidden layer {context.HiddenIndex} " +
                    $"produced shape [{string.Join(", ", hEval.Shape.ToArray())}].");
            int anchorSlot = context.HiddenIndex / AnchorStride;
            if (anchorSlot >= _auxU.Length)
                throw new InvalidOperationException(
                    $"Intermediate anchor block {anchorSlot} has no frame-alignment head: the rule was created with " +
                    $"maxAnchorBlocks = {_auxU.Length}. Create it with at least {anchorSlot + 1}.");
            int rows = hEval.Length / _hiddenDim;
            var h2d = hEval.Rank == 2 ? hEval : s.Engine.Reshape(hEval, [rows, _hiddenDim]);
            var latent = s.Engine.TensorMatMul(h2d, _auxU[anchorSlot]);
            var delta = s.Engine.TensorMatMulTransposed(latent, _auxV[anchorSlot]);
            var adapted2d = s.Engine.TensorAdd(h2d, delta);
            hEval = hEval.Rank == 2 ? adapted2d : s.Engine.Reshape(adapted2d, hEval.Shape.ToArray());
        }

        Tensor<T>? local = null;
        if (MaxCandidates < int.MaxValue && s.TrySampledReadoutLoss is not null)
        {
            int stepSeed = AdvanceStepSeedCore(s.Input);
            local = s.TrySampledReadoutLoss(
                hEval, s.Target, MaxCandidates, DetachReadoutOnHiddenBlocks, stepSeed);
        }

        local ??= DetachReadoutOnHiddenBlocks
            ? s.ReadoutLossDetached(hEval, s.Target)
            : s.ReadoutLoss(hEval, s.Target);

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
        if (_auxParams.Length == 0 || gradients.Count == 0)
            return;

        using var suspend = AiDotNet.Tensors.Engines.TensorScope.Suspend();
        using var noArena = AiDotNet.Tensors.Helpers.TensorArena.Suspend();
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();

        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        _auxAdamSteps++;
        const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8, wd = 0.01;
        double bc1 = 1.0 - Math.Pow(beta1, _auxAdamSteps);
        double bc2 = 1.0 - Math.Pow(beta2, _auxAdamSteps);
        double stepSize = _auxLr / bc1;
        double invSqrtBc2 = 1.0 / Math.Sqrt(bc2);

        T beta1T = Ops.FromDouble(beta1);
        T oneMinusBeta1T = Ops.FromDouble(1.0 - beta1);
        T beta2T = Ops.FromDouble(beta2);
        T oneMinusBeta2T = Ops.FromDouble(1.0 - beta2);
        T invSqrtBc2T = Ops.FromDouble(invSqrtBc2);
        T epsT = Ops.FromDouble(eps);
        T stepSizeT = Ops.FromDouble(stepSize);
        T wdScaleT = Ops.FromDouble(_auxLr * wd);

        for (int i = 0; i < _auxParams.Length; i++)
        {
            var p = _auxParams[i];
            if (!gradients.TryGetValue(p, out var g))
                continue;
            using var stepTmpScope = AiDotNet.Tensors.Engines.TensorScope.Begin();
            var mNew = AiDotNet.Tensors.Engines.TensorScope.Detach(engine.TensorAdd(
                engine.TensorMultiplyScalar(_auxM[i], beta1T),
                engine.TensorMultiplyScalar(g, oneMinusBeta1T)));
            var vNew = AiDotNet.Tensors.Engines.TensorScope.Detach(engine.TensorAdd(
                engine.TensorMultiplyScalar(_auxV2[i], beta2T),
                engine.TensorMultiplyScalar(engine.TensorMultiply(g, g), oneMinusBeta2T)));
            _auxM[i].Dispose();
            _auxV2[i].Dispose();
            _auxM[i] = mNew;
            _auxV2[i] = vNew;

            var denom = engine.TensorAddScalar(
                engine.TensorMultiplyScalar(engine.TensorSqrt(_auxV2[i]), invSqrtBc2T),
                epsT);
            var adamStep = engine.TensorMultiplyScalar(engine.TensorDivide(_auxM[i], denom), stepSizeT);
            var wdStep = engine.TensorMultiplyScalar(p, wdScaleT);
            var totalUpdate = engine.TensorAdd(adamStep, wdStep);
            engine.TensorSubtractInPlace(p, totalUpdate);
        }
    }
}
