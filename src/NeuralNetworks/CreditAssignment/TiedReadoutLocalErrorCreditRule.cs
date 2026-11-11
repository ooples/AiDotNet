using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// <b>Tied-readout local error</b>: a sequence-model form of Local Error Signals (Nøkland &amp; Eidnes, 2019) in which
/// every hidden layer's local classifier is the network's <b>own shared readout</b> (for a language model: the final
/// norm and the tied output head) rather than a per-layer learned matrix. Each hidden layer <c>k</c> is trained by
/// <c>λ · loss(readout(h_k), target)</c> with its input detached, so no gradient ever crosses a layer boundary below the
/// readout.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// <b>Why a shared readout.</b> The classic rule learns a <c>[features, classes]</c> classifier per layer over the
/// flattened layer output, which for a language model is <c>[sequence·width, vocabulary]</c> per layer — not
/// representable at real vocabularies. Re-using the readout costs zero extra parameters, so the model's parameter
/// count is unchanged, and every layer is asked the same question the network is: predict the next token.
/// </para>
/// <para>
/// <b>Readout detachment, stride, and depth-progressive weighting.</b> Setting <see cref="DetachReadoutOnHiddenLayers"/>
/// to <c>true</c> evaluates <see cref="LocalObjectiveStep{T}.ReadoutLossDetached"/> on intermediate hidden layers so
/// only the terminal layer trains the shared readout parameters (preventing <c>L − 1</c> shallow representations from
/// pulling the terminal readout away from final-layer geometry). Setting <see cref="LayerStride"/> &gt; 1 evaluates
/// local readout supervision every <c>LayerStride</c> blocks anchored below the terminal block, cutting local readout
/// compute linearly.
/// </para>
/// <para>
/// The last hidden layer feeds the readout directly and is already trained exactly by the output loss; it is given no
/// separate local objective (that would double-count it).
/// </para>
/// <para>
/// <b>For Beginners:</b> every layer is handed the model's own "answer sheet" (the output head) and is graded on how
/// well its output alone predicts the next word. No layer waits for blame to trickle down from the top.
/// </para>
/// </remarks>
public sealed class TiedReadoutLocalErrorCreditRule<T>
    : ILocalObjectiveCreditRule<T>, IOutputPredictionOptionalRule, IBatchedLocalReadoutRule
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly double _localWeight;

    /// <summary>Creates the rule with explicit readout-detachment, layer-stride, and depth-schedule options.</summary>
    /// <param name="readoutLayerCount">Trailing layers forming the shared readout (language model: norm + head = 2).</param>
    /// <param name="localWeight">Base weight λ of each hidden layer's local loss (default 1).</param>
    /// <param name="detachReadoutOnHiddenLayers">When <c>true</c>, intermediate hidden layers detach the shared readout's own trainable parameters so only the terminal output objective updates the readout.</param>
    /// <param name="layerStride">Stride across hidden layers receiving local readout losses (1 = every hidden layer; 2 = every second layer).</param>
    /// <param name="useDepthProgressiveWeight">When <c>true</c>, scales hidden layer <c>k</c>'s weight by <c>((k + 1) / H)²</c> so early feature-forming layers are not over-constrained to directly emit terminal logits.</param>
    public TiedReadoutLocalErrorCreditRule(
        int readoutLayerCount = 2,
        double localWeight = 1.0,
        bool detachReadoutOnHiddenLayers = false,
        int layerStride = 1,
        bool useDepthProgressiveWeight = false)
    {
        if (readoutLayerCount < 1) throw new ArgumentOutOfRangeException(nameof(readoutLayerCount), "Must be at least 1.");
        if (double.IsNaN(localWeight) || double.IsInfinity(localWeight) || localWeight <= 0) throw new ArgumentOutOfRangeException(nameof(localWeight), "Must be positive and finite.");
        if (layerStride < 1) throw new ArgumentOutOfRangeException(nameof(layerStride), "Must be at least 1.");
        ReadoutLayerCount = readoutLayerCount;
        _localWeight = localWeight;
        DetachReadoutOnHiddenLayers = detachReadoutOnHiddenLayers;
        LayerStride = layerStride;
        UseDepthProgressiveWeight = useDepthProgressiveWeight;
    }

    /// <inheritdoc />
    public string Name => "TiedReadoutLocalError";

    /// <inheritdoc />
    public bool IsExactBackprop => false;

    /// <inheritdoc />
    public int ReadoutLayerCount { get; }

    /// <summary>Gets whether intermediate hidden layers detach the shared readout's own parameters.</summary>
    public bool DetachReadoutOnHiddenLayers { get; }

    /// <summary>Gets the stride between hidden layers receiving local readout objectives.</summary>
    public int LayerStride { get; }

    /// <summary>Gets whether local loss weights scale quadratically with normalized layer depth.</summary>
    public bool UseDepthProgressiveWeight { get; }

    bool IOutputPredictionOptionalRule.UsesOutputPrediction => false;

    BatchedLocalReadoutMode IBatchedLocalReadoutRule.BatchedReadoutMode => BatchedLocalReadoutMode.FullVocabulary;
    double IBatchedLocalReadoutRule.BaseLocalWeight => _localWeight;
    int IBatchedLocalReadoutRule.SampledMaxCandidates => int.MaxValue;
    int IBatchedLocalReadoutRule.AdvanceAndGetStepSeed(object stepToken) => 0;

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
        if (context.HiddenIndex == lastHidden) return null; // trained exactly by the output loss
        if (LayerStride > 1 && ((lastHidden - 1 - context.HiddenIndex) % LayerStride) != 0)
            return null;

        var s = context.Step;
        var local = DetachReadoutOnHiddenLayers
            ? s.ReadoutLossDetached(context.Output, s.Target)
            : s.ReadoutLoss(context.Output, s.Target);

        double weight = _localWeight;
        if (UseDepthProgressiveWeight)
        {
            double depthFrac = (context.HiddenIndex + 1.0) / s.HiddenLayerCount;
            weight *= depthFrac * depthFrac;
        }

        // Skip scaling only for the exact identity coefficient.
        return weight == 1.0 ? local : s.Engine.TensorMultiplyScalar(local, Ops.FromDouble(weight));
    }

    /// <inheritdoc />
    public Tensor<T>? BuildOutputObjective(LocalObjectiveStep<T> step, Tensor<T> prediction, Tensor<T> defaultLoss) => null;

    /// <inheritdoc />
    public void ApplyAuxiliaryGradients(IReadOnlyDictionary<Tensor<T>, Tensor<T>> gradients) { }
}
