using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// <b>Tied-readout local error</b>: a sequence-model form of Local Error Signals (Nøkland &amp; Eidnes, 2019) in which
/// every hidden layer's local classifier is the network's <b>own shared readout</b> (for a language model: the final
/// norm and the tied output head) rather than a per-layer learned matrix. Each hidden layer <c>k</c> is trained by
/// <c>λ · loss(readout(h_k), target)</c> with its input detached, so no gradient ever crosses a layer boundary below the
/// readout; the readout itself is trained by the output loss plus every local loss routed through it.
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
/// The last hidden layer feeds the readout directly and is already trained exactly by the output loss; it is given no
/// separate local objective (that would double-count it).
/// </para>
/// <para>
/// <b>For Beginners:</b> every layer is handed the model's own "answer sheet" (the output head) and is graded on how
/// well its output alone predicts the next word. No layer waits for blame to trickle down from the top.
/// </para>
/// </remarks>
public sealed class TiedReadoutLocalErrorCreditRule<T> : ILocalObjectiveCreditRule<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly double _localWeight;

    /// <summary>Creates the rule.</summary>
    /// <param name="readoutLayerCount">Trailing layers forming the shared readout (language model: norm + head = 2).</param>
    /// <param name="localWeight">Weight λ of each hidden layer's local loss (default 1).</param>
    public TiedReadoutLocalErrorCreditRule(int readoutLayerCount = 2, double localWeight = 1.0)
    {
        if (readoutLayerCount < 1) throw new ArgumentOutOfRangeException(nameof(readoutLayerCount), "Must be at least 1.");
        if (double.IsNaN(localWeight) || double.IsInfinity(localWeight) || localWeight <= 0) throw new ArgumentOutOfRangeException(nameof(localWeight), "Must be positive and finite.");
        ReadoutLayerCount = readoutLayerCount;
        _localWeight = localWeight;
    }

    /// <inheritdoc />
    public string Name => "TiedReadoutLocalError";

    /// <inheritdoc />
    public bool IsExactBackprop => false;

    /// <inheritdoc />
    public int ReadoutLayerCount { get; }

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
        if (context.HiddenIndex == context.Step.HiddenLayerCount - 1) return null; // trained exactly by the output loss
        var s = context.Step;
        var local = s.Loss(s.Readout(context.Output), s.Target);
        // Skip scaling only for the exact identity coefficient.
        return _localWeight == 1.0 ? local : s.Engine.TensorMultiplyScalar(local, Ops.FromDouble(_localWeight));
    }

    /// <inheritdoc />
    public Tensor<T>? BuildOutputObjective(LocalObjectiveStep<T> step, Tensor<T> prediction, Tensor<T> defaultLoss) => null;

    /// <inheritdoc />
    public void ApplyAuxiliaryGradients(IReadOnlyDictionary<Tensor<T>, Tensor<T>> gradients) { }
}
