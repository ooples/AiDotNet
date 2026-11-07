using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Interfaces;

/// <summary>
/// A credit rule that trains every hidden layer from a <b>local objective</b> — a scalar loss defined on that layer's
/// own output — instead of a teaching signal routed back from the network output. Gradients never cross a layer
/// boundary below the shared readout: the input to each trainable layer is detached (stop-gradient), so a layer's
/// parameters are moved only by its own local objective. The trailing <see cref="ReadoutLayerCount"/> layers form a
/// shared readout that is trained exactly by the network's output loss.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// This is the form of local credit that applies to <b>sequence models</b> (language models, any network whose layer
/// outputs are <c>[batch, sequence, features]</c>). The classic <see cref="ICreditRule{T}"/> rules build per-layer
/// feedback matrices over the flattened feature size and a dense one-hot target, which at vocabulary 49,152 and
/// sequence 256 is not representable. A local-objective rule instead states each layer's loss directly (for example,
/// next-token cross-entropy through the shared readout, or a regression onto a teacher model's hidden state), and the
/// trainer adds all of them into one objective that is back-propagated once.
/// </para>
/// <para>
/// <b>For Beginners:</b> normal training sends one error signal from the very end of the network all the way back
/// through every layer. Here each layer gets its own small exam ("predict the next token from what you have so far",
/// or "match the teacher's layer") and studies only for that exam. Layers never pass blame to each other, so very deep
/// stacks do not suffer from signals fading or exploding on the way down.
/// </para>
/// </remarks>
public interface ILocalObjectiveCreditRule<T> : ICreditRule<T>
{
    /// <summary>
    /// Number of trailing layers of the network that form the shared readout (for a language model: final norm +
    /// output head = 2). These layers are trained exactly by the output loss and are the <c>readout</c> function
    /// handed to <see cref="BuildLocalObjective"/>. Must be at least 1.
    /// </summary>
    int ReadoutLayerCount { get; }

    /// <summary>
    /// Parameters owned by the rule itself (for example per-layer projections onto a teacher's hidden width). They are
    /// added as gradient sources of the combined objective and their gradients are handed back through
    /// <see cref="ApplyAuxiliaryGradients"/>; the network's optimizer never sees them. Empty when the rule owns none.
    /// </summary>
    IReadOnlyList<Tensor<T>> AuxiliaryParameters { get; }

    /// <summary>
    /// Builds the scalar, tape-recorded local objective for one hidden layer, or returns <c>null</c> to give that
    /// layer no local objective (its parameters then receive no gradient this step).
    /// </summary>
    /// <param name="context">The hidden layer's output node and the shared step state.</param>
    Tensor<T>? BuildLocalObjective(LocalObjectiveContext<T> context);

    /// <summary>
    /// Builds the scalar, tape-recorded output objective on the readout's prediction. Return <c>null</c> to use the
    /// network's own loss unchanged.
    /// </summary>
    Tensor<T>? BuildOutputObjective(LocalObjectiveStep<T> step, Tensor<T> prediction, Tensor<T> defaultLoss);

    /// <summary>Called once per step with the gradients of <see cref="AuxiliaryParameters"/> (missing = zero).</summary>
    void ApplyAuxiliaryGradients(IReadOnlyDictionary<Tensor<T>, Tensor<T>> gradients);
}

/// <summary>State shared by every local objective of one training step.</summary>
/// <typeparam name="T">The numeric data type.</typeparam>
public sealed class LocalObjectiveStep<T>
{
    internal LocalObjectiveStep(Tensor<T> input, Tensor<T> target, IEngine engine,
        Func<Tensor<T>, Tensor<T>> readout, Func<Tensor<T>, Tensor<T>, Tensor<T>> loss, int hiddenLayerCount,
        Func<Tensor<T>, Tensor<T>, Tensor<T>>? readoutLoss = null)
    {
        Input = input; Target = target; Engine = engine; Readout = readout; Loss = loss; HiddenLayerCount = hiddenLayerCount;
        ReadoutLoss = readoutLoss ?? ((h, t) => loss(readout(h), t));
    }

    /// <summary>
    /// The scalar task loss of a hidden state through the shared readout, <c>Loss(Readout(h), target)</c>. When the
    /// readout is a single layer implementing <see cref="IFusedReadoutLossLayer{T}"/> (a language-model head fused with
    /// its cross-entropy) the readout output is never materialised on the tape; otherwise this is exactly
    /// <c>Loss(Readout(h), target)</c>. Local-objective rules should prefer it over composing the two.
    /// </summary>
    public Func<Tensor<T>, Tensor<T>, Tensor<T>> ReadoutLoss { get; }

    /// <summary>The step's network input (for a language model: token ids). Not tape-tracked.</summary>
    public Tensor<T> Input { get; }

    /// <summary>The step's target, aligned the same way the network's own training step aligns it.</summary>
    public Tensor<T> Target { get; }

    /// <summary>The engine recording the step's tape. Every op of a local objective must go through it.</summary>
    public IEngine Engine { get; }

    /// <summary>Runs the shared readout (the trailing <see cref="ILocalObjectiveCreditRule{T}.ReadoutLayerCount"/> layers) on a hidden state.</summary>
    public Func<Tensor<T>, Tensor<T>> Readout { get; }

    /// <summary>The network's tape loss <c>(prediction, target) → scalar</c>.</summary>
    public Func<Tensor<T>, Tensor<T>, Tensor<T>> Loss { get; }

    /// <summary>Number of hidden (pre-readout) trainable layers that are offered a local objective this step.</summary>
    public int HiddenLayerCount { get; }
}

/// <summary>One hidden layer's view for <see cref="ILocalObjectiveCreditRule{T}.BuildLocalObjective"/>.</summary>
/// <typeparam name="T">The numeric data type.</typeparam>
public sealed class LocalObjectiveContext<T>
{
    internal LocalObjectiveContext(LocalObjectiveStep<T> step, int hiddenIndex, int networkLayerIndex, ILayer<T> layer, Tensor<T> output)
    {
        Step = step; HiddenIndex = hiddenIndex; NetworkLayerIndex = networkLayerIndex; Layer = layer; Output = output;
    }

    /// <summary>The step state (input, target, engine, readout, loss).</summary>
    public LocalObjectiveStep<T> Step { get; }

    /// <summary>Position of this layer among the hidden trainable layers, 0-based from the input side.</summary>
    public int HiddenIndex { get; }

    /// <summary>Index of this layer in the network's <c>Layers</c> list.</summary>
    public int NetworkLayerIndex { get; }

    /// <summary>The layer itself.</summary>
    public ILayer<T> Layer { get; }

    /// <summary>The layer's tape-recorded output, a function of this layer's parameters only (its input is detached).</summary>
    public Tensor<T> Output { get; }
}
