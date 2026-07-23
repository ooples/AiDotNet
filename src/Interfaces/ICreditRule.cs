using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// A pluggable <b>credit-assignment</b> (a.k.a. "learning") rule that decides how the per-parameter updates are
/// produced during neural-network training. The default rule is standard back-propagation; alternative published
/// rules — Feedback Alignment, Direct Feedback Alignment, Sign-Symmetric, and any custom rule you implement —
/// replace <i>how the error signal is routed to each layer</i> while reusing the same forward pass, optimizer,
/// batching, scheduler and observability.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g. <see cref="float"/>, <see cref="double"/>).</typeparam>
/// <remarks>
/// <para>
/// <b>What a rule does.</b> Back-propagation sends the output error backward through the exact transpose of every
/// weight. A credit rule instead decides, for every trainable layer, a <i>teaching signal</i> at that layer's
/// output — the error the layer should be trained against. The training engine then converts each teaching signal
/// into that layer's parameter gradients via a <b>local</b> vector-Jacobian product (one backward step through
/// only that layer). Because the engine handles the local gradient mechanics, a rule can be applied to <i>any</i>
/// layer type — dense, multi-head attention, feed-forward, LayerNorm, embedding — which is exactly why Direct
/// Feedback Alignment scales to Transformers (Nøkland 2016; Launay et al. 2020).
/// </para>
/// <para>
/// <b>For Beginners:</b> imagine every layer needs to be told "here's how wrong your output was" so it can adjust.
/// Back-prop computes that message exactly by chaining backwards through all later layers. Direct Feedback
/// Alignment instead hands each layer a <i>fixed random</i> shortcut of the final error — and the network still
/// learns, because the forward weights rotate to align with the random feedback. This interface is where you
/// define how that per-layer "how wrong were you" message is produced.
/// </para>
/// <para>
/// <b>Contract.</b> Given an <see cref="ICreditAssignmentContext{T}"/> (the ordered trainable layers, their output
/// shapes, and the network output error), a rule fills in <see cref="ICreditLayer{T}.TeachingSignal"/> for every
/// <i>hidden</i> trainable layer. The final (output) layer is trained with the exact loss gradient by the engine,
/// so a rule may leave its teaching signal unset. A rule may hold internal state (e.g. fixed random feedback
/// matrices) across steps; allocate it in <see cref="Initialize"/>.
/// </para>
/// </remarks>
public interface ICreditRule<T>
{
    /// <summary>A short human-readable identifier for this rule (used in diagnostics), e.g. "DirectFeedbackAlignment".</summary>
    string Name { get; }

    /// <summary>
    /// True only for the reference back-propagation rule. When true, the training engine bypasses the
    /// teaching-signal machinery and uses the network's exact reverse-mode gradient (identical to the default
    /// path). All non-backprop rules return false.
    /// </summary>
    bool IsExactBackprop { get; }

    /// <summary>
    /// Called once, before the first update, so the rule can allocate any fixed internal state sized to the
    /// concrete network (for example the per-layer random feedback matrices used by Feedback Alignment and Direct
    /// Feedback Alignment). Implementations should be idempotent: a no-op if already initialized for a context
    /// whose layer shapes match. Rules with no persistent state may leave this empty.
    /// </summary>
    /// <param name="context">The credit-assignment context describing the network shape.</param>
    void Initialize(ICreditAssignmentContext<T> context);

    /// <summary>
    /// Produces the teaching signals for one training step by writing <see cref="ICreditLayer{T}.TeachingSignal"/>
    /// on every <i>hidden</i> trainable layer of <paramref name="context"/> (the output layer is handled by the
    /// engine's exact loss gradient). Each teaching signal must be a tensor matching that layer's output shape.
    /// Must not mutate the network weights.
    /// </summary>
    /// <param name="context">The per-layer output shapes and network output error for this step.</param>
    void ComputeTeachingSignals(ICreditAssignmentContext<T> context);
}

/// <summary>
/// The information an <see cref="ICreditRule{T}"/> receives for a single training step: the ordered trainable
/// layers (with output shapes and current weights) and the network output error signal.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
public interface ICreditAssignmentContext<T>
{
    /// <summary>The trainable layers in forward (input → output) order. The last entry is the output layer.</summary>
    IReadOnlyList<ICreditLayer<T>> Layers { get; }

    /// <summary>
    /// The output error signal <c>prediction − target</c>, shaped <c>[batchSize, outputFeatures]</c>. This is the
    /// gradient of the loss w.r.t. the output logits for the canonical matched setups (cross-entropy+softmax,
    /// MSE+linear) — the signal a credit rule routes back to each layer.
    /// </summary>
    Tensor<T> OutputError { get; }

    /// <summary>The number of samples in this batch.</summary>
    int BatchSize { get; }

    /// <summary>Numeric operations for the type <typeparamref name="T"/>.</summary>
    INumericOperations<T> NumOps { get; }

    /// <summary>A deterministic random source the rule may use to build fixed feedback matrices, for reproducibility.</summary>
    Random Random { get; }

    /// <summary>
    /// The training target aligned to the prediction shape <c>[batchSize, outputFeatures]</c> — a one-hot matrix
    /// for classification, or the raw regression target. Related to <see cref="OutputError"/> by
    /// <c>OutputError = prediction − Target</c>. Used by target-driven rules such as Direct Random Target
    /// Projection (DRTP), which project the target rather than the error.
    /// </summary>
    Tensor<T> Target { get; }
}

/// <summary>
/// Optional extension of <see cref="ICreditRule{T}"/> for rules whose feedback state is <b>learned</b> across
/// training steps (e.g. Kolen-Pollack and Direct Kolen-Pollack, whose feedback matrices are updated each step to
/// align with the forward weights). The training engine calls <see cref="OnParametersUpdated"/> once per gradient
/// computation, after the step's teaching signals and gradients have been produced, while the per-layer forward
/// activations are still available on the context. Rules with fixed (non-learned) feedback do not implement this
/// interface, so existing rules are entirely unaffected.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
public interface IFeedbackLearningRule<T> : ICreditRule<T>
{
    /// <summary>
    /// Updates the rule's learned feedback state for this training step, using the same context (forward
    /// activations, output error and teaching signals) that produced the step's gradients. Invoked once per
    /// gradient computation. Must not mutate the network's forward weights.
    /// </summary>
    /// <param name="context">The credit-assignment context for the just-computed step.</param>
    void OnParametersUpdated(ICreditAssignmentContext<T> context);
}

/// <summary>
/// A single trainable layer as seen by an <see cref="ICreditRule{T}"/>: its output shape, its weight matrix (when
/// it has a single one), and the teaching-signal slot the rule fills.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
public interface ICreditLayer<T>
{
    /// <summary>Position of this layer among the trainable layers in forward order (0 = first).</summary>
    int Index { get; }

    /// <summary>True if this is the final trainable layer (trained with the exact loss gradient by the engine).</summary>
    bool IsOutputLayer { get; }

    /// <summary>The shape of this layer's forward output, e.g. <c>[batch, features]</c> or <c>[batch, seq, dim]</c>.</summary>
    int[] OutputShape { get; }

    /// <summary>The number of output features per sample (product of <see cref="OutputShape"/> excluding the batch axis).</summary>
    int FlatFeatureSize { get; }

    /// <summary>The layer's forward output tensor for this step (available if a rule needs the activations; most rules use only shapes).</summary>
    Tensor<T> Output { get; }

    /// <summary>
    /// The tensor fed <i>into</i> this layer's forward pass for this step — the activation from the previous
    /// (sub-)layer, shaped like this layer's input. Feedback-learning rules such as Kolen-Pollack need it to form
    /// the outer-product update <c>Δfeedback ∝ teachingSignalᵀ · input</c> that aligns a learned feedback matrix to
    /// the forward weight. Fixed-feedback rules can ignore it.
    /// </summary>
    Tensor<T> Input { get; }

    /// <summary>
    /// The layer's weight matrix (shape <c>[outputDim, inputDim]</c>) if it exposes a single one (dense layers),
    /// else <c>null</c> (e.g. attention/normalization blocks). Used by weight-based rules such as Sign-Symmetric.
    /// </summary>
    Matrix<T>? Weights { get; }

    /// <summary>
    /// The teaching signal the rule produces for this layer — a tensor matching <see cref="OutputShape"/>. Set by
    /// <see cref="ICreditRule{T}.ComputeTeachingSignals"/> for hidden layers; left null for the output layer.
    /// </summary>
    Tensor<T>? TeachingSignal { get; set; }
}
