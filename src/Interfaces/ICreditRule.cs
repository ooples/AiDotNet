using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// A pluggable <b>credit-assignment</b> (a.k.a. "learning") rule that decides how the per-parameter
/// updates are produced during neural-network training. The default rule is standard back-propagation;
/// alternative published rules (Feedback Alignment, Direct Feedback Alignment, Sign-Symmetric, and any
/// custom rule you implement) replace <i>how the error signal is routed back to each layer</i> while
/// reusing the same forward pass, optimizer, batching, scheduler and observability.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g. <see cref="float"/>, <see cref="double"/>).</typeparam>
/// <remarks>
/// <para>
/// <b>What is "credit assignment"?</b> When a network makes a mistake, training has to decide how much
/// each individual weight was responsible ("to blame") for that mistake, so it knows how to change it.
/// Back-propagation solves this by sending the error backwards through the exact transpose of every
/// weight matrix. That is powerful but biologically implausible and requires a symmetric backward path.
/// </para>
/// <para>
/// <b>For Beginners:</b> think of a relay race where the final runner is told "we lost by 3 seconds".
/// Back-prop passes that blame backward through the <i>exact</i> path each runner ran. Feedback-alignment
/// rules instead pass the blame through a <i>fixed random</i> shortcut — and, surprisingly, the network
/// still learns because the forward weights gradually rotate to <i>align</i> with the random feedback.
/// This interface lets you swap in those alternative "blame-routing" schemes.
/// </para>
/// <para>
/// <b>Contract.</b> Implementations are given, per training step, an <see cref="ICreditAssignmentContext{T}"/>
/// containing the ordered trainable layers (with their current weights and cached forward activations) and
/// the output error signal. The rule must fill in <see cref="ICreditLayer{T}.WeightGradient"/> and
/// <see cref="ICreditLayer{T}.BiasGradient"/> for every layer. The training loop then flattens those into
/// the gradient vector the optimizer consumes — so the rule never touches the optimizer, batching or
/// scheduler. A rule may hold internal state (e.g. fixed random feedback matrices) across steps; use
/// <see cref="Initialize"/> to allocate it once against the concrete network shape.
/// </para>
/// <para>
/// <b>Scope.</b> The built-in feedback-alignment family targets dense feed-forward stacks (each trainable
/// layer is a <c>FullyConnectedLayer&lt;T&gt;</c> carrying its own element-wise activation, with a matched
/// output loss — MSE+linear or cross-entropy+softmax). Custom rules receive the same general context and
/// are free to implement any credit-assignment scheme over those activations and the output error.
/// </para>
/// </remarks>
public interface ICreditRule<T>
{
    /// <summary>
    /// A short human-readable identifier for this rule (used in diagnostics/observability), e.g.
    /// "DirectFeedbackAlignment".
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Called once, before the first update, so the rule can allocate any fixed internal state sized to
    /// the concrete network (for example the per-layer random feedback matrices used by Feedback Alignment
    /// and Direct Feedback Alignment). Implementations should be idempotent: if already initialized for a
    /// context whose layer shapes match, this should be a no-op. Rules with no persistent state (e.g.
    /// Sign-Symmetric, which reads the live weights each step) may leave this empty.
    /// </summary>
    /// <param name="context">The credit-assignment context describing the network shape.</param>
    void Initialize(ICreditAssignmentContext<T> context);

    /// <summary>
    /// Produces the per-parameter updates for one training step by writing
    /// <see cref="ICreditLayer{T}.WeightGradient"/> and <see cref="ICreditLayer{T}.BiasGradient"/> on every
    /// layer of <paramref name="context"/>. Must not mutate the layer weights or activations.
    /// </summary>
    /// <param name="context">
    /// The forward activations, current weights and output error for this step (see
    /// <see cref="ICreditAssignmentContext{T}"/>).
    /// </param>
    void ComputeUpdates(ICreditAssignmentContext<T> context);
}

/// <summary>
/// The information an <see cref="ICreditRule{T}"/> receives for a single training step: the ordered trainable
/// layers (with current weights + cached forward activations), the network input/prediction/target, and the
/// output error signal.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
public interface ICreditAssignmentContext<T>
{
    /// <summary>The trainable layers in forward (input → output) order.</summary>
    IReadOnlyList<ICreditLayer<T>> Layers { get; }

    /// <summary>The network input for this batch, shaped <c>[batchSize, inputFeatures]</c>.</summary>
    Matrix<T> Input { get; }

    /// <summary>
    /// The network prediction for this batch, shaped <c>[batchSize, outputFeatures]</c> (the activated
    /// output of the final layer).
    /// </summary>
    Matrix<T> Prediction { get; }

    /// <summary>The target for this batch, shaped <c>[batchSize, outputFeatures]</c> (one-hot expanded for classification).</summary>
    Matrix<T> Target { get; }

    /// <summary>
    /// The output error signal <c>∂L/∂z_last = (prediction − target)</c>, shaped <c>[batchSize, outputFeatures]</c>.
    /// This is the exact gradient w.r.t. the final pre-activation for the two canonical matched setups
    /// (MSE+linear output, cross-entropy+softmax output) — the signal a credit rule routes back to each layer.
    /// </summary>
    Matrix<T> OutputError { get; }

    /// <summary>The number of samples in this batch.</summary>
    int BatchSize { get; }

    /// <summary>Numeric operations for the type <typeparamref name="T"/>.</summary>
    INumericOperations<T> NumOps { get; }

    /// <summary>
    /// A deterministic random source the rule may use to build fixed feedback matrices, so runs are
    /// reproducible.
    /// </summary>
    Random Random { get; }
}

/// <summary>
/// A single trainable dense layer as seen by an <see cref="ICreditRule{T}"/>: its current weights/bias, the
/// forward activations cached for this step, its element-wise activation derivative, and the gradient slots
/// the rule must fill.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
public interface ICreditLayer<T>
{
    /// <summary>Position of this layer in the forward stack (0 = first).</summary>
    int Index { get; }

    /// <summary>True if this is the final layer (its output is the network prediction).</summary>
    bool IsOutputLayer { get; }

    /// <summary>Number of inputs to this layer.</summary>
    int InputDim { get; }

    /// <summary>Number of outputs (units) of this layer.</summary>
    int OutputDim { get; }

    /// <summary>The layer's current weight matrix, shaped <c>[outputDim, inputDim]</c> (output = input · Wᵀ + b).</summary>
    Matrix<T> Weights { get; }

    /// <summary>The activations flowing <i>into</i> this layer, shaped <c>[batchSize, inputDim]</c>.</summary>
    Matrix<T> Input { get; }

    /// <summary>The pre-activation <c>z = input · Wᵀ + b</c>, shaped <c>[batchSize, outputDim]</c>.</summary>
    Matrix<T> PreActivation { get; }

    /// <summary>The activated output <c>a = f(z)</c>, shaped <c>[batchSize, outputDim]</c>.</summary>
    Matrix<T> Output { get; }

    /// <summary>
    /// The element-wise activation derivative <c>f'(z)</c> evaluated at this layer's pre-activation, shaped
    /// <c>[batchSize, outputDim]</c>. Used to gate the feedback signal on hidden layers.
    /// </summary>
    Matrix<T> ActivationDerivative();

    /// <summary>
    /// The weight gradient the rule produces, shaped <c>[outputDim, inputDim]</c> (same orientation as
    /// <see cref="Weights"/>). Set by <see cref="ICreditRule{T}.ComputeUpdates"/>.
    /// </summary>
    Matrix<T> WeightGradient { get; set; }

    /// <summary>
    /// The bias gradient the rule produces, length <c>outputDim</c>. Set by
    /// <see cref="ICreditRule{T}.ComputeUpdates"/>.
    /// </summary>
    Vector<T> BiasGradient { get; set; }
}
