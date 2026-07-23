using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// Shared implementation for credit rules that route the error <i>sequentially</i> layer-by-layer (Feedback
/// Alignment, Sign-Symmetric). Starting from the output error, each layer's teaching signal is obtained from the
/// next layer's teaching signal through a per-boundary feedback matrix (<see cref="FeedbackMatrix"/>). The engine
/// then supplies each layer's local activation Jacobian via a vector-Jacobian product.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
internal abstract class SequentialFeedbackRule<T> : CreditRuleBase<T>
{
    protected SequentialFeedbackRule(int? seed = null) : base(seed) { }

    /// <summary>
    /// The feedback matrix carrying the teaching signal from layer <c>index+1</c>'s output space
    /// (<c>M_{index+1}</c> features) to layer <c>index</c>'s output space (<c>M_index</c> features) — shape
    /// <c>[M_{index+1}, M_index]</c>. Feedback Alignment returns a fixed random matrix; Sign-Symmetric returns the
    /// element-wise sign of the next layer's weights.
    /// </summary>
    protected abstract Matrix<T> FeedbackMatrix(int index, ICreditAssignmentContext<T> context);

    public override void ComputeTeachingSignals(ICreditAssignmentContext<T> context)
    {
        var layers = context.Layers;
        int last = layers.Count - 1;

        // Teaching signal at the output layer = output error (the output layer itself is trained by the engine's
        // exact loss gradient; we only need this as the seed of the sequential recurrence).
        Matrix<T> current = ErrorMatrix(context); // [B, M_last]

        for (int i = last - 1; i >= 0; i--)
        {
            var feedback = FeedbackMatrix(i, context);         // [M_{i+1}, M_i]
            current = current.Multiply(feedback);              // [B, M_{i+1}] · [M_{i+1}, M_i] = [B, M_i]
            layers[i].TeachingSignal = ToTeachingSignal(current, layers[i].OutputShape);
        }
    }
}

/// <summary>
/// <b>Feedback Alignment</b> (Lillicrap et al., 2016): the error is routed sequentially through a <i>fixed random</i>
/// feedback matrix at each layer boundary instead of the transpose weights.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Feedback Alignment replaces each layer's exact backward matrix with a fixed random one and
/// passes the error back through those, one layer at a time. It still learns because the forward weights gradually
/// rotate to line up with the random feedback — the original demonstration that back-propagation's exact weights
/// are not strictly required.
/// </para>
/// </remarks>
internal sealed class FeedbackAlignmentCreditRule<T> : SequentialFeedbackRule<T>
{
    public FeedbackAlignmentCreditRule(int? seed = null) : base(seed) { }

    public override string Name => "FeedbackAlignment";

    protected override Matrix<T> FeedbackMatrix(int index, ICreditAssignmentContext<T> context)
    {
        // Boundary matrix i maps layer i+1's output space to layer i's output space: [M_{i+1}, M_i].
        var feedback = EnsureFeedback(context, (layers, i) =>
            i < layers.Count - 1 ? (layers[i + 1].FlatFeatureSize, layers[i].FlatFeatureSize) : ((int, int)?)null);
        return feedback[index]!;
    }
}

/// <summary>
/// <b>Sign-Symmetric feedback</b> (Liao et al., 2016 / Xiao et al., 2018): the error is routed back through the
/// element-wise <i>sign</i> of the next layer's transpose weights. This requires each trainable layer to expose a
/// single weight matrix (dense layers); it is not defined for attention/normalization blocks, which have no single
/// weight — use Direct Feedback Alignment for those architectures.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Sign-Symmetric keeps only the <i>sign</i> (+/−) of each real weight for the backward pass
/// and throws away the magnitude. Unlike the random rules its feedback tracks the live weight signs every step, so
/// it stays closer to true back-propagation while still avoiding exact weight transport.
/// </para>
/// </remarks>
internal sealed class SignSymmetricCreditRule<T> : SequentialFeedbackRule<T>
{
    public SignSymmetricCreditRule(int? seed = null) : base(seed) { }

    public override string Name => "SignSymmetric";

    protected override Matrix<T> FeedbackMatrix(int index, ICreditAssignmentContext<T> context)
    {
        var thisLayer = context.Layers[index];
        var nextLayer = context.Layers[index + 1];
        var weights = nextLayer.Weights;

        // Sign-Symmetric routes the error through sign(W_{index+1}), which only makes sense when each layer is a
        // single dense weight whose shape chains the consecutive feature sizes ([M_{index+1}, M_index]). Attention
        // / normalization blocks have no single weight (null) or a non-chaining shape — reject those clearly.
        if (weights is null || weights.Rows != nextLayer.FlatFeatureSize || weights.Columns != thisLayer.FlatFeatureSize)
        {
            throw new NotSupportedException(
                $"SignSymmetric requires every trainable layer to be a single dense weight matrix that chains the " +
                $"feature sizes; trainable layer {index + 1} is not (e.g. an attention/normalization block). Use " +
                $"CreditRule.DirectFeedbackAlignment for Transformer / attention architectures.");
        }

        // weights: [outNext, inNext] = [M_{index+1}, M_index]; sign gives the sign-symmetric feedback.
        return SignMatrix(weights, context.NumOps);
    }
}

/// <summary>
/// Standard reverse-mode back-propagation, exposed as a credit rule for API symmetry. Selecting it uses the
/// network's exact gradient (identical to the default path); it produces no teaching signals.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
internal sealed class BackpropCreditRule<T> : CreditRuleBase<T>
{
    public override string Name => "Backprop";
    public override bool IsExactBackprop => true;
    public override void ComputeTeachingSignals(ICreditAssignmentContext<T> context) { }
}
