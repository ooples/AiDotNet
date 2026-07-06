using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// Shared implementation for the credit rules that propagate the error <i>sequentially</i> layer-by-layer
/// (Back-prop, Feedback Alignment, Sign-Symmetric). They differ only in <b>which matrix</b> carries the error
/// from a layer back to the previous layer (<see cref="FeedbackMatrix"/>); the rest of the backward chain is
/// identical.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
internal abstract class SequentialFeedbackRule<T> : CreditRuleBase<T>
{
    /// <summary>
    /// The feedback matrix (shape <c>[outputDim, inputDim]</c>, same orientation as the layer weights) used to
    /// carry <c>delta</c> back from <paramref name="layer"/> to its input. Back-prop returns the true weights;
    /// Feedback Alignment returns a fixed random matrix; Sign-Symmetric returns the element-wise sign of the
    /// weights.
    /// </summary>
    protected abstract Matrix<T> FeedbackMatrix(ICreditLayer<T> layer, ICreditAssignmentContext<T> context);

    /// <inheritdoc />
    public override void ComputeUpdates(ICreditAssignmentContext<T> context)
    {
        var ops = context.NumOps;
        var layers = context.Layers;
        int last = layers.Count - 1;

        // delta at the output pre-activation = ∂L/∂z_last = (prediction − target).
        Matrix<T> delta = context.OutputError;

        for (int i = last; i >= 0; i--)
        {
            var layer = layers[i];
            SetParameterGradients(layer, delta, ops);

            if (i > 0)
            {
                // Route the error back to the previous layer through this layer's feedback matrix,
                // then gate it by the previous layer's activation derivative.
                var feedback = FeedbackMatrix(layer, context);   // [out_i, in_i]
                var preDelta = delta.Multiply(feedback);          // [B, out_i] · [out_i, in_i] = [B, in_i]
                var prevDeriv = layers[i - 1].ActivationDerivative();
                delta = Hadamard(preDelta, prevDeriv, ops);
            }
        }
    }
}

/// <summary>
/// Standard reverse-mode back-propagation expressed through the credit-rule interface: the feedback matrix is
/// the layer's own (transpose) weights. Used as the reference the alternative rules are validated against.
/// </summary>
internal sealed class BackpropCreditRule<T> : SequentialFeedbackRule<T>
{
    public override string Name => "Backprop";

    protected override Matrix<T> FeedbackMatrix(ICreditLayer<T> layer, ICreditAssignmentContext<T> context)
        => layer.Weights;
}

/// <summary>
/// <b>Feedback Alignment</b> (Lillicrap et al., 2016): each layer's transpose-weight feedback is replaced by a
/// <i>fixed random</i> matrix of the same shape, allocated once in <see cref="Initialize"/>.
/// </summary>
internal sealed class FeedbackAlignmentCreditRule<T> : SequentialFeedbackRule<T>
{
    private Matrix<T>[]? _feedback;
    private int[]? _shapeSignature;

    public override string Name => "FeedbackAlignment";

    public override void Initialize(ICreditAssignmentContext<T> context)
    {
        if (IsInitializedFor(context)) return;

        var layers = context.Layers;
        _feedback = new Matrix<T>[layers.Count];
        _shapeSignature = new int[layers.Count * 2];
        for (int i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];
            // Same shape as the weights [out, in]; scale by fan-in for stability.
            _feedback[i] = RandomGaussian(layer.OutputDim, layer.InputDim, layer.InputDim, context.Random, context.NumOps);
            _shapeSignature[i * 2] = layer.OutputDim;
            _shapeSignature[i * 2 + 1] = layer.InputDim;
        }
    }

    protected override Matrix<T> FeedbackMatrix(ICreditLayer<T> layer, ICreditAssignmentContext<T> context)
    {
        if (_feedback is null) Initialize(context);
        return _feedback![layer.Index];
    }

    private bool IsInitializedFor(ICreditAssignmentContext<T> context)
    {
        if (_feedback is null || _shapeSignature is null) return false;
        var layers = context.Layers;
        if (_feedback.Length != layers.Count) return false;
        for (int i = 0; i < layers.Count; i++)
            if (_shapeSignature[i * 2] != layers[i].OutputDim || _shapeSignature[i * 2 + 1] != layers[i].InputDim)
                return false;
        return true;
    }
}

/// <summary>
/// <b>Sign-Symmetric feedback</b> (Liao et al., 2016): the error is routed back through the element-wise
/// <i>sign</i> of the transpose weights (magnitude discarded). Unlike Feedback Alignment the feedback tracks
/// the live weight signs each step, so there is no fixed random state.
/// </summary>
internal sealed class SignSymmetricCreditRule<T> : SequentialFeedbackRule<T>
{
    public override string Name => "SignSymmetric";

    protected override Matrix<T> FeedbackMatrix(ICreditLayer<T> layer, ICreditAssignmentContext<T> context)
    {
        var ops = context.NumOps;
        var w = layer.Weights;
        var sign = new Matrix<T>(w.Rows, w.Columns);
        for (int i = 0; i < w.Rows; i++)
            for (int j = 0; j < w.Columns; j++)
                sign[i, j] = ops.SignOrZero(w[i, j]);
        return sign;
    }
}
