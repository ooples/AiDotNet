using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Helpers;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Base class for loss function implementations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
public abstract class LossFunctionBase<T> : ILossFunction<T>
{
    /// <summary>
    /// Provides operations for the numeric type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Initializes a new instance of the LossFunctionBase class.
    /// </summary>
    protected LossFunctionBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The loss value.</returns>
    public abstract T CalculateLoss(Vector<T> predicted, Vector<T> actual);

    /// <summary>
    /// Computes the loss as a scalar tensor using tape-differentiable engine operations.
    /// Gradients flow through this computation automatically via the gradient tape.
    /// </summary>
    /// <param name="predicted">The predicted tensor from the forward pass.</param>
    /// <param name="target">The target tensor.</param>
    /// <returns>A rank-0 scalar tensor containing the loss value.</returns>
    /// <remarks>
    /// Implementations define their math here and nowhere else. There is no hand-written
    /// derivative to keep in step with it, because the tape differentiates this method.
    /// </remarks>
    public abstract Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target);

    /// <summary>
    /// Whether this loss keeps a GPU result from one <see cref="ComputeTapeLoss"/> call and reads it in a later one (a
    /// basis or scale built on the device by its first call). A <see cref="AiDotNet.Tensors.Engines.TensorScope"/>
    /// around the call frees every GPU result it creates, so a caller scopes it only when this is false. The library's
    /// own losses keep none (a table they reuse is built on the host); a loss defined elsewhere, a derived one included,
    /// has not been checked, and a library loss that runs a caller's network, layer, callback or nested loss overrides
    /// this to answer for them.
    /// </summary>
    internal virtual bool KeepsResultsBetweenCalls => GetType().Assembly != typeof(LossFunctionBase<T>).Assembly;

    /// <summary>
    /// Whether <see cref="ComputeTapeLoss"/> reads the target's values on the host (decoding encoded labels into index
    /// arrays, say) rather than through engine ops. A compiled training plan replays only engine ops, so it would keep
    /// the target decoded when it was traced for the whole run; a model training with such a loss takes the eager step.
    /// </summary>
    internal virtual bool ReadsTargetOnHost => false;

    /// <summary>
    /// Calculates both loss and gradient on GPU in a single pass.
    /// Default implementation differentiates <see cref="ComputeTapeLoss"/> on the CPU and uploads
    /// the result; override to compute both in a single GPU kernel invocation.
    /// </summary>
    /// <param name="predicted">The predicted GPU tensor from the model.</param>
    /// <param name="actual">The actual (target) GPU tensor.</param>
    /// <returns>A tuple containing the loss value and gradient tensor.</returns>
    /// <remarks>
    /// <para>
    /// The loss value and the gradient are read from a <b>single</b> tape recording rather than from
    /// two independent computations. Previously this method called the forward and a separately
    /// hand-written derivative, so the pair could disagree with no way to detect it; taking both from
    /// one differentiated forward makes that class of mismatch unrepresentable.
    /// </para>
    /// <para>
    /// The tape is scoped to this call because a GPU caller supplies already-materialized tensors and
    /// has no surrounding tape to record onto. Callers that do own a training tape should record
    /// <see cref="ComputeTapeLoss"/> on it directly instead of coming through here.
    /// </para>
    /// </remarks>
    public virtual (T Loss, Tensor<T> Gradient) CalculateLossAndGradientGpu(Tensor<T> predicted, Tensor<T> actual)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (actual is null) throw new ArgumentNullException(nameof(actual));

        var engine = AiDotNetEngine.Current as DirectGpuTensorEngine;
        var backend = engine?.GetBackend() ?? throw new InvalidOperationException("GPU backend not available");

        var (loss, gradientTensor) = LossFunctionExtensions.ComputeLossAndGradient<T>(this, predicted, actual);
        var gradientGpu = GpuTensorHelper.UploadToGpu<T>(backend, gradientTensor, GpuTensorRole.Gradient);

        return (loss, gradientGpu);
    }

    /// <summary>
    /// Validates that the predicted and actual vectors have the same length.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual values vector.</param>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    protected void ValidateVectorLengths(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
        {
            throw new ArgumentException("Predicted and actual vectors must have the same length.");
        }
    }

    /// <summary>
    /// When the target has fewer dimensions than the prediction (e.g., integer
    /// class indices <c>[B, S]</c> vs logits <c>[B, S, V]</c>), auto-converts
    /// to one-hot encoding so pointwise loss operations (multiply, subtract)
    /// work without shape mismatch. Returns the target unchanged when shapes
    /// already match. Fixes #1114 — affects all classification loss functions.
    /// </summary>
    /// <remarks>
    /// <para>Only use this in classification-specific loss functions (cross-entropy,
    /// focal, sparse categorical). Do NOT use for contrastive, Wasserstein, or
    /// other non-classification losses where targets have different semantics.</para>
    /// <para>Validates that target shape is a prefix of predicted shape (all dims
    /// except the last class dimension must match). Throws on invalid class indices.</para>
    /// </remarks>
    protected Tensor<T> EnsureTargetMatchesPredicted(Tensor<T> predicted, Tensor<T> target)
    {
        if (target.Shape.Length >= predicted.Shape.Length)
            return target;

        int numClasses = predicted.Shape[predicted.Shape.Length - 1];

        // Validate shape prefix: target shape must match predicted shape
        // for all dimensions except the final class dimension.
        // e.g., predicted [B, S, V] requires target [B, S]
        if (target.Shape.Length != predicted.Shape.Length - 1)
        {
            throw new ArgumentException(
                $"Target rank ({target.Shape.Length}) must be exactly one less than predicted rank " +
                $"({predicted.Shape.Length}) for one-hot encoding. " +
                $"Target shape: [{string.Join(", ", target.Shape.ToArray())}], " +
                $"Predicted shape: [{string.Join(", ", predicted.Shape.ToArray())}].");
        }

        for (int d = 0; d < target.Shape.Length; d++)
        {
            if (target.Shape[d] != predicted.Shape[d])
            {
                throw new ArgumentException(
                    $"Target shape dimension {d} ({target.Shape[d]}) does not match " +
                    $"predicted shape dimension {d} ({predicted.Shape[d]}). " +
                    $"Target shape: [{string.Join(", ", target.Shape.ToArray())}], " +
                    $"Predicted shape: [{string.Join(", ", predicted.Shape.ToArray())}].");
            }
        }

        // Validate every target value at once: it must be an integer class index in [0, numClasses).
        // A value is not an integer class index when it has a fractional part (NaN included) or lies
        // outside the int range, where the old (int) cast could not represent it.
        var flatTarget = Engine.Reshape(target, new[] { target.Length });
        var notInteger = Engine.TensorMax(
            Engine.TensorNotEquals(flatTarget, Engine.TensorTrunc(flatTarget)),
            Engine.TensorGreaterThan(Engine.TensorAbs(flatTarget), NumOps.FromDouble(int.MaxValue)));
        var outOfRange = Engine.TensorMax(
            Engine.TensorLessThan(flatTarget, NumOps.Zero),
            Engine.TensorGreaterThan(flatTarget, NumOps.FromDouble(numClasses - 1)));
        var invalid = Engine.TensorMax(notInteger, outOfRange);
        if (target.Length > 0 && NumOps.GreaterThan(Engine.TensorMaxValue(invalid), NumOps.Zero))
        {
            // First offending position, reported exactly as the per-element check did.
            int i = Engine.TensorArgMax(invalid, 0).GetFlat(0);
            T rawVal = flatTarget.GetFlat(i);
            if (NumOps.GreaterThan(notInteger.GetFlat(i), NumOps.Zero))
            {
                throw new ArgumentException(
                    $"Target value {rawVal} at position {i} is not an integer class index.");
            }

            // An integral value within the int range: the class index is exact.
            int classIdx = NumOps.ToInt32(rawVal);
            throw new ArgumentOutOfRangeException(nameof(target),
                $"Class index {classIdx} at position {i} is out of range [0, {numClasses}). " +
                "Target values must be valid class indices.");
        }

        // One-hot along the last axis: compare each class index against 0..numClasses-1.
        var oneHotShape = predicted.Shape.ToArray();
        var columnShape = (int[])oneHotShape.Clone();
        columnShape[columnShape.Length - 1] = 1;
        var classIds = Engine.TensorArange<T>(0, numClasses, 1);
        return Engine.TensorEquals(
            Engine.TensorBroadcastTo(Engine.Reshape(target, columnShape), oneHotShape),
            Engine.TensorBroadcastTo(classIds, oneHotShape));
    }

    /// <summary>
    /// Element-wise <c>NumericalStabilityHelper.SafeLog</c>: log(max(x, epsilon)).
    /// </summary>
    private protected Tensor<T> SafeLogTensor(Tensor<T> values, double epsilon = NumericalStabilityHelper.DefaultEpsilon)
    {
        return Engine.TensorLog(Engine.TensorClampMin(values, NumOps.FromDouble(epsilon)));
    }

    /// <summary>
    /// Element-wise <c>NumericalStabilityHelper.SafeDiv</c>: a denominator whose
    /// magnitude is below epsilon is replaced by epsilon carrying the denominator's sign (+epsilon for 0).
    /// </summary>
    private protected Tensor<T> SafeDivideTensor(
        Tensor<T> numerator, Tensor<T> denominator, double epsilon = NumericalStabilityHelper.DefaultEpsilon)
    {
        T eps = NumOps.FromDouble(epsilon);
        var signedEps = Engine.TensorWhere(
            Engine.TensorLessThan(denominator, NumOps.Zero),
            Engine.TensorFullLike(denominator, NumOps.Negate(eps)),
            Engine.TensorFullLike(denominator, eps));
        var safeDenominator = Engine.TensorWhere(
            Engine.TensorLessThan(Engine.TensorAbs(denominator), eps), signedEps, denominator);
        return Engine.TensorDivide(numerator, safeDenominator);
    }
}
