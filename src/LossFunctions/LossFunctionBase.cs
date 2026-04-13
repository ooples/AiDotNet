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
    /// Calculates the derivative (gradient) of the loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivatives of the loss with respect to each prediction.</returns>
    public abstract Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual);

    /// <summary>
    /// Computes the loss as a scalar tensor using tape-differentiable engine operations.
    /// Gradients flow through this computation automatically via the gradient tape.
    /// </summary>
    /// <param name="predicted">The predicted tensor from the forward pass.</param>
    /// <param name="target">The target tensor.</param>
    /// <returns>A scalar tensor containing the loss value.</returns>
    public abstract Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target);

    /// <summary>
    /// Calculates both loss and gradient on GPU in a single pass.
    /// Default implementation falls back to separate calls.
    /// </summary>
    /// <param name="predicted">The predicted GPU tensor from the model.</param>
    /// <param name="actual">The actual (target) GPU tensor.</param>
    /// <returns>A tuple containing the loss value and gradient tensor.</returns>
    public virtual (T Loss, Tensor<T> Gradient) CalculateLossAndGradientGpu(Tensor<T> predicted, Tensor<T> actual)
    {
        // Default: fall back to CPU
        var predictedCpu = predicted;
        var actualCpu = actual;

        var loss = CalculateLoss(predictedCpu.ToVector(), actualCpu.ToVector());
        var gradientCpu = CalculateDerivative(predictedCpu.ToVector(), actualCpu.ToVector());

        var gradientTensor = new Tensor<T>(predictedCpu._shape);
        Array.Copy(gradientCpu.ToArray(), gradientTensor.Data.ToArray(), gradientCpu.Length);

        var engine = AiDotNetEngine.Current as DirectGpuTensorEngine;
        var backend = engine?.GetBackend() ?? throw new InvalidOperationException("GPU backend not available");
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

        var oneHot = new Tensor<T>(predicted.Shape.ToArray());
        int batchElements = target.Length;

        for (int i = 0; i < batchElements; i++)
        {
            double rawVal = NumOps.ToDouble(target[i]);
            int classIdx = (int)rawVal;
            if (rawVal != classIdx)
            {
                throw new ArgumentException(
                    $"Target value {rawVal} at position {i} is not an integer class index.");
            }
            if (classIdx < 0 || classIdx >= numClasses)
            {
                throw new ArgumentOutOfRangeException(nameof(target),
                    $"Class index {classIdx} at position {i} is out of range [0, {numClasses}). " +
                    "Target values must be valid class indices.");
            }
            oneHot[i * numClasses + classIdx] = NumOps.One;
        }

        return oneHot;
    }
}
