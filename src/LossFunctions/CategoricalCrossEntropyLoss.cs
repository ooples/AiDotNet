

using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Helpers;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Categorical Cross Entropy loss function for multi-class classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Categorical Cross Entropy is used for multi-class classification problems,
/// where you need to assign inputs to one of several categories (like classifying images as dog, cat, bird, etc.).
/// 
/// It measures how well the predicted probability distribution matches the actual distribution of classes.
/// 
/// The formula is: CCE = -(1/n) * ?[?(actual_j * log(predicted_j))]
/// 
/// Where:
/// - actual_j is usually a one-hot encoded vector (1 for the correct class, 0 for others)
/// - predicted_j is the predicted probability for each class (typically from a softmax output)
/// - The inner sum is over all classes, and the outer sum is over all samples
/// 
/// Key properties:
/// - Predicted values should be probabilities (between 0 and 1) that sum to 1 across classes
/// - It heavily penalizes confident incorrect predictions
/// - It's the standard loss function for multi-class neural network classifiers
/// - Often used together with the softmax activation function in the output layer
/// 
/// This loss function is ideal when your model needs to choose one option from multiple possibilities.
/// </para>
/// </remarks>
[LossCategory(LossCategory.Classification)]
[LossTask(LossTask.MultiClass)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = false, RequiresProbabilityInputs = true, SupportsClassWeights = true, TestInputFormat = LossTestInputFormat.ProbabilityDistribution, ExpectedOutput = OutputType.Probabilities)]
public class CategoricalCrossEntropyLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Initializes a new instance of the CategoricalCrossEntropyLoss class.
    /// </summary>
    public CategoricalCrossEntropyLoss()
    {
    }

    /// <summary>
    /// Calculates the Categorical Cross Entropy loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values (probabilities that sum to 1 across categories).</param>
    /// <param name="actual">The actual (target) values (typically one-hot encoded).</param>
    /// <returns>The categorical cross entropy loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            // -Σ(actual * log(predicted))
            sum = NumOps.Add(sum, NumOps.Multiply(actual[i], NumericalStabilityHelper.SafeLog(predicted[i], NumericalStabilityHelper.SmallEpsilon)));
        }

        return NumOps.Negate(sum);
    }

    /// <summary>
    /// Calculates the derivative of the Categorical Cross Entropy loss function.
    /// </summary>
    /// <param name="predicted">The predicted values (probabilities that sum to 1 across categories).</param>
    /// <param name="actual">The actual (target) values (typically one-hot encoded).</param>
    /// <returns>A vector containing the derivatives of CCE for each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // Derivative of -Σ actual_i * log(predicted_i) with respect to predicted_i = -actual_i / predicted_i
        // Note: When composed with softmax, this simplifies to (predicted - actual),
        // but the standalone derivative must use the correct formula.
        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = NumOps.Negate(
                NumericalStabilityHelper.SafeDiv(actual[i], predicted[i], NumericalStabilityHelper.SmallEpsilon));
        }

        return derivative;
    }

    /// <summary>
    /// Calculates both Categorical Cross Entropy loss and gradient on GPU in a single efficient pass.
    /// </summary>
    /// <param name="predicted">The predicted GPU tensor from the model.</param>
    /// <param name="actual">The actual (target) GPU tensor.</param>
    /// <returns>A tuple containing the loss value and gradient tensor.</returns>
    public override (T Loss, Tensor<T> Gradient) CalculateLossAndGradientGpu(Tensor<T> predicted, Tensor<T> actual)
    {
        var engine = AiDotNetEngine.Current as DirectGpuTensorEngine;
        var backend = engine?.GetBackend();

        if (backend == null)
        {
            return base.CalculateLossAndGradientGpu(predicted, actual);
        }

        int size = predicted.Length;

        // Compute loss on GPU
        float lossValue = backend.CategoricalCrossEntropyLoss(predicted.Buffer, actual.Buffer, size);

        // Allocate gradient buffer and compute gradient on GPU
        var gradientBuffer = backend.AllocateBuffer(size);
        backend.CategoricalCrossEntropyBackward(predicted.Buffer, actual.Buffer, gradientBuffer, size);

        // Create gradient tensor
        var gradientTensor = GpuTensorHelper.UploadToGpu<T>(backend, gradientBuffer, predicted._shape, GpuTensorRole.Gradient);

        return (NumOps.FromDouble(lossValue), gradientTensor);
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Contract: <paramref name="predicted"/> must already be a probability
    /// distribution (the output of a softmax layer or otherwise
    /// non-negative with a last-axis sum of ~1). This matches the class
    /// docstring's formula <c>CCE = -Σ actual * log(predicted)</c> and
    /// the <see cref="CalculateLoss(Vector{T}, Vector{T})"/> branch
    /// above — both of which treat <paramref name="predicted"/> as
    /// probabilities.
    /// </para>
    /// <para>
    /// If your model produces raw logits instead, use
    /// <see cref="CrossEntropyWithLogitsLoss{T}"/>, which applies
    /// <c>log_softmax</c> internally. Using this loss on top of a
    /// softmax output used to silently apply softmax a second time
    /// here, which squashed the already-normalized distribution toward
    /// uniform and made gradients vanish at initialization — the
    /// symptom in issue #1187 where <c>Transformer&lt;T&gt;.Train()</c>
    /// plateaued at <c>log(V)/V</c> from epoch 1.
    /// </para>
    /// </remarks>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        target = EnsureTargetMatchesPredicted(predicted, target);
        // Categorical CE per sample = -Σ_class target * log(predicted + eps),
        // summed over the class (last) axis, then averaged over any remaining
        // batch / sequence axes. This matches PyTorch's
        // nn.CrossEntropyLoss(reduction='mean') and the scalar
        // CalculateLoss(Vector, Vector) above — both of which sum (do not
        // average) across classes. Averaging over the class axis here would
        // silently divide the loss and the back-propagated gradient by V,
        // which is the 1/V scaling reported in issue #1191.
        var safePredicted = Engine.TensorAddScalar(predicted, NumOps.FromDouble(1e-7));
        var logP = Engine.TensorLog(safePredicted);
        var product = Engine.TensorMultiply(target, logP);

        int lastAxis = product.Shape.Length - 1;
        var perSample = Engine.ReduceSum(product, new[] { lastAxis }, keepDims: false);

        // For 1D inputs (a single unbatched sample), perSample is rank-0 —
        // there are no batch axes to average over.
        if (perSample.Shape.Length == 0)
            return Engine.TensorNegate(perSample);

        var batchAxes = Enumerable.Range(0, perSample.Shape.Length).ToArray();
        var mean = Engine.ReduceMean(perSample, batchAxes, keepDims: false);
        return Engine.TensorNegate(mean);
    }
}
