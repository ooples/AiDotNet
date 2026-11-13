using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements Binary Cross-Entropy loss that accepts raw logits (not probabilities).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This loss function is equivalent to PyTorch's
/// <c>nn.BCEWithLogitsLoss</c>. It combines a sigmoid activation and binary
/// cross-entropy into a single numerically stable computation.
/// </para>
/// <para>
/// Unlike <see cref="BinaryCrossEntropyLoss{T}"/> which expects probability inputs
/// (after sigmoid), this version accepts raw logits (unbounded model outputs) and
/// applies the sigmoid internally. This is the correct choice when your model's
/// final layer outputs raw scores without sigmoid activation — for example, the
/// classification heads in DenseNet/EfficientNet/etc. emit logits because applying
/// sigmoid in the model and then again inside the loss would compose two non-linear
/// functions and produce wrong gradients.
/// </para>
/// <para>
/// The numerically stable form (avoids exp overflow for large positive x and
/// log(0) for very negative x) is:
/// </para>
/// <para>
/// <c>loss = max(x, 0) - x * y + log(1 + exp(-|x|))</c>
/// </para>
/// <para>
/// Derivation: BCE on probability p = sigmoid(x) is
/// <c>-(y log p + (1-y) log(1-p))</c>. Substituting <c>p = 1/(1+exp(-x))</c> and
/// using <c>log(1-sigmoid(x)) = -x - log(1+exp(-x))</c>, the expression simplifies
/// to <c>x - x*y + log(1+exp(-x))</c>. The <c>max(x, 0) + log(1 + exp(-|x|))</c>
/// rewrite is the standard log-sum-exp trick that keeps both the very-positive and
/// very-negative tails well-conditioned.
/// </para>
/// <para>
/// The gradient with respect to logits has the elegant form
/// <c>d(loss)/d(x) = sigmoid(x) - y</c>, just like cross-entropy with logits.
/// </para>
/// </remarks>
[LossCategory(LossCategory.Classification)]
[LossTask(LossTask.BinaryClassification)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = false, RequiresProbabilityInputs = false, TestInputFormat = LossTestInputFormat.RawLogits, ExpectedOutput = OutputType.Logits)]
public class BinaryCrossEntropyWithLogitsLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Initializes a new instance of the BinaryCrossEntropyWithLogitsLoss class.
    /// </summary>
    public BinaryCrossEntropyWithLogitsLoss()
    {
    }

    /// <summary>
    /// Calculates BCE loss directly from raw logits using the numerically stable form.
    /// </summary>
    /// <param name="predicted">Raw logits (unbounded model outputs, NOT probabilities).</param>
    /// <param name="actual">Binary target values (0 or 1, or soft targets in [0, 1]).</param>
    /// <returns>The mean binary cross-entropy loss.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        var x = predicted.AsTensor();
        var y = actual.AsTensor();

        // max(x, 0)
        var maxXZero = Engine.TensorMax(x, NumOps.Zero);

        // x * y
        var xy = Engine.TensorMultiply(x, y);

        // log(1 + exp(-|x|)) — softplus(-|x|), guaranteed non-overflowing because
        // -|x| <= 0 so exp(-|x|) is in (0, 1], so 1 + exp(-|x|) >= 1 never needs the SafeLog floor.
        var expNegAbsX = Engine.TensorExp(Engine.TensorNegate(Engine.TensorAbs(x)));
        var logTerm = Engine.TensorLog(Engine.TensorAddScalar(expNegAbsX, NumOps.One));

        // loss_i = max(x,0) - xy + log(1 + exp(-|x|))
        T sum = Engine.TensorSum(Engine.TensorAdd(Engine.TensorSubtract(maxXZero, xy), logTerm));

        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }

    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        // BCE-with-logits is softplus(x) - x*y. The primitive keeps the smooth sigmoid
        // derivative at zero; a ReLU/Abs expansion has ambiguous subgradients there.
        var softplus = Engine.Softplus(predicted);
        var xy = Engine.TensorMultiply(predicted, target);
        var perElement = Engine.TensorSubtract(softplus, xy);
        var allAxes = Enumerable.Range(0, perElement.Shape.Length).ToArray();
        return Engine.ReduceMean(perElement, allAxes, keepDims: false);
    }
}
