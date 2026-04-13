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

        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T x = predicted[i];
            T y = actual[i];

            // max(x, 0)
            T maxXZero = NumOps.GreaterThan(x, NumOps.Zero) ? x : NumOps.Zero;

            // x * y
            T xy = NumOps.Multiply(x, y);

            // log(1 + exp(-|x|)) — softplus(-|x|), guaranteed non-overflowing because
            // -|x| <= 0 so exp(-|x|) is in (0, 1].
            T absX = NumOps.GreaterThan(x, NumOps.Zero) ? x : NumOps.Negate(x);
            T expNegAbsX = NumOps.Exp(NumOps.Negate(absX));
            T logTerm = NumericalStabilityHelper.SafeLog(NumOps.Add(NumOps.One, expNegAbsX));

            // loss_i = max(x,0) - xy + log(1 + exp(-|x|))
            T term = NumOps.Add(NumOps.Subtract(maxXZero, xy), logTerm);
            sum = NumOps.Add(sum, term);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of BCE loss with respect to logits.
    /// </summary>
    /// <param name="predicted">Raw logits.</param>
    /// <param name="actual">Binary target values.</param>
    /// <returns>Gradient: (sigmoid(x) - y) / N.</returns>
    /// <remarks>
    /// <para>
    /// The gradient simplification <c>sigmoid(x) - y</c> is what makes the fused
    /// sigmoid-plus-BCE loss preferable to applying them separately: it avoids the
    /// catastrophic cancellation that occurs in <c>-y/p + (1-y)/(1-p)</c> when
    /// <c>p</c> approaches 0 or 1.
    /// </para>
    /// </remarks>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        var derivative = new Vector<T>(predicted.Length);
        T n = NumOps.FromDouble(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            // Numerically stable sigmoid:
            //   if x >= 0: 1 / (1 + exp(-x))
            //   if x <  0: exp(x) / (1 + exp(x))
            T x = predicted[i];
            T sigmoid;
            if (NumOps.GreaterThanOrEquals(x, NumOps.Zero))
            {
                T expNegX = NumOps.Exp(NumOps.Negate(x));
                sigmoid = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegX));
            }
            else
            {
                T expX = NumOps.Exp(x);
                sigmoid = NumOps.Divide(expX, NumOps.Add(NumOps.One, expX));
            }

            derivative[i] = NumOps.Divide(NumOps.Subtract(sigmoid, actual[i]), n);
        }

        return derivative;
    }

    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        // Numerically stable BCE-with-logits via Engine ops:
        //   loss = max(x, 0) - x * y + log(1 + exp(-|x|))
        //
        // Engine.ReLU gives us max(x, 0) and the rest composes from elementary ops
        // that the tape already tracks (used by other loss functions in this file).
        var maxXZero = Engine.ReLU(predicted);

        var xy = Engine.TensorMultiply(predicted, target);

        var absX = Engine.TensorAbs(predicted);
        var negAbsX = Engine.TensorNegate(absX);
        var expNegAbsX = Engine.TensorExp(negAbsX);
        var onePlusExp = Engine.TensorAddScalar(expNegAbsX, NumOps.One);
        var logTerm = Engine.TensorLog(onePlusExp);

        var perElement = Engine.TensorAdd(Engine.TensorSubtract(maxXZero, xy), logTerm);
        var allAxes = Enumerable.Range(0, perElement.Shape.Length).ToArray();
        return Engine.ReduceMean(perElement, allAxes, keepDims: false);
    }
}
