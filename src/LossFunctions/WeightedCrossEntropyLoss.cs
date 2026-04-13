using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Weighted Cross Entropy loss function for classification problems with uneven class importance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Weighted Cross Entropy is a variation of the standard cross-entropy loss that applies
/// different weights to different samples or classes.
/// 
/// The regular cross-entropy penalizes all misclassifications equally, but in some cases:
/// - Some classes might be more important to classify correctly
/// - Some classes might be rare in the training data but important in practice
/// - Some samples might be more reliable or representative than others
/// 
/// Weighted Cross Entropy lets you control the importance of different samples by applying weights
/// to them. Higher weights mean the model will focus more on getting those specific samples right.
/// 
/// This loss function is particularly useful for:
/// - Imbalanced datasets where some classes are underrepresented
/// - Problems where misclassifying certain classes is more costly than others
/// - Situations where you have varying confidence in your training data
/// </para>
/// </remarks>
[LossCategory(LossCategory.Classification)]
[LossTask(LossTask.BinaryClassification)]
[LossTask(LossTask.MultiLabel)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = false, SupportsClassWeights = true, HandlesImbalancedData = true, RequiresProbabilityInputs = true, ExpectedOutput = OutputType.Probabilities)]
public class WeightedCrossEntropyLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// The weights to apply to each sample.
    /// </summary>
    private readonly Vector<T> _weights;

    /// <summary>
    /// Initializes a new instance of the WeightedCrossEntropyLoss class.
    /// </summary>
    /// <param name="weights">The weights vector for each sample. If null, all samples will have weight 1.</param>
    public WeightedCrossEntropyLoss(Vector<T>? weights = null)
    {
        if (weights != null)
        {
            _weights = weights;
        }
        else
        {
            // Create a default single-element vector with weight 1
            // Note: The actual weights will be recreated in CalculateLoss if the length doesn't match
            _weights = new Vector<T>(1);
            _weights[0] = NumOps.One;
        }
    }

    /// <summary>
    /// Calculates the Weighted Cross Entropy loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values (probabilities between 0 and 1).</param>
    /// <param name="actual">The actual (target) values (typically 0 or 1).</param>
    /// <returns>The weighted cross entropy loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // If weights are not provided, use uniform weights
        Vector<T> weights = _weights;
        if (weights == null || weights.Length != predicted.Length)
        {
            weights = new Vector<T>(predicted.Length);
            for (int i = 0; i < predicted.Length; i++)
            {
                weights[i] = NumOps.One;
            }
        }

        T loss = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            // -weight * [y*log(p) + (1-y)*log(1-p)]
            loss = NumOps.Add(loss, NumOps.Multiply(weights[i],
                NumOps.Add(
                    NumOps.Multiply(actual[i], NumericalStabilityHelper.SafeLog(predicted[i], NumericalStabilityHelper.SmallEpsilon)),
                    NumOps.Multiply(
                        NumOps.Subtract(NumOps.One, actual[i]),
                        NumericalStabilityHelper.SafeLog(NumOps.Subtract(NumOps.One, predicted[i]), NumericalStabilityHelper.SmallEpsilon)
                    )
                )
            ));
        }

        // Return the average loss (consistent with BinaryCrossEntropyLoss)
        return NumOps.Negate(NumOps.Divide(loss, NumOps.FromDouble(predicted.Length)));
    }

    /// <summary>
    /// Calculates the derivative of the Weighted Cross Entropy loss function.
    /// </summary>
    /// <param name="predicted">The predicted values (probabilities between 0 and 1).</param>
    /// <param name="actual">The actual (target) values (typically 0 or 1).</param>
    /// <returns>A vector containing the derivatives of the weighted cross entropy loss with respect to each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // If weights are not provided, use uniform weights
        Vector<T> weights = _weights;
        if (weights == null || weights.Length != predicted.Length)
        {
            weights = new Vector<T>(predicted.Length);
            for (int i = 0; i < predicted.Length; i++)
            {
                weights[i] = NumOps.One;
            }
        }

        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            // weight * [(p - y)/(p*(1-p))]
            T denominator = NumOps.Multiply(predicted[i], NumOps.Subtract(NumOps.One, predicted[i]));
            derivative[i] = NumOps.Multiply(
                weights[i],
                NumericalStabilityHelper.SafeDiv(
                    NumOps.Subtract(predicted[i], actual[i]),
                    denominator,
                    NumericalStabilityHelper.SmallEpsilon
                )
            );
        }

        // Return the average derivative (consistent with BinaryCrossEntropyLoss)
        return derivative.Divide(NumOps.FromDouble(predicted.Length));
    }

    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        // Only one-hot encode for multi-class (numClasses > 1).
        // For binary (predicted=[B,1], target=[B]), just reshape.
        if (target.Shape.Length < predicted.Shape.Length)
        {
            if (predicted.Shape[predicted.Shape.Length - 1] > 1)
            {
                target = EnsureTargetMatchesPredicted(predicted, target);
            }
            else if (target.Shape.Length == predicted.Shape.Length - 1
                     && predicted.Shape[predicted.Shape.Length - 1] == 1
                     && target.Length == predicted.Length)
            {
                target = Engine.Reshape(target, predicted.Shape.ToArray());
            }
        }

        // Full weighted BCE: -mean(weights * (target * log(p) + (1-target) * log(1-p)))
        var eps = NumOps.FromDouble(1e-7);
        var oneMinusEps = NumOps.FromDouble(1.0 - 1e-7);
        var clamped = Engine.TensorClamp(predicted, eps, oneMinusEps);

        var logP = Engine.TensorLog(clamped);
        var oneMinusP = Engine.ScalarMinusTensor(NumOps.One, clamped);
        var logOneMinusP = Engine.TensorLog(oneMinusP);
        var oneMinusTarget = Engine.ScalarMinusTensor(NumOps.One, target);

        // target * log(p) + (1-target) * log(1-p)
        var positiveTerm = Engine.TensorMultiply(target, logP);
        var negativeTerm = Engine.TensorMultiply(oneMinusTarget, logOneMinusP);
        var bce = Engine.TensorAdd(positiveTerm, negativeTerm);

        if (_weights.Length > 1 && _weights.Length == bce.Length)
        {
            var weightTensor = Tensor<T>.FromVector(_weights);
            bce = Engine.TensorMultiply(bce, weightTensor);
        }
        var allAxes = Enumerable.Range(0, bce.Shape.Length).ToArray();
        var mean = Engine.ReduceMean(bce, allAxes, keepDims: false);
        return Engine.TensorNegate(mean);
    }
}
