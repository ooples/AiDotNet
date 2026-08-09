

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Dice loss function, commonly used for image segmentation tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Dice loss measures the overlap between predicted and actual segments in an image.
/// It's based on the Dice coefficient (also known as F1 score), which is a statistical measure of similarity.
/// 
/// The formula is: DiceLoss = 1 - (2 * intersection) / (sum of predicted + sum of actual)
/// 
/// Where:
/// - intersection is the sum of element-wise multiplication of predicted and actual values
/// - A value of 0 means perfect overlap (ideal predictions)
/// - A value of 1 means no overlap at all (worst predictions)
/// 
/// Key properties:
/// - It's ideal for problems where the positive class (what you're trying to detect) is rare
/// - Handles imbalanced data better than cross-entropy in many cases
/// - Focuses on maximizing the overlap between predictions and ground truth
/// - Commonly used in medical image segmentation, satellite imagery, and other segmentation tasks
/// 
/// Unlike cross-entropy, which treats each pixel independently, Dice loss considers the global
/// relationship between predicted and actual masks, which often leads to better segmentation results.
/// </para>
/// </remarks>
[LossCategory(LossCategory.Segmentation)]
[LossTask(LossTask.SemanticSegmentation)]
[LossTask(LossTask.InstanceSegmentation)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = false, HandlesImbalancedData = true, RequiresProbabilityInputs = true, TestInputFormat = LossTestInputFormat.SegmentationMask, ExpectedOutput = OutputType.Probabilities)]
public class DiceLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Initializes a new instance of the DiceLoss class.
    /// </summary>
    public DiceLoss()
    {
    }

    /// <summary>
    /// Calculates the Dice loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values (typically probabilities between 0 and 1).</param>
    /// <param name="actual">The actual (target) values (typically 0 or 1).</param>
    /// <returns>The Dice loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T intersection = Engine.DotProduct(predicted, actual);

        // Sum vectors using dot product with ones vector
        T sumPredicted = NumOps.Zero;
        T sumActual = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            sumPredicted = NumOps.Add(sumPredicted, predicted[i]);
            sumActual = NumOps.Add(sumActual, actual[i]);
        }

        // Use NumericalStabilityHelper.SafeDiv to prevent division by zero
        T denominator = NumOps.Add(sumPredicted, sumActual);
        T diceCoefficient = NumericalStabilityHelper.SafeDiv(
            NumOps.Multiply(NumOps.FromDouble(2), intersection),
            denominator,
            NumericalStabilityHelper.SmallEpsilon
        );

        return NumOps.Subtract(NumOps.One, diceCoefficient);
    }


    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        // Dice = 1 - (2 * intersection + smooth) / (|pred| + |target| + smooth)
        var intersection = Engine.TensorMultiply(predicted, target);
        var allAxes = Enumerable.Range(0, intersection.Shape.Length).ToArray();
        var interSum = Engine.ReduceSum(intersection, allAxes, keepDims: false);
        var predSum = Engine.ReduceSum(predicted, allAxes, keepDims: false);
        var targSum = Engine.ReduceSum(target, allAxes, keepDims: false);
        var twoInter = Engine.TensorMultiplyScalar(interSum, NumOps.FromDouble(2.0));
        var numerator = Engine.TensorAddScalar(twoInter, NumOps.One);
        var denominator = Engine.TensorAddScalar(Engine.TensorAdd(predSum, targSum), NumOps.One);

        // Divide RANK-1 [1] tensors, not rank-0 scalars. ReduceSum with keepDims: false collapses to
        // rank-0 [], and while the forward divide is fine, its BACKWARD is not: DivideBackward
        // receives a [1] upstream gradient against a [] input and throws
        // "Tensor shapes must match. Got [1] and []" from CpuEngine.TensorDivide, aborting
        // ComputeGradients. That made DiceLoss unusable for TRAINING whenever it was applied on its
        // own -- the failure surfaces as an exception mid-Train, which callers see as "no parameters
        // changed", not as a loss bug. Reshaping is an Engine op so the tape is preserved, and the
        // returned rank-1 loss matches what the other loss functions here already produce.
        var numerator1 = Engine.Reshape(numerator, new[] { 1 });
        var denominator1 = Engine.Reshape(denominator, new[] { 1 });
        var dice = Engine.TensorDivide(numerator1, denominator1);
        return Engine.ScalarMinusTensor(NumOps.One, dice);
    }
}
