using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Margin loss function, specifically designed for Capsule Networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Margin loss is a special loss function used in Capsule Networks.
/// 
/// The formula is:
/// T_c * max(0, m+ - ||v_c||)^2 + lambda * (1 - T_c) * max(0, ||v_c|| - m-)^2
/// 
/// Where:
/// - T_c is 1 if class c is present, 0 otherwise
/// - ||v_c|| is the length of the output vector of the capsule for class c
/// - m+ is the upper bound (usually 0.9)
/// - m- is the lower bound (usually 0.1)
/// - lambda is a down-weighting factor (usually 0.5)
/// 
/// Key properties:
/// - Encourages the network to output high values for correct classes
/// - Discourages high outputs for incorrect classes
/// - Helps in learning to represent different aspects of the input
/// 
/// Margin loss is ideal for Capsule Networks because:
/// - It allows multiple classes to be present in the same image
/// - It encourages the network to learn to represent different viewpoints and transformations
/// - It helps in achieving equivariance, a key property of Capsule Networks
/// </para>
/// </remarks>
[LossCategory(LossCategory.Classification)]
[LossCategory(LossCategory.Ranking)]
[LossTask(LossTask.BinaryClassification)]
[LossTask(LossTask.Ranking)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = false, TestInputFormat = LossTestInputFormat.MarginBased, ExpectedOutput = OutputType.Logits)]
public class MarginLoss<T> : LossFunctionBase<T>
{
    private readonly T _mPlus;
    private readonly T _mMinus;
    private readonly T _lambda;

    /// <summary>
    /// Initializes a new instance of the MarginLoss class with the specified parameters.
    /// </summary>
    /// <param name="mPlus">The upper bound. Default is 0.9.</param>
    /// <param name="mMinus">The lower bound. Default is 0.1.</param>
    /// <param name="lambda">The down-weighting factor. Default is 0.5.</param>
    public MarginLoss(double mPlus = 0.9, double mMinus = 0.1, double lambda = 0.5)
    {
        _mPlus = NumOps.FromDouble(mPlus);
        _mMinus = NumOps.FromDouble(mMinus);
        _lambda = NumOps.FromDouble(lambda);
    }

    /// <summary>
    /// Calculates the Margin loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The Margin loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        var v = predicted.AsTensor();
        var y = actual.AsTensor();
        ValidateTargets(y, nameof(actual));

        // Sabour et al. (2017): y * max(0, m+ - v)² + λ * (1 - y) * max(0, v - m-)². The label weights the squared
        // hinge (it is not squared itself), matching ComputeTapeLoss for fractional labels as well as 0/1 labels.
        var hinge1 = Engine.TensorMax(Engine.ScalarMinusTensor(_mPlus, v), NumOps.Zero);
        var term1 = Engine.TensorMultiply(y, Engine.TensorMultiply(hinge1, hinge1));

        var hinge2 = Engine.TensorMax(Engine.TensorSubtractScalar(v, _mMinus), NumOps.Zero);
        var term2 = Engine.TensorMultiply(Engine.ScalarMinusTensor(NumOps.One, y), Engine.TensorMultiply(hinge2, hinge2));

        T loss = Engine.TensorSum(Engine.TensorAdd(term1, Engine.TensorMultiplyScalar(term2, _lambda)));

        return NumOps.Divide(loss, NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Rejects targets outside [0, 1] (including NaN and infinities): the label weights y and 1 - y must be
    /// nonnegative, or the loss can turn negative and push training the wrong way.
    /// </summary>
    private void ValidateTargets(Tensor<T> targets, string paramName)
    {
        if (targets.Length == 0)
        {
            return;
        }

        var invalid = Engine.TensorMax(
            Engine.TensorMax(
                Engine.TensorLessThan(targets, NumOps.Zero),
                Engine.TensorGreaterThan(targets, NumOps.One)),
            Engine.TensorNotEquals(targets, targets));
        if (NumOps.GreaterThan(Engine.TensorMaxValue(invalid), NumOps.Zero))
        {
            int position = Engine.TensorNonzero(invalid).GetFlat(0);
            throw new ArgumentOutOfRangeException(paramName,
                $"Margin loss targets must be finite values in [0, 1]; position {position} is {targets.GetFlat(position)}.");
        }
    }

    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        ValidateTargets(target, nameof(target));

        // Margin (capsule): mean(t * max(0, m+ - p)² + λ * (1-t) * max(0, p - m-)²)
        var posDiff = Engine.ScalarMinusTensor(_mPlus, predicted);
        var posHinged = Engine.ReLU(posDiff);
        var posSq = Engine.TensorMultiply(posHinged, posHinged);
        var negDiff = Engine.TensorSubtractScalar(predicted, _mMinus);
        var negHinged = Engine.ReLU(negDiff);
        var negSq = Engine.TensorMultiply(negHinged, negHinged);
        var oneMinusT = Engine.ScalarMinusTensor(NumOps.One, target);
        var posLoss = Engine.TensorMultiply(target, posSq);
        var negLoss = Engine.TensorMultiplyScalar(Engine.TensorMultiply(oneMinusT, negSq), _lambda);
        var result = Engine.TensorAdd(posLoss, negLoss);
        var allAxes = Enumerable.Range(0, result.Shape.Length).ToArray();
        return Engine.ReduceMean(result, allAxes, keepDims: false);
    }
}
