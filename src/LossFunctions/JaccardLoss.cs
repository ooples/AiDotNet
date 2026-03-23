

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Jaccard loss function, commonly used for measuring dissimilarity between sets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Jaccard loss measures how dissimilar two sets are. It's calculated as 1 minus 
/// the size of the intersection divided by the size of the union.
/// 
/// The formula is: 1 - |A n B| / |A ? B|
/// Where:
/// - A n B is the intersection of sets A and B (elements in both)
/// - A ? B is the union of sets A and B (elements in either)
/// 
/// For continuous values (like probabilities), the intersection is the sum of the minimum values,
/// and the union is the sum of the maximum values at each position.
/// 
/// Key properties:
/// - A value of 0 means perfect overlap (identical sets)
/// - A value of 1 means no overlap at all
/// - It's symmetric: Jaccard(A,B) = Jaccard(B,A)
/// - It's a proper distance metric, suitable for measuring dissimilarity
/// 
/// Jaccard loss is particularly useful for:
/// - Image segmentation tasks
/// - Set similarity problems
/// - Binary classification problems
/// - Tasks where the positive class is rare (imbalanced data)
/// 
/// It's often a better choice than pixel-wise losses (like MSE) for segmentation tasks 
/// because it directly optimizes for the overlap of segments.
/// </para>
/// </remarks>
[LossCategory(LossCategory.Segmentation)]
[LossTask(LossTask.SemanticSegmentation)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = false, HandlesImbalancedData = true, RequiresProbabilityInputs = true, TestInputFormat = LossTestInputFormat.SegmentationMask, ExpectedOutput = OutputType.Probabilities)]
public class JaccardLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Initializes a new instance of the JaccardLoss class.
    /// </summary>
    public JaccardLoss()
    {
    }

    /// <summary>
    /// Calculates the Jaccard loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <returns>The Jaccard loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // Soft IoU (differentiable): intersection = Σ(p*a), union = Σp + Σa - Σ(p*a)
        // This is the standard formulation used by PyTorch and matches the derivative.
        T intersection = NumOps.Zero;
        T sumPredicted = NumOps.Zero;
        T sumActual = NumOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            intersection = NumOps.Add(intersection, NumOps.Multiply(predicted[i], actual[i]));
            sumPredicted = NumOps.Add(sumPredicted, predicted[i]);
            sumActual = NumOps.Add(sumActual, actual[i]);
        }

        T union = NumOps.Subtract(NumOps.Add(sumPredicted, sumActual), intersection);

        // Jaccard loss = 1 - IoU
        return NumOps.Subtract(NumOps.One, NumericalStabilityHelper.SafeDiv(intersection, union, NumericalStabilityHelper.SmallEpsilon));
    }

    /// <summary>
    /// Calculates the derivative of the Jaccard loss function.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <returns>A vector containing the derivatives of the Jaccard loss with respect to each predicted value.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // Soft IoU derivative: d(1 - IoU)/d(p_i)
        // IoU = intersection / union
        // intersection = Σ(p*a), union = Σp + Σa - Σ(p*a)
        // d(IoU)/d(p_i) = (a_i * union - intersection * (1 - a_i)) / union²
        // d(loss)/d(p_i) = -d(IoU)/d(p_i)
        T intersection = NumOps.Zero;
        T sumPredicted = NumOps.Zero;
        T sumActual = NumOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            intersection = NumOps.Add(intersection, NumOps.Multiply(predicted[i], actual[i]));
            sumPredicted = NumOps.Add(sumPredicted, predicted[i]);
            sumActual = NumOps.Add(sumActual, actual[i]);
        }

        T union = NumOps.Subtract(NumOps.Add(sumPredicted, sumActual), intersection);
        T unionSquared = NumOps.Multiply(union, union);

        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            // d(IoU)/d(p_i) = (a_i * union - intersection * (1 - a_i)) / union²
            T numerator = NumOps.Subtract(
                NumOps.Multiply(actual[i], union),
                NumOps.Multiply(intersection, NumOps.Subtract(NumOps.One, actual[i]))
            );
            // loss = 1 - IoU, so derivative is negated
            derivative[i] = NumOps.Negate(
                NumericalStabilityHelper.SafeDiv(numerator, unionSquared, NumericalStabilityHelper.SmallEpsilon)
            );
        }

        return derivative;
    }
}
