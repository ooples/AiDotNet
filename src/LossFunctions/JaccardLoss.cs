

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

        T intersection = NumOps.Zero;
        T union = NumOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            // Intersection is the sum of minimums
            intersection = NumOps.Add(intersection, MathHelper.Min(predicted[i], actual[i]));

            // Union is the sum of maximums
            union = NumOps.Add(union, MathHelper.Max(predicted[i], actual[i]));
        }

        // Jaccard loss = 1 - Jaccard Index
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

        // Calculate intersection and union first
        T intersection = NumOps.Zero;
        T union = NumOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            intersection = NumOps.Add(intersection, MathHelper.Min(predicted[i], actual[i]));
            union = NumOps.Add(union, MathHelper.Max(predicted[i], actual[i]));
        }


        // Calculate derivative for each element
        Vector<T> derivative = new Vector<T>(predicted.Length);
        T unionSquared = NumOps.Power(union, NumOps.FromDouble(2));
        T numerator = NumOps.Subtract(union, intersection);

        for (int i = 0; i < predicted.Length; i++)
        {
            if (NumOps.GreaterThan(predicted[i], actual[i]))
            {
                // If predicted > actual, derivative = (union - intersection) / union²
                derivative[i] = NumericalStabilityHelper.SafeDiv(numerator, unionSquared, NumericalStabilityHelper.SmallEpsilon);
            }
            else if (NumOps.LessThan(predicted[i], actual[i]))
            {
                // If predicted < actual, derivative = -(union - intersection) / union²
                derivative[i] = NumOps.Negate(
                    NumericalStabilityHelper.SafeDiv(numerator, unionSquared, NumericalStabilityHelper.SmallEpsilon)
                );
            }
            else
            {
                // If predicted == actual, derivative = 0
                derivative[i] = NumOps.Zero;
            }
        }

        return derivative;
    }
}
