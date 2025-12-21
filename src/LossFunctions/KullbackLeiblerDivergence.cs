

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Kullback-Leibler Divergence, a measure of how one probability distribution differs from another.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Kullback-Leibler (KL) Divergence measures how one probability distribution differs from another.
/// It's often interpreted as the "information loss" when using one distribution to approximate another.
/// 
/// The formula is: KL(P||Q) = sum(P(i) * log(P(i)/Q(i))
/// Where:
/// - P is the true distribution
/// - Q is the approximating distribution
/// 
/// Key properties:
/// - It's always non-negative (zero only when the distributions are identical)
/// - It's not symmetric: KL(P||Q) ? KL(Q||P)
/// - It's not a true distance metric due to this asymmetry
/// 
/// KL divergence is commonly used in:
/// - Variational Autoencoders (VAEs)
/// - Reinforcement learning algorithms
/// - Information theory applications
/// - Distribution approximation tasks
/// 
/// When training models, KL divergence helps push the predicted distribution (Q) to match the target distribution (P).
/// </para>
/// </remarks>
public class KullbackLeiblerDivergence<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Initializes a new instance of the KullbackLeiblerDivergence class.
    /// </summary>
    public KullbackLeiblerDivergence()
    {
    }

    /// <summary>
    /// Calculates the Kullback-Leibler Divergence between predicted and actual probability distributions.
    /// </summary>
    /// <param name="predicted">The predicted probability distribution.</param>
    /// <param name="actual">The actual (target) probability distribution.</param>
    /// <returns>The KL divergence value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            // KL(P||Q) = sum(P(i) * log(P(i)/Q(i))
            T ratio = NumericalStabilityHelper.SafeDiv(actual[i], predicted[i], NumericalStabilityHelper.SmallEpsilon);
            sum = NumOps.Add(sum, NumOps.Multiply(actual[i], NumericalStabilityHelper.SafeLog(ratio, NumericalStabilityHelper.SmallEpsilon)));
        }

        return sum;
    }

    /// <summary>
    /// Calculates the derivative of the Kullback-Leibler Divergence.
    /// </summary>
    /// <param name="predicted">The predicted probability distribution.</param>
    /// <param name="actual">The actual (target) probability distribution.</param>
    /// <returns>A vector containing the gradient of the KL divergence with respect to each prediction.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            // The derivative of KL(P||Q) with respect to Q is -P/Q
            derivative[i] = NumOps.Negate(NumericalStabilityHelper.SafeDiv(actual[i], predicted[i], NumericalStabilityHelper.SmallEpsilon));
        }

        return derivative;
    }
}
