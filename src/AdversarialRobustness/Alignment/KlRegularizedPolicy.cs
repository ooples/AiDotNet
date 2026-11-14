using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AdversarialRobustness.Alignment;

/// <summary>
/// The optimal policy of the KL-regularized RLHF objective over a model's output classes.
/// </summary>
/// <remarks>
/// <para>
/// RLHF maximizes E_{y ~ pi}[r(x, y)] - beta KL(pi(. | x) || pi_ref(. | x)) (Ziegler et al. 2019, "Fine-Tuning Language
/// Models from Human Preferences"; Ouyang et al. 2022). Its maximizer is
/// pi*(y | x) = pi_ref(y | x) exp(r(x, y) / beta) / Z(x) (Rafailov et al. 2023, "Direct Preference Optimization",
/// eq. 4). Reading the base model's output z as the logits of pi_ref over K actions and scoring action k with
/// r(x, e_k), the aligned distribution is softmax(z + r / beta): beta (the KL coefficient) trades reward against
/// staying near the reference model, from the reward-greedy policy as beta -> 0 to the reference model as beta grows.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type.</typeparam>
internal static class KlRegularizedPolicy<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The aligned action distribution softmax(z + r / beta) for the reference logits <paramref name="referenceLogits"/>.
    /// </summary>
    /// <param name="input">The prompt x passed to the reward.</param>
    /// <param name="referenceLogits">The reference model's output z (logits over the actions).</param>
    /// <param name="reward">The reward r(x, y), evaluated at every one-hot action e_k.</param>
    /// <param name="klCoefficient">beta, the weight of the KL penalty; must be positive and finite.</param>
    /// <returns>The aligned distribution over the actions.</returns>
    public static Vector<T> Apply(Vector<T> input, Vector<T> referenceLogits, Func<Vector<T>, Vector<T>, double> reward, double klCoefficient)
    {
        if (!(klCoefficient > 0.0 && !double.IsInfinity(klCoefficient)))
        {
            throw new InvalidOperationException(
                $"AlignmentMethodOptions.KLCoefficient must be a positive finite number (was {klCoefficient}).");
        }

        int actions = referenceLogits.Length;
        if (actions == 0)
        {
            return referenceLogits;
        }

        // One reward query per action: the loop runs over reward-model calls, not over arithmetic.
        var scaledRewards = new double[actions];
        for (int k = 0; k < actions; k++)
        {
            scaledRewards[k] = reward(input, Vector<T>.CreateStandardBasis(actions, k)) / klCoefficient;
        }

        // softmax is shift-invariant: subtracting the largest scaled reward (in double) keeps every term at most 0,
        // so a small beta cannot overflow narrow types such as Half (which saturates at 65504).
        double largest = scaledRewards.Max();
        for (int k = 0; k < actions; k++)
        {
            scaledRewards[k] -= largest;
        }

        var engine = AiDotNetEngine.Current;
        var logits = engine.TensorAdd(referenceLogits.AsTensor(), new Tensor<double>(scaledRewards, new[] { actions }).Cast<T>());
        T maximum = engine.TensorMaxValue(logits);
        var exponentials = engine.TensorExp(engine.TensorAddScalar(logits, NumOps.Negate(maximum)));
        T total = NumOps.Multiply(engine.TensorMean(exponentials), NumOps.FromDouble(actions));
        return engine.TensorDivideScalar(exponentials, total).ToVector();
    }
}
