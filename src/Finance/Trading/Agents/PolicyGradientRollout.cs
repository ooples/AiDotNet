using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Finance.Trading.Agents;

/// <summary>
/// The Gaussian-policy REINFORCE update the portfolio policy agents share: minimize
/// Σ_t adv_t · ½‖a_t − μ_t‖² / σ², with adv the whitened discounted return.
/// </summary>
/// <remarks>
/// An episode is packed as one training step: the states as an input <c>[stateDim, n]</c>, and the actions stacked over
/// the per-step loss weights as a target <c>[actionDim + 1, n]</c>. A fused compiled plan rebinds only its input and
/// target on each replay, so every value an episode contributes travels in them.
/// </remarks>
internal static class PolicyGradientRollout<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>The per-step loss weights adv_t · 0.5 / σ², from the discounted returns whitened over the episode.</summary>
    public static double[] LossWeights(IReadOnlyList<double> rewards, double gamma, double sigma)
    {
        int n = rewards.Count;
        var returns = new double[n];
        double g = 0;
        for (int t = n - 1; t >= 0; t--)
        {
            g = rewards[t] + gamma * g;
            returns[t] = g;
        }
        double mean = 0;
        for (int t = 0; t < n; t++) mean += returns[t];
        mean /= n;
        double var = 0;
        for (int t = 0; t < n; t++) var += (returns[t] - mean) * (returns[t] - mean);
        double std = Math.Sqrt(var / Math.Max(1, n)) + 1e-8;

        double halfInvSigma2 = 0.5 / (sigma * sigma);
        var weights = new double[n];
        for (int t = 0; t < n; t++) weights[t] = halfInvSigma2 * (returns[t] - mean) / std;
        return weights;
    }

    /// <summary>The states as <c>[stateDim, n]</c>, one column per step.</summary>
    public static Tensor<T> States(IReadOnlyList<T[]> states, int stateDim)
    {
        int n = states.Count;
        var packed = new Tensor<T>(new[] { stateDim, n });
        for (int t = 0; t < n; t++)
            for (int i = 0; i < stateDim; i++) packed[i, t] = states[t][i];
        return packed;
    }

    /// <summary>The actions <c>[actionDim, n]</c> with the loss weights as one more row.</summary>
    public static Tensor<T> Target(IReadOnlyList<T[]> actions, double[] weights, int actionDim)
    {
        int n = actions.Count;
        var packed = new Tensor<T>(new[] { actionDim + 1, n });
        for (int t = 0; t < n; t++)
        {
            for (int a = 0; a < actionDim; a++) packed[a, t] = actions[t][a];
            packed[actionDim, t] = NumOps.FromDouble(weights[t]);
        }
        return packed;
    }

    /// <summary>Σ_t w_t ‖a_t − μ_t‖² for the policy means <c>[actionDim, n]</c> against a packed target.</summary>
    public static Tensor<T> Loss(Tensor<T> means, Tensor<T> target, int actionDim)
    {
        var actions = Engine.TensorNarrow(target, 0, 0, actionDim);
        var weights = Engine.TensorNarrow(target, 0, actionDim, 1);
        var diff = Engine.TensorSubtract(actions, means);
        var squared = Engine.ReduceSum(Engine.TensorMultiply(diff, diff), new[] { 0 }, keepDims: true); // [1, n]
        return Engine.ReduceSum(Engine.TensorMultiply(squared, weights), new[] { 0, 1 }, keepDims: false);
    }
}
