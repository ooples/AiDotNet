using System;
using System.Collections.Generic;
using System.Linq;
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
    public static Tensor<T> LossWeights(IReadOnlyList<T> rewards, T gamma, T sigma)
    {
        int n = rewards.Count;
        if (n == 0)
        {
            return new Tensor<T>(new[] { 0 });
        }

        var returns = AiDotNet.Helpers.LinearRecurrence<T>.DiscountedReturns(
            new Tensor<T>(rewards.ToArray(), new[] { n }), gamma);
        var centred = Engine.TensorSubtractScalar(returns, Engine.TensorMean(returns));
        T std = NumOps.Add(
            NumOps.Sqrt(NumOps.Divide(Engine.TensorSumOfSquares(centred), NumOps.FromDouble(n))),
            NumOps.FromDouble(1e-8));

        T halfInvSigma2 = NumOps.Divide(NumOps.FromDouble(0.5), NumOps.Multiply(sigma, sigma));
        return Engine.TensorMultiplyScalar(centred, NumOps.Divide(halfInvSigma2, std));
    }

    /// <summary>
    /// The first <paramref name="dim"/> entries of <paramref name="values"/> as a <c>[dim]</c> tensor, zero-padded
    /// when it is shorter.
    /// </summary>
    public static Tensor<T> Column(Vector<T> values, int dim)
    {
        var all = Tensor<T>.FromVector(values);
        int kept = Math.Min(dim, values.Length);
        var head = kept == values.Length ? all : Engine.TensorSlice(all, new[] { 0 }, new[] { kept });
        return kept == dim ? head : Engine.TensorConcatenate(new[] { head, new Tensor<T>(new[] { dim - kept }) }, 0);
    }

    /// <summary>The states as <c>[stateDim, n]</c>, one column per step.</summary>
    public static Tensor<T> States(IReadOnlyList<Tensor<T>> states) => Engine.TensorStack(states.ToArray(), 1);

    /// <summary>The actions <c>[actionDim, n]</c> with the loss weights as one more row.</summary>
    public static Tensor<T> Target(IReadOnlyList<Tensor<T>> actions, Tensor<T> weights)
        => Engine.TensorConcatenate(new[]
        {
            Engine.TensorStack(actions.ToArray(), 1),
            Engine.Reshape(weights, new[] { 1, weights.Length }),
        }, 0);

    /// <summary>
    /// The explored action <c>clip(μ + σ·z, −1, 1)</c> for policy means and standard-normal draws of the same shape;
    /// without exploration the clipped mean.
    /// </summary>
    public static Vector<T> Act(Tensor<T> mean, Tensor<T>? draws, T sigma)
    {
        var perturbed = draws is null ? mean : Engine.TensorAdd(mean, Engine.TensorMultiplyScalar(draws, sigma));
        var flat = Engine.Reshape(perturbed, new[] { perturbed.Length });
        return Engine.TensorClamp(flat, NumOps.Negate(NumOps.One), NumOps.One).ToVector();
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
