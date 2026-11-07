using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Trust Region Policy Optimization (TRPO) agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TRPO ensures monotonic improvement by constraining policy updates to a "trust region"
/// using KL divergence. This prevents destructively large updates.
/// </para>
/// <para><b>For Beginners:</b>
/// TRPO is like learning carefully - it never makes a change that's "too big".
/// By limiting how much the policy can change, it guarantees that performance
/// never gets worse (monotonic improvement).
///
/// Key features:
/// - **Trust Region**: Limits policy change per update (via KL divergence)
/// - **Monotonic Improvement**: Guarantees performance doesn't degrade
/// - **Conjugate Gradient**: Efficiently solves constrained optimization
/// - **Line Search**: Ensures constraints are satisfied
///
/// Think of it like taking small, safe steps when walking on uncertain terrain
/// rather than making large leaps that might cause you to fall.
///
/// Famous for: OpenAI's robotics research, predecessor to PPO
/// </para>
/// </remarks>
public class TRPOOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; init; } = 1;
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; init; } = 1;
    /// <summary>Whether the action space is continuous rather than discrete.</summary>
    public bool IsContinuous { get; init; } = false;
    /// <summary>Adam learning rate of the value baseline. Default: 0.001.</summary>
    public T ValueLearningRate { get; init; }

    /// <summary>GAE lambda of the advantage estimate. Default: 1, the paper's discounted return minus the baseline
    /// (single path, section 5.1).</summary>
    public T GaeLambda { get; init; }

    /// <summary>The trust region δ, the largest mean KL divergence of a step. Default: 0.01, used "for all
    /// experiments" (section 8).</summary>
    public T MaxKL { get; init; }

    /// <summary>Damping added to the Fisher-vector products of the conjugate gradient solve. Default: 0.1 (the
    /// authors' implementation).</summary>
    public double Damping { get; init; } = 0.1;

    /// <summary>Conjugate gradient iterations of the search direction (appendix C). Default: 10.</summary>
    public int ConjugateGradientIterations { get; init; } = 10;

    /// <summary>Backtracking steps of the line search. Default: 10.</summary>
    public int LineSearchSteps { get; init; } = 10;

    /// <summary>The smallest accepted fraction of the surrogate's linearly predicted improvement. Default: 0.1.</summary>
    public double LineSearchAcceptRatio { get; init; } = 0.1;

    /// <summary>The line search's exponential shrink factor (appendix C). Default: 0.5.</summary>
    public double LineSearchBacktrackCoeff { get; init; } = 0.5;

    /// <summary>Number of environment steps collected before each policy update.</summary>
    public int StepsPerUpdate { get; init; } = 2048;
    /// <summary>Number of value-function gradient steps per update.</summary>
    public int ValueIterations { get; init; } = 5;
    /// <summary>Loss function used to train the value network.</summary>
    public ILossFunction<T> ValueLossFunction { get; init; } = new MeanSquaredErrorLoss<T>();
    /// <summary>Hidden layer widths of the policy (tanh). Default: one layer of 30 units, the swimmer policy of
    /// figure 3 (the hopper and walker used 50, table 2).</summary>
    public List<int> PolicyHiddenLayers { get; init; } = new List<int> { 30 };

    /// <summary>Hidden layer widths of the value baseline (tanh). Default: { 64, 64 }; the paper does not describe its
    /// baseline.</summary>
    public List<int> ValueHiddenLayers { get; init; } = new List<int> { 64, 64 };

    /// <summary>Creates options with the default values.</summary>
    public TRPOOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        ValueLearningRate = numOps.FromDouble(0.001);
        GaeLambda = numOps.FromDouble(1.0);
        MaxKL = numOps.FromDouble(0.01);
    }
}
