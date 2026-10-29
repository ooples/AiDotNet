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
    /// <summary>Learning rate of the value network.</summary>
    public T ValueLearningRate { get; init; }
    /// <summary>Lambda of generalized advantage estimation, trading bias against variance.</summary>
    public T GaeLambda { get; init; }

    // TRPO-specific parameters
    /// <summary>Maximum KL divergence between the old and new policies: the size of the trust region.</summary>
    public T MaxKL { get; init; }  // Maximum KL divergence (trust region size)
    /// <summary>Damping added to the Fisher-vector product for numerical stability of conjugate gradient.</summary>
    public double Damping { get; init; } = 0.1;  // Damping coefficient for conjugate gradient
    /// <summary>Number of conjugate-gradient iterations used to compute the natural-gradient direction.</summary>
    public int ConjugateGradientIterations { get; init; } = 10;
    /// <summary>Maximum number of backtracking steps of the line search.</summary>
    public int LineSearchSteps { get; init; } = 10;
    /// <summary>Minimum ratio of actual to expected improvement for the line search to accept a step.</summary>
    public double LineSearchAcceptRatio { get; init; } = 0.1;
    /// <summary>Factor by which the backtracking line search shrinks the step each time it rejects one.</summary>
    public double LineSearchBacktrackCoeff { get; init; } = 0.8;

    /// <summary>Number of environment steps collected before each policy update.</summary>
    public int StepsPerUpdate { get; init; } = 2048;
    /// <summary>Number of value-function gradient steps per update.</summary>
    public int ValueIterations { get; init; } = 5;
    /// <summary>Loss function used to train the value network.</summary>
    public ILossFunction<T> ValueLossFunction { get; init; } = new MeanSquaredErrorLoss<T>();
    /// <summary>Hidden layer sizes of the policy network.</summary>
    public List<int> PolicyHiddenLayers { get; init; } = new List<int> { 64, 64 };
    /// <summary>Hidden layer sizes of the value network.</summary>
    public List<int> ValueHiddenLayers { get; init; } = new List<int> { 64, 64 };

    /// <summary>
    /// The optimizer used for updating network parameters. If null, Adam optimizer will be used by default.
    /// </summary>
    public IOptimizer<T, Vector<T>, Vector<T>>? Optimizer { get; init; }

    /// <summary>Creates options with the default values.</summary>
    public TRPOOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        ValueLearningRate = numOps.FromDouble(0.001);
        GaeLambda = numOps.FromDouble(0.95);
        MaxKL = numOps.FromDouble(0.01);
    }
}
