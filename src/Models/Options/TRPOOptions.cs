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
    public int StateSize { get; init; } = 1;
    public int ActionSize { get; init; } = 1;
    public bool IsContinuous { get; init; } = false;
    public T ValueLearningRate { get; init; }
    public T GaeLambda { get; init; }

    // TRPO-specific parameters
    public T MaxKL { get; init; }  // Maximum KL divergence (trust region size)
    public double Damping { get; init; } = 0.1;  // Damping coefficient for conjugate gradient
    public int ConjugateGradientIterations { get; init; } = 10;
    public int LineSearchSteps { get; init; } = 10;
    public double LineSearchAcceptRatio { get; init; } = 0.1;
    public double LineSearchBacktrackCoeff { get; init; } = 0.8;

    public int StepsPerUpdate { get; init; } = 2048;
    public int ValueIterations { get; init; } = 5;
    public ILossFunction<T> ValueLossFunction { get; init; } = new MeanSquaredErrorLoss<T>();
    public List<int> PolicyHiddenLayers { get; init; } = new List<int> { 64, 64 };
    public List<int> ValueHiddenLayers { get; init; } = new List<int> { 64, 64 };

    /// <summary>
    /// The optimizer used for updating network parameters. If null, Adam optimizer will be used by default.
    /// </summary>
    public IOptimizer<T, Vector<T>, Vector<T>>? Optimizer { get; init; }

    public TRPOOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        ValueLearningRate = numOps.FromDouble(0.001);
        GaeLambda = numOps.FromDouble(0.95);
        MaxKL = numOps.FromDouble(0.01);
    }
}
