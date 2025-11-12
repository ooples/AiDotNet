using AiDotNet.LossFunctions;

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
public class TRPOOptions<T>
{
    public int StateSize { get; set; }
    public int ActionSize { get; set; }
    public bool IsContinuous { get; set; } = false;
    public T ValueLearningRate { get; set; }
    public T DiscountFactor { get; set; }
    public T GaeLambda { get; set; }

    // TRPO-specific parameters
    public T MaxKL { get; set; }  // Maximum KL divergence (trust region size)
    public double Damping { get; set; } = 0.1;  // Damping coefficient for conjugate gradient
    public int ConjugateGradientIterations { get; set; } = 10;
    public int LineSearchSteps { get; set; } = 10;
    public double LineSearchAcceptRatio { get; set; } = 0.1;
    public double LineSearchBacktrackCoeff { get; set; } = 0.8;

    public int StepsPerUpdate { get; set; } = 2048;
    public int ValueIterations { get; set; } = 5;
    public ILossFunction<T> ValueLossFunction { get; set; } = new MeanSquaredError<T>();
    public List<int> PolicyHiddenLayers { get; set; } = [64, 64];
    public List<int> ValueHiddenLayers { get; set; } = [64, 64];
    public int? Seed { get; set; }

    public TRPOOptions()
    {
        var numOps = NumericOperations<T>.Instance;
        ValueLearningRate = numOps.FromDouble(0.001);
        DiscountFactor = numOps.FromDouble(0.99);
        GaeLambda = numOps.FromDouble(0.95);
        MaxKL = numOps.FromDouble(0.01);
    }
}
