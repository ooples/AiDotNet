using AiDotNet.LossFunctions;

namespace AiDotNet.ReinforcementLearning.Agents.A2C;

/// <summary>
/// Configuration options for Advantage Actor-Critic (A2C) agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A2C is a synchronous version of A3C that is simpler and often more sample-efficient.
/// It combines policy gradients with value function learning for stable, efficient training.
/// </para>
/// <para><b>For Beginners:</b>
/// A2C learns two things simultaneously:
/// - **Actor (Policy)**: What action to take in each state
/// - **Critic (Value Function)**: How good each state is
///
/// The critic helps the actor learn faster by providing better feedback than just rewards alone.
/// Think of the critic as a coach giving targeted advice rather than just "good" or "bad".
///
/// A2C is the foundation for many modern RL algorithms including PPO.
/// </para>
/// </remarks>
public class A2COptions<T>
{
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public bool IsContinuous { get; init; } = false;
    public T PolicyLearningRate { get; init; }
    public T ValueLearningRate { get; init; }
    public T DiscountFactor { get; init; }
    public T EntropyCoefficient { get; init; }
    public T ValueLossCoefficient { get; init; }
    public int StepsPerUpdate { get; init; } = 5;
    public ILossFunction<T> ValueLossFunction { get; init; }
    public int[] PolicyHiddenLayers { get; init; } = new[] { 64, 64 };
    public int[] ValueHiddenLayers { get; init; } = new[] { 64, 64 };
    public int? Seed { get; init; }

    public static A2COptions<T> Default(int stateSize, int actionSize, T policyLr, T valueLr, T gamma, bool isContinuous = false)
    {
        var numOps = NumericOperations<T>.Instance;
        return new A2COptions<T>
        {
            StateSize = stateSize,
            ActionSize = actionSize,
            IsContinuous = isContinuous,
            PolicyLearningRate = policyLr,
            ValueLearningRate = valueLr,
            DiscountFactor = gamma,
            EntropyCoefficient = numOps.FromDouble(0.01),
            ValueLossCoefficient = numOps.FromDouble(0.5),
            StepsPerUpdate = 5,
            ValueLossFunction = new MeanSquaredError<T>(),
            PolicyHiddenLayers = new[] { 64, 64 },
            ValueHiddenLayers = new[] { 64, 64 }
        };
    }
}
