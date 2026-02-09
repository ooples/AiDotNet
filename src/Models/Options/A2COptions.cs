using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

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
public class A2COptions<T> : ModelOptions
{
    public int StateSize { get; set; }
    public int ActionSize { get; set; }
    public bool IsContinuous { get; set; } = false;
    public T PolicyLearningRate { get; set; }
    public T ValueLearningRate { get; set; }
    public T DiscountFactor { get; set; }
    public T EntropyCoefficient { get; set; }
    public T ValueLossCoefficient { get; set; }
    public int StepsPerUpdate { get; set; } = 5;
    public ILossFunction<T> ValueLossFunction { get; set; } = new MeanSquaredErrorLoss<T>();
    public List<int> PolicyHiddenLayers { get; set; } = new List<int> { 64, 64 };
    public List<int> ValueHiddenLayers { get; set; } = new List<int> { 64, 64 };

    public A2COptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        PolicyLearningRate = numOps.FromDouble(0.0007);
        ValueLearningRate = numOps.FromDouble(0.001);
        DiscountFactor = numOps.FromDouble(0.99);
        EntropyCoefficient = numOps.FromDouble(0.01);
        ValueLossCoefficient = numOps.FromDouble(0.5);
    }
}
