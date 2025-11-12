namespace AiDotNet.ReinforcementLearning.Agents.REINFORCE;

/// <summary>
/// Configuration options for REINFORCE agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// REINFORCE is the simplest policy gradient algorithm. It directly optimizes
/// the policy by following the gradient of expected returns.
/// </para>
/// <para><b>For Beginners:</b>
/// REINFORCE is the "hello world" of policy gradient methods. It's simple but powerful:
/// - Play an entire episode
/// - See which actions led to good rewards
/// - Make those actions more likely in the future
///
/// Think of it like learning to play a game: you play a round, see your score,
/// then adjust your strategy to do better next time.
///
/// Simple, but can be slow to learn and high variance.
/// Modern algorithms like PPO improve on REINFORCE's ideas.
/// </para>
/// </remarks>
public class REINFORCEOptions<T>
{
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public bool IsContinuous { get; init; } = false;
    public T LearningRate { get; init; }
    public T DiscountFactor { get; init; }
    public int[] HiddenLayers { get; init; } = new[] { 32, 32 };
    public int? Seed { get; init; }

    public static REINFORCEOptions<T> Default(int stateSize, int actionSize, T learningRate, T gamma, bool isContinuous = false)
    {
        return new REINFORCEOptions<T>
        {
            StateSize = stateSize,
            ActionSize = actionSize,
            IsContinuous = isContinuous,
            LearningRate = learningRate,
            DiscountFactor = gamma,
            HiddenLayers = new[] { 32, 32 }
        };
    }
}
