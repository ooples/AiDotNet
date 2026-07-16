using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.IntrinsicMotivation;

/// <summary>
/// Produces an intrinsic (curiosity) reward for a state, driving a reinforcement-learning agent toward
/// novel states even when the environment's extrinsic reward is sparse.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Some environments only reward the agent rarely (e.g. at the very end of a
/// long task). Without a signal in between, the agent flails. An intrinsic-reward module gives the agent
/// a small bonus for visiting <i>unfamiliar</i> states, so it keeps exploring purposefully until it finds
/// the real reward. The bonus shrinks as a state becomes familiar.</para>
/// </remarks>
public interface IIntrinsicRewardModule<T>
{
    /// <summary>
    /// Returns the intrinsic novelty reward for <paramref name="state"/> (larger = more novel). Does not
    /// change the module's estimate — call <see cref="Update"/> to learn from a visited state.
    /// </summary>
    T ComputeIntrinsicReward(Vector<T> state);

    /// <summary>Learns from a visited state so it becomes less novel next time.</summary>
    void Update(Vector<T> state);

    /// <summary>Resets any per-episode state.</summary>
    void Reset();
}
