namespace AiDotNet.Interfaces;

/// <summary>
/// Exposes the state-conditional action-value function Q(s, ·) of a value-based agent,
/// independent of the greedy argmax projection applied by <c>SelectAction</c>/<c>Predict</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Value-based reinforcement-learning agents (DQN, Double DQN, Dueling DQN, …) choose an action by
/// taking the argmax over a learned action-value function Q(s, ·). The argmax is a lossy projection:
/// at random initialization it can be constant over the entire input domain (one action's value
/// dominating everywhere) even though the underlying Q-function is genuinely state-conditional.
/// </para>
/// <para><b>For Beginners:</b>
/// The agent scores every possible action for a given state — those scores are the "action values"
/// (Q-values). Normally you only see the single best action it picked. This method lets you see the
/// raw scores, which always respond to the input state, making it possible to verify the policy is
/// actually paying attention to the state rather than blindly returning one fixed action.
/// </para>
/// </remarks>
public interface IActionValueProvider<T>
{
    /// <summary>
    /// Returns the estimated value of each available action for the given state.
    /// </summary>
    /// <param name="state">The state to evaluate.</param>
    /// <returns>A vector of action values (one entry per action).</returns>
    Vector<T> GetActionValues(Vector<T> state);
}
