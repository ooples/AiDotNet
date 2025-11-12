using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Interfaces;

/// <summary>
/// Represents a policy that maps states to action probabilities or selections.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// A policy is a strategy that defines how an agent selects actions given states. It can be:
/// - Deterministic: Always selects the same action for a given state
/// - Stochastic: Samples actions from a probability distribution
/// - Exploratory: Includes mechanisms to try new actions (e.g., epsilon-greedy)
///
/// Policies are central to reinforcement learning, as the goal is to learn an optimal policy
/// that maximizes expected cumulative reward.
/// </para>
/// <para><b>For Beginners:</b> A policy is the agent's decision-making strategy or rulebook.
///
/// Think of a policy as answering: "What should I do in this situation?"
///
/// Types of policies:
/// - Deterministic: Always makes the same choice (like always taking the shortest path)
/// - Stochastic: Sometimes makes different choices (like occasionally trying a different route)
/// - Exploratory: Intentionally tries new things sometimes to learn
///
/// For example, in a maze:
/// - A deterministic policy might always turn right at a junction
/// - A stochastic policy might turn right 70% of the time and left 30%
/// - An exploratory policy might usually follow the known best path but occasionally try a random direction
///
/// The agent's goal is to learn the policy that leads to the highest rewards over time.
/// </para>
/// </remarks>
public interface IPolicy<T>
{
    /// <summary>
    /// Selects an action based on the current state and Q-values.
    /// </summary>
    /// <param name="state">The current state of the environment.</param>
    /// <param name="qValues">The Q-values for each possible action (optional, may be null for some policies).</param>
    /// <returns>The index of the selected action.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the policy's action selection mechanism. The exact behavior depends
    /// on the policy type (e.g., epsilon-greedy, softmax, random). Some policies use Q-values
    /// to inform their decision, while others might use different criteria.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the policy decides which action to take.
    ///
    /// How it works:
    /// - Takes in the current situation (state)
    /// - Optionally takes in quality scores for each action (Q-values)
    /// - Returns which action to take (as a number/index)
    ///
    /// Different policies make this decision differently:
    /// - Greedy: Always pick the action with highest Q-value
    /// - Epsilon-Greedy: Usually pick best, sometimes pick random
    /// - Softmax: Pick actions randomly but favor higher Q-values
    /// - Random: Just pick any action randomly
    ///
    /// The Q-values represent how good each action is expected to be.
    /// Think of them as scores - higher is better.
    /// </para>
    /// </remarks>
    int SelectAction(Tensor<T> state, Tensor<T>? qValues = null);

    /// <summary>
    /// Gets the probability distribution over actions for the given state and Q-values.
    /// </summary>
    /// <param name="state">The current state of the environment.</param>
    /// <param name="qValues">The Q-values for each possible action (optional).</param>
    /// <returns>A tensor where each element represents the probability of taking that action.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the full probability distribution over actions. For deterministic policies,
    /// this will be a one-hot vector. For stochastic policies, this represents the sampling probabilities.
    /// This is useful for policy gradient algorithms that need to compute the gradient with respect to
    /// the policy parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This shows the chances of picking each action.
    ///
    /// What you get back:
    /// - A list of numbers, one for each possible action
    /// - Each number is between 0 and 1
    /// - They add up to 1 (or 100% if you think of them as percentages)
    /// - Higher number = more likely to pick that action
    ///
    /// Examples:
    /// - [1.0, 0.0, 0.0, 0.0] means "definitely pick action 0"
    /// - [0.25, 0.25, 0.25, 0.25] means "equal chance of picking any action"
    /// - [0.7, 0.2, 0.05, 0.05] means "strongly favor action 0"
    ///
    /// This is useful for:
    /// - Understanding what the agent is thinking
    /// - Training algorithms that need probability information
    /// - Debugging policy behavior
    /// </para>
    /// </remarks>
    Tensor<T> GetActionProbabilities(Tensor<T> state, Tensor<T>? qValues = null);

    /// <summary>
    /// Updates the policy parameters (e.g., decreases exploration rate over time).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method allows the policy to adapt over time. For example, an epsilon-greedy policy
    /// might decrease its exploration rate (epsilon) as training progresses. This implements
    /// the common pattern of starting with high exploration and gradually shifting to exploitation
    /// as the agent learns more about the environment.
    /// </para>
    /// <para><b>For Beginners:</b> This adjusts how the policy makes decisions as training progresses.
    ///
    /// Common adjustments:
    /// - Decrease exploration: Try random actions less often as the agent learns
    /// - Adjust temperature: Make action selection more or less random
    /// - Adapt to performance: Change behavior based on how well the agent is doing
    ///
    /// Why adjust over time:
    /// - Early in training: Explore a lot to discover what works
    /// - Later in training: Exploit what you've learned more
    /// - Eventually: Mostly or entirely exploit the best known strategy
    ///
    /// Think of it like learning to play a game:
    /// - At first, try all sorts of random things to see what happens
    /// - As you learn, stick more to strategies that work
    /// - Eventually, play your best strategy most or all of the time
    ///
    /// This method is typically called periodically during training (e.g., after each episode).
    /// </para>
    /// </remarks>
    void Update();
}
