using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Interfaces;

namespace AiDotNet.ReinforcementLearning.Interfaces;

/// <summary>
/// Represents a reinforcement learning agent that can interact with an environment and learn from experience.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// An RL agent is an entity that learns to make decisions by interacting with an environment.
/// It observes states, takes actions, receives rewards, and updates its policy or value function
/// to improve future performance. This interface provides the core methods that all RL agents should implement.
/// </para>
/// <para><b>For Beginners:</b> Think of an RL agent as a learner or player in a game.
///
/// Key responsibilities of an agent:
/// - Observe: Look at the current situation (state)
/// - Decide: Choose what action to take
/// - Learn: Update its knowledge based on what happened
/// - Improve: Get better at making decisions over time
///
/// For example, an agent learning to play chess:
/// - Observes the board position
/// - Decides which move to make
/// - Learns from the outcome (win, loss, or draw)
/// - Improves its strategy for future games
/// </para>
/// </remarks>
public interface IRLAgent<T>
{
    /// <summary>
    /// Selects an action to take given the current state.
    /// </summary>
    /// <param name="state">The current state of the environment.</param>
    /// <param name="training">Whether the agent is in training mode (may use exploration) or evaluation mode (purely greedy).</param>
    /// <returns>The index of the selected action.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the agent's policy - how it maps states to actions. During training,
    /// the agent typically uses an exploration strategy (like epsilon-greedy) to discover new strategies.
    /// During evaluation, it usually acts greedily by always selecting the best known action.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the agent decides what to do next.
    ///
    /// The decision process differs based on the mode:
    /// - Training mode: Sometimes tries random actions to explore and discover new strategies
    /// - Evaluation mode: Always picks the best action it knows about
    ///
    /// Think of it like learning to cook:
    /// - While learning (training): Sometimes try new ingredient combinations to experiment
    /// - When cooking for guests (evaluation): Use your best known recipes
    ///
    /// This balance between exploration (trying new things) and exploitation (using what works)
    /// is crucial for effective learning.
    /// </para>
    /// </remarks>
    int SelectAction(Tensor<T> state, bool training = true);

    /// <summary>
    /// Updates the agent's knowledge based on an experience or batch of experiences.
    /// </summary>
    /// <returns>The training loss or performance metric from this update.</returns>
    /// <remarks>
    /// <para>
    /// This method performs one step of learning, updating the agent's policy or value function.
    /// The exact learning mechanism depends on the algorithm (e.g., Q-learning updates, policy gradient updates).
    /// The returned value typically represents the loss or error, which can be used to monitor training progress.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the agent learns from its experiences.
    ///
    /// During training:
    /// - The agent reviews what happened (experiences from memory)
    /// - Figures out what it should have done differently
    /// - Updates its internal decision-making process
    /// - The returned loss shows how much it needs to improve
    ///
    /// Think of it like studying:
    /// - Review your past quizzes (experiences)
    /// - Figure out what you got wrong
    /// - Update your understanding
    /// - The loss is like your error rate - you want it to decrease over time
    ///
    /// Lower loss means the agent is making better predictions and decisions.
    /// </para>
    /// </remarks>
    T Train();

    /// <summary>
    /// Stores an experience for later learning.
    /// </summary>
    /// <param name="state">The state before taking the action.</param>
    /// <param name="action">The action that was taken.</param>
    /// <param name="reward">The reward received.</param>
    /// <param name="nextState">The state after taking the action.</param>
    /// <param name="done">Whether the episode ended.</param>
    /// <remarks>
    /// <para>
    /// This method adds a new experience tuple to the agent's memory (typically a replay buffer).
    /// These stored experiences are later sampled for training. Not all algorithms use experience replay,
    /// but many off-policy algorithms like DQN rely heavily on it.
    /// </para>
    /// <para><b>For Beginners:</b> This saves a memory of what just happened.
    ///
    /// Each time the agent acts, it remembers:
    /// - What the situation was (state)
    /// - What it decided to do (action)
    /// - What feedback it got (reward)
    /// - What the new situation is (nextState)
    /// - Whether the task ended (done)
    ///
    /// Think of it like keeping a diary:
    /// - "When I was at position X (state)"
    /// - "I moved right (action)"
    /// - "I got 10 points (reward)"
    /// - "Now I'm at position Y (nextState)"
    /// - "And the level ended (done = true)"
    ///
    /// Later, the agent can review these diary entries to learn what works and what doesn't.
    /// </para>
    /// </remarks>
    void StoreExperience(Tensor<T> state, int action, T reward, Tensor<T> nextState, bool done);

    /// <summary>
    /// Resets the agent's episode-specific state (if any).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method is called at the start of each episode to reset any episode-specific state.
    /// For example, recurrent neural networks might need to reset their hidden states.
    /// Many agents don't need episode-specific state and can leave this method empty.
    /// </para>
    /// <para><b>For Beginners:</b> This prepares the agent for a new episode or game.
    ///
    /// Why reset:
    /// - Some agents have "memory" that carries over between steps
    /// - When starting a new episode, this memory should be cleared
    /// - It's like clearing your mind before starting a new round of a game
    ///
    /// Examples of what might be reset:
    /// - Recurrent neural network hidden states
    /// - Episode-specific counters or statistics
    /// - Temporary buffers
    ///
    /// Many simple agents don't need this, but it's important for agents with
    /// short-term memory mechanisms.
    /// </para>
    /// </remarks>
    void Reset();

    /// <summary>
    /// Saves the agent's learned policy or value function to disk.
    /// </summary>
    /// <param name="filepath">The path where the agent should be saved.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the agent's learned parameters (neural network weights, Q-tables, etc.)
    /// to disk. This allows you to train once and reuse the agent later, or to pause and resume training.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the agent's learned knowledge to a file.
    ///
    /// Why save an agent:
    /// - Training can take hours or days - you don't want to lose that!
    /// - You can load it later to use or continue training
    /// - You can share trained agents with others
    /// - You can keep backups of good versions
    ///
    /// Think of it like saving your game progress:
    /// - All the skills and knowledge the agent learned are written to disk
    /// - Later, you can load it and pick up right where you left off
    /// - No need to retrain from scratch
    /// </para>
    /// </remarks>
    void Save(string filepath);

    /// <summary>
    /// Loads a previously saved agent from disk.
    /// </summary>
    /// <param name="filepath">The path from which to load the agent.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes a previously saved agent from disk, restoring its learned parameters.
    /// After loading, the agent can continue training or be used for evaluation.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a previously saved agent from a file.
    ///
    /// Why load an agent:
    /// - Continue training from where you left off
    /// - Use a fully trained agent for production
    /// - Compare different versions of your agent
    /// - Share and reuse agents trained by others
    ///
    /// Think of it like loading a saved game:
    /// - All the skills and knowledge are restored
    /// - The agent is ready to use immediately
    /// - You can keep training or just use it as-is
    /// </para>
    /// </remarks>
    void Load(string filepath);
}
