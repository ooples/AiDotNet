using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Common;

/// <summary>
/// Represents a trajectory of experience for on-policy RL algorithms (PPO, A2C, etc.).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A trajectory is a sequence of states, actions, and rewards collected by an agent
/// interacting with an environment. Unlike experience replay (used in DQN), trajectories
/// are used immediately for training and then discarded in on-policy algorithms.
/// </para>
/// <para><b>For Beginners:</b>
/// A trajectory is like recording a game session. It contains:
/// - Every state you saw
/// - Every action you took
/// - Every reward you got
/// - Additional info (value estimates, action probabilities)
///
/// On-policy algorithms like PPO collect these trajectories, learn from them immediately,
/// then throw them away and collect new ones. This is different from DQN which stores
/// experiences in a replay buffer and samples from them multiple times.
/// </para>
/// </remarks>
public class Trajectory<T>
{
    /// <summary>
    /// States observed during the trajectory.
    /// </summary>
    public List<Vector<T>> States { get; init; }

    /// <summary>
    /// Actions taken during the trajectory.
    /// </summary>
    public List<Vector<T>> Actions { get; init; }

    /// <summary>
    /// Rewards received during the trajectory.
    /// </summary>
    public List<T> Rewards { get; init; }

    /// <summary>
    /// Value estimates for each state (from critic).
    /// </summary>
    public List<T> Values { get; init; }

    /// <summary>
    /// Log probabilities of actions taken (for policy gradient).
    /// </summary>
    public List<T> LogProbs { get; init; }

    /// <summary>
    /// Whether each step was terminal (episode ended).
    /// </summary>
    public List<bool> Dones { get; init; }

    /// <summary>
    /// Computed advantages (used during training).
    /// </summary>
    public List<T>? Advantages { get; set; }

    /// <summary>
    /// Computed returns (discounted sum of rewards).
    /// </summary>
    public List<T>? Returns { get; set; }

    /// <summary>
    /// Initializes an empty trajectory.
    /// </summary>
    public Trajectory()
    {
        States = new List<Vector<T>>();
        Actions = new List<Vector<T>>();
        Rewards = new List<T>();
        Values = new List<T>();
        LogProbs = new List<T>();
        Dones = new List<bool>();
    }

    /// <summary>
    /// Adds a step to the trajectory.
    /// </summary>
    public void AddStep(Vector<T> state, Vector<T> action, T reward, T value, T logProb, bool done)
    {
        States.Add(state);
        Actions.Add(action);
        Rewards.Add(reward);
        Values.Add(value);
        LogProbs.Add(logProb);
        Dones.Add(done);
    }

    /// <summary>
    /// Gets the number of steps in the trajectory.
    /// </summary>
    public int Length => States.Count;

    /// <summary>
    /// Clears the trajectory.
    /// </summary>
    public void Clear()
    {
        States.Clear();
        Actions.Clear();
        Rewards.Clear();
        Values.Clear();
        LogProbs.Clear();
        Dones.Clear();
        Advantages = null;
        Returns = null;
    }
}
