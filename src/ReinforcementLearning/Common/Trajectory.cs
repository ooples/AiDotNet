using System.Linq;
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
    /// Whether each step was terminal: the episode reached a terminal state, so nothing is bootstrapped after it.
    /// </summary>
    public List<bool> Dones { get; init; }

    /// <summary>
    /// Whether each step was truncated: the episode was cut off (a time limit or the end of the rollout) in a
    /// non-terminal state, so the return after it is bootstrapped from <see cref="BootstrapValues"/>.
    /// </summary>
    /// <remarks>
    /// Gymnasium's terminated/truncated split; Stable-Baselines3 and CleanRL bootstrap V(s_{t+1}) on truncation
    /// and not on termination.
    /// </remarks>
    public List<bool> Truncations { get; init; }

    /// <summary>
    /// V(s_{t+1}) for each truncated step (zero for the other steps).
    /// </summary>
    public List<T> BootstrapValues { get; init; }

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
        Truncations = new List<bool>();
        BootstrapValues = new List<T>();
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
        Truncations.Add(false);
        BootstrapValues.Add(MathHelper.GetNumericOperations<T>().Zero);
    }

    /// <summary>
    /// Adds a step that ended its episode by termination, by truncation, or neither.
    /// </summary>
    /// <param name="state">The state before the action.</param>
    /// <param name="action">The action taken.</param>
    /// <param name="reward">The reward received.</param>
    /// <param name="value">V(s_t).</param>
    /// <param name="logProb">log pi(a_t | s_t).</param>
    /// <param name="terminated">Whether the episode reached a terminal state.</param>
    /// <param name="truncated">Whether the episode was cut off in a non-terminal state.</param>
    /// <param name="bootstrapValue">V(s_{t+1}), used only when <paramref name="truncated"/> is set.</param>
    public void AddStep(Vector<T> state, Vector<T> action, T reward, T value, T logProb, bool terminated, bool truncated,
        T bootstrapValue)
    {
        States.Add(state);
        Actions.Add(action);
        Rewards.Add(reward);
        Values.Add(value);
        LogProbs.Add(logProb);
        Dones.Add(terminated);
        bool bootstraps = truncated && !terminated;
        Truncations.Add(bootstraps);
        BootstrapValues.Add(bootstraps ? bootstrapValue : MathHelper.GetNumericOperations<T>().Zero);
    }

    /// <summary>
    /// Marks the last step as truncated with the given V(s_{t+1}) when it ended neither by termination nor by
    /// truncation: the rollout itself cut the episode off there.
    /// </summary>
    public void TruncateLastStep(T bootstrapValue)
    {
        int last = States.Count - 1;
        if (last < 0 || Dones[last] || Truncations[last])
        {
            return;
        }

        Truncations[last] = true;
        BootstrapValues[last] = bootstrapValue;
    }

    /// <summary>
    /// V'_t per step, the value bootstrapped after step t: zero after a terminal step, the stored V(s_{t+1}) after a
    /// truncated one, and V of the next stored step otherwise.
    /// </summary>
    public Vector<T> NextValues()
    {
        var engine = AiDotNetEngine.Current;
        int n = States.Count;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var values = Vector<T>.FromEnumerable(Values);
        var lastZero = Vector<T>.CreateDefault(1, ops.Zero).AsTensor();
        var following = n == 1
            ? lastZero
            : engine.TensorConcatenate([engine.TensorSlice(values.AsTensor(), [1], [n - 1]), lastZero], 0);
        var bootstrapped = engine.TensorWhere(Truncations.ToArray(), Vector<T>.FromEnumerable(BootstrapValues).AsTensor(), following);
        return engine.TensorWhere(Dones.ToArray(), Vector<T>.CreateDefault(n, ops.Zero).AsTensor(), bootstrapped).ToVector();
    }

    /// <summary>Whether the return chain stops after each step: a terminal or truncated step, or the last one.</summary>
    public bool[] ChainCuts()
    {
        var cuts = Dones.Zip(Truncations, (terminated, truncated) => terminated || truncated).ToArray();
        if (cuts.Length > 0)
        {
            cuts[cuts.Length - 1] = true;
        }

        return cuts;
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
        Truncations.Clear();
        BootstrapValues.Clear();
        Advantages = null;
        Returns = null;
    }
}
