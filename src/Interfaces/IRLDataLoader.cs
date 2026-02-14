using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.ReplayBuffers;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for data loaders that provide experience data for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This interface is for reinforcement learning scenarios where an agent interacts with an
/// environment to collect experience data. The loader manages:
/// - Environment interactions (stepping through episodes)
/// - Experience collection and storage
/// - Replay buffer management for batch sampling
/// </para>
/// <para><b>For Beginners:</b> Reinforcement learning is learning through trial and error.
///
/// **How it works:**
/// - An agent takes actions in an environment
/// - The environment returns rewards and new states
/// - The agent learns to maximize total rewards
///
/// **Example: Game Playing**
/// - Environment: The game (e.g., CartPole, Atari, Chess)
/// - State: What the agent sees (game screen, piece positions)
/// - Action: What the agent does (move left, jump, place piece)
/// - Reward: Score or outcome (+1 for winning, -1 for losing)
///
/// **This data loader:**
/// - Runs episodes in the environment
/// - Collects experience tuples (state, action, reward, next_state, done)
/// - Stores them in a replay buffer for training
/// - Provides batches of experiences for learning
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("RLDataLoader")]
public interface IRLDataLoader<T> : IDataLoader<T>, IBatchIterable<Experience<T, Vector<T>, Vector<T>>>
{
    /// <summary>
    /// Gets the environment that the agent interacts with.
    /// </summary>
    IEnvironment<T> Environment { get; }

    /// <summary>
    /// Gets the total number of episodes to run during training.
    /// </summary>
    int Episodes { get; }

    /// <summary>
    /// Gets the maximum number of steps per episode (prevents infinite episodes).
    /// </summary>
    int MaxStepsPerEpisode { get; }

    /// <summary>
    /// Gets whether to print training progress to console.
    /// </summary>
    bool Verbose { get; }

    /// <summary>
    /// Gets the replay buffer used for storing and sampling experiences.
    /// </summary>
    IReplayBuffer<T, Vector<T>, Vector<T>> ReplayBuffer { get; }

    /// <summary>
    /// Gets the minimum number of experiences required before training can begin.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> We need some experiences before we can learn from random samples.
    /// This ensures the replay buffer has enough diverse experiences for effective learning.
    /// </remarks>
    int MinExperiencesBeforeTraining { get; }

    /// <summary>
    /// Gets the current episode number (0-indexed).
    /// </summary>
    int CurrentEpisode { get; }

    /// <summary>
    /// Gets the total number of steps taken across all episodes.
    /// </summary>
    int TotalSteps { get; }

    /// <summary>
    /// Samples a batch of experiences from the replay buffer.
    /// </summary>
    /// <param name="batchSize">Number of experiences to sample.</param>
    /// <returns>List of sampled experiences for training.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Instead of learning from experiences in order (which can cause issues),
    /// we randomly sample from past experiences. This makes learning more stable.
    /// </remarks>
    IReadOnlyList<Experience<T, Vector<T>, Vector<T>>> SampleBatch(int batchSize);

    /// <summary>
    /// Checks if there are enough experiences to begin training.
    /// </summary>
    /// <param name="batchSize">The desired batch size for training.</param>
    /// <returns>True if training can proceed, false if more experiences are needed.</returns>
    bool CanTrain(int batchSize);

    /// <summary>
    /// Runs a single episode and collects experiences.
    /// </summary>
    /// <param name="agent">Optional agent to use for action selection. If null, uses random actions.</param>
    /// <returns>Episode result containing total reward, steps, and whether it was successful.</returns>
    EpisodeResult<T> RunEpisode(IRLAgent<T>? agent = null);

    /// <summary>
    /// Runs multiple episodes and collects experiences.
    /// </summary>
    /// <param name="numEpisodes">Number of episodes to run.</param>
    /// <param name="agent">Optional agent to use for action selection.</param>
    /// <returns>List of episode results.</returns>
    IReadOnlyList<EpisodeResult<T>> RunEpisodes(int numEpisodes, IRLAgent<T>? agent = null);

    /// <summary>
    /// Adds an experience to the replay buffer.
    /// </summary>
    /// <param name="experience">The experience to add.</param>
    void AddExperience(Experience<T, Vector<T>, Vector<T>> experience);

    /// <summary>
    /// Resets the data loader state (clears buffer, resets counters).
    /// </summary>
    void ResetTraining();

    /// <summary>
    /// Sets the random seed for reproducible training.
    /// </summary>
    /// <param name="seed">Random seed value.</param>
    void SetSeed(int seed);
}

/// <summary>
/// Result of running a single RL episode.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This contains statistics about one complete episode:
/// - How much total reward the agent earned
/// - How many steps it took
/// - Whether it succeeded (depends on environment definition)
/// </remarks>
public record EpisodeResult<T>(
    int EpisodeNumber,
    T TotalReward,
    int Steps,
    bool Completed,
    bool Success,
    TimeSpan Duration)
{
    /// <summary>
    /// Gets whether the episode ended due to reaching max steps (truncated).
    /// </summary>
    public bool Truncated => !Completed && Steps > 0;
}
