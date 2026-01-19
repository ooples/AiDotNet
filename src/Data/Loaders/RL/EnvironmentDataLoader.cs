using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Data.Loaders.RL;

/// <summary>
/// Data loader for reinforcement learning that wraps an environment for experience collection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// EnvironmentDataLoader provides a clean facade for RL training by wrapping an environment
/// and managing experience collection. Use this with AiModelBuilder for unified training.
/// </para>
/// <para><b>For Beginners:</b> This is the main way to set up RL training:
///
/// <code>
/// var result = await new AiModelBuilder&lt;double, Vector&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureDataLoader(new EnvironmentDataLoader&lt;double&gt;(
///         environment: new CartPoleEnvironment&lt;double&gt;(),
///         episodes: 1000))
///     .ConfigureModel(new DQNAgent&lt;double&gt;(options))
///     .BuildAsync();
/// </code>
///
/// **What it does:**
/// - Wraps your RL environment
/// - Creates a replay buffer for storing experiences
/// - Manages episode running during training
/// - Provides experience batches for learning
/// </para>
/// </remarks>
public class EnvironmentDataLoader<T> : RLDataLoaderBase<T>
{
    private readonly string _environmentName;

    /// <inheritdoc/>
    public override string Name => $"EnvironmentDataLoader({_environmentName})";

    /// <summary>
    /// Initializes a new EnvironmentDataLoader with the specified environment.
    /// </summary>
    /// <param name="environment">The RL environment to collect experiences from.</param>
    /// <param name="episodes">Number of episodes to run during training.</param>
    /// <param name="maxStepsPerEpisode">Maximum steps per episode (prevents infinite loops).</param>
    /// <param name="replayBufferCapacity">Size of the replay buffer for storing experiences.</param>
    /// <param name="minExperiencesBeforeTraining">Minimum experiences before training begins.</param>
    /// <param name="verbose">Whether to print progress during training.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <b>For Beginners:</b> The default values work well for most environments:
    /// - 1000 episodes is enough for simple environments, increase for complex ones
    /// - 500 max steps prevents infinite loops in broken environments
    /// - 10000 buffer size provides good experience diversity
    /// - 1000 min experiences ensures some exploration before learning
    /// </remarks>
    public EnvironmentDataLoader(
        IEnvironment<T> environment,
        int episodes = 1000,
        int maxStepsPerEpisode = 500,
        int replayBufferCapacity = 10000,
        int minExperiencesBeforeTraining = 1000,
        bool verbose = true,
        int? seed = null)
        : base(
            environment,
            new UniformReplayBuffer<T, Vector<T>, Vector<T>>(replayBufferCapacity, seed),
            episodes,
            maxStepsPerEpisode,
            minExperiencesBeforeTraining,
            verbose,
            seed)
    {
        _environmentName = environment.GetType().Name;
    }

    /// <summary>
    /// Initializes a new EnvironmentDataLoader with a custom replay buffer.
    /// </summary>
    /// <param name="environment">The RL environment to collect experiences from.</param>
    /// <param name="replayBuffer">Custom replay buffer implementation (e.g., prioritized).</param>
    /// <param name="episodes">Number of episodes to run during training.</param>
    /// <param name="maxStepsPerEpisode">Maximum steps per episode.</param>
    /// <param name="minExperiencesBeforeTraining">Minimum experiences before training begins.</param>
    /// <param name="verbose">Whether to print progress during training.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Use this constructor if you want to provide a custom replay buffer,
    /// such as a prioritized experience replay buffer that samples important experiences more often.
    /// </remarks>
    public EnvironmentDataLoader(
        IEnvironment<T> environment,
        IReplayBuffer<T, Vector<T>, Vector<T>> replayBuffer,
        int episodes = 1000,
        int maxStepsPerEpisode = 500,
        int minExperiencesBeforeTraining = 1000,
        bool verbose = true,
        int? seed = null)
        : base(
            environment,
            replayBuffer,
            episodes,
            maxStepsPerEpisode,
            minExperiencesBeforeTraining,
            verbose,
            seed)
    {
        _environmentName = environment.GetType().Name;
    }
}
