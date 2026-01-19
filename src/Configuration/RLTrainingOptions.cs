using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Policies.Exploration;
using AiDotNet.ReinforcementLearning.ReplayBuffers;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for reinforcement learning training loops via AiModelBuilder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This class provides comprehensive configuration for RL training loops, following industry-standard
/// patterns from libraries like Stable-Baselines3, RLlib, and CleanRL.
/// </para>
/// <para>
/// <b>Note:</b> This class is for configuring the training loop (episodes, steps, callbacks).
/// For agent-specific options (learning rate, discount factor), see each agent's options class.
/// </para>
/// <para><b>For Beginners:</b> Reinforcement learning trains an agent through trial and error
/// in an environment. This options class lets you customize every aspect of that training process:
/// - How many episodes to run
/// - How to explore vs exploit
/// - How to store and sample experiences
/// - When to receive progress updates
///
/// **Quick Start Example:**
/// <code>
/// var options = new RLTrainingOptions&lt;double&gt;
/// {
///     Environment = new CartPoleEnvironment&lt;double&gt;(),
///     Episodes = 1000,
///     MaxStepsPerEpisode = 500
/// };
///
/// var result = await new AiModelBuilder&lt;double, Vector&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureReinforcementLearning(options)
///     .ConfigureModel(new DQNAgent&lt;double&gt;(agentOptions))
///     .BuildAsync();
/// </code>
/// </para>
/// </remarks>
public class RLTrainingOptions<T>
{
    /// <summary>
    /// Gets or sets the environment for the agent to interact with.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The environment is the "world" where your agent learns.
    /// It could be a game, simulation, or any system with states, actions, and rewards.
    /// </remarks>
    public IEnvironment<T>? Environment { get; set; }

    /// <summary>
    /// Gets or sets the number of episodes to train for.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> An episode is one complete run through the environment
    /// from start to finish (or until max steps). More episodes generally means better learning.
    /// Default: 1000 episodes.
    /// </remarks>
    public int Episodes { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the maximum steps per episode to prevent infinite loops.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Some environments might never end naturally.
    /// This limit ensures episodes don't run forever.
    /// Default: 500 steps per episode.
    /// </remarks>
    public int MaxStepsPerEpisode { get; set; } = 500;

    /// <summary>
    /// Gets or sets the number of initial random steps before training begins.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Before the agent starts learning, it's helpful to fill
    /// the replay buffer with some random experiences. This provides diverse starting data.
    /// Default: 1000 warmup steps.
    /// </remarks>
    public int WarmupSteps { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the frequency of training updates (every N steps).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The agent doesn't have to learn after every single step.
    /// Training every N steps can be more efficient. Set to 1 for training every step.
    /// Default: 1 (train every step).
    /// </remarks>
    public int TrainFrequency { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of gradient steps per training update.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Each training update can perform multiple gradient descent steps.
    /// More gradient steps can speed up learning but may cause instability.
    /// Default: 1 gradient step per update.
    /// </remarks>
    public int GradientSteps { get; set; } = 1;

    /// <summary>
    /// Gets or sets the batch size for sampling from the replay buffer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> When learning, the agent samples a batch of past experiences.
    /// Larger batches give more stable gradients but use more memory.
    /// Default: 64 experiences per batch.
    /// </remarks>
    public int BatchSize { get; set; } = 64;

    /// <summary>
    /// Gets or sets the optional exploration strategy to use during training.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Exploration strategies help the agent try new things instead
    /// of always doing what it thinks is best. Common strategies:
    /// - EpsilonGreedy: Random action with probability epsilon
    /// - Boltzmann: Softmax over Q-values
    /// - GaussianNoise: Add noise to continuous actions
    /// If null, the agent's default exploration is used.
    /// </remarks>
    public IExplorationStrategy<T>? ExplorationStrategy { get; set; }

    /// <summary>
    /// Gets or sets the optional replay buffer for experience storage.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A replay buffer stores past experiences for learning.
    /// If null, the agent's internal buffer is used. You can provide:
    /// - UniformReplayBuffer: All experiences equally likely
    /// - PrioritizedReplayBuffer: Important experiences sampled more often
    /// </remarks>
    public IReplayBuffer<T, Vector<T>, Vector<T>>? ReplayBuffer { get; set; }

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Setting a seed makes training reproducible -
    /// you'll get the same results if you run training again with the same seed.
    /// If null, results will vary between runs.
    /// </remarks>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets whether to normalize observations.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Normalizing observations (scaling them to similar ranges)
    /// often helps neural networks learn faster and more stably.
    /// Default: false.
    /// </remarks>
    public bool NormalizeObservations { get; set; }

    /// <summary>
    /// Gets or sets whether to normalize rewards.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Normalizing rewards can help when reward scales vary
    /// widely during training. Default: false.
    /// </remarks>
    public bool NormalizeRewards { get; set; }

    /// <summary>
    /// Gets or sets the callback invoked after each episode completes.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This callback lets you monitor training progress.
    /// It receives detailed metrics about the completed episode.
    ///
    /// Example:
    /// <code>
    /// options.OnEpisodeComplete = (metrics) =>
    /// {
    ///     Console.WriteLine($"Episode {metrics.Episode}: Reward = {metrics.TotalReward}");
    /// };
    /// </code>
    /// </remarks>
    public Action<RLEpisodeMetrics<T>>? OnEpisodeComplete { get; set; }

    /// <summary>
    /// Gets or sets the callback invoked after each training step.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This callback fires more frequently (every step or training update).
    /// Useful for detailed logging or progress bars.
    /// </remarks>
    public Action<RLStepMetrics<T>>? OnStepComplete { get; set; }

    /// <summary>
    /// Gets or sets the callback invoked when training starts.
    /// </summary>
    public Action? OnTrainingStart { get; set; }

    /// <summary>
    /// Gets or sets the callback invoked when training ends.
    /// </summary>
    /// <remarks>
    /// Receives the final training summary with aggregated metrics.
    /// </remarks>
    public Action<RLTrainingSummary<T>>? OnTrainingComplete { get; set; }

    /// <summary>
    /// Gets or sets how often to log progress (every N episodes).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Set to 0 to disable automatic console logging.
    /// Set to 10 to log every 10 episodes, etc.
    /// Default: 10 (log every 10 episodes).
    /// </remarks>
    public int LogFrequency { get; set; } = 10;

    /// <summary>
    /// Gets or sets the checkpoint configuration for saving models during training.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Checkpointing saves your model periodically during training.
    /// This protects against crashes and lets you resume training later.
    /// </remarks>
    public RLCheckpointConfig? CheckpointConfig { get; set; }

    /// <summary>
    /// Gets or sets the early stopping configuration.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Early stopping automatically stops training when
    /// the agent stops improving, preventing overfitting and saving time.
    /// </remarks>
    public RLEarlyStoppingConfig<T>? EarlyStoppingConfig { get; set; }

    /// <summary>
    /// Gets or sets the target network configuration for DQN-family algorithms.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Target networks help stabilize learning in DQN-based algorithms
    /// by providing stable Q-value targets. This prevents the "moving target" problem.
    /// </remarks>
    public TargetNetworkConfig<T>? TargetNetworkConfig { get; set; }

    /// <summary>
    /// Gets or sets the exploration schedule configuration.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This controls how exploration (trying random actions) decreases
    /// over time as the agent learns. Common schedule: start at 1.0 (fully random),
    /// decay to 0.01 (mostly learned policy).
    /// </remarks>
    public ExplorationScheduleConfig<T>? ExplorationSchedule { get; set; }

    /// <summary>
    /// Gets or sets the reward clipping bounds.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Clipping rewards to a range (e.g., -1 to 1) can stabilize training
    /// when raw rewards have very different scales. Used famously in Atari DQN paper.
    /// If null, rewards are not clipped.
    /// </remarks>
    public RewardClippingConfig<T>? RewardClipping { get; set; }

    /// <summary>
    /// Gets or sets the evaluation configuration for assessing agent performance during training.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Evaluation runs the agent without exploration to measure
    /// true performance. This gives you an unbiased estimate of how well the agent learned.
    /// </remarks>
    public RLEvaluationConfig? EvaluationConfig { get; set; }

    /// <summary>
    /// Gets or sets whether to use prioritized experience replay.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Prioritized replay samples important experiences more often.
    /// "Important" usually means experiences with high TD-error (surprising outcomes).
    /// This can speed up learning but adds computational overhead.
    /// Default: false (uniform sampling).
    /// </remarks>
    public bool UsePrioritizedReplay { get; set; }

    /// <summary>
    /// Gets or sets the prioritized replay configuration.
    /// </summary>
    /// <remarks>
    /// Only used if UsePrioritizedReplay is true.
    /// </remarks>
    public PrioritizedReplayConfig<T>? PrioritizedReplayConfig { get; set; }

    /// <summary>
    /// Creates default options with sensible values for most use cases.
    /// </summary>
    /// <param name="environment">The environment to train in.</param>
    /// <returns>Options with recommended defaults.</returns>
    public static RLTrainingOptions<T> Default(IEnvironment<T> environment)
    {
        return new RLTrainingOptions<T>
        {
            Environment = environment,
            Episodes = 1000,
            MaxStepsPerEpisode = 500,
            WarmupSteps = 1000,
            TrainFrequency = 1,
            GradientSteps = 1,
            BatchSize = 64,
            LogFrequency = 10
        };
    }
}
