using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Abstract base class for RL data loaders providing common reinforcement learning functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// RLDataLoaderBase provides shared implementation for all RL data loaders including:
/// - Environment interaction management
/// - Replay buffer management
/// - Episode running and experience collection
/// - Batch sampling for training
/// </para>
/// <para><b>For Beginners:</b> This base class handles common RL operations:
/// - Stepping through the environment collecting experiences
/// - Storing experiences in a replay buffer
/// - Sampling batches for training
///
/// Concrete implementations extend this to work with specific environments or
/// provide specialized experience collection strategies.
/// </para>
/// </remarks>
public abstract class RLDataLoaderBase<T> : DataLoaderBase<T>, IRLDataLoader<T>
{
    /// <summary>
    /// Numeric operations helper for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IEnvironment<T> _environment;
    private readonly IReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;
    private readonly int _episodes;
    private readonly int _maxStepsPerEpisode;
    private readonly bool _verbose;
    private readonly int _minExperiencesBeforeTraining;

    private int _currentEpisode;
    private int _totalSteps;
    private int _batchSize = 32;
    private int _currentBatchIndex;
    private Random _random;

    /// <summary>
    /// Lock object for thread-safe access to _random during batch generation.
    /// </summary>
    private readonly object _randomLock = new object();

    /// <summary>
    /// Initializes a new instance of the RLDataLoaderBase class.
    /// </summary>
    /// <param name="environment">The RL environment to interact with.</param>
    /// <param name="replayBuffer">The replay buffer for storing experiences.</param>
    /// <param name="episodes">Total number of episodes for training.</param>
    /// <param name="maxStepsPerEpisode">Maximum steps per episode (prevents infinite loops).</param>
    /// <param name="minExperiencesBeforeTraining">Minimum experiences needed before training can start.</param>
    /// <param name="verbose">Whether to print progress to console.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    protected RLDataLoaderBase(
        IEnvironment<T> environment,
        IReplayBuffer<T, Vector<T>, Vector<T>> replayBuffer,
        int episodes = 1000,
        int maxStepsPerEpisode = 500,
        int minExperiencesBeforeTraining = 1000,
        bool verbose = true,
        int? seed = null)
    {
        _environment = environment ?? throw new ArgumentNullException(nameof(environment));
        _replayBuffer = replayBuffer ?? throw new ArgumentNullException(nameof(replayBuffer));
        _episodes = episodes;
        _maxStepsPerEpisode = maxStepsPerEpisode;
        _minExperiencesBeforeTraining = minExperiencesBeforeTraining;
        _verbose = verbose;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        if (seed.HasValue)
        {
            _environment.Seed(seed.Value);
        }
    }

    /// <inheritdoc/>
    public IEnvironment<T> Environment => _environment;

    /// <inheritdoc/>
    public int Episodes => _episodes;

    /// <inheritdoc/>
    public int MaxStepsPerEpisode => _maxStepsPerEpisode;

    /// <inheritdoc/>
    public bool Verbose => _verbose;

    /// <inheritdoc/>
    public IReplayBuffer<T, Vector<T>, Vector<T>> ReplayBuffer => _replayBuffer;

    /// <inheritdoc/>
    public int MinExperiencesBeforeTraining => _minExperiencesBeforeTraining;

    /// <inheritdoc/>
    public int CurrentEpisode => _currentEpisode;

    /// <inheritdoc/>
    public int TotalSteps => _totalSteps;

    /// <inheritdoc/>
    public override int TotalCount => _replayBuffer.Count;

    /// <inheritdoc/>
    public override int BatchSize
    {
        get => _batchSize;
        set => _batchSize = Math.Max(1, value);
    }

    /// <inheritdoc/>
    public bool HasNext => _currentBatchIndex * _batchSize < _replayBuffer.Count;

    /// <inheritdoc/>
    public Experience<T, Vector<T>, Vector<T>> GetNextBatch()
    {
        if (!HasNext)
        {
            throw new InvalidOperationException("No more batches available. Call Reset() to start over.");
        }

        var batch = _replayBuffer.Sample(1);
        _currentBatchIndex++;
        AdvanceBatchIndex();

        return batch[0];
    }

    /// <inheritdoc/>
    public bool TryGetNextBatch(out Experience<T, Vector<T>, Vector<T>> batch)
    {
        if (!HasNext || !_replayBuffer.CanSample(1))
        {
            batch = new Experience<T, Vector<T>, Vector<T>>(
                new Vector<T>(0),
                new Vector<T>(0),
                NumOps.Zero,
                new Vector<T>(0),
                false);
            return false;
        }

        batch = GetNextBatch();
        return true;
    }

    /// <inheritdoc/>
    public IReadOnlyList<Experience<T, Vector<T>, Vector<T>>> SampleBatch(int batchSize)
    {
        if (!_replayBuffer.CanSample(batchSize))
        {
            throw new InvalidOperationException(
                $"Not enough experiences in buffer. Have {_replayBuffer.Count}, need {batchSize}.");
        }

        return _replayBuffer.Sample(batchSize);
    }

    /// <inheritdoc/>
    public bool CanTrain(int batchSize)
    {
        return _replayBuffer.Count >= _minExperiencesBeforeTraining &&
               _replayBuffer.CanSample(batchSize);
    }

    /// <inheritdoc/>
    public virtual EpisodeResult<T> RunEpisode(IRLAgent<T>? agent = null)
    {
        var stopwatch = Stopwatch.StartNew();
        var state = _environment.Reset();
        var totalReward = NumOps.Zero;
        int steps = 0;
        bool done = false;
        bool success = false;

        while (!done && steps < _maxStepsPerEpisode)
        {
            // Select action (random if no agent provided)
            Vector<T> action;
            if (agent != null)
            {
                action = agent.SelectAction(state, explore: true);
            }
            else
            {
                action = SelectRandomAction();
            }

            // Take step in environment
            var (nextState, reward, isDone, info) = _environment.Step(action);

            // Create and store experience
            var experience = new Experience<T, Vector<T>, Vector<T>>(state, action, reward, nextState, isDone);
            _replayBuffer.Add(experience);

            // Update state and counters
            totalReward = NumOps.Add(totalReward, reward);
            state = nextState;
            done = isDone;
            steps++;
            _totalSteps++;

            // Check for success (if environment provides this info)
            if (info.TryGetValue("success", out var successObj) && successObj is bool s)
            {
                success = s;
            }
        }

        stopwatch.Stop();
        _currentEpisode++;

        var result = new EpisodeResult<T>(
            EpisodeNumber: _currentEpisode,
            TotalReward: totalReward,
            Steps: steps,
            Completed: done,
            Success: success,
            Duration: stopwatch.Elapsed);

        if (_verbose && _currentEpisode % 10 == 0)
        {
            Console.WriteLine($"Episode {_currentEpisode}: Reward = {totalReward}, Steps = {steps}, Success = {success}");
        }

        return result;
    }

    /// <inheritdoc/>
    public IReadOnlyList<EpisodeResult<T>> RunEpisodes(int numEpisodes, IRLAgent<T>? agent = null)
    {
        var results = new List<EpisodeResult<T>>(numEpisodes);

        for (int i = 0; i < numEpisodes; i++)
        {
            results.Add(RunEpisode(agent));
        }

        return results;
    }

    /// <inheritdoc/>
    public void AddExperience(Experience<T, Vector<T>, Vector<T>> experience)
    {
        _replayBuffer.Add(experience);
    }

    /// <inheritdoc/>
    public void ResetTraining()
    {
        _replayBuffer.Clear();
        _currentEpisode = 0;
        _totalSteps = 0;
        _currentBatchIndex = 0;
    }

    /// <inheritdoc/>
    public void SetSeed(int seed)
    {
        lock (_randomLock)
        {
            _random = RandomHelper.CreateSeededRandom(seed);
        }
        _environment.Seed(seed);
    }

    /// <inheritdoc/>
    protected override void OnReset()
    {
        _currentBatchIndex = 0;
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // RL data loaders don't load data upfront - they collect it during training
        // Mark as loaded immediately
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        // RL data loaders manage their own buffer
        _replayBuffer.Clear();
    }

    /// <summary>
    /// Selects a random action for exploration.
    /// </summary>
    /// <returns>A random action vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>Thread Safety:</b> This method uses locking to ensure thread-safe access
    /// to the random number generator.
    /// </para>
    /// </remarks>
    protected virtual Vector<T> SelectRandomAction()
    {
        lock (_randomLock)
        {
            if (_environment.IsContinuousActionSpace)
            {
                // Random continuous action in [-1, 1] range
                var action = new Vector<T>(_environment.ActionSpaceSize);
                for (int i = 0; i < _environment.ActionSpaceSize; i++)
                {
                    action[i] = NumOps.FromDouble(_random.NextDouble() * 2 - 1);
                }
                return action;
            }
            else
            {
                // Random discrete action (one-hot encoded)
                var action = new Vector<T>(_environment.ActionSpaceSize);
                int randomAction = _random.Next(_environment.ActionSpaceSize);
                action[randomAction] = NumOps.One;
                return action;
            }
        }
    }

    #region PyTorch-Style Batch Iteration

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// For RL data loaders, this method samples experiences from the replay buffer.
    /// Unlike supervised learning, RL uses random sampling from the buffer rather than
    /// sequential iteration, which helps break temporal correlations.
    /// </para>
    /// <para>
    /// <b>Performance Notes:</b>
    /// - Each batch is independently sampled from the replay buffer
    /// - The shuffle parameter is ignored since RL inherently uses random sampling
    /// - Uses lazy evaluation via yield return for memory efficiency
    /// </para>
    /// </remarks>
    public virtual IEnumerable<Experience<T, Vector<T>, Vector<T>>> GetBatches(
        int? batchSize = null,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null)
    {
        int effectiveBatchSize = batchSize ?? BatchSize;
        if (effectiveBatchSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be at least 1.");
        }

        int bufferCount = _replayBuffer.Count;
        if (bufferCount == 0 || !_replayBuffer.CanSample(1))
        {
            yield break;
        }

        // Note: The seed parameter is not used for RL batch iteration because
        // the replay buffer performs its own random sampling. Seeding for RL
        // should be done at construction time via the seed constructor parameter,
        // which seeds both the random instance (for action selection) and the environment.

        // Calculate how many batches we can provide
        // For RL, we typically do multiple passes through the data
        int numBatches = bufferCount / effectiveBatchSize;
        if (!dropLast && bufferCount % effectiveBatchSize > 0)
        {
            numBatches++;
        }

        // Yield sampled batches
        for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
        {
            // Calculate actual batch size for this iteration
            int currentBatchSize = (batchIdx == numBatches - 1 && !dropLast)
                ? Math.Min(effectiveBatchSize, bufferCount - batchIdx * effectiveBatchSize)
                : effectiveBatchSize;

            if (currentBatchSize <= 0 || !_replayBuffer.CanSample(currentBatchSize))
            {
                yield break;
            }

            // Sample batch from replay buffer (RL uses random sampling)
            var samples = _replayBuffer.Sample(currentBatchSize);

            // Yield each experience in the batch
            foreach (var experience in samples)
            {
                yield return experience;
            }
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// This async implementation uses a bounded Channel for prefetching sampled experiences,
    /// allowing background sampling while the current batch is being used for training.
    /// </para>
    /// <para>
    /// <b>Implementation Details:</b>
    /// - Uses System.Threading.Channels for thread-safe producer-consumer pattern
    /// - Bounded channel capacity = prefetchCount to limit memory usage
    /// - Particularly useful when training is GPU-bound and sampling is CPU-bound
    /// </para>
    /// </remarks>
    public virtual async IAsyncEnumerable<Experience<T, Vector<T>, Vector<T>>> GetBatchesAsync(
        int? batchSize = null,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null,
        int prefetchCount = 2,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (prefetchCount < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(prefetchCount), "Prefetch count must be at least 1.");
        }

        int effectiveBatchSize = batchSize ?? BatchSize;
        if (effectiveBatchSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be at least 1.");
        }

        int bufferCount = _replayBuffer.Count;
        if (bufferCount == 0 || !_replayBuffer.CanSample(1))
        {
            yield break;
        }

        // Create bounded channel for prefetching
        var channel = Channel.CreateBounded<Experience<T, Vector<T>, Vector<T>>>(
            new BoundedChannelOptions(prefetchCount * effectiveBatchSize)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = true,
                SingleWriter = true
            });

        // Start producer task
        var producerTask = Task.Run(async () =>
        {
            try
            {
                // Note: The seed parameter is not used for RL batch iteration because
                // the replay buffer performs its own random sampling. Seeding for RL
                // should be done at construction time via the seed constructor parameter.

                // Calculate number of batches
                int numBatches = bufferCount / effectiveBatchSize;
                if (!dropLast && bufferCount % effectiveBatchSize > 0)
                {
                    numBatches++;
                }

                // Produce sampled experiences
                for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    int currentBatchSize = (batchIdx == numBatches - 1 && !dropLast)
                        ? Math.Min(effectiveBatchSize, bufferCount - batchIdx * effectiveBatchSize)
                        : effectiveBatchSize;

                    if (currentBatchSize <= 0 || !_replayBuffer.CanSample(currentBatchSize))
                    {
                        break;
                    }

                    var samples = _replayBuffer.Sample(currentBatchSize);

                    foreach (var experience in samples)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        await channel.Writer.WriteAsync(experience, cancellationToken);
                    }
                }
            }
            finally
            {
                channel.Writer.Complete();
            }
        }, cancellationToken);

        // Consume experiences (net471 compatible - no ReadAllAsync)
        while (await channel.Reader.WaitToReadAsync(cancellationToken))
        {
            while (channel.Reader.TryRead(out var experience))
            {
                yield return experience;
            }
        }

        // Ensure producer task completed without errors
        await producerTask;
    }

    #endregion
}
