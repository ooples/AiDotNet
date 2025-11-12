using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Interfaces;

namespace AiDotNet.ReinforcementLearning.ReplayBuffers;

/// <summary>
/// Implements a prioritized experience replay buffer that samples experiences based on their TD error.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Prioritized Experience Replay (PER) improves upon uniform sampling by prioritizing experiences
/// that the agent can learn more from. Experiences with higher TD (Temporal Difference) errors
/// are sampled more frequently, as they represent situations where the agent's predictions were
/// most wrong. This leads to more efficient learning, especially in environments with sparse rewards.
/// </para>
/// <para><b>For Beginners:</b> This is a smarter memory bank that focuses on the most important experiences.
///
/// How it differs from uniform sampling:
/// - Experiences aren't all equally important
/// - Some experiences are more surprising or informative than others
/// - This buffer prioritizes "surprising" experiences (where predictions were very wrong)
/// - Those important experiences get sampled more often for learning
///
/// Think of it like studying for a test:
/// - Uniform buffer: Review all topics equally, even ones you already know well
/// - Prioritized buffer: Focus more on topics you struggle with
///
/// Benefits:
/// - Learn faster by focusing on important experiences
/// - Better performance, especially when rewards are rare
/// - More sample-efficient (need fewer experiences to learn well)
///
/// The "TD error" measures how wrong the agent's prediction was. Higher error = more surprising = more important.
/// </para>
/// </remarks>
public class PrioritizedReplayBuffer<T> : IReplayBuffer<T>
{
    private readonly List<Experience<T>> _buffer;
    private readonly List<double> _priorities;
    private readonly Random _random;
    private readonly INumericOperations<T> _numOps;
    private readonly double _alpha; // Priority exponent
    private readonly double _beta; // Importance sampling exponent
    private readonly double _betaIncrement;
    private readonly double _epsilon; // Small constant to ensure non-zero priorities
    private double _maxPriority;
    private int _nextIndex;

    /// <inheritdoc/>
    public int Count => _buffer.Count;

    /// <inheritdoc/>
    public int Capacity { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="PrioritizedReplayBuffer{T}"/> class.
    /// </summary>
    /// <param name="capacity">The maximum number of experiences to store.</param>
    /// <param name="alpha">Priority exponent (0 = uniform sampling, 1 = full prioritization). Default is 0.6.</param>
    /// <param name="beta">Importance sampling exponent (starts low, anneals to 1). Default is 0.4.</param>
    /// <param name="betaIncrement">How much to increase beta each time Update is called. Default is 0.001.</param>
    /// <param name="epsilon">Small constant to ensure non-zero priorities. Default is 1e-6.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// Creates a new prioritized replay buffer. The alpha parameter controls how much prioritization to use:
    /// - alpha = 0: Uniform sampling (no prioritization)
    /// - alpha = 1: Full prioritization based on TD error
    /// - alpha between 0 and 1: Partial prioritization (typically 0.6)
    ///
    /// The beta parameter controls importance sampling correction, which compensates for the bias
    /// introduced by prioritized sampling. It typically starts at around 0.4 and is annealed to 1.0
    /// over the course of training.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a prioritized memory bank with several tuning knobs.
    ///
    /// Parameters explained:
    /// - capacity: How many experiences to store (same as uniform buffer)
    /// - alpha: How much to prioritize important experiences
    ///   * 0 = treat all experiences equally (like uniform buffer)
    ///   * 1 = strongly prioritize important experiences
    ///   * 0.6 is a good default (moderate prioritization)
    /// - beta: Correction factor to keep learning unbiased
    ///   * Starts around 0.4 and increases to 1.0 during training
    ///   * This compensates for seeing important experiences more often
    /// - betaIncrement: How fast to increase beta (default 0.001 works well)
    /// - epsilon: Tiny number to ensure all experiences have some chance of being selected
    ///
    /// In most cases, the defaults work well! Only adjust if you're an advanced user
    /// experimenting with the algorithm.
    /// </para>
    /// </remarks>
    public PrioritizedReplayBuffer(
        int capacity,
        double alpha = 0.6,
        double beta = 0.4,
        double betaIncrement = 0.001,
        double epsilon = 1e-6,
        int? seed = null)
    {
        if (capacity <= 0)
        {
            throw new ArgumentException("Capacity must be positive", nameof(capacity));
        }

        if (alpha < 0 || alpha > 1)
        {
            throw new ArgumentException("Alpha must be between 0 and 1", nameof(alpha));
        }

        if (beta < 0 || beta > 1)
        {
            throw new ArgumentException("Beta must be between 0 and 1", nameof(beta));
        }

        Capacity = capacity;
        _alpha = alpha;
        _beta = beta;
        _betaIncrement = betaIncrement;
        _epsilon = epsilon;
        _buffer = new List<Experience<T>>(capacity);
        _priorities = new List<double>(capacity);
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        _numOps = NumericOperations<T>.Instance;
        _maxPriority = 1.0;
        _nextIndex = 0;
    }

    /// <inheritdoc/>
    public void Add(Experience<T> experience)
    {
        if (experience == null)
        {
            throw new ArgumentNullException(nameof(experience));
        }

        // New experiences get maximum priority to ensure they're sampled at least once
        double priority = _maxPriority;

        if (_buffer.Count < Capacity)
        {
            // Buffer not full yet, just add
            _buffer.Add(experience);
            _priorities.Add(priority);
        }
        else
        {
            // Buffer full, replace oldest (circular buffer)
            _buffer[_nextIndex] = experience;
            _priorities[_nextIndex] = priority;
            _nextIndex = (_nextIndex + 1) % Capacity;
        }
    }

    /// <inheritdoc/>
    public Experience<T>[] Sample(int batchSize)
    {
        if (batchSize <= 0)
        {
            throw new ArgumentException("Batch size must be positive", nameof(batchSize));
        }

        if (batchSize > _buffer.Count)
        {
            throw new InvalidOperationException(
                $"Cannot sample {batchSize} experiences from buffer with only {_buffer.Count} experiences");
        }

        // Convert priorities to sampling probabilities
        var probabilities = new double[_buffer.Count];
        double totalPriority = 0.0;

        for (int i = 0; i < _buffer.Count; i++)
        {
            // Apply priority exponent alpha
            probabilities[i] = Math.Pow(_priorities[i] + _epsilon, _alpha);
            totalPriority += probabilities[i];
        }

        // Normalize to get probabilities
        for (int i = 0; i < probabilities.Length; i++)
        {
            probabilities[i] /= totalPriority;
        }

        // Sample based on probabilities
        var batch = new Experience<T>[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            double rand = _random.NextDouble();
            double cumulative = 0.0;
            int selectedIndex = 0;

            for (int j = 0; j < probabilities.Length; j++)
            {
                cumulative += probabilities[j];
                if (rand <= cumulative)
                {
                    selectedIndex = j;
                    break;
                }
            }

            batch[i] = _buffer[selectedIndex];
        }

        return batch;
    }

    /// <summary>
    /// Updates the priorities of experiences based on their TD errors.
    /// </summary>
    /// <param name="indices">The indices of the experiences to update.</param>
    /// <param name="tdErrors">The TD errors for these experiences.</param>
    /// <remarks>
    /// <para>
    /// This method should be called after training on a batch to update the priorities of the
    /// sampled experiences based on their new TD errors. Higher TD errors result in higher priorities,
    /// making those experiences more likely to be sampled in the future.
    /// </para>
    /// <para><b>For Beginners:</b> This updates how important each experience is based on what was learned.
    ///
    /// After training on a batch:
    /// - We know which experiences were most surprising (high TD error)
    /// - We update their priority so they're more likely to be sampled again
    /// - Experiences we predicted well get lower priority
    /// - This focuses future learning on difficult cases
    ///
    /// Think of it like adjusting study priorities:
    /// - After practicing, you know which problems you got wrong
    /// - Mark those problems for extra review
    /// - Spend less time on problems you already understand well
    /// </para>
    /// </remarks>
    public void UpdatePriorities(int[] indices, T[] tdErrors)
    {
        if (indices == null)
        {
            throw new ArgumentNullException(nameof(indices));
        }

        if (tdErrors == null)
        {
            throw new ArgumentNullException(nameof(tdErrors));
        }

        if (indices.Length != tdErrors.Length)
        {
            throw new ArgumentException("Indices and TD errors must have the same length");
        }

        for (int i = 0; i < indices.Length; i++)
        {
            // Priority is based on absolute TD error
            double priority = Math.Abs(_numOps.ToDouble(tdErrors[i]));
            _priorities[indices[i]] = priority;

            // Track maximum priority
            if (priority > _maxPriority)
            {
                _maxPriority = priority;
            }
        }
    }

    /// <inheritdoc/>
    public void Clear()
    {
        _buffer.Clear();
        _priorities.Clear();
        _nextIndex = 0;
        _maxPriority = 1.0;
    }

    /// <inheritdoc/>
    public bool CanSample(int batchSize)
    {
        return _buffer.Count >= batchSize;
    }
}
