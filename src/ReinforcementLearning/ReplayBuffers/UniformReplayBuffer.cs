using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Interfaces;

namespace AiDotNet.ReinforcementLearning.ReplayBuffers;

/// <summary>
/// Implements a uniform replay buffer that stores experiences and samples them uniformly at random.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// The UniformReplayBuffer is the standard experience replay mechanism used in DQN and related algorithms.
/// It stores experiences in a circular buffer and samples them uniformly at random during training.
/// This breaks temporal correlations in the training data, leading to more stable learning.
/// </para>
/// <para><b>For Beginners:</b> This is a memory bank that stores the agent's experiences and lets it learn from random past memories.
///
/// How it works:
/// - Experiences are stored in a fixed-size buffer
/// - When the buffer fills up, old experiences are replaced by new ones
/// - During training, experiences are randomly selected for learning
/// - All experiences have an equal chance of being selected
///
/// Think of it like a jar of memory marbles:
/// - Each marble represents one experience
/// - The jar has a maximum capacity
/// - When it's full, you remove old marbles to add new ones
/// - When studying, you randomly grab a handful of marbles
///
/// This "uniform" sampling means every experience is equally likely to be selected,
/// which is simple and works well for many problems.
/// </para>
/// </remarks>
public class UniformReplayBuffer<T> : IReplayBuffer<T>
{
    private readonly List<Experience<T>> _buffer;
    private readonly Random _random;
    private int _nextIndex;

    /// <inheritdoc/>
    public int Count => _buffer.Count;

    /// <inheritdoc/>
    public int Capacity { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="UniformReplayBuffer{T}"/> class with the specified capacity.
    /// </summary>
    /// <param name="capacity">The maximum number of experiences to store.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// Creates a new replay buffer with the specified maximum capacity. The capacity determines how many
    /// experiences can be stored before old ones start being replaced. Larger capacities allow more diverse
    /// experiences but use more memory.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new memory bank with a specific size limit.
    ///
    /// Parameters:
    /// - capacity: How many experiences you can store (typical values: 10,000 to 1,000,000)
    /// - seed: Optional number to make randomness predictable (useful for debugging)
    ///
    /// Memory usage:
    /// - Larger capacity = more diverse memories but more RAM usage
    /// - Smaller capacity = less memory but less diversity
    /// - Choose based on your problem complexity and available memory
    ///
    /// Typical capacities:
    /// - Simple problems: 10,000 - 50,000
    /// - Medium problems: 100,000 - 500,000
    /// - Complex problems (like Atari): 1,000,000
    /// </para>
    /// </remarks>
    public UniformReplayBuffer(int capacity, int? seed = null)
    {
        if (capacity <= 0)
        {
            throw new ArgumentException("Capacity must be positive", nameof(capacity));
        }

        Capacity = capacity;
        _buffer = new List<Experience<T>>(capacity);
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        _nextIndex = 0;
    }

    /// <inheritdoc/>
    public void Add(Experience<T> experience)
    {
        if (experience == null)
        {
            throw new ArgumentNullException(nameof(experience));
        }

        if (_buffer.Count < Capacity)
        {
            // Buffer not full yet, just add
            _buffer.Add(experience);
        }
        else
        {
            // Buffer full, replace oldest (circular buffer)
            _buffer[_nextIndex] = experience;
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

        var batch = new Experience<T>[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            int randomIndex = _random.Next(_buffer.Count);
            batch[i] = _buffer[randomIndex];
        }

        return batch;
    }

    /// <inheritdoc/>
    public void Clear()
    {
        _buffer.Clear();
        _nextIndex = 0;
    }

    /// <inheritdoc/>
    public bool CanSample(int batchSize)
    {
        return _buffer.Count >= batchSize;
    }
}
