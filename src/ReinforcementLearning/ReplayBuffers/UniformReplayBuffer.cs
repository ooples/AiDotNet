namespace AiDotNet.ReinforcementLearning.ReplayBuffers;

/// <summary>
/// A replay buffer that samples experiences uniformly at random.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TState">The type representing the state observation (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TAction">The type representing the action (e.g., Vector&lt;T&gt; for continuous, int for discrete).</typeparam>
/// <remarks>
/// <para>
/// This is the standard replay buffer used in algorithms like DQN. Experiences are stored
/// in a circular buffer and sampled uniformly at random for training. All experiences have
/// an equal probability of being selected, regardless of their importance or recency.
/// </para>
/// <para><b>For Beginners:</b>
/// This replay buffer treats all experiences equally - it's like having a bag of memories
/// and pulling out random ones to learn from. When the buffer is full, the oldest memories
/// get replaced with new ones.
///
/// **Key Properties:**
/// - **Uniform Sampling**: Every experience has an equal chance of being picked
/// - **Circular Buffer**: Old experiences are automatically removed when capacity is reached
/// - **No Prioritization**: Unlike prioritized replay, doesn't favor "important" experiences
///
/// **When to Use:**
/// - Good starting point for most RL algorithms
/// - Works well when all experiences are roughly equally valuable
/// - Simpler and faster than prioritized variants
/// </para>
/// </remarks>
public class UniformReplayBuffer<T, TState, TAction> : IReplayBuffer<T, TState, TAction>
{
    private readonly List<Experience<T, TState, TAction>> _buffer;
    private readonly Random _random;
    private int _position;

    /// <inheritdoc/>
    public int Capacity { get; }

    /// <inheritdoc/>
    public int Count => _buffer.Count;

    /// <summary>
    /// Initializes a new instance of the UniformReplayBuffer class.
    /// </summary>
    /// <param name="capacity">Maximum number of experiences to store.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Capacity determines how many experiences the buffer remembers.
    /// Larger buffers have more diverse experiences but use more memory.
    /// Common values: 10,000 for simple problems, 100,000-1,000,000 for complex ones.
    /// </remarks>
    public UniformReplayBuffer(int capacity, int? seed = null)
    {
        if (capacity <= 0)
            throw new ArgumentException("Capacity must be positive", nameof(capacity));

        Capacity = capacity;
        _buffer = new List<Experience<T, TState, TAction>>(capacity);
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        _position = 0;
    }

    /// <inheritdoc/>
    public void Add(Experience<T, TState, TAction> experience)
    {
        if (_buffer.Count < Capacity)
        {
            _buffer.Add(experience);
        }
        else
        {
            // Circular buffer - overwrite oldest experience
            _buffer[_position] = experience;
            _position = (_position + 1) % Capacity;
        }
    }

    /// <inheritdoc/>
    public List<Experience<T, TState, TAction>> Sample(int batchSize)
    {
        if (!CanSample(batchSize))
            throw new InvalidOperationException($"Cannot sample {batchSize} experiences. Buffer only contains {Count} experiences.");

        var sampled = new List<Experience<T, TState, TAction>>(batchSize);
        var indices = new HashSet<int>();

        // Sample without replacement
        while (indices.Count < batchSize)
        {
            indices.Add(_random.Next(_buffer.Count));
        }

        foreach (var index in indices)
        {
            sampled.Add(_buffer[index]);
        }

        return sampled;
    }

    /// <summary>
    /// Samples a batch of experiences with their buffer indices.
    /// </summary>
    /// <param name="batchSize">Number of experiences to sample.</param>
    /// <returns>A tuple containing the list of sampled experiences and their corresponding buffer indices.</returns>
    /// <remarks>
    /// <para>
    /// This method is useful for multi-agent scenarios where additional per-agent data is stored
    /// separately and needs to be retrieved using the buffer index.
    /// </para>
    /// <para><b>For Beginners:</b> Sometimes you need to know where in the buffer each sampled
    /// experience came from. This method returns both the experiences and their positions,
    /// which is useful for advanced techniques like updating priorities in prioritized replay.
    /// </para>
    /// </remarks>
    public (List<Experience<T, TState, TAction>> Experiences, List<int> Indices) SampleWithIndices(int batchSize)
    {
        if (!CanSample(batchSize))
            throw new InvalidOperationException($"Cannot sample {batchSize} experiences. Buffer only contains {Count} experiences.");

        var sampled = new List<Experience<T, TState, TAction>>(batchSize);
        var sampledIndices = new List<int>(batchSize);
        var indices = new HashSet<int>();

        // Sample without replacement
        while (indices.Count < batchSize)
        {
            indices.Add(_random.Next(_buffer.Count));
        }

        foreach (var index in indices)
        {
            sampled.Add(_buffer[index]);
            sampledIndices.Add(index);
        }

        return (sampled, sampledIndices);
    }

    /// <inheritdoc/>
    public bool CanSample(int batchSize)
    {
        return _buffer.Count >= batchSize;
    }

    /// <inheritdoc/>
    public void Clear()
    {
        _buffer.Clear();
        _position = 0;
    }
}
