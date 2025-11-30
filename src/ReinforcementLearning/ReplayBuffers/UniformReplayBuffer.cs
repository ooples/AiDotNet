namespace AiDotNet.ReinforcementLearning.ReplayBuffers;

/// <summary>
/// A replay buffer that samples experiences uniformly at random.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This is the standard replay buffer used in algorithms like DQN. Experiences are stored
/// in a circular buffer and sampled uniformly at random for training.
/// </para>
/// <para><b>For Beginners:</b>
/// This replay buffer treats all experiences equally - it's like having a bag of memories
/// and pulling out random ones to learn from. When the buffer is full, the oldest memories
/// get replaced with new ones.
/// </para>
/// </remarks>
public class UniformReplayBuffer<T> : IReplayBuffer<T>
{
    private readonly List<Experience<T>> _buffer;
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
    public UniformReplayBuffer(int capacity, int? seed = null)
    {
        if (capacity <= 0)
            throw new ArgumentException("Capacity must be positive", nameof(capacity));

        Capacity = capacity;
        _buffer = new List<Experience<T>>(capacity);
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        _position = 0;
    }

    /// <inheritdoc/>
    public void Add(Experience<T> experience)
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
    public List<Experience<T>> Sample(int batchSize)
    {
        if (!CanSample(batchSize))
            throw new InvalidOperationException($"Cannot sample {batchSize} experiences. Buffer only contains {Count} experiences.");

        var sampled = new List<Experience<T>>(batchSize);
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
    /// This method is useful for multi-agent scenarios where additional per-agent data is stored
    /// separately and needs to be retrieved using the buffer index.
    /// </remarks>
    public (List<Experience<T>> Experiences, List<int> Indices) SampleWithIndices(int batchSize)
    {
        if (!CanSample(batchSize))
            throw new InvalidOperationException($"Cannot sample {batchSize} experiences. Buffer only contains {Count} experiences.");

        var sampled = new List<Experience<T>>(batchSize);
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
