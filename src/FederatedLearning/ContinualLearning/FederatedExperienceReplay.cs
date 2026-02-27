namespace AiDotNet.FederatedLearning.ContinualLearning;

/// <summary>
/// Implements Federated Experience Replay for continual learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Experience replay is the simplest anti-forgetting technique:
/// keep a small "memory" of representative examples from old tasks, and mix them in when
/// training on new data. In federated ER, each client maintains their own replay buffer
/// locally (no data sharing). When training on task T+1, each client trains on a mix of
/// new task data and old examples from their buffer.</para>
///
/// <para>Algorithm per client:</para>
/// <code>
/// 1. For each training batch on new task:
///    a. Sample mini-batch from new task data
///    b. Sample mini-batch from replay buffer
///    c. Train on combined batch
/// 2. After task: add representative examples to buffer (reservoir sampling)
/// </code>
///
/// <para>Reference: Federated Experience Replay for Continual FL (2023).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FederatedExperienceReplay<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedContinualLearningStrategy<T>
{
    private readonly int _bufferCapacity;
    private readonly double _replayRatio;
    private readonly List<(T[] Features, int Label)> _buffer;
    private int _totalSeen;
    private readonly Random _rng;

    /// <summary>
    /// Creates a new Federated Experience Replay strategy.
    /// </summary>
    /// <param name="bufferCapacity">Maximum number of examples in replay buffer. Default: 500.</param>
    /// <param name="replayRatio">Fraction of each training batch from replay buffer. Default: 0.3.</param>
    /// <param name="seed">Random seed. Default: 42.</param>
    public FederatedExperienceReplay(int bufferCapacity = 500, double replayRatio = 0.3, int seed = 42)
    {
        if (bufferCapacity < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(bufferCapacity), "Buffer capacity must be at least 1.");
        }

        if (replayRatio < 0 || replayRatio > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(replayRatio), "Replay ratio must be in [0, 1].");
        }

        _bufferCapacity = bufferCapacity;
        _replayRatio = replayRatio;
        _buffer = new List<(T[], int)>(_bufferCapacity);
        _rng = new Random(seed);
    }

    /// <summary>
    /// Adds an example to the replay buffer using reservoir sampling.
    /// </summary>
    public void AddToBuffer(T[] features, int label)
    {
        _totalSeen++;
        if (_buffer.Count < _bufferCapacity)
        {
            _buffer.Add((features, label));
        }
        else
        {
            int idx = _rng.Next(_totalSeen);
            if (idx < _bufferCapacity)
            {
                _buffer[idx] = (features, label);
            }
        }
    }

    /// <summary>
    /// Samples a batch from the replay buffer.
    /// </summary>
    /// <param name="batchSize">Number of examples to sample.</param>
    /// <returns>Sampled examples.</returns>
    public List<(T[] Features, int Label)> SampleReplay(int batchSize)
    {
        if (_buffer.Count == 0)
        {
            return new List<(T[], int)>();
        }

        int effectiveSize = Math.Min(batchSize, _buffer.Count);
        var sampled = new List<(T[], int)>(effectiveSize);
        for (int i = 0; i < effectiveSize; i++)
        {
            sampled.Add(_buffer[_rng.Next(_buffer.Count)]);
        }

        return sampled;
    }

    /// <inheritdoc/>
    public Vector<T> ComputeImportance(Vector<T> modelParameters, Matrix<T> taskData)
    {
        var importance = new T[modelParameters.Length];
        for (int i = 0; i < importance.Length; i++)
        {
            importance[i] = NumOps.FromDouble(1.0); // Uniform importance for ER.
        }

        return new Vector<T>(importance);
    }

    /// <inheritdoc/>
    public T ComputeRegularizationPenalty(
        Vector<T> currentParameters, Vector<T> referenceParameters,
        Vector<T> importanceWeights, double regularizationStrength)
    {
        // ER uses replay data instead of explicit regularization.
        return NumOps.Zero;
    }

    /// <inheritdoc/>
    public Vector<T> ProjectGradient(Vector<T> gradient, Vector<T> importanceWeights)
    {
        return gradient; // No gradient projection — ER relies on data replay.
    }

    /// <inheritdoc/>
    public Vector<T> AggregateImportance(
        Dictionary<int, Vector<T>> clientImportances,
        Dictionary<int, double>? clientWeights)
    {
        int d = clientImportances.Values.First().Length;
        var aggregated = new T[d];
        double totalWeight = clientWeights?.Values.Sum() ?? clientImportances.Count;

        foreach (var (clientId, importance) in clientImportances)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            var wT = NumOps.FromDouble(w / totalWeight);
            for (int i = 0; i < d; i++)
            {
                aggregated[i] = NumOps.Add(aggregated[i], NumOps.Multiply(importance[i], wT));
            }
        }

        return new Vector<T>(aggregated);
    }

    /// <summary>Gets the buffer capacity.</summary>
    public int BufferCapacity => _bufferCapacity;

    /// <summary>Gets the current buffer size.</summary>
    public int BufferSize => _buffer.Count;

    /// <summary>Gets the replay ratio.</summary>
    public double ReplayRatio => _replayRatio;
}
