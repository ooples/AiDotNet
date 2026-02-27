namespace AiDotNet.FederatedLearning.Trainers;

/// <summary>
/// Implements FedBuff — Buffered asynchronous federated aggregation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In pure async FL, the server aggregates each client's update
/// as soon as it arrives. This can lead to "stale" updates from slow clients being applied
/// to a model that has already moved on. FedBuff adds a buffer: the server waits until
/// K updates arrive (from any clients), then aggregates them all at once. This balances
/// freshness (not too stale) with efficiency (don't wait for all clients).</para>
///
/// <para>Algorithm:</para>
/// <code>
/// buffer = []
/// while True:
///   update = receive_any_client_update()
///   buffer.append(update)
///   if len(buffer) >= K:
///     global_model = aggregate(buffer)
///     buffer = []
///     broadcast(global_model)
/// </code>
///
/// <para>Reference: Nguyen, J., et al. (2022). "Federated Learning with Buffered
/// Asynchronous Aggregation." AISTATS 2022.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class BufferedAsyncFederatedTrainer<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly int _bufferSize;
    private readonly double _stalenessDiscount;
    private readonly object _bufferLock = new();
    private readonly List<(int ClientId, Dictionary<string, T[]> Update, int Staleness)> _buffer;
    private int _currentGlobalRound;

    /// <summary>
    /// Creates a new FedBuff trainer.
    /// </summary>
    /// <param name="bufferSize">Number of updates to collect before aggregating (K). Default: 10.</param>
    /// <param name="stalenessDiscount">Discount factor for stale updates (per round). Default: 0.9.</param>
    public BufferedAsyncFederatedTrainer(int bufferSize = 10, double stalenessDiscount = 0.9)
    {
        if (bufferSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(bufferSize), "Buffer size must be at least 1.");
        }

        if (stalenessDiscount <= 0 || stalenessDiscount > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(stalenessDiscount), "Staleness discount must be in (0, 1].");
        }

        _bufferSize = bufferSize;
        _stalenessDiscount = stalenessDiscount;
        _buffer = new List<(int, Dictionary<string, T[]>, int)>();
    }

    /// <summary>
    /// Submits a client update to the buffer.
    /// </summary>
    /// <param name="clientId">Client ID.</param>
    /// <param name="update">Model update from client.</param>
    /// <param name="clientRound">The global round the client's update was based on.</param>
    public void SubmitUpdate(int clientId, Dictionary<string, T[]> update, int clientRound)
    {
        Guard.NotNull(update);
        int staleness = _currentGlobalRound - clientRound;
        lock (_bufferLock)
        {
            _buffer.Add((clientId, update, Math.Max(0, staleness)));
        }
    }

    /// <summary>
    /// Checks if the buffer is full and ready for aggregation.
    /// </summary>
    public bool IsBufferReady { get { lock (_bufferLock) { return _buffer.Count >= _bufferSize; } } }

    /// <summary>
    /// Aggregates the buffered updates with staleness-weighted averaging.
    /// </summary>
    /// <returns>Aggregated model update, or null if buffer is empty.</returns>
    public Dictionary<string, T[]>? AggregateBuffer()
    {
        List<(int ClientId, Dictionary<string, T[]> Update, int Staleness)> snapshot;
        lock (_bufferLock)
        {
            if (_buffer.Count == 0)
            {
                return null;
            }

            snapshot = new List<(int, Dictionary<string, T[]>, int)>(_buffer);
            _buffer.Clear();
        }

        var reference = snapshot[0].Update;
        var layerNames = reference.Keys.ToArray();

        // Compute staleness-discounted weights.
        double totalWeight = 0;
        var weights = new double[snapshot.Count];
        for (int i = 0; i < snapshot.Count; i++)
        {
            weights[i] = Math.Pow(_stalenessDiscount, snapshot[i].Staleness);
            totalWeight += weights[i];
        }

        // Guard against totalWeight underflowing to 0 (e.g., 0.9^large → 0).
        if (totalWeight <= 0)
        {
            // Fall back to uniform weights.
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = 1.0;
            }

            totalWeight = weights.Length;
        }

        var result = new Dictionary<string, T[]>(reference.Count);
        foreach (var layerName in layerNames)
        {
            int layerLen = reference[layerName].Length;
            var aggregated = new T[layerLen];
            for (int j = 0; j < aggregated.Length; j++)
            {
                aggregated[j] = NumOps.Zero;
            }

            for (int i = 0; i < snapshot.Count; i++)
            {
                if (!snapshot[i].Update.TryGetValue(layerName, out var cp))
                {
                    continue; // Skip clients missing this layer.
                }

                var nw = NumOps.FromDouble(weights[i] / totalWeight);
                int len = Math.Min(cp.Length, layerLen);
                for (int j = 0; j < len; j++)
                {
                    aggregated[j] = NumOps.Add(aggregated[j], NumOps.Multiply(cp[j], nw));
                }
            }

            result[layerName] = aggregated;
        }

        Interlocked.Increment(ref _currentGlobalRound);
        return result;
    }

    /// <summary>Gets the buffer size (K).</summary>
    public int BufferSize => _bufferSize;

    /// <summary>Gets the current buffer count.</summary>
    public int CurrentBufferCount { get { lock (_bufferLock) { return _buffer.Count; } } }

    /// <summary>Gets the current global round.</summary>
    public int CurrentGlobalRound => _currentGlobalRound;
}
