namespace AiDotNet.FederatedLearning.Trainers;

/// <summary>
/// Implements Semi-Asynchronous Federated Learning — hybrid sync/async with periodic barriers.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Pure synchronous FL waits for ALL clients each round (slow, wastes
/// time waiting for stragglers). Pure async FL processes each update immediately (fast, but stale
/// updates can hurt convergence). Semi-Async is the middle ground: the server accepts async updates
/// between synchronization barriers that occur every K rounds. During async phases, fast clients
/// can contribute multiple updates. At barriers, all pending updates are aggregated and a new
/// global model is broadcast. This balances speed with convergence quality.</para>
///
/// <para>Algorithm:</para>
/// <code>
/// for each global epoch:
///   for round = 1 to K (async phase):
///     accept any arriving client update
///     apply with staleness discount: w_global += lr * decay^staleness * delta_k
///   barrier:
///     wait for all active clients
///     aggregate all pending updates (FedAvg-style)
///     broadcast new global model
/// </code>
///
/// <para>Reference: Wu, X., et al. (2023). "Semi-Asynchronous Federated Learning:
/// Convergence and Efficiency." IEEE TPDS 2023.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class SemiAsyncFederatedTrainer<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly int _asyncRoundsPerBarrier;
    private readonly double _stalenessDiscount;
    private readonly double _asyncLearningRate;
    private readonly List<PendingUpdate> _updateBuffer;
    private int _currentRound;

    /// <summary>
    /// Creates a new Semi-Async FL trainer.
    /// </summary>
    /// <param name="asyncRoundsPerBarrier">Number of async rounds between sync barriers. Default: 5.</param>
    /// <param name="stalenessDiscount">Discount factor per round of staleness. Default: 0.9.</param>
    /// <param name="asyncLearningRate">Learning rate for async update application. Default: 0.1.</param>
    public SemiAsyncFederatedTrainer(
        int asyncRoundsPerBarrier = 5,
        double stalenessDiscount = 0.9,
        double asyncLearningRate = 0.1)
    {
        if (asyncRoundsPerBarrier <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(asyncRoundsPerBarrier), "Must have at least 1 async round per barrier.");
        }

        if (stalenessDiscount <= 0 || stalenessDiscount > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(stalenessDiscount), "Staleness discount must be in (0, 1].");
        }

        if (asyncLearningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(asyncLearningRate), "Learning rate must be positive.");
        }

        _asyncRoundsPerBarrier = asyncRoundsPerBarrier;
        _stalenessDiscount = stalenessDiscount;
        _asyncLearningRate = asyncLearningRate;
        _updateBuffer = new List<PendingUpdate>();
        _currentRound = 0;
    }

    /// <summary>
    /// Receives an async client update and buffers it for application.
    /// </summary>
    /// <param name="clientId">The client identifier.</param>
    /// <param name="update">The parameter update (delta).</param>
    /// <param name="clientRound">The global round when the client started training.</param>
    public void ReceiveUpdate(int clientId, Dictionary<string, T[]> update, int clientRound)
    {
        _updateBuffer.Add(new PendingUpdate(clientId, update, clientRound));
    }

    /// <summary>
    /// Applies buffered async updates to the global model with staleness discounting.
    /// </summary>
    /// <param name="globalModel">Current global model parameters.</param>
    /// <returns>Updated global model after applying async updates.</returns>
    public Dictionary<string, T[]> ApplyAsyncUpdates(Dictionary<string, T[]> globalModel)
    {
        if (_updateBuffer.Count == 0)
        {
            return globalModel;
        }

        var result = new Dictionary<string, T[]>();

        // Deep copy global model.
        foreach (var (layerName, layerParams) in globalModel)
        {
            var copy = new T[layerParams.Length];
            Array.Copy(layerParams, copy, layerParams.Length);
            result[layerName] = copy;
        }

        // Apply each buffered update with staleness discount.
        foreach (var pending in _updateBuffer)
        {
            int staleness = _currentRound - pending.ClientRound;
            double discount = Math.Pow(_stalenessDiscount, Math.Max(staleness, 0));
            double effectiveLR = _asyncLearningRate * discount;

            foreach (var (layerName, delta) in pending.Update)
            {
                if (result.TryGetValue(layerName, out var globalLayer))
                {
                    for (int i = 0; i < Math.Min(delta.Length, globalLayer.Length); i++)
                    {
                        double g = NumOps.ToDouble(globalLayer[i]);
                        double d = NumOps.ToDouble(delta[i]);
                        globalLayer[i] = NumOps.FromDouble(g + effectiveLR * d);
                    }
                }
            }
        }

        _updateBuffer.Clear();
        return result;
    }

    /// <summary>
    /// Performs a synchronization barrier: aggregates all buffered updates via weighted average.
    /// </summary>
    /// <param name="globalModel">Current global model.</param>
    /// <param name="clientModels">Full client models collected at the barrier.</param>
    /// <param name="clientSampleCounts">Sample counts per client.</param>
    /// <returns>Synchronized global model.</returns>
    public Dictionary<string, T[]> SynchronizationBarrier(
        Dictionary<string, T[]> globalModel,
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, int> clientSampleCounts)
    {
        // First apply any remaining async updates.
        var updated = ApplyAsyncUpdates(globalModel);

        if (clientModels.Count == 0)
        {
            return updated;
        }

        // FedAvg-style weighted average at the barrier.
        double totalSamples = clientSampleCounts.Values.Sum();
        var result = new Dictionary<string, T[]>();

        foreach (var (layerName, layerParams) in updated)
        {
            var merged = new double[layerParams.Length];

            foreach (var (clientId, model) in clientModels)
            {
                double w = totalSamples > 0
                    ? clientSampleCounts.GetValueOrDefault(clientId, 1) / totalSamples
                    : 1.0 / clientModels.Count;

                if (model.TryGetValue(layerName, out var clientLayer))
                {
                    for (int i = 0; i < Math.Min(clientLayer.Length, merged.Length); i++)
                    {
                        merged[i] += w * NumOps.ToDouble(clientLayer[i]);
                    }
                }
            }

            var mergedT = new T[layerParams.Length];
            for (int i = 0; i < mergedT.Length; i++)
            {
                mergedT[i] = NumOps.FromDouble(merged[i]);
            }

            result[layerName] = mergedT;
        }

        return result;
    }

    /// <summary>
    /// Determines whether the current round is a synchronization barrier.
    /// </summary>
    /// <param name="round">The current round number.</param>
    /// <returns>True if this round should trigger a sync barrier.</returns>
    public bool IsBarrierRound(int round)
    {
        return round > 0 && round % _asyncRoundsPerBarrier == 0;
    }

    /// <summary>
    /// Advances the trainer to the next round.
    /// </summary>
    public void AdvanceRound()
    {
        _currentRound++;
    }

    /// <summary>Gets the number of async rounds per barrier.</summary>
    public int AsyncRoundsPerBarrier => _asyncRoundsPerBarrier;

    /// <summary>Gets the staleness discount factor.</summary>
    public double StalenessDiscount => _stalenessDiscount;

    /// <summary>Gets the async learning rate.</summary>
    public double AsyncLearningRate => _asyncLearningRate;

    /// <summary>Gets the current round number.</summary>
    public int CurrentRound => _currentRound;

    /// <summary>Gets the number of pending buffered updates.</summary>
    public int PendingUpdateCount => _updateBuffer.Count;

    private sealed class PendingUpdate
    {
        public PendingUpdate(int clientId, Dictionary<string, T[]> update, int clientRound)
        {
            ClientId = clientId;
            Update = update;
            ClientRound = clientRound;
        }

        public int ClientId { get; }
        public Dictionary<string, T[]> Update { get; }
        public int ClientRound { get; }
    }
}
