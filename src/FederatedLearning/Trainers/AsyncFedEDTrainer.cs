namespace AiDotNet.FederatedLearning.Trainers;

/// <summary>
/// Implements AsyncFedED — Asynchronous FL with Entropy-Driven client scheduling.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In async FL, the server doesn't wait for all clients — it processes
/// updates as they arrive. But which clients should train next? AsyncFedED uses an information-
/// theoretic approach: it estimates each client's "information gain" (how much the global model
/// would improve from that client's update) using entropy of the client's local loss distribution.
/// Clients with higher entropy (more uncertain predictions, more to learn from) are scheduled first.
/// This prioritizes the most informative clients, converging faster than random scheduling.</para>
///
/// <para>Scheduling:</para>
/// <code>
/// 1. Server maintains entropy estimate per client: H_k = -sum(p_c * log(p_c))
/// 2. Higher entropy → more uncertain → more informative → higher priority
/// 3. Apply staleness discount: priority_k = H_k * decay^(t - t_k)
/// 4. Select top-K clients with highest priority
/// </code>
///
/// <para>Reference: AsyncFedED: Entropy-Driven Scheduling for Asynchronous Federated Learning (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class AsyncFedEDTrainer<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly object _stateLock = new();
    private readonly double _stalenessDecay;
    private readonly double _explorationBonus;
    private readonly int _selectionBudget;
    private readonly Dictionary<int, double> _clientEntropies;
    private readonly Dictionary<int, int> _lastParticipationRound;

    /// <summary>
    /// Creates a new AsyncFedED trainer.
    /// </summary>
    /// <param name="stalenessDecay">Decay factor for stale client priorities. Default: 0.95.</param>
    /// <param name="explorationBonus">Bonus for clients that haven't participated recently. Default: 0.1.</param>
    /// <param name="selectionBudget">Maximum clients to select per round. Default: 10.</param>
    public AsyncFedEDTrainer(
        double stalenessDecay = 0.95,
        double explorationBonus = 0.1,
        int selectionBudget = 10)
    {
        if (stalenessDecay <= 0 || stalenessDecay > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(stalenessDecay), "Staleness decay must be in (0, 1].");
        }

        if (explorationBonus < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(explorationBonus), "Exploration bonus must be non-negative.");
        }

        if (selectionBudget <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(selectionBudget), "Selection budget must be positive.");
        }

        _stalenessDecay = stalenessDecay;
        _explorationBonus = explorationBonus;
        _selectionBudget = selectionBudget;
        _clientEntropies = new Dictionary<int, double>();
        _lastParticipationRound = new Dictionary<int, int>();
    }

    /// <summary>
    /// Updates the entropy estimate for a client based on their local loss distribution.
    /// </summary>
    /// <param name="clientId">The client identifier.</param>
    /// <param name="classLosses">Per-class or per-sample losses from the client's local evaluation.</param>
    /// <param name="currentRound">The current communication round.</param>
    public void UpdateClientEntropy(int clientId, double[] classLosses, int currentRound)
    {
        Guard.NotNull(classLosses);
        if (currentRound < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(currentRound), "Current round must be non-negative.");
        }

        if (classLosses.Length == 0)
        {
            lock (_stateLock)
            {
                _clientEntropies[clientId] = 0;
                _lastParticipationRound[clientId] = currentRound;
            }

            return;
        }

        // Convert losses to probability distribution via softmax.
        double maxLoss = classLosses.Max();
        double expSum = 0;
        var probs = new double[classLosses.Length];

        for (int i = 0; i < classLosses.Length; i++)
        {
            probs[i] = Math.Exp(classLosses[i] - maxLoss);
            expSum += probs[i];
        }

        // Compute Shannon entropy.
        double entropy = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            double p = probs[i] / expSum;
            if (p > 1e-10)
            {
                entropy -= p * Math.Log(p);
            }
        }

        lock (_stateLock)
        {
            _clientEntropies[clientId] = entropy;
            _lastParticipationRound[clientId] = currentRound;
        }
    }

    /// <summary>
    /// Selects the most informative clients for the next round.
    /// </summary>
    /// <param name="availableClients">Set of currently available client IDs.</param>
    /// <param name="currentRound">The current communication round.</param>
    /// <returns>Ordered list of selected client IDs (highest priority first).</returns>
    public List<int> SelectClients(IReadOnlyCollection<int> availableClients, int currentRound)
    {
        Guard.NotNull(availableClients);
        if (currentRound < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(currentRound), "Current round must be non-negative.");
        }

        if (availableClients.Count == 0)
        {
            return new List<int>();
        }

        var priorities = new Dictionary<int, double>();

        lock (_stateLock)
        {
            foreach (var clientId in availableClients)
            {
                double entropy = _clientEntropies.GetValueOrDefault(clientId, 1.0); // default high entropy for new clients
                int lastRound = _lastParticipationRound.GetValueOrDefault(clientId, -1);

                // Staleness discount: older info → less reliable entropy estimate.
                int staleness = lastRound >= 0 ? currentRound - lastRound : currentRound;
                double stalenessDiscount = Math.Pow(_stalenessDecay, staleness);

                // Exploration bonus for clients that haven't participated recently.
                double exploration = _explorationBonus * staleness;

                priorities[clientId] = entropy * stalenessDiscount + exploration;
            }
        }

        // Select top-K by priority.
        return priorities
            .OrderByDescending(kvp => kvp.Value)
            .Take(Math.Min(_selectionBudget, availableClients.Count))
            .Select(kvp => kvp.Key)
            .ToList();
    }

    /// <summary>
    /// Aggregates client updates with entropy-weighted contributions.
    /// Clients with higher entropy (more informative) get higher aggregation weight.
    /// </summary>
    /// <param name="clientUpdates">Parameter updates from selected clients.</param>
    /// <param name="currentRound">The current round for staleness computation.</param>
    /// <returns>Entropy-weighted aggregated update.</returns>
    public Dictionary<string, T[]> AggregateWithEntropyWeights(
        Dictionary<int, Dictionary<string, T[]>> clientUpdates,
        int currentRound)
    {
        Guard.NotNull(clientUpdates);
        if (clientUpdates.Count == 0)
        {
            throw new ArgumentException("Client updates cannot be empty.", nameof(clientUpdates));
        }

        if (currentRound < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(currentRound), "Current round must be non-negative.");
        }

        // Compute entropy-based aggregation weights.
        var weights = new Dictionary<int, double>();
        double totalWeight = 0;

        lock (_stateLock)
        {
            foreach (var clientId in clientUpdates.Keys)
            {
                double entropy = _clientEntropies.GetValueOrDefault(clientId, 1.0);
                int lastRound = _lastParticipationRound.GetValueOrDefault(clientId, currentRound);
                int staleness = currentRound - lastRound;
                double w = entropy * Math.Pow(_stalenessDecay, Math.Max(staleness, 0));
                weights[clientId] = w;
                totalWeight += w;
            }
        }

        var result = new Dictionary<string, T[]>();
        var template = clientUpdates.Values.First();

        foreach (var (layerName, layerParams) in template)
        {
            var merged = new double[layerParams.Length];
            double layerWeight = 0;

            foreach (var (clientId, update) in clientUpdates)
            {
                if (update.TryGetValue(layerName, out var clientLayer))
                {
                    if (clientLayer.Length != layerParams.Length)
                    {
                        throw new ArgumentException(
                            $"Client {clientId} layer '{layerName}' length {clientLayer.Length} differs from expected {layerParams.Length}.");
                    }

                    double w = weights[clientId];
                    layerWeight += w;

                    for (int i = 0; i < clientLayer.Length; i++)
                    {
                        merged[i] += w * NumOps.ToDouble(clientLayer[i]);
                    }
                }
            }

            var mergedT = new T[layerParams.Length];
            for (int i = 0; i < mergedT.Length; i++)
            {
                mergedT[i] = NumOps.FromDouble(layerWeight > 0 ? merged[i] / layerWeight : 0);
            }

            result[layerName] = mergedT;
        }

        return result;
    }

    /// <summary>Gets the staleness decay factor.</summary>
    public double StalenessDecay => _stalenessDecay;

    /// <summary>Gets the exploration bonus.</summary>
    public double ExplorationBonus => _explorationBonus;

    /// <summary>Gets the selection budget.</summary>
    public int SelectionBudget => _selectionBudget;

    /// <summary>Gets a snapshot of the current entropy estimates for all tracked clients.</summary>
    public IReadOnlyDictionary<int, double> ClientEntropies
    {
        get
        {
            lock (_stateLock)
            {
                return new Dictionary<int, double>(_clientEntropies);
            }
        }
    }
}
