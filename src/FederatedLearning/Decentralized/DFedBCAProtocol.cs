namespace AiDotNet.FederatedLearning.Decentralized;

/// <summary>
/// Implements DFedBCA — Decentralized Federated Learning via Block Coordinate Ascent.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In decentralized FL, each client only communicates with its neighbors
/// in a network graph (no central server). DFedBCA improves communication efficiency by only
/// sharing a subset (block) of model parameters per round. Instead of exchanging the full model,
/// each client and its neighbors agree on which "block" of parameters to synchronize this round
/// (e.g., layers 1-3 in round 1, layers 4-6 in round 2). Over multiple rounds, all blocks get
/// synchronized, but each round requires much less bandwidth.</para>
///
/// <para>Algorithm per round:</para>
/// <code>
/// 1. Select block b_t for this round (cyclic or random selection)
/// 2. Each client trains locally for E epochs
/// 3. Clients exchange only block b_t with neighbors
/// 4. Average block b_t with neighbors: w_k[b_t] = sum(mixing_kj * w_j[b_t])
/// 5. Non-selected blocks remain unchanged (local only)
/// </code>
///
/// <para>Reference: DFedBCA: Block Coordinate Ascent for Decentralized Federated Learning (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
internal class DFedBCAProtocol<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly int _numBlocks;
    private readonly BlockSelectionStrategy _selectionStrategy;
    private int _currentBlock;
    private double[]? _blockImportanceScores;

    /// <summary>
    /// Creates a new DFedBCA protocol.
    /// </summary>
    /// <param name="numBlocks">Number of blocks to partition the model into. Default: 4.</param>
    /// <param name="selectionStrategy">How to select which block to synchronize each round. Default: Cyclic.</param>
    public DFedBCAProtocol(
        int numBlocks = 4,
        BlockSelectionStrategy selectionStrategy = BlockSelectionStrategy.Cyclic)
    {
        if (numBlocks <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numBlocks), "Must have at least 1 block.");
        }

        _numBlocks = numBlocks;
        _selectionStrategy = selectionStrategy;
        _currentBlock = 0;
    }

    /// <summary>
    /// Determines which block index to synchronize for the given round.
    /// </summary>
    /// <param name="round">The current communication round.</param>
    /// <param name="seed">Optional random seed for random selection strategy.</param>
    /// <returns>The block index to synchronize (0-based).</returns>
    public int SelectBlock(int round, int? seed = null)
    {
        _currentBlock = _selectionStrategy switch
        {
            BlockSelectionStrategy.Cyclic => round % _numBlocks,
            BlockSelectionStrategy.Random => seed.HasValue
                ? new Random(seed.Value + round).Next(_numBlocks)
                : new Random(round).Next(_numBlocks),
            BlockSelectionStrategy.ImportanceBased => SelectByImportance(round),
            _ => round % _numBlocks
        };

        return _currentBlock;
    }

    /// <summary>
    /// Determines which block to synchronize using importance-based selection.
    /// Selects the block with the highest accumulated change since its last synchronization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of blindly cycling through blocks, importance-based
    /// selection picks the block that has changed the most since it was last synchronized.
    /// This is measured by the accumulated "staleness" (L2 norm of gradient updates) for each
    /// block. Blocks with large unsynced changes are prioritized to reduce convergence error.</para>
    /// </remarks>
    /// <param name="round">The current round number.</param>
    /// <returns>Block index with the highest importance.</returns>
    private int SelectByImportance(int round)
    {
        if (_blockImportanceScores == null || _blockImportanceScores.Length != _numBlocks)
        {
            // No importance scores yet — fall back to cyclic until scores are available.
            return round % _numBlocks;
        }

        // Select the block with the highest importance score.
        int bestBlock = 0;
        double bestScore = _blockImportanceScores[0];
        for (int b = 1; b < _numBlocks; b++)
        {
            if (_blockImportanceScores[b] > bestScore)
            {
                bestScore = _blockImportanceScores[b];
                bestBlock = b;
            }
        }

        // After selecting, decay this block's importance (it's being synced now).
        _blockImportanceScores[bestBlock] = 0;

        return bestBlock;
    }

    /// <summary>
    /// Updates importance scores for all blocks based on gradient magnitudes per block.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After each local training step, the gradient magnitudes per
    /// block tell us how much that block needs to be synchronized. We accumulate these scores
    /// across rounds. When a block is finally synchronized, its score resets to zero. This
    /// creates a "staleness-aware" selection that prioritizes the most out-of-date blocks.</para>
    /// </remarks>
    /// <param name="gradient">Full model gradient dictionary after local training.</param>
    public void UpdateBlockImportanceScores(Dictionary<string, T[]> gradient)
    {
        Guard.NotNull(gradient);
        _blockImportanceScores ??= new double[_numBlocks];

        var layerNames = gradient.Keys.OrderBy(k => k, StringComparer.Ordinal).ToList();
        int layersPerBlock = Math.Max(1, (layerNames.Count + _numBlocks - 1) / _numBlocks);

        for (int b = 0; b < _numBlocks; b++)
        {
            int start = b * layersPerBlock;
            int end = Math.Min(start + layersPerBlock, layerNames.Count);

            double blockNorm = 0;
            for (int i = start; i < end; i++)
            {
                var layerGrad = gradient[layerNames[i]];
                for (int j = 0; j < layerGrad.Length; j++)
                {
                    double v = NumOps.ToDouble(layerGrad[j]);
                    blockNorm += v * v;
                }
            }

            _blockImportanceScores[b] += Math.Sqrt(blockNorm); // Accumulate L2 norm.
        }
    }

    /// <summary>
    /// Extracts a specific block from a client's full model parameters.
    /// </summary>
    /// <param name="fullParameters">The full model parameter dictionary.</param>
    /// <param name="blockIndex">Which block to extract.</param>
    /// <returns>The parameter subset for the given block.</returns>
    public Dictionary<string, T[]> ExtractBlock(Dictionary<string, T[]> fullParameters, int blockIndex)
    {
        Guard.NotNull(fullParameters);
        if (blockIndex < 0 || blockIndex >= _numBlocks)
        {
            throw new ArgumentOutOfRangeException(nameof(blockIndex),
                $"Block index {blockIndex} is out of bounds [0, {_numBlocks}).");
        }

        var layerNames = fullParameters.Keys.OrderBy(k => k, StringComparer.Ordinal).ToList();
        int layersPerBlock = Math.Max(1, (layerNames.Count + _numBlocks - 1) / _numBlocks);
        int start = blockIndex * layersPerBlock;
        int end = Math.Min(start + layersPerBlock, layerNames.Count);

        var block = new Dictionary<string, T[]>();
        for (int i = start; i < end; i++)
        {
            block[layerNames[i]] = fullParameters[layerNames[i]];
        }

        return block;
    }

    /// <summary>
    /// Merges a synchronized block back into the full model parameters.
    /// </summary>
    /// <param name="fullParameters">The full local model (will be updated in-place conceptually).</param>
    /// <param name="synchronizedBlock">The block after neighbor averaging.</param>
    /// <returns>Updated full parameters with the synchronized block merged in.</returns>
    public Dictionary<string, T[]> MergeBlock(
        Dictionary<string, T[]> fullParameters,
        Dictionary<string, T[]> synchronizedBlock)
    {
        Guard.NotNull(fullParameters);
        Guard.NotNull(synchronizedBlock);
        var result = new Dictionary<string, T[]>(fullParameters);

        foreach (var (layerName, layerParams) in synchronizedBlock)
        {
            result[layerName] = layerParams;
        }

        return result;
    }

    /// <summary>
    /// Performs block-coordinate neighbor averaging for the selected block.
    /// </summary>
    /// <param name="clientBlocks">Each neighbor's block parameters (keyed by neighbor ID).</param>
    /// <param name="mixingWeights">Mixing weights (from topology matrix row for this client).</param>
    /// <returns>The averaged block parameters.</returns>
    public Dictionary<string, T[]> AverageBlock(
        Dictionary<int, Dictionary<string, T[]>> clientBlocks,
        Dictionary<int, double> mixingWeights)
    {
        Guard.NotNull(clientBlocks);
        Guard.NotNull(mixingWeights);
        if (clientBlocks.Count == 0)
        {
            throw new ArgumentException("Client blocks cannot be empty.", nameof(clientBlocks));
        }

        var result = new Dictionary<string, T[]>();
        var template = clientBlocks.Values.First();

        foreach (var (layerName, layerParams) in template)
        {
            var averaged = new double[layerParams.Length];
            double layerWeight = 0;

            foreach (var (neighborId, block) in clientBlocks)
            {
                double w = mixingWeights.GetValueOrDefault(neighborId, 0);
                if (block.TryGetValue(layerName, out var neighborLayer))
                {
                    if (neighborLayer.Length != layerParams.Length)
                    {
                        throw new ArgumentException(
                            $"Neighbor {neighborId} layer '{layerName}' length {neighborLayer.Length} differs from expected {layerParams.Length}.");
                    }

                    for (int i = 0; i < neighborLayer.Length; i++)
                    {
                        averaged[i] += w * NumOps.ToDouble(neighborLayer[i]);
                    }

                    layerWeight += w;
                }
            }

            var averagedT = new T[layerParams.Length];
            for (int i = 0; i < averagedT.Length; i++)
            {
                averagedT[i] = NumOps.FromDouble(layerWeight > 0 ? averaged[i] / layerWeight : 0);
            }

            result[layerName] = averagedT;
        }

        return result;
    }

    /// <summary>
    /// Computes the communication savings ratio for this round.
    /// Only 1/numBlocks of the model is transmitted.
    /// </summary>
    /// <returns>Fraction of full-model communication used (e.g., 0.25 for 4 blocks).</returns>
    public double CommunicationRatio => 1.0 / _numBlocks;

    /// <summary>Gets the number of blocks.</summary>
    public int NumBlocks => _numBlocks;

    /// <summary>Gets the block selection strategy.</summary>
    public BlockSelectionStrategy SelectionStrategy => _selectionStrategy;

    /// <summary>Gets the index of the most recently selected block.</summary>
    public int CurrentBlock => _currentBlock;
}

/// <summary>
/// Strategy for selecting which block to synchronize each round.
/// </summary>
public enum BlockSelectionStrategy
{
    /// <summary>Rotate through blocks in order (round-robin).</summary>
    Cyclic = 0,

    /// <summary>Select blocks randomly each round.</summary>
    Random = 1,

    /// <summary>Select blocks based on gradient importance (highest-change block first).</summary>
    ImportanceBased = 2
}
