namespace AiDotNet.Pruning;

/// <summary>
/// Implements the Lottery Ticket Hypothesis (Frankle &amp; Carbin, 2019).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Lottery Ticket Hypothesis states that dense neural networks contain sparse subnetworks
/// (winning tickets) that, when trained in isolation from initialization, can match the performance
/// of the original network. This strategy finds these winning tickets through iterative pruning
/// and resetting weights to their initial values.
/// </para>
/// <para><b>For Beginners:</b> This strategy is based on a fascinating discovery in neural networks!
///
/// The Lottery Ticket Hypothesis says:
/// "Inside every large neural network, there's a smaller 'winning lottery ticket' network
/// that could have achieved the same performance if trained from the start."
///
/// The analogy:
/// Imagine you're building a team:
/// - You start with 100 people (full network)
/// - After working together, you realize only 20 people did most of the work
/// - If you had started with just those 20 people from day one, you'd achieve the same results!
///
/// How it works:
/// 1. Train the network to completion
/// 2. Find which weights became large (important)
/// 3. Create a mask keeping only those weights
/// 4. Reset the KEPT weights to their original random initialization
/// 5. Retrain with the mask - you'll match the original accuracy!
///
/// This is different from regular pruning because:
/// - Regular pruning: Train → Prune → Fine-tune
/// - Lottery ticket: Train → Prune → Reset to init → Retrain from scratch
///
/// Why this matters:
/// - Shows that the structure (which connections) matters more than learned values
/// - Enables training sparse networks from scratch
/// - Challenges assumptions about why neural networks work
///
/// Example workflow:
/// 1. Initialize network with random weights W₀
/// 2. Train to get final weights W_final
/// 3. Create mask M based on |W_final| (keep largest 30%)
/// 4. Reset: W = W₀ ⊙ M (original weights, masked)
/// 5. Retrain this sparse network - it matches full network performance!
/// </para>
/// </remarks>
public class LotteryTicketPruningStrategy<T> : IPruningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Dictionary<string, Matrix<T>> _initialWeights;
    private readonly int _iterativeRounds;

    /// <summary>
    /// Gets whether this strategy requires gradients (false for lottery ticket).
    /// </summary>
    public bool RequiresGradients => false;

    /// <summary>
    /// Gets whether this is structured pruning (false for lottery ticket).
    /// </summary>
    public bool IsStructured => false;

    /// <summary>
    /// Creates a new lottery ticket pruning strategy.
    /// </summary>
    /// <param name="iterativeRounds">Number of iterative pruning rounds (default 5)</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Iterative rounds control how gradually we prune.
    ///
    /// Instead of removing 90% of weights at once, iterative pruning:
    /// - Round 1: Remove 20%
    /// - Round 2: Remove 20% of remaining
    /// - ...and so on
    ///
    /// This gentler approach often finds better "lottery tickets."
    /// More rounds = more gradual = often better results, but slower.
    /// </para>
    /// </remarks>
    public LotteryTicketPruningStrategy(int iterativeRounds = 5)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _initialWeights = new Dictionary<string, Matrix<T>>();
        _iterativeRounds = iterativeRounds;
    }

    /// <summary>
    /// Stores initial weights before training (critical for lottery ticket).
    /// </summary>
    /// <param name="layerName">Name/identifier for the layer</param>
    /// <param name="weights">Initial weight matrix</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This saves the random initialization before training.
    ///
    /// This is THE KEY step for lottery tickets:
    /// - You MUST save weights before any training
    /// - Later, you'll reset pruned networks to these exact values
    /// - Without this, lottery ticket hypothesis doesn't work!
    ///
    /// Call this right after initializing your network, before the first training step.
    /// </para>
    /// </remarks>
    public void StoreInitialWeights(string layerName, Matrix<T> weights)
    {
        _initialWeights[layerName] = weights.Clone();
    }

    /// <summary>
    /// Gets the stored initial weights for a layer.
    /// </summary>
    /// <param name="layerName">Name/identifier for the layer</param>
    /// <returns>Initial weight matrix</returns>
    /// <exception cref="InvalidOperationException">Thrown when no weights stored for the layer</exception>
    public Matrix<T> GetInitialWeights(string layerName)
    {
        if (!_initialWeights.ContainsKey(layerName))
            throw new InvalidOperationException($"No initial weights stored for layer {layerName}");

        return _initialWeights[layerName].Clone();
    }

    /// <summary>
    /// Computes importance scores using magnitude-based scoring.
    /// </summary>
    /// <param name="weights">Weight matrix (typically after training)</param>
    /// <param name="gradients">Gradients (not used for lottery ticket)</param>
    /// <returns>Matrix of importance scores</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For lottery tickets, we use the final trained weights' magnitudes.
    ///
    /// The intuition: Weights that became large during training found useful features.
    /// Those are the connections that form our "winning lottery ticket."
    /// </para>
    /// </remarks>
    public Matrix<T> ComputeImportanceScores(Matrix<T> weights, Matrix<T>? gradients = null)
    {
        // Use magnitude-based scores (lottery ticket uses magnitude pruning)
        var scores = new Matrix<T>(weights.Rows, weights.Columns);

        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = 0; j < weights.Columns; j++)
            {
                scores[i, j] = _numOps.Abs(weights[i, j]);
            }
        }

        return scores;
    }

    /// <summary>
    /// Creates a pruning mask using iterative magnitude pruning.
    /// </summary>
    /// <param name="importanceScores">Importance scores from ComputeImportanceScores</param>
    /// <param name="targetSparsity">Target sparsity ratio (0 to 1)</param>
    /// <returns>Binary mask (1 = keep, 0 = prune)</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates the mask in multiple gentle rounds rather than all at once.
    ///
    /// For example, to reach 90% sparsity in 5 rounds:
    /// - Each round prunes about 37% of REMAINING weights
    /// - Round 1: 100 → 63 weights remain
    /// - Round 2: 63 → 40 weights remain
    /// - Round 3: 40 → 25 weights remain
    /// - Round 4: 25 → 16 weights remain
    /// - Round 5: 16 → 10 weights remain (90% total sparsity)
    ///
    /// This gradual approach helps find better lottery tickets.
    /// </para>
    /// </remarks>
    public IPruningMask<T> CreateMask(Matrix<T> importanceScores, double targetSparsity)
    {
        // Iterative magnitude pruning to target sparsity
        // Each round prunes (1 - (1 - targetSparsity)^(1/rounds)) of remaining weights
        double prunePerRound = 1.0 - Math.Pow(1.0 - targetSparsity, 1.0 / _iterativeRounds);

        var currentMask = new PruningMask<T>(importanceScores.Rows, importanceScores.Columns);

        for (int round = 0; round < _iterativeRounds; round++)
        {
            // Compute scores for current non-pruned weights
            var maskedScores = currentMask.Apply(importanceScores);

            // Find threshold for this round
            int totalRemaining = CountNonZero(maskedScores);
            int numToPrune = (int)(totalRemaining * prunePerRound);

            var flatScores = new List<(int row, int col, double score)>();

            for (int i = 0; i < maskedScores.Rows; i++)
            {
                for (int j = 0; j < maskedScores.Columns; j++)
                {
                    if (!_numOps.Equals(maskedScores[i, j], _numOps.Zero))
                    {
                        double scoreVal = Convert.ToDouble(maskedScores[i, j]);
                        flatScores.Add((i, j, scoreVal));
                    }
                }
            }

            flatScores.Sort((a, b) => a.score.CompareTo(b.score));

            var keepIndices = new bool[importanceScores.Rows, importanceScores.Columns];

            for (int i = 0; i < importanceScores.Rows; i++)
                for (int j = 0; j < importanceScores.Columns; j++)
                    keepIndices[i, j] = !_numOps.Equals(currentMask.Apply(importanceScores)[i, j], _numOps.Zero);

            for (int i = 0; i < numToPrune && i < flatScores.Count; i++)
            {
                var (row, col, _) = flatScores[i];
                keepIndices[row, col] = false;
            }

            currentMask.UpdateMask(keepIndices);
        }

        return currentMask;
    }

    /// <summary>
    /// Applies the pruning mask to weights in-place.
    /// </summary>
    /// <param name="weights">Weight matrix to prune</param>
    /// <param name="mask">Pruning mask to apply</param>
    public void ApplyPruning(Matrix<T> weights, IPruningMask<T> mask)
    {
        var pruned = mask.Apply(weights);

        for (int i = 0; i < weights.Rows; i++)
            for (int j = 0; j < weights.Columns; j++)
                weights[i, j] = pruned[i, j];
    }

    /// <summary>
    /// Resets pruned weights to their initial values (key step in lottery ticket).
    /// </summary>
    /// <param name="layerName">Name/identifier for the layer</param>
    /// <param name="weights">Weight matrix to reset</param>
    /// <param name="mask">Pruning mask indicating which weights to keep</param>
    /// <exception cref="ArgumentException">Thrown when weight dimensions don't match initial weights</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is what makes lottery ticket special!
    ///
    /// After finding which connections are important:
    /// 1. Take the initial random weights (before any training)
    /// 2. Apply the mask to keep only the "winning ticket" connections
    /// 3. Set weights to these masked initial values
    /// 4. Now retrain - you'll match the original network's performance!
    ///
    /// This proves that the structure (which connections exist) is more important
    /// than the specific learned values. The winning lottery ticket could have
    /// succeeded from the start - we just needed to know which connections to keep!
    /// </para>
    /// </remarks>
    public void ResetToInitialWeights(string layerName, Matrix<T> weights, IPruningMask<T> mask)
    {
        var initial = GetInitialWeights(layerName);

        if (initial.Rows != weights.Rows || initial.Columns != weights.Columns)
            throw new ArgumentException("Weight dimensions don't match initial weights");

        // Reset non-pruned weights to their initialization
        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = 0; j < weights.Columns; j++)
            {
                // Keep initial value where mask is 1, zero otherwise
                var maskValue = mask.Apply(initial)[i, j];
                weights[i, j] = maskValue;
            }
        }
    }

    /// <summary>
    /// Counts the number of non-zero elements in a matrix.
    /// </summary>
    private int CountNonZero(Matrix<T> matrix)
    {
        int count = 0;
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
                if (!_numOps.Equals(matrix[i, j], _numOps.Zero))
                    count++;
        return count;
    }
}
