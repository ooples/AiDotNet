using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.ModelCompression;

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
    /// Gets the name of this pruning strategy.
    /// </summary>
    public string Name => "LotteryTicket";

    /// <summary>
    /// Gets whether this strategy requires gradients (false for lottery ticket).
    /// </summary>
    public bool RequiresGradients => false;

    /// <summary>
    /// Gets whether this is structured pruning (false for lottery ticket).
    /// </summary>
    public bool IsStructured => false;

    /// <summary>
    /// Gets supported sparsity patterns (unstructured and N:M patterns).
    /// </summary>
    public IReadOnlyList<SparsityPattern> SupportedPatterns => new[]
    {
        SparsityPattern.Unstructured,
        SparsityPattern.Structured2to4,
        SparsityPattern.StructuredNtoM
    };

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
        if (iterativeRounds <= 0)
            throw new ArgumentOutOfRangeException(nameof(iterativeRounds), "iterativeRounds must be greater than 0.");

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
        if (!_initialWeights.TryGetValue(layerName, out var weights))
            throw new InvalidOperationException($"No initial weights stored for layer {layerName}");

        return weights.Clone();
    }

    /// <summary>
    /// Computes importance scores using magnitude-based scoring for vectors.
    /// </summary>
    /// <param name="weights">Weight vector (typically after training)</param>
    /// <param name="gradients">Gradients (not used for lottery ticket)</param>
    /// <returns>Vector of importance scores</returns>
    public Vector<T> ComputeImportanceScores(Vector<T> weights, Vector<T>? gradients = null)
    {
        var scores = new T[weights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            scores[i] = _numOps.Abs(weights[i]);
        }
        return new Vector<T>(scores);
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
    /// Computes importance scores using magnitude-based scoring for tensors.
    /// </summary>
    /// <param name="weights">Weight tensor (typically after training)</param>
    /// <param name="gradients">Gradients (not used for lottery ticket)</param>
    /// <returns>Tensor of importance scores</returns>
    public Tensor<T> ComputeImportanceScores(Tensor<T> weights, Tensor<T>? gradients = null)
    {
        var flatWeights = weights.ToVector();
        var scores = new T[flatWeights.Length];
        for (int i = 0; i < flatWeights.Length; i++)
        {
            scores[i] = _numOps.Abs(flatWeights[i]);
        }
        return Tensor<T>.FromVector(new Vector<T>(scores), (int[])weights.Shape.Clone());
    }

    /// <summary>
    /// Creates a pruning mask using iterative magnitude pruning for vectors.
    /// </summary>
    /// <param name="importanceScores">Importance scores from ComputeImportanceScores</param>
    /// <param name="targetSparsity">Target sparsity ratio (0 to 1)</param>
    /// <returns>Binary mask (1 = keep, 0 = prune)</returns>
    public IPruningMask<T> CreateMask(Vector<T> importanceScores, double targetSparsity)
    {
        if (targetSparsity < 0.0 || targetSparsity > 1.0)
            throw new ArgumentException("targetSparsity must be between 0 and 1 (inclusive).", nameof(targetSparsity));

        double prunePerRound = 1.0 - Math.Pow(1.0 - targetSparsity, 1.0 / _iterativeRounds);
        var currentMask = new PruningMask<T>(1, importanceScores.Length);

        for (int round = 0; round < _iterativeRounds; round++)
        {
            var maskedScores = currentMask.Apply(importanceScores);
            int totalRemaining = CountNonZero(maskedScores);
            // Use rounding instead of truncation to achieve target sparsity more accurately
            int numToPrune = (int)Math.Round(totalRemaining * prunePerRound);

            var flatScores = new List<(int idx, double score)>();
            for (int i = 0; i < maskedScores.Length; i++)
            {
                if (!_numOps.Equals(maskedScores[i], _numOps.Zero))
                {
                    flatScores.Add((i, _numOps.ToDouble(maskedScores[i])));
                }
            }

            flatScores.Sort((a, b) => a.score.CompareTo(b.score));

            // Reuse maskedScores instead of recomputing currentMask.Apply(importanceScores) in loop
            var keepIndices = new bool[importanceScores.Length];
            for (int i = 0; i < importanceScores.Length; i++)
                keepIndices[i] = !_numOps.Equals(maskedScores[i], _numOps.Zero);

            for (int i = 0; i < numToPrune && i < flatScores.Count; i++)
            {
                keepIndices[flatScores[i].idx] = false;
            }

            currentMask.UpdateMask(keepIndices);
        }

        return currentMask;
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
        if (targetSparsity < 0.0 || targetSparsity > 1.0)
            throw new ArgumentException("targetSparsity must be between 0 and 1 (inclusive).", nameof(targetSparsity));

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
            // Use rounding instead of truncation to achieve target sparsity more accurately
            int numToPrune = (int)Math.Round(totalRemaining * prunePerRound);

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

            // Reuse maskedScores instead of recomputing currentMask.Apply(importanceScores) in loop (O(n⁴) → O(n²))
            var keepIndices = new bool[importanceScores.Rows, importanceScores.Columns];

            for (int i = 0; i < importanceScores.Rows; i++)
                for (int j = 0; j < importanceScores.Columns; j++)
                    keepIndices[i, j] = !_numOps.Equals(maskedScores[i, j], _numOps.Zero);

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
    /// Creates a pruning mask using iterative magnitude pruning for tensors.
    /// </summary>
    /// <param name="importanceScores">Importance scores from ComputeImportanceScores</param>
    /// <param name="targetSparsity">Target sparsity ratio (0 to 1)</param>
    /// <returns>Binary mask (1 = keep, 0 = prune)</returns>
    public IPruningMask<T> CreateMask(Tensor<T> importanceScores, double targetSparsity)
    {
        if (targetSparsity < 0.0 || targetSparsity > 1.0)
            throw new ArgumentException("targetSparsity must be between 0 and 1 (inclusive).", nameof(targetSparsity));

        double prunePerRound = 1.0 - Math.Pow(1.0 - targetSparsity, 1.0 / _iterativeRounds);
        var flatScoresInit = importanceScores.ToVector();
        var keepIndicesInit = new bool[flatScoresInit.Length];
        ArrayPolyfill.Fill(keepIndicesInit, true);
        var currentMask = new PruningMask<T>(keepIndicesInit);

        for (int round = 0; round < _iterativeRounds; round++)
        {
            var maskedScores = currentMask.Apply(importanceScores);
            var flatMasked = maskedScores.ToVector();

            int totalRemaining = CountNonZero(flatMasked);
            // Use rounding instead of truncation to achieve target sparsity more accurately
            int numToPrune = (int)Math.Round(totalRemaining * prunePerRound);

            var sortedScores = GetSortedNonZeroScores(flatMasked);
            var keepIndices = BuildKeepIndicesFromMasked(flatMasked);

            PruneLowestScores(keepIndices, sortedScores, numToPrune);
            currentMask.UpdateMask(keepIndices);
        }

        return currentMask;
    }

    /// <summary>
    /// Gets non-zero scores sorted by value in ascending order.
    /// </summary>
    /// <param name="maskedScores">Vector of masked importance scores.</param>
    /// <returns>
    /// List of tuples containing the index and double-converted score for each non-zero element,
    /// sorted by score in ascending order (lowest scores first).
    /// </returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method extracts only the non-zero scores from a masked
    /// vector and sorts them so the lowest-scoring (least important) weights come first.
    /// These are the candidates for pruning in the next round.</para>
    /// </remarks>
    private List<(int idx, double score)> GetSortedNonZeroScores(Vector<T> maskedScores)
    {
        var scores = new List<(int idx, double score)>();
        for (int i = 0; i < maskedScores.Length; i++)
        {
            if (!_numOps.Equals(maskedScores[i], _numOps.Zero))
            {
                scores.Add((i, _numOps.ToDouble(maskedScores[i])));
            }
        }
        scores.Sort((a, b) => a.score.CompareTo(b.score));
        return scores;
    }

    /// <summary>
    /// Builds a boolean array indicating which indices currently have non-zero values.
    /// </summary>
    /// <param name="maskedScores">Vector of masked importance scores.</param>
    /// <returns>
    /// Boolean array where true indicates the weight at that index should be kept (non-zero),
    /// and false indicates it has already been pruned (zero).
    /// </returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a "keep/prune" map from the current
    /// state of the masked scores. Weights that are already zero stay pruned (false),
    /// while non-zero weights are candidates to keep (true) unless they get pruned
    /// in this round.</para>
    /// </remarks>
    private bool[] BuildKeepIndicesFromMasked(Vector<T> maskedScores)
    {
        var keepIndices = new bool[maskedScores.Length];
        for (int i = 0; i < maskedScores.Length; i++)
            keepIndices[i] = !_numOps.Equals(maskedScores[i], _numOps.Zero);
        return keepIndices;
    }

    /// <summary>
    /// Marks the lowest-scoring indices as pruned by setting their keep flags to false.
    /// </summary>
    /// <param name="keepIndices">Boolean array to modify, where false means pruned.</param>
    /// <param name="sortedScores">List of scores sorted in ascending order (lowest first).</param>
    /// <param name="numToPrune">Number of weights to prune from the lowest-scoring elements.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes the pre-sorted list of scores and
    /// marks the first N (lowest-scoring) weights for removal. After this, those weights
    /// will have their keepIndices set to false, indicating they should be zeroed out.</para>
    /// </remarks>
    private static void PruneLowestScores(bool[] keepIndices, List<(int idx, double score)> sortedScores, int numToPrune)
    {
        for (int i = 0; i < numToPrune && i < sortedScores.Count; i++)
        {
            keepIndices[sortedScores[i].idx] = false;
        }
    }

    /// <summary>
    /// Creates a 2:4 structured sparsity mask (NVIDIA Ampere compatible).
    /// </summary>
    /// <param name="importanceScores">Importance scores.</param>
    /// <returns>2:4 structured mask (exactly 2 zeros per 4 elements).</returns>
    public IPruningMask<T> Create2to4Mask(Tensor<T> importanceScores)
    {
        return CreateNtoMMask(importanceScores, 2, 4);
    }

    /// <summary>
    /// Creates an N:M structured sparsity mask.
    /// </summary>
    /// <param name="importanceScores">Importance scores.</param>
    /// <param name="n">Number of zeros per group.</param>
    /// <param name="m">Group size.</param>
    /// <returns>N:M structured mask.</returns>
    public IPruningMask<T> CreateNtoMMask(Tensor<T> importanceScores, int n, int m)
    {
        if (m <= 0)
            throw new ArgumentOutOfRangeException(nameof(m), "m must be greater than 0.");
        if (n < 0)
            throw new ArgumentOutOfRangeException(nameof(n), "n must be greater than or equal to 0.");
        if (n > m)
            throw new ArgumentException($"n ({n}) cannot be greater than m ({m}).", nameof(n));

        var flatScores = importanceScores.ToVector();
        int totalElements = flatScores.Length;
        var keepIndices = new bool[totalElements];
        ArrayPolyfill.Fill(keepIndices, true);

        // Process in groups of m elements
        for (int groupStart = 0; groupStart < totalElements; groupStart += m)
        {
            int groupEnd = Math.Min(groupStart + m, totalElements);
            int groupSize = groupEnd - groupStart;

            var groupScores = new List<(int idx, double score)>();
            for (int i = groupStart; i < groupEnd; i++)
                groupScores.Add((i, _numOps.ToDouble(flatScores[i])));

            groupScores.Sort((a, b) => a.score.CompareTo(b.score));

            int numToPrune = Math.Min(n, groupSize);
            for (int i = 0; i < numToPrune; i++)
                keepIndices[groupScores[i].idx] = false;
        }

        return new PruningMask<T>(keepIndices);
    }

    /// <summary>
    /// Applies pruning mask to vector weights in-place.
    /// </summary>
    /// <param name="weights">Weight vector to prune</param>
    /// <param name="mask">Pruning mask to apply</param>
    public void ApplyPruning(Vector<T> weights, IPruningMask<T> mask)
    {
        var pruned = mask.Apply(weights);
        for (int i = 0; i < weights.Length; i++)
            weights[i] = pruned[i];
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
    /// Applies pruning mask to tensor weights in-place.
    /// </summary>
    /// <param name="weights">Weight tensor to prune</param>
    /// <param name="mask">Pruning mask to apply</param>
    public void ApplyPruning(Tensor<T> weights, IPruningMask<T> mask)
    {
        var pruned = mask.Apply(weights);
        var flatPruned = pruned.ToVector();

        // Write pruned values back to the original tensor using flat indexer
        for (int i = 0; i < flatPruned.Length; i++)
            weights[i] = flatPruned[i];
    }

    /// <summary>
    /// Converts pruned weights to sparse format for efficient storage.
    /// </summary>
    /// <param name="weights">Pruned weights (containing zeros).</param>
    /// <param name="format">Target sparse format.</param>
    /// <returns>Sparse representation.</returns>
    /// <remarks>
    /// For StructuredNtoM format, use the overload that accepts n and m parameters.
    /// </remarks>
    public SparseCompressionResult<T> ToSparseFormat(Tensor<T> weights, SparseFormat format)
    {
        // For N:M formats, delegate to the overload with explicit parameters
        if (format == SparseFormat.Structured2to4)
            return ToSparseFormat(weights, format, 2, 4);
        if (format == SparseFormat.StructuredNtoM)
            return ToSparseFormat(weights, format, 2, 4); // Default to 2:4 pattern

        var flatWeights = weights.ToVector();
        var nonZeroValues = new List<T>();
        var rowIndices = new List<int>();
        var colIndices = new List<int>();

        // For simplicity, we'll treat the tensor as a 2D matrix by reshaping
        // First dimension vs rest of dimensions
        int rows = weights.Shape[0];
        int cols = flatWeights.Length / rows;

        switch (format)
        {
            case SparseFormat.COO:
                // Coordinate format: store (row, col, value) triplets
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        int idx = i * cols + j;
                        if (!_numOps.Equals(flatWeights[idx], _numOps.Zero))
                        {
                            nonZeroValues.Add(flatWeights[idx]);
                            rowIndices.Add(i);
                            colIndices.Add(j);
                        }
                    }
                }

                return new SparseCompressionResult<T>
                {
                    Format = SparseFormat.COO,
                    Values = nonZeroValues.ToArray(),
                    RowIndices = rowIndices.ToArray(),
                    ColumnIndices = colIndices.ToArray(),
                    OriginalShape = weights.Shape.ToArray()
                };

            case SparseFormat.CSR:
                // Compressed Sparse Row: row pointers + column indices + values
                var rowPointers = new List<int> { 0 };

                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        int idx = i * cols + j;
                        if (!_numOps.Equals(flatWeights[idx], _numOps.Zero))
                        {
                            nonZeroValues.Add(flatWeights[idx]);
                            colIndices.Add(j);
                        }
                    }
                    rowPointers.Add(nonZeroValues.Count);
                }

                return new SparseCompressionResult<T>
                {
                    Format = SparseFormat.CSR,
                    Values = nonZeroValues.ToArray(),
                    RowPointers = rowPointers.ToArray(),
                    ColumnIndices = colIndices.ToArray(),
                    OriginalShape = weights.Shape.ToArray()
                };

            case SparseFormat.CSC:
                // Compressed Sparse Column: column pointers + row indices + values
                var colPointers = new List<int> { 0 };

                for (int j = 0; j < cols; j++)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        int idx = i * cols + j;
                        if (!_numOps.Equals(flatWeights[idx], _numOps.Zero))
                        {
                            nonZeroValues.Add(flatWeights[idx]);
                            rowIndices.Add(i);
                        }
                    }
                    colPointers.Add(nonZeroValues.Count);
                }

                return new SparseCompressionResult<T>
                {
                    Format = SparseFormat.CSC,
                    Values = nonZeroValues.ToArray(),
                    ColumnPointers = colPointers.ToArray(),
                    RowIndices = rowIndices.ToArray(),
                    OriginalShape = weights.Shape.ToArray()
                };

            default:
                throw new NotSupportedException($"Sparse format {format} is not supported");
        }
    }

    /// <summary>
    /// Converts pruned weights to N:M structured sparse format for efficient storage.
    /// </summary>
    /// <param name="weights">Pruned weights (containing zeros).</param>
    /// <param name="format">Target sparse format (should be Structured2to4 or StructuredNtoM).</param>
    /// <param name="n">Number of zeros per group in N:M sparsity pattern.</param>
    /// <param name="m">Group size in N:M sparsity pattern.</param>
    /// <returns>Sparse representation with N:M pattern metadata.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when n or m are invalid.</exception>
    public SparseCompressionResult<T> ToSparseFormat(Tensor<T> weights, SparseFormat format, int n, int m)
    {
        if (m <= 0)
            throw new ArgumentOutOfRangeException(nameof(m), "m must be greater than 0.");
        if (n < 0)
            throw new ArgumentOutOfRangeException(nameof(n), "n must be greater than or equal to 0.");
        if (n > m)
            throw new ArgumentException($"n ({n}) cannot be greater than m ({m}).", nameof(n));

        // For non-N:M formats, delegate to the standard method
        if (format != SparseFormat.Structured2to4 && format != SparseFormat.StructuredNtoM)
            return ToSparseFormat(weights, format);

        var flatWeights = weights.ToVector();
        var nonZeroValues = new List<T>();
        var mask = new List<byte>();

        for (int i = 0; i < flatWeights.Length; i++)
        {
            if (!_numOps.Equals(flatWeights[i], _numOps.Zero))
            {
                nonZeroValues.Add(flatWeights[i]);
                mask.Add(1);
            }
            else
            {
                mask.Add(0);
            }
        }

        return new SparseCompressionResult<T>
        {
            Format = format,
            Values = nonZeroValues.ToArray(),
            SparsityMask = mask.ToArray(),
            SparsityN = n,
            SparsityM = m,
            OriginalShape = weights.Shape.ToArray()
        };
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

        // Compute masked initial weights once (O(n²) instead of O(n⁴))
        var maskedInitial = mask.Apply(initial);

        // Reset non-pruned weights to their initialization
        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = 0; j < weights.Columns; j++)
            {
                // Keep initial value where mask is 1, zero otherwise
                weights[i, j] = maskedInitial[i, j];
            }
        }
    }

    /// <summary>
    /// Counts the number of non-zero elements in a vector.
    /// </summary>
    private int CountNonZero(Vector<T> vector)
    {
        int count = 0;
        for (int i = 0; i < vector.Length; i++)
            if (!_numOps.Equals(vector[i], _numOps.Zero))
                count++;
        return count;
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
