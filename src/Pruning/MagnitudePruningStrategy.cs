using AiDotNet.ModelCompression;

namespace AiDotNet.Pruning;

/// <summary>
/// Prunes weights with smallest absolute values.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Magnitude-based pruning is one of the simplest and most effective pruning strategies.
/// It removes weights with the smallest absolute values, based on the intuition that
/// weights with small magnitudes contribute less to the network's output.
/// </para>
/// <para><b>For Beginners:</b> This strategy removes the weakest connections in your neural network.
///
/// Think of it like trimming weak branches from a tree:
/// - Thick, strong branches (large weight values) carry lots of nutrients and stay
/// - Thin, weak branches (small weight values) don't contribute much and get trimmed
///
/// In mathematical terms:
/// - Each weight gets an importance score equal to its absolute value |w|
/// - Weights with the smallest scores are pruned (set to zero)
/// - This is simple but surprisingly effective!
///
/// For example:
/// - A weight of 0.001 has low importance and might be pruned
/// - A weight of 0.9 has high importance and will likely be kept
/// - A weight of -0.8 has high importance too (|-0.8| = 0.8)
///
/// This technique can often remove 50-90% of weights with minimal accuracy loss!
/// </para>
/// </remarks>
public class MagnitudePruningStrategy<T> : IPruningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Gets the name of this pruning strategy.
    /// </summary>
    public string Name => "Magnitude";

    /// <summary>
    /// Gets whether this strategy requires gradients (false for magnitude-based).
    /// </summary>
    public bool RequiresGradients => false;

    /// <summary>
    /// Gets whether this is structured pruning (false for magnitude-based).
    /// </summary>
    public bool IsStructured => false;

    /// <summary>
    /// Gets supported sparsity patterns.
    /// </summary>
    public IReadOnlyList<SparsityPattern> SupportedPatterns { get; } = new[]
    {
        SparsityPattern.Unstructured,
        SparsityPattern.Structured2to4,
        SparsityPattern.StructuredNtoM
    };

    /// <summary>
    /// Initializes a new instance of MagnitudePruningStrategy.
    /// </summary>
    public MagnitudePruningStrategy()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    #region Importance Scoring

    /// <summary>
    /// Computes importance scores as absolute values of weights for vectors.
    /// </summary>
    /// <param name="weights">Weight vector</param>
    /// <param name="gradients">Gradients (not used for magnitude-based pruning)</param>
    /// <returns>Vector of importance scores</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates how important each weight is.
    /// For magnitude pruning, importance is simply the absolute value of the weight.
    /// Larger absolute values mean more important weights.
    /// </para>
    /// </remarks>
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
    /// Computes importance scores as absolute values of weights for matrices.
    /// </summary>
    /// <param name="weights">Weight matrix</param>
    /// <param name="gradients">Gradients (not used for magnitude-based pruning)</param>
    /// <returns>Matrix of importance scores</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates how important each weight is.
    /// For magnitude pruning, importance is simply the absolute value of the weight.
    /// Larger absolute values mean more important weights.
    /// </para>
    /// </remarks>
    public Matrix<T> ComputeImportanceScores(Matrix<T> weights, Matrix<T>? gradients = null)
    {
        // Importance = absolute value of weight
        var scores = new Matrix<T>(weights.Rows, weights.Columns);

        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = 0; j < weights.Columns; j++)
            {
                // |w_ij|
                scores[i, j] = _numOps.Abs(weights[i, j]);
            }
        }

        return scores;
    }

    /// <summary>
    /// Computes importance scores as absolute values of weights for tensors.
    /// </summary>
    /// <param name="weights">Weight tensor</param>
    /// <param name="gradients">Gradients (not used for magnitude-based pruning)</param>
    /// <returns>Tensor of importance scores</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates how important each weight is.
    /// For magnitude pruning, importance is simply the absolute value of the weight.
    /// Larger absolute values mean more important weights.
    /// </para>
    /// </remarks>
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

    #endregion

    #region Mask Creation

    /// <summary>
    /// Creates a pruning mask for vectors by selecting the smallest magnitude weights to prune.
    /// </summary>
    /// <param name="importanceScores">Importance scores from ComputeImportanceScores</param>
    /// <param name="targetSparsity">Target sparsity ratio (0 to 1)</param>
    /// <returns>Binary mask (1 = keep, 0 = prune)</returns>
    /// <exception cref="ArgumentException">Thrown when targetSparsity is not between 0 and 1</exception>
    public IPruningMask<T> CreateMask(Vector<T> importanceScores, double targetSparsity)
    {
        if (targetSparsity < 0 || targetSparsity > 1)
            throw new ArgumentException("targetSparsity must be between 0 and 1");

        // Separate zero and non-zero scores
        // Zero scores indicate already-pruned weights (from previous pruning rounds)
        var zeroScores = new List<int>();
        var nonZeroScores = new List<(int idx, T score)>();

        for (int i = 0; i < importanceScores.Length; i++)
        {
            if (_numOps.Equals(importanceScores[i], _numOps.Zero))
            {
                zeroScores.Add(i);
            }
            else
            {
                nonZeroScores.Add((i, importanceScores[i]));
            }
        }

        // Sort non-zero scores by importance (ascending, so smallest are first)
        nonZeroScores.Sort((a, b) => _numOps.ToDouble(a.score).CompareTo(_numOps.ToDouble(b.score)));

        var keepIndices = new bool[importanceScores.Length];
        ArrayPolyfill.Fill(keepIndices, true);

        // Mark already-zero weights as pruned (they stay pruned)
        foreach (int idx in zeroScores)
        {
            keepIndices[idx] = false;
        }

        // For iterative pruning: targetSparsity applies to REMAINING non-zero weights
        // This allows users to prune "X% more" in each round
        // Example: 25% sparsity in round 1, then 33% of remaining = ~50% total
        int numToPruneFromNonZero = (int)Math.Round(nonZeroScores.Count * targetSparsity);

        // Prune the smallest non-zero weights
        for (int i = 0; i < numToPruneFromNonZero && i < nonZeroScores.Count; i++)
        {
            keepIndices[nonZeroScores[i].idx] = false;
        }

        // Special case: if ALL weights are zero, handle based on targetSparsity
        if (nonZeroScores.Count == 0)
        {
            if (targetSparsity > 0)
            {
                // Honor the sparsity target by marking some as "pruned" (they're already zero anyway)
                int numToPrune = (int)Math.Round(importanceScores.Length * targetSparsity);
                // All weights are zero - just mark the first numToPrune as pruned
                for (int i = 0; i < numToPrune && i < zeroScores.Count; i++)
                {
                    keepIndices[zeroScores[i]] = false;
                }
                // Reset the remaining to "kept" to achieve target sparsity
                for (int i = numToPrune; i < zeroScores.Count; i++)
                {
                    keepIndices[zeroScores[i]] = true;
                }
            }
            else
            {
                // targetSparsity == 0 means keep all weights (no pruning)
                // Set all zero-score indices to kept so nothing remains pruned
                foreach (int idx in zeroScores)
                {
                    keepIndices[idx] = true;
                }
            }
        }

        var mask = new PruningMask<T>(keepIndices);
        mask.UpdateMask(keepIndices);

        return mask;
    }

    /// <summary>
    /// Creates a pruning mask for matrices by selecting the smallest magnitude weights to prune.
    /// </summary>
    /// <param name="importanceScores">Importance scores from ComputeImportanceScores</param>
    /// <param name="targetSparsity">Target sparsity ratio (0 to 1)</param>
    /// <returns>Binary mask (1 = keep, 0 = prune)</returns>
    /// <exception cref="ArgumentException">Thrown when targetSparsity is not between 0 and 1</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates the mask that decides which weights to remove.
    ///
    /// If targetSparsity is 0.7:
    /// - 70% of the weights with the smallest importance scores will be pruned (set to 0)
    /// - 30% of the weights with the highest importance scores will be kept
    ///
    /// The method sorts all weights by importance and draws a line where the bottom 70% get pruned.
    /// </para>
    /// </remarks>
    public IPruningMask<T> CreateMask(Matrix<T> importanceScores, double targetSparsity)
    {
        if (targetSparsity < 0 || targetSparsity > 1)
            throw new ArgumentException("targetSparsity must be between 0 and 1");

        int totalElements = importanceScores.Rows * importanceScores.Columns;
        int numToPrune = (int)(totalElements * targetSparsity);

        // Flatten scores and find threshold
        var flatScores = new List<(int row, int col, T score)>();

        for (int i = 0; i < importanceScores.Rows; i++)
        {
            for (int j = 0; j < importanceScores.Columns; j++)
            {
                flatScores.Add((i, j, importanceScores[i, j]));
            }
        }

        // Sort by importance (ascending, so smallest are first)
        flatScores.Sort((a, b) => _numOps.ToDouble(a.score).CompareTo(_numOps.ToDouble(b.score)));

        // Create mask: prune the smallest numToPrune elements
        var keepIndices = new bool[importanceScores.Rows, importanceScores.Columns];

        for (int i = 0; i < importanceScores.Rows; i++)
            for (int j = 0; j < importanceScores.Columns; j++)
                keepIndices[i, j] = true;

        for (int i = 0; i < numToPrune && i < flatScores.Count; i++)
        {
            var (row, col, _) = flatScores[i];
            keepIndices[row, col] = false;
        }

        var mask = new PruningMask<T>(keepIndices);
        mask.UpdateMask(keepIndices);

        return mask;
    }

    /// <summary>
    /// Creates a pruning mask for tensors by selecting the smallest magnitude weights to prune.
    /// </summary>
    /// <param name="importanceScores">Importance scores from ComputeImportanceScores</param>
    /// <param name="targetSparsity">Target sparsity ratio (0 to 1)</param>
    /// <returns>Binary mask (1 = keep, 0 = prune)</returns>
    /// <exception cref="ArgumentException">Thrown when targetSparsity is not between 0 and 1</exception>
    public IPruningMask<T> CreateMask(Tensor<T> importanceScores, double targetSparsity)
    {
        if (targetSparsity < 0 || targetSparsity > 1)
            throw new ArgumentException("targetSparsity must be between 0 and 1");

        var flatScores = importanceScores.ToVector();
        int totalElements = flatScores.Length;
        int numToPrune = (int)(totalElements * targetSparsity);

        // Create indexed list of scores
        var indexedScores = new List<(int idx, T score)>();
        for (int i = 0; i < flatScores.Length; i++)
        {
            indexedScores.Add((i, flatScores[i]));
        }

        // Sort by importance (ascending, so smallest are first)
        indexedScores.Sort((a, b) => _numOps.ToDouble(a.score).CompareTo(_numOps.ToDouble(b.score)));

        // Create mask: prune the smallest numToPrune elements
        var keepIndices = new bool[totalElements];
        ArrayPolyfill.Fill(keepIndices, true);

        for (int i = 0; i < numToPrune && i < indexedScores.Count; i++)
        {
            keepIndices[indexedScores[i].idx] = false;
        }

        return new PruningMask<T>(keepIndices);
    }

    /// <summary>
    /// Creates a 2:4 structured sparsity mask (NVIDIA Ampere compatible).
    /// </summary>
    /// <param name="importanceScores">Importance scores.</param>
    /// <returns>2:4 structured mask (exactly 2 zeros per 4 elements).</returns>
    public IPruningMask<T> Create2to4Mask(Tensor<T> importanceScores)
    {
        // Create 2:4 structured sparsity mask (2 zeros per 4 elements)
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
        var flatScores = importanceScores.ToVector();
        int totalElements = flatScores.Length;
        var keepIndices = new bool[totalElements];

        // Initialize all to true
        ArrayPolyfill.Fill(keepIndices, true);

        // Process in groups of m elements
        for (int groupStart = 0; groupStart < totalElements; groupStart += m)
        {
            int groupEnd = Math.Min(groupStart + m, totalElements);
            int groupSize = groupEnd - groupStart;

            // Collect scores for this group
            var groupScores = new List<(int idx, T score)>();
            for (int i = groupStart; i < groupEnd; i++)
            {
                groupScores.Add((i, flatScores[i]));
            }

            // Sort by importance (ascending)
            groupScores.Sort((a, b) => _numOps.ToDouble(a.score).CompareTo(_numOps.ToDouble(b.score)));

            // Prune the n smallest in this group
            int numToPrune = Math.Min(n, groupSize);
            for (int i = 0; i < numToPrune; i++)
            {
                keepIndices[groupScores[i].idx] = false;
            }
        }

        return new PruningMask<T>(keepIndices);
    }

    #endregion

    #region Pruning Application

    /// <summary>
    /// Applies the pruning mask to vector weights in-place.
    /// </summary>
    /// <param name="weights">Weight vector to prune</param>
    /// <param name="mask">Pruning mask to apply</param>
    public void ApplyPruning(Vector<T> weights, IPruningMask<T> mask)
    {
        var pruned = mask.Apply(weights);

        // Update weights in-place
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = pruned[i];
        }
    }

    /// <summary>
    /// Applies the pruning mask to matrix weights in-place.
    /// </summary>
    /// <param name="weights">Weight matrix to prune</param>
    /// <param name="mask">Pruning mask to apply</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This actually removes the weights by setting them to zero.
    /// After this method, the weight matrix will have zeros wherever the mask was 0.
    /// </para>
    /// </remarks>
    public void ApplyPruning(Matrix<T> weights, IPruningMask<T> mask)
    {
        var pruned = mask.Apply(weights);

        // Update weights in-place
        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = 0; j < weights.Columns; j++)
            {
                weights[i, j] = pruned[i, j];
            }
        }
    }

    /// <summary>
    /// Applies the pruning mask to tensor weights in-place.
    /// </summary>
    /// <param name="weights">Weight tensor to prune</param>
    /// <param name="mask">Pruning mask to apply</param>
    public void ApplyPruning(Tensor<T> weights, IPruningMask<T> mask)
    {
        var pruned = mask.Apply(weights);
        var prunedFlat = pruned.ToVector();

        // Update tensor in-place by copying pruned data back
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = prunedFlat[i];
        }
    }

    /// <summary>
    /// Converts pruned weights to sparse format for efficient storage.
    /// </summary>
    /// <param name="weights">Pruned weights (containing zeros).</param>
    /// <param name="format">Target sparse format.</param>
    /// <returns>Sparse representation.</returns>
    public SparseCompressionResult<T> ToSparseFormat(Tensor<T> weights, SparseFormat format)
    {
        var flatWeightsVec = weights.ToVector();
        var flatWeights = new T[flatWeightsVec.Length];
        for (int i = 0; i < flatWeightsVec.Length; i++)
            flatWeights[i] = flatWeightsVec[i];
        var dims = (int[])weights.Shape.Clone();

        return format switch
        {
            SparseFormat.COO => ConvertToCOO(flatWeights, dims),
            SparseFormat.CSR => ConvertToCSR(flatWeights, dims),
            SparseFormat.CSC => ConvertToCSC(flatWeights, dims),
            SparseFormat.Structured2to4 => ConvertTo2to4(flatWeights, dims),
            SparseFormat.StructuredNtoM => ConvertToNtoM(flatWeights, dims, 2, 4),
            _ => throw new ArgumentException($"Unsupported sparse format: {format}")
        };
    }

    #endregion

    #region Sparse Format Conversion

    private SparseCompressionResult<T> ConvertToCOO(T[] flatWeights, int[] dims)
    {
        var values = new List<T>();
        var rowIndices = new List<int>();
        var colIndices = new List<int>();

        // For simplicity, treat as 2D (flatten to matrix if higher dimensional)
        // dims[0] represents rows, remaining dimensions are flattened to cols
        int cols = dims.Length > 1 ? dims.Skip(1).Aggregate(1, (a, b) => a * b) : 1;

        for (int i = 0; i < flatWeights.Length; i++)
        {
            if (!_numOps.Equals(flatWeights[i], _numOps.Zero))
            {
                values.Add(flatWeights[i]);
                rowIndices.Add(i / cols);
                colIndices.Add(i % cols);
            }
        }

        return new SparseCompressionResult<T>
        {
            Format = SparseFormat.COO,
            Values = values.ToArray(),
            RowIndices = rowIndices.ToArray(),
            ColumnIndices = colIndices.ToArray(),
            OriginalShape = dims
        };
    }

    private SparseCompressionResult<T> ConvertToCSR(T[] flatWeights, int[] dims)
    {
        var values = new List<T>();
        var colIndices = new List<int>();
        var rowPointers = new List<int>();

        // For simplicity, treat as 2D
        int rows = dims[0];
        int cols = dims.Length > 1 ? dims.Skip(1).Aggregate(1, (a, b) => a * b) : 1;

        rowPointers.Add(0);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                int idx = i * cols + j;
                if (!_numOps.Equals(flatWeights[idx], _numOps.Zero))
                {
                    values.Add(flatWeights[idx]);
                    colIndices.Add(j);
                }
            }
            rowPointers.Add(values.Count);
        }

        return new SparseCompressionResult<T>
        {
            Format = SparseFormat.CSR,
            Values = values.ToArray(),
            ColumnIndices = colIndices.ToArray(),
            RowPointers = rowPointers.ToArray(),
            OriginalShape = dims
        };
    }

    private SparseCompressionResult<T> ConvertToCSC(T[] flatWeights, int[] dims)
    {
        var values = new List<T>();
        var rowIndices = new List<int>();
        var colPointers = new List<int>();

        // For simplicity, treat as 2D
        int rows = dims[0];
        int cols = dims.Length > 1 ? dims.Skip(1).Aggregate(1, (a, b) => a * b) : 1;

        colPointers.Add(0);
        for (int j = 0; j < cols; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                int idx = i * cols + j;
                if (!_numOps.Equals(flatWeights[idx], _numOps.Zero))
                {
                    values.Add(flatWeights[idx]);
                    rowIndices.Add(i);
                }
            }
            colPointers.Add(values.Count);
        }

        return new SparseCompressionResult<T>
        {
            Format = SparseFormat.CSC,
            Values = values.ToArray(),
            RowIndices = rowIndices.ToArray(),
            ColumnPointers = colPointers.ToArray(),
            OriginalShape = dims
        };
    }

    private SparseCompressionResult<T> ConvertTo2to4(T[] flatWeights, int[] dims)
    {
        return ConvertToNtoM(flatWeights, dims, 2, 4);
    }

    private SparseCompressionResult<T> ConvertToNtoM(T[] flatWeights, int[] dims, int n, int m)
    {
        var values = new List<T>();
        var mask = new List<byte>();

        // Process in groups of M elements
        for (int i = 0; i < flatWeights.Length; i += m)
        {
            int groupSize = Math.Min(m, flatWeights.Length - i);
            byte groupMask = 0;

            for (int j = 0; j < groupSize; j++)
            {
                if (!_numOps.Equals(flatWeights[i + j], _numOps.Zero))
                {
                    values.Add(flatWeights[i + j]);
                    groupMask |= (byte)(1 << j);
                }
            }

            mask.Add(groupMask);
        }

        return new SparseCompressionResult<T>
        {
            Format = n == 2 && m == 4 ? SparseFormat.Structured2to4 : SparseFormat.StructuredNtoM,
            Values = values.ToArray(),
            SparsityMask = mask.ToArray(),
            SparsityN = n,
            SparsityM = m,
            OriginalShape = dims
        };
    }

    #endregion
}
