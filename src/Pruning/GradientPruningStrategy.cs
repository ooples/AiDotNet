using AiDotNet.Interfaces;
using AiDotNet.ModelCompression;

namespace AiDotNet.Pruning;

/// <summary>
/// Prunes weights based on gradient magnitude (sensitivity).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Gradient-based pruning uses gradient information to determine weight importance.
/// Weights with small gradients have little impact on the loss function and can be safely removed.
/// This approach considers both the weight value and how much it affects learning.
/// </para>
/// <para><b>For Beginners:</b> This strategy removes connections that don't learn much.
///
/// Think of it like identifying which team members contribute to a project:
/// - High gradient = This weight changes a lot during training, it's learning something important
/// - Low gradient = This weight barely changes, it's not contributing much to learning
///
/// The importance score is calculated as |weight × gradient|:
/// - If a weight is large BUT has tiny gradients, it might not be doing much
/// - If a weight is learning slowly (small gradient), removing it won't hurt performance
///
/// This is smarter than magnitude-based pruning because it considers learning dynamics,
/// not just weight size. However, it requires gradient information from training.
///
/// Example:
/// - Weight = 0.5, Gradient = 0.001 → Importance = |0.5 × 0.001| = 0.0005 (low, prune it)
/// - Weight = 0.3, Gradient = 0.9 → Importance = |0.3 × 0.9| = 0.27 (high, keep it)
/// </para>
/// </remarks>
public class GradientPruningStrategy<T> : IPruningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Gets whether this strategy requires gradients (true for gradient-based).
    /// </summary>
    public bool RequiresGradients => true;

    /// <summary>
    /// Gets whether this is structured pruning (false for gradient-based).
    /// </summary>
    public bool IsStructured => false;

    /// <summary>
    /// Gets the name of this pruning strategy.
    /// </summary>
    public string Name => "Gradient";

    /// <summary>
    /// Gets supported sparsity patterns.
    /// </summary>
    public IReadOnlyList<SparsityPattern> SupportedPatterns => new[]
    {
        SparsityPattern.Unstructured,
        SparsityPattern.Structured2to4,
        SparsityPattern.StructuredNtoM
    };

    /// <summary>
    /// Initializes a new instance of GradientPruningStrategy.
    /// </summary>
    public GradientPruningStrategy()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Computes importance scores for vector weights.
    /// </summary>
    /// <param name="weights">Weight vector.</param>
    /// <param name="gradients">Gradient vector (required for gradient-based pruning).</param>
    /// <returns>Importance score for each weight (higher = more important).</returns>
    /// <exception cref="ArgumentException">Thrown when gradients are null or shape doesn't match weights</exception>
    public Vector<T> ComputeImportanceScores(Vector<T> weights, Vector<T>? gradients = null)
    {
        if (gradients == null)
            throw new ArgumentException("GradientPruningStrategy requires gradients");

        if (weights.Length != gradients.Length)
            throw new ArgumentException("Weights and gradients must have same length");

        var scores = new T[weights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            var product = _numOps.Multiply(weights[i], gradients[i]);
            scores[i] = _numOps.Abs(product);
        }

        return new Vector<T>(scores);
    }

    /// <summary>
    /// Computes importance scores as the product of weight magnitude and gradient magnitude.
    /// </summary>
    /// <param name="weights">Weight matrix</param>
    /// <param name="gradients">Gradient matrix (required for this strategy)</param>
    /// <returns>Matrix of importance scores</returns>
    /// <exception cref="ArgumentException">Thrown when gradients are null or shape doesn't match weights</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates how important each weight is by looking at both:
    /// 1. The weight's value
    /// 2. How much the weight is learning (its gradient)
    ///
    /// The importance is |weight × gradient|. This tells us how much removing the weight
    /// would affect the model's learning and output.
    /// </para>
    /// </remarks>
    public Matrix<T> ComputeImportanceScores(Matrix<T> weights, Matrix<T>? gradients = null)
    {
        if (gradients == null)
            throw new ArgumentException("GradientPruningStrategy requires gradients");

        if (weights.Rows != gradients.Rows || weights.Columns != gradients.Columns)
            throw new ArgumentException("Weights and gradients must have same shape");

        // Importance = |weight * gradient|
        // This measures how much removing the weight affects the loss
        var scores = new Matrix<T>(weights.Rows, weights.Columns);

        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = 0; j < weights.Columns; j++)
            {
                // |w_ij * g_ij|
                var product = _numOps.Multiply(weights[i, j], gradients[i, j]);
                scores[i, j] = _numOps.Abs(product);
            }
        }

        return scores;
    }

    /// <summary>
    /// Computes importance scores for tensor weights.
    /// </summary>
    /// <param name="weights">Weight tensor.</param>
    /// <param name="gradients">Gradient tensor (required for gradient-based pruning).</param>
    /// <returns>Importance score for each weight (higher = more important).</returns>
    /// <exception cref="ArgumentException">Thrown when gradients are null or shape doesn't match weights</exception>
    public Tensor<T> ComputeImportanceScores(Tensor<T> weights, Tensor<T>? gradients = null)
    {
        if (gradients == null)
            throw new ArgumentException("GradientPruningStrategy requires gradients");

        if (!weights.Shape.SequenceEqual(gradients.Shape))
            throw new ArgumentException("Weights and gradients must have same shape");

        var weightsFlat = weights.ToVector();
        var gradientsFlat = gradients.ToVector();
        var scores = new T[weightsFlat.Length];

        for (int i = 0; i < weightsFlat.Length; i++)
        {
            var product = _numOps.Multiply(weightsFlat[i], gradientsFlat[i]);
            scores[i] = _numOps.Abs(product);
        }

        return Tensor<T>.FromVector(new Vector<T>(scores), (int[])weights.Shape.Clone());
    }

    /// <summary>
    /// Creates a pruning mask for vector weights based on target sparsity.
    /// </summary>
    /// <param name="importanceScores">Importance scores from ComputeImportanceScores.</param>
    /// <param name="targetSparsity">Target sparsity ratio (0 to 1).</param>
    /// <returns>Binary mask (1 = keep, 0 = prune).</returns>
    /// <exception cref="ArgumentException">Thrown when targetSparsity is not between 0 and 1</exception>
    public IPruningMask<T> CreateMask(Vector<T> importanceScores, double targetSparsity)
    {
        if (targetSparsity < 0 || targetSparsity > 1)
            throw new ArgumentException("targetSparsity must be between 0 and 1");

        int numToPrune = (int)(importanceScores.Length * targetSparsity);

        var flatScores = new List<(int idx, T score)>();
        for (int i = 0; i < importanceScores.Length; i++)
            flatScores.Add((i, importanceScores[i]));

        flatScores.Sort((a, b) =>
        {
            double aVal = _numOps.ToDouble(a.score);
            double bVal = _numOps.ToDouble(b.score);
            return aVal.CompareTo(bVal);
        });

        var keepIndices = new bool[importanceScores.Length];
        ArrayPolyfill.Fill(keepIndices, true);

        for (int i = 0; i < numToPrune && i < flatScores.Count; i++)
        {
            keepIndices[flatScores[i].idx] = false;
        }

        return new PruningMask<T>(keepIndices);
    }

    /// <summary>
    /// Creates a pruning mask by selecting weights with lowest gradient-based importance.
    /// </summary>
    /// <param name="importanceScores">Importance scores from ComputeImportanceScores</param>
    /// <param name="targetSparsity">Target sparsity ratio (0 to 1)</param>
    /// <returns>Binary mask (1 = keep, 0 = prune)</returns>
    /// <exception cref="ArgumentException">Thrown when targetSparsity is not between 0 and 1</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates the mask that decides which weights to remove.
    ///
    /// Similar to magnitude pruning, but using gradient-based scores:
    /// - Weights with low |weight × gradient| scores are pruned
    /// - Weights with high scores are kept
    ///
    /// This tends to preserve weights that are actively contributing to learning.
    /// </para>
    /// </remarks>
    public IPruningMask<T> CreateMask(Matrix<T> importanceScores, double targetSparsity)
    {
        // Same logic as magnitude pruning, but with gradient-based scores
        if (targetSparsity < 0 || targetSparsity > 1)
            throw new ArgumentException("targetSparsity must be between 0 and 1");

        int totalElements = importanceScores.Rows * importanceScores.Columns;
        int numToPrune = (int)(totalElements * targetSparsity);

        var flatScores = new List<(int row, int col, T score)>();

        for (int i = 0; i < importanceScores.Rows; i++)
            for (int j = 0; j < importanceScores.Columns; j++)
                flatScores.Add((i, j, importanceScores[i, j]));

        flatScores.Sort((a, b) =>
        {
            double aVal = Convert.ToDouble(a.score);
            double bVal = Convert.ToDouble(b.score);
            return aVal.CompareTo(bVal);
        });

        var keepIndices = new bool[importanceScores.Rows, importanceScores.Columns];

        for (int i = 0; i < importanceScores.Rows; i++)
            for (int j = 0; j < importanceScores.Columns; j++)
                keepIndices[i, j] = true;

        for (int i = 0; i < numToPrune && i < flatScores.Count; i++)
        {
            var (row, col, _) = flatScores[i];
            keepIndices[row, col] = false;
        }

        return new PruningMask<T>(keepIndices);
    }

    /// <summary>
    /// Creates a pruning mask for tensor weights based on target sparsity.
    /// </summary>
    /// <param name="importanceScores">Importance scores from ComputeImportanceScores.</param>
    /// <param name="targetSparsity">Target sparsity ratio (0 to 1).</param>
    /// <returns>Binary mask (1 = keep, 0 = prune).</returns>
    /// <exception cref="ArgumentException">Thrown when targetSparsity is not between 0 and 1</exception>
    public IPruningMask<T> CreateMask(Tensor<T> importanceScores, double targetSparsity)
    {
        if (targetSparsity < 0 || targetSparsity > 1)
            throw new ArgumentException("targetSparsity must be between 0 and 1");

        var flatScores = importanceScores.ToVector();
        int numToPrune = (int)(flatScores.Length * targetSparsity);

        var scoredIndices = new List<(int idx, T score)>();
        for (int i = 0; i < flatScores.Length; i++)
            scoredIndices.Add((i, flatScores[i]));

        scoredIndices.Sort((a, b) =>
        {
            double aVal = _numOps.ToDouble(a.score);
            double bVal = _numOps.ToDouble(b.score);
            return aVal.CompareTo(bVal);
        });

        var keepIndices = new bool[flatScores.Length];
        ArrayPolyfill.Fill(keepIndices, true);

        for (int i = 0; i < numToPrune && i < scoredIndices.Count; i++)
        {
            keepIndices[scoredIndices[i].idx] = false;
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
        ArrayPolyfill.Fill(keepIndices, true);

        for (int groupStart = 0; groupStart < totalElements; groupStart += m)
        {
            int groupEnd = Math.Min(groupStart + m, totalElements);
            int groupSize = groupEnd - groupStart;

            var groupScores = new List<(int idx, T score)>();
            for (int i = groupStart; i < groupEnd; i++)
                groupScores.Add((i, flatScores[i]));

            groupScores.Sort((a, b) => _numOps.ToDouble(a.score).CompareTo(_numOps.ToDouble(b.score)));

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
    /// <remarks>
    /// <para><b>For Beginners:</b> This actually removes the weights by setting them to zero.
    /// The pruned weights are those identified as having low gradient-based importance.
    /// </para>
    /// </remarks>
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
        var prunedFlat = pruned.ToVector();

        for (int i = 0; i < weights.Length; i++)
            weights[i] = prunedFlat[i];
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
        var nonZeroValues = new List<T>();
        var rowIndices = new List<int>();
        var colIndices = new List<int>();

        // For simplicity, treat tensors as flattened 1D for COO format
        // or as 2D (first dimension × product of remaining dimensions)
        var dims = (int[])weights.Shape.Clone();
        int rows = dims.Length > 0 ? dims[0] : 1;
        int cols = dims.Length > 1 ? dims.Skip(1).Aggregate(1, (a, b) => a * b) : flatWeights.Length / rows;

        if (format == SparseFormat.COO)
        {
            for (int i = 0; i < flatWeights.Length; i++)
            {
                if (!_numOps.Equals(flatWeights[i], _numOps.Zero))
                {
                    nonZeroValues.Add(flatWeights[i]);
                    int row = i / cols;
                    int col = i % cols;
                    rowIndices.Add(row);
                    colIndices.Add(col);
                }
            }

            return new SparseCompressionResult<T>
            {
                Format = SparseFormat.COO,
                Values = nonZeroValues.ToArray(),
                RowIndices = rowIndices.ToArray(),
                ColumnIndices = colIndices.ToArray(),
                OriginalShape = dims
            };
        }
        else if (format == SparseFormat.CSR)
        {
            var rowPointers = new List<int> { 0 };

            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++)
                {
                    int idx = row * cols + col;
                    if (idx < flatWeights.Length && !_numOps.Equals(flatWeights[idx], _numOps.Zero))
                    {
                        nonZeroValues.Add(flatWeights[idx]);
                        colIndices.Add(col);
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
                OriginalShape = dims
            };
        }
        else if (format == SparseFormat.CSC)
        {
            var colPointers = new List<int> { 0 };

            for (int col = 0; col < cols; col++)
            {
                for (int row = 0; row < rows; row++)
                {
                    int idx = row * cols + col;
                    if (idx < flatWeights.Length && !_numOps.Equals(flatWeights[idx], _numOps.Zero))
                    {
                        nonZeroValues.Add(flatWeights[idx]);
                        rowIndices.Add(row);
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
                OriginalShape = dims
            };
        }
        else if (format == SparseFormat.Structured2to4)
        {
            // For 2:4 sparsity, store non-zero values and their positions within each group
            var mask = new List<byte>();

            for (int i = 0; i < flatWeights.Length; i += 4)
            {
                byte groupMask = 0;
                int groupSize = Math.Min(4, flatWeights.Length - i);

                for (int j = 0; j < groupSize; j++)
                {
                    if (!_numOps.Equals(flatWeights[i + j], _numOps.Zero))
                    {
                        nonZeroValues.Add(flatWeights[i + j]);
                        groupMask |= (byte)(1 << j);
                    }
                }
                mask.Add(groupMask);
            }

            return new SparseCompressionResult<T>
            {
                Format = SparseFormat.Structured2to4,
                Values = nonZeroValues.ToArray(),
                SparsityMask = mask.ToArray(),
                SparsityN = 2,
                SparsityM = 4,
                OriginalShape = dims
            };
        }
        else if (format == SparseFormat.StructuredNtoM)
        {
            // N:M format requires explicit N and M parameters which aren't available through this method.
            // Use CreateNtoMMask() to create masks with specific N:M patterns, then call ToSparseFormat
            // with Structured2to4 format, or implement an overload that accepts N and M parameters.
            throw new NotSupportedException(
                "StructuredNtoM format requires explicit N and M parameters. " +
                "Use Structured2to4 for 2:4 sparsity, or implement ToSparseFormat(weights, format, n, m) overload.");
        }
        else
        {
            throw new NotSupportedException($"Sparse format {format} is not supported");
        }
    }
}
