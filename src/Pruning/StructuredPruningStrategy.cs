using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ModelCompression;

namespace AiDotNet.Pruning;

/// <summary>
/// Structured pruning removes entire neurons/filters/channels.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Structured pruning removes entire structural units (neurons, filters, channels) rather than
/// individual weights. This results in smaller dense networks that are easier to deploy and
/// can achieve actual speedups on standard hardware, unlike unstructured pruning which creates
/// sparse matrices that require specialized libraries for acceleration.
/// </para>
/// <para><b>For Beginners:</b> This strategy removes entire building blocks, not just individual connections.
///
/// The difference between structured and unstructured pruning:
///
/// Unstructured pruning (like magnitude or gradient):
/// - Removes individual connections randomly scattered throughout the network
/// - Creates a "swiss cheese" pattern with holes everywhere
/// - Requires special sparse matrix libraries to run faster
/// - Harder to deploy on mobile or edge devices
///
/// Structured pruning:
/// - Removes entire neurons, filters, or channels
/// - Creates a smaller but still dense (solid) network
/// - Runs faster on ANY hardware - no special libraries needed!
/// - Easier to deploy and understand
///
/// Analogy: Building a smaller car
/// - Unstructured: Remove random bolts and parts everywhere (car still same size, just hollow)
/// - Structured: Remove entire seats or components (car is actually smaller)
///
/// Types of structured pruning:
/// 1. **Neuron pruning**: Remove entire neurons (columns in weight matrix)
///    - Reduces layer width
///    - Common in fully connected layers
///
/// 2. **Filter pruning**: Remove entire convolutional filters
///    - Reduces number of feature maps
///    - Very effective for CNNs
///
/// 3. **Channel pruning**: Remove input/output channels
///    - Reduces both computation and memory
///    - Commonly used with filter pruning
///
/// Example:
/// Original layer: 100 neurons
/// After 40% structured pruning: 60 neurons (actually smaller!)
/// After 40% unstructured pruning: 100 neurons (60% of weights are zero, but layer size unchanged)
///
/// Trade-offs:
/// - Structured pruning: Less flexibility, but real speedups
/// - Unstructured pruning: More flexibility, but needs special hardware/software
/// </para>
/// </remarks>
public class StructuredPruningStrategy<T> : IPruningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly StructurePruningType _pruningType;

    /// <summary>
    /// Gets whether this strategy requires gradients (false for structured pruning).
    /// </summary>
    public bool RequiresGradients => false;

    /// <summary>
    /// Gets whether this is structured pruning (true).
    /// </summary>
    public bool IsStructured => true;

    /// <summary>
    /// Gets the name of this pruning strategy.
    /// </summary>
    public string Name => "Structured";

    /// <summary>
    /// Gets supported sparsity patterns for structured pruning.
    /// </summary>
    public IReadOnlyList<SparsityPattern> SupportedPatterns { get; }

    /// <summary>
    /// Defines the type of structural unit to prune.
    /// </summary>
    public enum StructurePruningType
    {
        /// <summary>
        /// Prune entire output neurons (columns in weight matrix).
        /// </summary>
        /// <remarks>
        /// <para><b>For Beginners:</b> This removes entire neurons from a layer.
        /// In a weight matrix, each column represents one neuron's connections.
        /// Removing a column = removing that neuron entirely.
        /// </para>
        /// </remarks>
        Neuron,

        /// <summary>
        /// Prune entire filters in convolutional layers.
        /// </summary>
        /// <remarks>
        /// <para><b>For Beginners:</b> In CNNs, filters detect patterns (edges, textures, etc.).
        /// This removes entire filters that don't contribute much.
        /// Fewer filters = faster convolutions and less memory.
        /// </para>
        /// </remarks>
        Filter,

        /// <summary>
        /// Prune entire channels in convolutional layers.
        /// </summary>
        /// <remarks>
        /// <para><b>For Beginners:</b> Channels are different "views" or feature maps.
        /// For example, RGB images have 3 input channels (red, green, blue).
        /// This removes entire channels that aren't important.
        /// </para>
        /// </remarks>
        Channel
    }

    /// <summary>
    /// Creates a new structured pruning strategy.
    /// </summary>
    /// <param name="pruningType">Type of structural pruning to perform</param>
    public StructuredPruningStrategy(StructurePruningType pruningType = StructurePruningType.Neuron)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _pruningType = pruningType;

        // Initialize supported patterns based on pruning type
        var patterns = new List<SparsityPattern>();
        switch (pruningType)
        {
            case StructurePruningType.Neuron:
                patterns.Add(SparsityPattern.RowStructured);
                patterns.Add(SparsityPattern.ColumnStructured);
                break;
            case StructurePruningType.Filter:
                patterns.Add(SparsityPattern.FilterStructured);
                break;
            case StructurePruningType.Channel:
                patterns.Add(SparsityPattern.ChannelStructured);
                break;
        }
        SupportedPatterns = patterns.AsReadOnly();
    }

    /// <summary>
    /// Computes importance scores for structural units.
    /// </summary>
    /// <param name="weights">Weight matrix</param>
    /// <param name="gradients">Gradients (not used for basic structured pruning)</param>
    /// <returns>Matrix of importance scores (same value per structural unit)</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This scores entire groups (neurons/filters) instead of individual weights.
    ///
    /// For neuron pruning:
    /// - Each neuron is scored by the L2 norm (magnitude) of all its incoming weights
    /// - Higher norm = more important neuron
    /// - All weights in the same column get the same score (they're part of the same neuron)
    ///
    /// L2 norm intuition:
    /// If a neuron has strong connections (large weights), it's probably important.
    /// If all its weights are tiny, it's not contributing much.
    /// </para>
    /// </remarks>
    public Matrix<T> ComputeImportanceScores(Matrix<T> weights, Matrix<T>? gradients = null)
    {
        var scores = new Matrix<T>(weights.Rows, weights.Columns);

        switch (_pruningType)
        {
            case StructurePruningType.Neuron:
                // Score for each neuron (column) = L2 norm of its weights
                for (int col = 0; col < weights.Columns; col++)
                {
                    double columnNorm = 0;
                    for (int row = 0; row < weights.Rows; row++)
                    {
                        double val = _numOps.ToDouble(weights[row, col]);
                        columnNorm += val * val;
                    }
                    columnNorm = Math.Sqrt(columnNorm);

                    // Assign same score to all weights in column
                    for (int row = 0; row < weights.Rows; row++)
                    {
                        scores[row, col] = _numOps.FromDouble(columnNorm);
                    }
                }
                break;

            case StructurePruningType.Filter:
                // For 2D matrix, Filter pruning treats rows as filters
                // Score for each filter (row) = L2 norm of its weights
                for (int row = 0; row < weights.Rows; row++)
                {
                    double rowNorm = 0;
                    for (int col = 0; col < weights.Columns; col++)
                    {
                        double val = _numOps.ToDouble(weights[row, col]);
                        rowNorm += val * val;
                    }
                    rowNorm = Math.Sqrt(rowNorm);

                    // Assign same score to all weights in row
                    for (int col = 0; col < weights.Columns; col++)
                    {
                        scores[row, col] = _numOps.FromDouble(rowNorm);
                    }
                }
                break;

            case StructurePruningType.Channel:
                // For 2D matrix, Channel pruning treats columns as channels (same as Neuron)
                // Score for each channel (column) = L2 norm of its weights
                for (int col = 0; col < weights.Columns; col++)
                {
                    double columnNorm = 0;
                    for (int row = 0; row < weights.Rows; row++)
                    {
                        double val = _numOps.ToDouble(weights[row, col]);
                        columnNorm += val * val;
                    }
                    columnNorm = Math.Sqrt(columnNorm);

                    // Assign same score to all weights in column
                    for (int row = 0; row < weights.Rows; row++)
                    {
                        scores[row, col] = _numOps.FromDouble(columnNorm);
                    }
                }
                break;
        }

        return scores;
    }

    /// <summary>
    /// Creates a structured pruning mask.
    /// </summary>
    /// <param name="importanceScores">Importance scores from ComputeImportanceScores</param>
    /// <param name="targetSparsity">Target sparsity ratio (0 to 1)</param>
    /// <returns>Binary mask (1 = keep, 0 = prune)</returns>
    /// <exception cref="ArgumentException">Thrown when targetSparsity is not between 0 and 1</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a mask where entire columns/rows are either all 1s or all 0s.
    ///
    /// For neuron pruning with 50% sparsity:
    /// 1. Score each neuron (column) by its L2 norm
    /// 2. Sort neurons by score
    /// 3. Keep top 50%, prune bottom 50%
    /// 4. Entire columns are set to either all 1s (keep) or all 0s (prune)
    ///
    /// This is different from unstructured pruning where individual elements can be 0 or 1.
    /// </para>
    /// </remarks>
    public IPruningMask<T> CreateMask(Matrix<T> importanceScores, double targetSparsity)
    {
        if (double.IsNaN(targetSparsity) || double.IsInfinity(targetSparsity))
            throw new ArgumentException("targetSparsity cannot be NaN or Infinity.", nameof(targetSparsity));

        if (targetSparsity < 0 || targetSparsity > 1)
            throw new ArgumentException("targetSparsity must be between 0 and 1", nameof(targetSparsity));

        if (importanceScores.Rows == 0 || importanceScores.Columns == 0)
            throw new ArgumentException("importanceScores matrix cannot be empty.", nameof(importanceScores));

        var keepIndices = new bool[importanceScores.Rows, importanceScores.Columns];

        switch (_pruningType)
        {
            case StructurePruningType.Neuron:
            case StructurePruningType.Channel:
                // Prune entire columns (neurons/channels)
                int totalColumns = importanceScores.Columns;
                int columnsToPrune = (int)(totalColumns * targetSparsity);

                // Get score for each column (all rows in column have same score)
                var columnScores = new List<(int col, double score)>();
                for (int col = 0; col < importanceScores.Columns; col++)
                {
                    double score = _numOps.ToDouble(importanceScores[0, col]);
                    columnScores.Add((col, score));
                }

                // Sort by score (ascending - lowest scores get pruned first)
                columnScores.Sort((a, b) => a.score.CompareTo(b.score));

                // Mark columns to keep
                var keepColumns = new bool[importanceScores.Columns];
                for (int i = 0; i < importanceScores.Columns; i++)
                    keepColumns[i] = true;

                for (int i = 0; i < columnsToPrune; i++)
                {
                    keepColumns[columnScores[i].col] = false;
                }

                // Create mask
                for (int row = 0; row < importanceScores.Rows; row++)
                {
                    for (int col = 0; col < importanceScores.Columns; col++)
                    {
                        keepIndices[row, col] = keepColumns[col];
                    }
                }
                break;

            case StructurePruningType.Filter:
                // Prune entire rows (filters)
                int totalRows = importanceScores.Rows;
                int rowsToPrune = (int)(totalRows * targetSparsity);

                // Get score for each row (all columns in row have same score)
                var rowScores = new List<(int row, double score)>();
                for (int row = 0; row < importanceScores.Rows; row++)
                {
                    double score = _numOps.ToDouble(importanceScores[row, 0]);
                    rowScores.Add((row, score));
                }

                // Sort by score (ascending - lowest scores get pruned first)
                rowScores.Sort((a, b) => a.score.CompareTo(b.score));

                // Mark rows to keep
                var keepRows = new bool[importanceScores.Rows];
                for (int i = 0; i < importanceScores.Rows; i++)
                    keepRows[i] = true;

                for (int i = 0; i < rowsToPrune; i++)
                {
                    keepRows[rowScores[i].row] = false;
                }

                // Create mask
                for (int row = 0; row < importanceScores.Rows; row++)
                {
                    for (int col = 0; col < importanceScores.Columns; col++)
                    {
                        keepIndices[row, col] = keepRows[row];
                    }
                }
                break;
        }

        var mask = new PruningMask<T>(importanceScores.Rows, importanceScores.Columns);
        mask.UpdateMask(keepIndices);

        return mask;
    }

    /// <summary>
    /// Computes importance scores for vector weights.
    /// </summary>
    /// <param name="weights">Weight vector</param>
    /// <param name="gradients">Gradients (not used for structured pruning)</param>
    /// <returns>Vector of importance scores</returns>
    public Vector<T> ComputeImportanceScores(Vector<T> weights, Vector<T>? gradients = null)
    {
        // For vectors, importance is just the absolute value
        var scores = new T[weights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            scores[i] = _numOps.FromDouble(Math.Abs(_numOps.ToDouble(weights[i])));
        }
        return new Vector<T>(scores);
    }

    /// <summary>
    /// Computes importance scores for tensor weights.
    /// </summary>
    /// <param name="weights">Weight tensor</param>
    /// <param name="gradients">Gradients (not used for structured pruning)</param>
    /// <returns>Tensor of importance scores</returns>
    /// <exception cref="ArgumentException">Thrown when tensor rank doesn't match the pruning type requirements.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Structured pruning requires specific tensor shapes:
    /// - Neuron pruning: Requires 2D tensors (matrices). For 2D, use ComputeImportanceScores(Matrix) instead.
    /// - Filter pruning: Requires 4D tensors [filters, channels, height, width].
    /// - Channel pruning: Requires 4D tensors [filters, channels, height, width].
    /// </para>
    /// </remarks>
    public Tensor<T> ComputeImportanceScores(Tensor<T> weights, Tensor<T>? gradients = null)
    {
        switch (_pruningType)
        {
            case StructurePruningType.Filter:
                return ComputeFilterImportanceScores(weights);
            case StructurePruningType.Channel:
                return ComputeChannelImportanceScores(weights);
            case StructurePruningType.Neuron:
                // Neuron pruning is only meaningful for 2D matrices
                if (weights.Rank != 2)
                {
                    throw new ArgumentException(
                        $"Neuron structured pruning requires 2D tensor (matrix). Got {weights.Rank}D tensor. " +
                        "For 4D convolutional tensors, use Filter or Channel pruning type.",
                        nameof(weights));
                }
                // For 2D tensors, convert to matrix and use matrix-based scoring
                var matrix = weights.ToMatrix();
                var matrixScores = ComputeImportanceScores(matrix, gradients: null);
                return Tensor<T>.FromMatrix(matrixScores);
            default:
                throw new NotSupportedException($"Pruning type {_pruningType} is not supported for tensor operations.");
        }
    }

    private Tensor<T> ComputeFilterImportanceScores(Tensor<T> weights)
    {
        if (weights.Rank != 4)
            throw new ArgumentException("Filter pruning requires 4D tensor [filters, channels, height, width]");

        var dims = weights.Shape;
        int filters = dims[0];
        int channels = dims[1];
        int height = dims[2];
        int width = dims[3];
        int elementsPerFilter = channels * height * width;

        var flatData = weights.ToVector();
        var scores = new T[flatData.Length];

        // Compute L2 norm for each filter
        for (int f = 0; f < filters; f++)
        {
            double norm = 0;
            int baseIdx = f * elementsPerFilter;
            for (int i = 0; i < elementsPerFilter; i++)
            {
                double val = _numOps.ToDouble(flatData[baseIdx + i]);
                norm += val * val;
            }
            norm = Math.Sqrt(norm);

            // Assign same score to all weights in filter
            for (int i = 0; i < elementsPerFilter; i++)
            {
                scores[baseIdx + i] = _numOps.FromDouble(norm);
            }
        }

        return Tensor<T>.FromVector(new Vector<T>(scores), (int[])dims.Clone());
    }

    private Tensor<T> ComputeChannelImportanceScores(Tensor<T> weights)
    {
        if (weights.Rank != 4)
            throw new ArgumentException("Channel pruning requires 4D tensor [filters, channels, height, width]");

        var dims = weights.Shape;
        int filters = dims[0];
        int channels = dims[1];
        int height = dims[2];
        int width = dims[3];
        int elementsPerChannel = height * width;
        int elementsPerFilter = channels * elementsPerChannel;

        var flatData = weights.ToVector();
        var scores = new T[flatData.Length];

        // Compute L2 norm for each channel across all filters
        for (int c = 0; c < channels; c++)
        {
            double norm = 0;
            for (int f = 0; f < filters; f++)
            {
                int baseIdx = f * elementsPerFilter + c * elementsPerChannel;
                for (int i = 0; i < elementsPerChannel; i++)
                {
                    double val = _numOps.ToDouble(flatData[baseIdx + i]);
                    norm += val * val;
                }
            }
            norm = Math.Sqrt(norm);

            // Assign same score to all weights in channel
            for (int f = 0; f < filters; f++)
            {
                int baseIdx = f * elementsPerFilter + c * elementsPerChannel;
                for (int i = 0; i < elementsPerChannel; i++)
                {
                    scores[baseIdx + i] = _numOps.FromDouble(norm);
                }
            }
        }

        return Tensor<T>.FromVector(new Vector<T>(scores), (int[])dims.Clone());
    }

    /// <summary>
    /// Creates a pruning mask for vector weights.
    /// </summary>
    /// <param name="importanceScores">Importance scores from ComputeImportanceScores</param>
    /// <param name="targetSparsity">Target sparsity ratio (0 to 1)</param>
    /// <returns>Binary mask (1 = keep, 0 = prune)</returns>
    public IPruningMask<T> CreateMask(Vector<T> importanceScores, double targetSparsity)
    {
        if (double.IsNaN(targetSparsity) || double.IsInfinity(targetSparsity))
            throw new ArgumentException("targetSparsity cannot be NaN or Infinity.", nameof(targetSparsity));

        if (targetSparsity < 0 || targetSparsity > 1)
            throw new ArgumentException("targetSparsity must be between 0 and 1", nameof(targetSparsity));

        int totalElements = importanceScores.Length;
        int elementsToPrune = (int)(totalElements * targetSparsity);

        // Sort by importance
        var indexed = importanceScores.Select((score, idx) => (idx, score: _numOps.ToDouble(score))).ToArray();
        Array.Sort(indexed, (a, b) => a.score.CompareTo(b.score));

        // Create mask
        var keepIndices = new bool[totalElements];
        ArrayPolyfill.Fill(keepIndices, true);

        for (int i = 0; i < elementsToPrune; i++)
        {
            keepIndices[indexed[i].idx] = false;
        }

        var mask = new PruningMask<T>(keepIndices);
        return mask;
    }

    /// <summary>
    /// Creates a pruning mask for tensor weights.
    /// </summary>
    /// <param name="importanceScores">Importance scores from ComputeImportanceScores</param>
    /// <param name="targetSparsity">Target sparsity ratio (0 to 1)</param>
    /// <returns>Binary mask (1 = keep, 0 = prune)</returns>
    public IPruningMask<T> CreateMask(Tensor<T> importanceScores, double targetSparsity)
    {
        if (double.IsNaN(targetSparsity) || double.IsInfinity(targetSparsity))
            throw new ArgumentException("targetSparsity cannot be NaN or Infinity.", nameof(targetSparsity));

        if (targetSparsity < 0 || targetSparsity > 1)
            throw new ArgumentException("targetSparsity must be between 0 and 1", nameof(targetSparsity));

        var flatScores = importanceScores.ToVector();
        int totalElements = flatScores.Length;
        var keepIndices = new bool[totalElements];
        ArrayPolyfill.Fill(keepIndices, true);

        // For 4D tensors, apply proper structured pruning
        if (importanceScores.Rank == 4 && (_pruningType == StructurePruningType.Filter || _pruningType == StructurePruningType.Channel))
        {
            int filters = importanceScores.Shape[0];
            int channels = importanceScores.Shape[1];
            int height = importanceScores.Shape[2];
            int width = importanceScores.Shape[3];
            int elementsPerFilter = channels * height * width;
            int elementsPerChannel = height * width;

            if (_pruningType == StructurePruningType.Filter)
            {
                // Prune entire filters - get one score per filter (they're identical within filter)
                int filtersToPrune = (int)(filters * targetSparsity);
                var filterScores = new List<(int filterIdx, double score)>();

                for (int f = 0; f < filters; f++)
                {
                    // All elements in a filter have the same score from ComputeFilterImportanceScores
                    double score = _numOps.ToDouble(flatScores[f * elementsPerFilter]);
                    filterScores.Add((f, score));
                }

                filterScores.Sort((a, b) => a.score.CompareTo(b.score));

                // Mark entire filters as pruned
                for (int i = 0; i < filtersToPrune && i < filterScores.Count; i++)
                {
                    int filterIdx = filterScores[i].filterIdx;
                    int baseIdx = filterIdx * elementsPerFilter;
                    for (int j = 0; j < elementsPerFilter; j++)
                    {
                        keepIndices[baseIdx + j] = false;
                    }
                }
            }
            else // Channel pruning
            {
                // Prune entire channels across all filters
                int channelsToPrune = (int)(channels * targetSparsity);
                var channelScores = new List<(int channelIdx, double score)>();

                for (int c = 0; c < channels; c++)
                {
                    // All elements in a channel have the same score from ComputeChannelImportanceScores
                    double score = _numOps.ToDouble(flatScores[c * elementsPerChannel]);
                    channelScores.Add((c, score));
                }

                channelScores.Sort((a, b) => a.score.CompareTo(b.score));

                // Mark entire channels as pruned across all filters
                for (int i = 0; i < channelsToPrune && i < channelScores.Count; i++)
                {
                    int channelIdx = channelScores[i].channelIdx;
                    for (int f = 0; f < filters; f++)
                    {
                        int baseIdx = f * elementsPerFilter + channelIdx * elementsPerChannel;
                        for (int j = 0; j < elementsPerChannel; j++)
                        {
                            keepIndices[baseIdx + j] = false;
                        }
                    }
                }
            }
        }
        else if (importanceScores.Rank == 2 && _pruningType == StructurePruningType.Neuron)
        {
            // For 2D tensors with Neuron pruning, convert to matrix and use structured matrix pruning
            var matrix = importanceScores.ToMatrix();
            var matrixMask = CreateMask(matrix, targetSparsity);

            // Apply matrix mask once and transfer pattern to flat keepIndices
            var matrixResult = matrixMask.Apply(matrix);
            int cols = importanceScores.Shape[1];
            for (int i = 0; i < importanceScores.Shape[0]; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    int flatIdx = i * cols + j;
                    keepIndices[flatIdx] = !_numOps.Equals(matrixResult[i, j], _numOps.Zero);
                }
            }
        }
        else
        {
            // For unsupported tensor ranks, throw an exception to maintain structured semantics
            throw new ArgumentException(
                $"Structured pruning type {_pruningType} does not support {importanceScores.Rank}D tensors. " +
                "Use Neuron pruning for 2D tensors, or Filter/Channel pruning for 4D tensors.",
                nameof(importanceScores));
        }

        return new PruningMask<T>(keepIndices);
    }

    /// <summary>
    /// Creates a 2:4 structured sparsity mask (NVIDIA Ampere compatible).
    /// </summary>
    /// <param name="importanceScores">Importance scores</param>
    /// <returns>2:4 structured mask (exactly 2 zeros per 4 elements)</returns>
    public IPruningMask<T> Create2to4Mask(Tensor<T> importanceScores)
    {
        return CreateNtoMMask(importanceScores, 2, 4);
    }

    /// <summary>
    /// Creates an N:M structured sparsity mask.
    /// </summary>
    /// <param name="importanceScores">Importance scores</param>
    /// <param name="n">Number of zeros per group</param>
    /// <param name="m">Group size</param>
    /// <returns>N:M structured mask</returns>
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
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets entire columns to zero, effectively removing those neurons.
    /// After this, you can actually create a smaller weight matrix if you want,
    /// since entire neurons are gone!
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

        // Write pruned values back to the original tensor using flat indexer
        for (int i = 0; i < prunedFlat.Length; i++)
            weights[i] = prunedFlat[i];
    }

    /// <summary>
    /// Converts pruned weights to sparse format for efficient storage.
    /// </summary>
    /// <param name="weights">Pruned weights (containing zeros)</param>
    /// <param name="format">Target sparse format</param>
    /// <returns>Sparse representation</returns>
    public SparseCompressionResult<T> ToSparseFormat(Tensor<T> weights, SparseFormat format)
    {
        var flatWeights = weights.ToVector();
        var nonZeroValues = new List<T>();
        var rowIndices = new List<int>();
        var columnIndices = new List<int>();

        // Extract non-zero values and their positions
        for (int i = 0; i < flatWeights.Length; i++)
        {
            if (!_numOps.Equals(flatWeights[i], _numOps.Zero))
            {
                nonZeroValues.Add(flatWeights[i]);

                // For 2D tensors, calculate row and column
                if (weights.Rank == 2)
                {
                    int cols = weights.Shape[1];
                    rowIndices.Add(i / cols);
                    columnIndices.Add(i % cols);
                }
                else
                {
                    // For higher-dimensional tensors, use flat index
                    rowIndices.Add(i);
                    columnIndices.Add(0);
                }
            }
        }

        // Build result based on format
        switch (format)
        {
            case SparseFormat.COO:
                return new SparseCompressionResult<T>
                {
                    Format = SparseFormat.COO,
                    Values = nonZeroValues.ToArray(),
                    RowIndices = rowIndices.ToArray(),
                    ColumnIndices = columnIndices.ToArray(),
                    OriginalShape = weights.Shape.ToArray()
                };

            case SparseFormat.CSR:
                return ConvertToCSR(nonZeroValues, rowIndices, columnIndices, weights.Shape.ToArray());

            case SparseFormat.CSC:
                return ConvertToCSC(nonZeroValues, rowIndices, columnIndices, weights.Shape.ToArray());

            case SparseFormat.Structured2to4:
                return new SparseCompressionResult<T>
                {
                    Format = SparseFormat.Structured2to4,
                    Values = nonZeroValues.ToArray(),
                    OriginalShape = weights.Shape.ToArray(),
                    SparsityN = 2,
                    SparsityM = 4
                };

            case SparseFormat.StructuredNtoM:
                // Default to 2:4 pattern for StructuredNtoM when no explicit n, m are provided
                return ToSparseFormat(weights, format, 2, 4);

            default:
                throw new NotSupportedException($"Sparse format {format} is not supported for structured pruning");
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
    /// <remarks>
    /// <para><b>For Beginners:</b> N:M structured sparsity is a special pattern where exactly N elements
    /// out of every M consecutive elements are zero. For example, 2:4 sparsity means 2 zeros per 4 elements.
    /// This is hardware-friendly on NVIDIA Ampere GPUs which have specialized support for 2:4 patterns.
    /// </para>
    /// </remarks>
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

    private SparseCompressionResult<T> ConvertToCSR(List<T> values, List<int> rowIndices, List<int> columnIndices, int[] shape)
    {
        if (shape.Length != 2)
            throw new ArgumentException("CSR format requires 2D tensor");

        int rows = shape[0];
        var rowPointers = new int[rows + 1];

        // Count non-zeros per row
        foreach (var row in rowIndices)
            rowPointers[row + 1]++;

        // Convert counts to pointers
        for (int i = 1; i <= rows; i++)
            rowPointers[i] += rowPointers[i - 1];

        return new SparseCompressionResult<T>
        {
            Format = SparseFormat.CSR,
            Values = values.ToArray(),
            ColumnIndices = columnIndices.ToArray(),
            RowPointers = rowPointers,
            OriginalShape = shape
        };
    }

    private SparseCompressionResult<T> ConvertToCSC(List<T> values, List<int> rowIndices, List<int> columnIndices, int[] shape)
    {
        if (shape.Length != 2)
            throw new ArgumentException("CSC format requires 2D tensor");

        int cols = shape[1];
        int nnz = values.Count;

        // Build triplets and sort by (column, row) for proper CSC ordering
        var triplets = new List<(int col, int row, T value)>(nnz);
        for (int i = 0; i < nnz; i++)
        {
            triplets.Add((columnIndices[i], rowIndices[i], values[i]));
        }
        triplets.Sort((a, b) =>
        {
            int colCmp = a.col.CompareTo(b.col);
            return colCmp != 0 ? colCmp : a.row.CompareTo(b.row);
        });

        // Build CSC arrays from sorted triplets
        var sortedValues = new T[nnz];
        var sortedRowIndices = new int[nnz];
        var colPointers = new int[cols + 1];

        for (int i = 0; i < nnz; i++)
        {
            sortedValues[i] = triplets[i].value;
            sortedRowIndices[i] = triplets[i].row;
            colPointers[triplets[i].col + 1]++;
        }

        // Convert counts to pointers
        for (int i = 1; i <= cols; i++)
            colPointers[i] += colPointers[i - 1];

        return new SparseCompressionResult<T>
        {
            Format = SparseFormat.CSC,
            Values = sortedValues,
            RowIndices = sortedRowIndices,
            ColumnPointers = colPointers,
            OriginalShape = shape
        };
    }

    /// <summary>
    /// Applies layer-aware structured pruning to a model using per-category sparsity targets.
    /// </summary>
    /// <param name="model">The model to prune (must implement <see cref="ILayeredModel{T}"/>).</param>
    /// <param name="config">Pruning configuration with <see cref="PruningConfig.CategorySparsityTargets"/>.</param>
    /// <returns>A dictionary mapping layer index to the pruning mask applied, or empty if the model
    /// does not implement <see cref="ILayeredModel{T}"/>.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies different pruning strengths to different layer types.
    /// For example, attention layers might keep 90% of their weights (10% sparsity) while
    /// dense layers might only keep 50% (50% sparsity).</para>
    ///
    /// <para><b>How it works:</b></para>
    /// <list type="number">
    /// <item><description>Gets layer metadata from <see cref="ILayeredModel{T}"/>.</description></item>
    /// <item><description>For each trainable layer, looks up its <see cref="LayerCategory"/> in the config's
    /// <see cref="PruningConfig.CategorySparsityTargets"/>.</description></item>
    /// <item><description>If a name-based target exists in <see cref="PruningConfig.LayerSparsityTargets"/>,
    /// that takes precedence over category targets.</description></item>
    /// <item><description>Applies structured pruning to each layer's weight matrix at its effective sparsity level.</description></item>
    /// </list>
    ///
    /// <para><b>Research References:</b></para>
    /// <list type="bullet">
    /// <item><description>Lottery Ticket Hypothesis (Frankle &amp; Carlin, 2019): Different layers have different pruning sensitivity</description></item>
    /// <item><description>Layer-Adaptive Sparsity (LAMP, 2021): Per-layer sparsity allocation based on layer sensitivity</description></item>
    /// </list>
    /// </remarks>
    public Dictionary<int, IPruningMask<T>> ApplyLayerAwarePruning(
        IFullModel<T, Tensor<T>, Tensor<T>> model, PruningConfig config)
    {
        var result = new Dictionary<int, IPruningMask<T>>();

        if (model is not ILayeredModel<T> layeredModel)
        {
            return result;
        }

        var allLayerInfo = layeredModel.GetAllLayerInfo();
        double defaultSparsity = config.TargetSparsity;

        for (int i = 0; i < allLayerInfo.Count; i++)
        {
            var info = allLayerInfo[i];

            // Skip non-trainable layers
            if (!info.IsTrainable || info.ParameterCount == 0)
            {
                continue;
            }

            // Determine effective sparsity: name override > category override > default
            double layerSparsity = defaultSparsity;

            if (config.LayerSparsityTargets is not null &&
                config.LayerSparsityTargets.TryGetValue(info.Name, out double nameSparsity))
            {
                layerSparsity = nameSparsity;
            }
            else if (config.CategorySparsityTargets is not null &&
                     config.CategorySparsityTargets.TryGetValue(info.Category, out double categorySparsity))
            {
                layerSparsity = categorySparsity;
            }

            // Skip layers with zero sparsity target
            if (layerSparsity <= 0.0)
            {
                continue;
            }

            // Get the layer's weights
            var weights = info.Layer.GetWeights();
            if (weights is null)
            {
                continue;
            }

            // Compute importance scores and create mask
            var importanceScores = ComputeImportanceScores(weights);
            var mask = CreateMask(importanceScores, layerSparsity);

            // Apply the mask to the weights
            ApplyPruning(weights, mask);

            result[i] = mask;
        }

        return result;
    }
}
