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
                        double val = Convert.ToDouble(weights[row, col]);
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

            default:
                throw new NotImplementedException($"Pruning type {_pruningType} not yet implemented");
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
        if (targetSparsity < 0 || targetSparsity > 1)
            throw new ArgumentException("targetSparsity must be between 0 and 1");

        var keepIndices = new bool[importanceScores.Rows, importanceScores.Columns];

        switch (_pruningType)
        {
            case StructurePruningType.Neuron:
                // Prune entire columns (neurons)
                int totalNeurons = importanceScores.Columns;
                int neuronsToPrune = (int)(totalNeurons * targetSparsity);

                // Get score for each neuron (all rows in column have same score)
                var neuronScores = new List<(int col, double score)>();
                for (int col = 0; col < importanceScores.Columns; col++)
                {
                    double score = Convert.ToDouble(importanceScores[0, col]);
                    neuronScores.Add((col, score));
                }

                // Sort by score (ascending)
                neuronScores.Sort((a, b) => a.score.CompareTo(b.score));

                // Mark columns to keep
                var keepColumns = new bool[importanceScores.Columns];
                for (int i = 0; i < importanceScores.Columns; i++)
                    keepColumns[i] = true;

                for (int i = 0; i < neuronsToPrune; i++)
                {
                    keepColumns[neuronScores[i].col] = false;
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

            default:
                throw new NotImplementedException($"Pruning type {_pruningType} not yet implemented");
        }

        var mask = new PruningMask<T>(importanceScores.Rows, importanceScores.Columns);
        mask.UpdateMask(keepIndices);

        return mask;
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
}
