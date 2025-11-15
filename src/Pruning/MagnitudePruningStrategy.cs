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
    /// Gets whether this strategy requires gradients (false for magnitude-based).
    /// </summary>
    public bool RequiresGradients => false;

    /// <summary>
    /// Gets whether this is structured pruning (false for magnitude-based).
    /// </summary>
    public bool IsStructured => false;

    /// <summary>
    /// Initializes a new instance of MagnitudePruningStrategy.
    /// </summary>
    public MagnitudePruningStrategy()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Computes importance scores as absolute values of weights.
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
    /// Creates a pruning mask by selecting the smallest magnitude weights to prune.
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
        flatScores.Sort((a, b) =>
        {
            double aVal = Convert.ToDouble(a.score);
            double bVal = Convert.ToDouble(b.score);
            return aVal.CompareTo(bVal);
        });

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
}
