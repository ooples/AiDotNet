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
    /// Initializes a new instance of GradientPruningStrategy.
    /// </summary>
    public GradientPruningStrategy()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
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
}
