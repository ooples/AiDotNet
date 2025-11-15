namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a strategy for pruning neural network weights.
/// </summary>
/// <typeparam name="T">Numeric type for weights and gradients</typeparam>
/// <remarks>
/// <para>
/// A pruning strategy determines which weights in a neural network should be removed
/// to reduce model size and computational requirements while maintaining accuracy.
/// Different strategies use different criteria to measure weight importance.
/// </para>
/// <para><b>For Beginners:</b> A pruning strategy decides which connections to remove from a neural network.
///
/// Think of it like pruning a tree:
/// - You want to remove branches that don't contribute much to the tree's health
/// - You keep the important branches that carry nutrients and support the structure
/// - The goal is a healthier, more efficient tree
///
/// In neural networks:
/// - Different strategies measure "importance" differently
/// - Magnitude-based: Remove smallest weights (they contribute less to output)
/// - Gradient-based: Remove weights with smallest gradients (they learn slowly)
/// - Structured: Remove entire neurons or filters (cleaner architecture)
///
/// All strategies aim to compress the model while preserving its predictive power.
/// </para>
/// </remarks>
public interface IPruningStrategy<T>
{
    /// <summary>
    /// Computes importance scores for each weight.
    /// </summary>
    /// <param name="weights">Weight matrix</param>
    /// <param name="gradients">Gradient matrix (optional, can be null)</param>
    /// <returns>Importance score for each weight (higher = more important)</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method assigns each weight a score representing its importance.
    /// Higher scores mean the weight is more important and should be kept.
    /// Lower scores mean the weight can be safely removed.
    /// </para>
    /// </remarks>
    Matrix<T> ComputeImportanceScores(Matrix<T> weights, Matrix<T>? gradients = null);

    /// <summary>
    /// Creates a pruning mask based on target sparsity.
    /// </summary>
    /// <param name="importanceScores">Importance scores from ComputeImportanceScores</param>
    /// <param name="targetSparsity">Target sparsity ratio (0 to 1)</param>
    /// <returns>Binary mask (1 = keep, 0 = prune)</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates the actual mask that determines which weights to remove.
    /// If targetSparsity is 0.7, it will mark 70% of the least important weights for removal.
    /// </para>
    /// </remarks>
    IPruningMask<T> CreateMask(Matrix<T> importanceScores, double targetSparsity);

    /// <summary>
    /// Prunes a weight matrix in-place.
    /// </summary>
    /// <param name="weights">Weight matrix to prune</param>
    /// <param name="mask">Pruning mask to apply</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This actually removes the weights by applying the mask.
    /// After this operation, pruned weights become zero.
    /// </para>
    /// </remarks>
    void ApplyPruning(Matrix<T> weights, IPruningMask<T> mask);

    /// <summary>
    /// Gets whether this strategy requires gradients.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some strategies need gradient information to determine importance.
    /// If true, you must provide gradients when calling ComputeImportanceScores.
    /// </para>
    /// </remarks>
    bool RequiresGradients { get; }

    /// <summary>
    /// Gets whether this is structured pruning (removes entire rows/cols).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Structured pruning removes entire neurons or filters.
    /// Unstructured pruning (false) removes individual weights anywhere in the network.
    /// </para>
    /// </remarks>
    bool IsStructured { get; }
}
