namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a binary mask for pruning weights in a neural network layer.
/// </summary>
/// <typeparam name="T">Numeric type for mask values</typeparam>
/// <remarks>
/// <para>
/// A pruning mask is a binary matrix that determines which weights to keep (1) and which to remove (0)
/// during model compression. It enables selective removal of network parameters while maintaining the
/// ability to restore the network structure.
/// </para>
/// <para><b>For Beginners:</b> Think of a pruning mask as a stencil or template.
///
/// Imagine you're painting a picture and want to cover certain areas:
/// - The mask has holes (1s) where paint should go through (weights to keep)
/// - The mask is solid (0s) where paint should be blocked (weights to prune/remove)
///
/// In neural networks:
/// - A pruning mask helps you selectively remove less important connections
/// - This makes your model smaller and faster without losing too much accuracy
/// - The mask can be applied to weight matrices to zero out pruned weights
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("PruningMask")]
public interface IPruningMask<T>
{
    /// <summary>
    /// Gets the mask dimensions matching the weight matrix shape.
    /// </summary>
    int[] Shape { get; }

    /// <summary>
    /// Gets the sparsity ratio (proportion of zeros).
    /// </summary>
    /// <returns>Value between 0 (dense) and 1 (fully pruned)</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sparsity measures how many weights have been removed.
    /// - 0.0 means no weights removed (0% sparse, 100% dense)
    /// - 0.5 means half the weights removed (50% sparse)
    /// - 0.9 means 90% of weights removed (90% sparse)
    /// </para>
    /// </remarks>
    double GetSparsity();

    /// <summary>
    /// Applies the mask to a vector.
    /// </summary>
    Vector<T> Apply(Vector<T> weights);

    /// <summary>
    /// Applies the mask to a weight matrix (element-wise multiplication).
    /// </summary>
    /// <param name="weights">Weight matrix to prune</param>
    /// <returns>Pruned weights (zeros where mask is zero)</returns>
    Matrix<T> Apply(Matrix<T> weights);

    /// <summary>
    /// Applies the mask to a weight tensor (for convolutional layers).
    /// </summary>
    Tensor<T> Apply(Tensor<T> weights);

    /// <summary>
    /// Updates the mask with new keep/prune decisions.
    /// </summary>
    void UpdateMask(bool[] keepIndices);

    /// <summary>
    /// Updates the mask based on new pruning criteria.
    /// </summary>
    /// <param name="keepIndices">Indices of weights to keep (not prune)</param>
    void UpdateMask(bool[,] keepIndices);

    /// <summary>
    /// Updates the mask with new N-D keep/prune decisions.
    /// </summary>
    void UpdateMask(Array keepIndices);

    /// <summary>
    /// Combines this mask with another mask (logical AND).
    /// </summary>
    IPruningMask<T> CombineWith(IPruningMask<T> otherMask);

    /// <summary>
    /// Gets the sparsity pattern type.
    /// </summary>
    SparsityPattern Pattern { get; }

    /// <summary>
    /// Gets the raw mask data as a flat array.
    /// </summary>
    T[] GetMaskData();

    /// <summary>
    /// Gets indices of non-zero (kept) elements.
    /// </summary>
    int[] GetKeptIndices();

    /// <summary>
    /// Gets indices of zero (pruned) elements.
    /// </summary>
    int[] GetPrunedIndices();
}
