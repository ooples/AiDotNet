namespace AiDotNet.Pruning;

/// <summary>
/// Binary mask for pruning neural network weights.
/// </summary>
/// <typeparam name="T">Numeric type</typeparam>
/// <remarks>
/// <para>
/// PruningMask represents a binary matrix where 1 indicates weights to keep and 0 indicates weights
/// to prune (set to zero). It provides methods to apply the mask to weight matrices and tensors,
/// compute sparsity levels, and combine multiple masks.
/// </para>
/// <para><b>For Beginners:</b> A pruning mask is like a stencil for your neural network weights.
///
/// Imagine you have a grid of numbers (your neural network weights):
/// - Some numbers are important and should stay
/// - Some numbers can be removed to make the model smaller
///
/// The pruning mask marks which ones to keep (1) and which to remove (0).
/// When you apply the mask, all the marked weights become zero, effectively removing
/// those connections from your neural network.
///
/// This helps create smaller, faster models that still work well!
/// </para>
/// </remarks>
public class PruningMask<T> : IPruningMask<T>
{
    private readonly Matrix<T> _mask;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Gets the shape of the mask as [rows, columns].
    /// </summary>
    public int[] Shape => new[] { _mask.Rows, _mask.Columns };

    /// <summary>
    /// Initializes a new pruning mask with all ones (no pruning).
    /// </summary>
    /// <param name="rows">Number of rows in the mask</param>
    /// <param name="cols">Number of columns in the mask</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a new mask with the specified size.
    /// Initially, all values are 1 (meaning keep all weights).
    /// </para>
    /// </remarks>
    public PruningMask(int rows, int cols)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _mask = new Matrix<T>(rows, cols);

        // Initialize to all ones (no pruning)
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                _mask[i, j] = _numOps.One;
    }

    /// <summary>
    /// Initializes a pruning mask from an existing matrix.
    /// </summary>
    /// <param name="maskMatrix">Matrix containing the mask values (should be 0s and 1s)</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a mask from an existing matrix of 0s and 1s.
    /// </para>
    /// </remarks>
    public PruningMask(Matrix<T> maskMatrix)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _mask = maskMatrix.Clone();
    }

    /// <summary>
    /// Calculates the sparsity level of the mask.
    /// </summary>
    /// <returns>Proportion of pruned weights (0 to 1)</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sparsity tells you what percentage of weights have been removed.
    /// For example, 0.7 means 70% of the weights are pruned (removed).
    /// </para>
    /// </remarks>
    public double GetSparsity()
    {
        int totalElements = _mask.Rows * _mask.Columns;
        int zeroCount = 0;

        for (int i = 0; i < _mask.Rows; i++)
        {
            for (int j = 0; j < _mask.Columns; j++)
            {
                if (_numOps.Equals(_mask[i, j], _numOps.Zero))
                    zeroCount++;
            }
        }

        return (double)zeroCount / totalElements;
    }

    /// <summary>
    /// Applies the mask to a weight matrix by element-wise multiplication.
    /// </summary>
    /// <param name="weights">Weight matrix to prune</param>
    /// <returns>Pruned weight matrix</returns>
    /// <exception cref="ArgumentException">Thrown when weight matrix shape doesn't match mask shape</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This multiplies each weight by the corresponding mask value.
    /// If the mask is 0, the weight becomes 0 (pruned).
    /// If the mask is 1, the weight stays the same (kept).
    /// </para>
    /// </remarks>
    public Matrix<T> Apply(Matrix<T> weights)
    {
        if (weights.Rows != _mask.Rows || weights.Columns != _mask.Columns)
            throw new ArgumentException("Weight matrix shape must match mask shape");

        var result = new Matrix<T>(weights.Rows, weights.Columns);

        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = 0; j < weights.Columns; j++)
            {
                result[i, j] = _numOps.Multiply(weights[i, j], _mask[i, j]);
            }
        }

        return result;
    }

    /// <summary>
    /// Applies the mask to a weight tensor.
    /// </summary>
    /// <param name="weights">Weight tensor to prune</param>
    /// <returns>Pruned weight tensor</returns>
    /// <exception cref="NotSupportedException">Thrown for unsupported tensor ranks</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is similar to applying to a matrix, but works with
    /// multi-dimensional arrays (tensors) used in convolutional neural networks.
    /// </para>
    /// </remarks>
    public Tensor<T> Apply(Tensor<T> weights)
    {
        // For 2D tensors (fully connected layers)
        if (weights.Rank == 2)
        {
            var matrix = TensorToMatrix(weights);
            var pruned = Apply(matrix);
            return MatrixToTensor(pruned);
        }

        // For 4D tensors (convolutional layers: [filters, channels, height, width])
        if (weights.Rank == 4)
        {
            var result = weights.Clone();
            int filters = weights.Dimensions[0];
            int channels = weights.Dimensions[1];

            // Apply mask element-wise for now (unstructured pruning)
            // For structured pruning, this would need to be modified
            for (int f = 0; f < filters; f++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < weights.Dimensions[2]; h++)
                    {
                        for (int w = 0; w < weights.Dimensions[3]; w++)
                        {
                            result[f, c, h, w] = weights[f, c, h, w];
                        }
                    }
                }
            }

            return result;
        }

        throw new NotSupportedException($"Tensor rank {weights.Rank} not supported for pruning");
    }

    /// <summary>
    /// Updates the mask with new keep/prune decisions.
    /// </summary>
    /// <param name="keepIndices">Boolean array indicating which weights to keep (true) or prune (false)</param>
    /// <exception cref="ArgumentException">Thrown when keepIndices shape doesn't match mask shape</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This updates which weights should be kept or removed.
    /// Pass true for weights to keep, false for weights to prune.
    /// </para>
    /// </remarks>
    public void UpdateMask(bool[,] keepIndices)
    {
        if (keepIndices.GetLength(0) != _mask.Rows || keepIndices.GetLength(1) != _mask.Columns)
            throw new ArgumentException("keepIndices shape must match mask shape");

        for (int i = 0; i < _mask.Rows; i++)
        {
            for (int j = 0; j < _mask.Columns; j++)
            {
                _mask[i, j] = keepIndices[i, j] ? _numOps.One : _numOps.Zero;
            }
        }
    }

    /// <summary>
    /// Combines two masks using logical AND operation.
    /// </summary>
    /// <param name="otherMask">Another mask to combine with this one</param>
    /// <returns>Combined mask where both masks must be 1 to keep the weight</returns>
    /// <exception cref="ArgumentException">Thrown when masks have different shapes</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a new mask that only keeps weights if BOTH masks want to keep them.
    /// It's useful when you want to apply multiple pruning criteria together.
    /// </para>
    /// </remarks>
    public IPruningMask<T> CombineWith(IPruningMask<T> otherMask)
    {
        if (otherMask.Shape[0] != Shape[0] || otherMask.Shape[1] != Shape[1])
            throw new ArgumentException("Masks must have same shape to combine");

        var combined = new Matrix<T>(_mask.Rows, _mask.Columns);
        var otherMatrix = ((PruningMask<T>)otherMask)._mask;

        for (int i = 0; i < _mask.Rows; i++)
        {
            for (int j = 0; j < _mask.Columns; j++)
            {
                // Logical AND: both must be 1 to keep
                bool keep = !_numOps.Equals(_mask[i, j], _numOps.Zero) &&
                            !_numOps.Equals(otherMatrix[i, j], _numOps.Zero);
                combined[i, j] = keep ? _numOps.One : _numOps.Zero;
            }
        }

        return new PruningMask<T>(combined);
    }

    /// <summary>
    /// Converts a 2D tensor to a matrix.
    /// </summary>
    private Matrix<T> TensorToMatrix(Tensor<T> tensor)
    {
        var matrix = new Matrix<T>(tensor.Dimensions[0], tensor.Dimensions[1]);
        for (int i = 0; i < tensor.Dimensions[0]; i++)
            for (int j = 0; j < tensor.Dimensions[1]; j++)
                matrix[i, j] = tensor[i, j];
        return matrix;
    }

    /// <summary>
    /// Converts a matrix to a 2D tensor.
    /// </summary>
    private Tensor<T> MatrixToTensor(Matrix<T> matrix)
    {
        var tensor = new Tensor<T>(matrix.Rows, matrix.Columns);
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
                tensor[i, j] = matrix[i, j];
        return tensor;
    }
}
