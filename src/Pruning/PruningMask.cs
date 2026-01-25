using AiDotNet.Extensions;
using AiDotNet.Interfaces;

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
    /// Initializes a pruning mask from a 1D boolean array (for vectors).
    /// </summary>
    /// <param name="keepIndices">Boolean array where true means keep the weight.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a mask from a simple array of true/false values.
    /// True means keep the weight, false means prune (remove) it.
    /// </para>
    /// </remarks>
    public PruningMask(bool[] keepIndices)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _mask = new Matrix<T>(1, keepIndices.Length);

        for (int i = 0; i < keepIndices.Length; i++)
        {
            _mask[0, i] = keepIndices[i] ? _numOps.One : _numOps.Zero;
        }
    }

    /// <summary>
    /// Initializes a pruning mask from a 2D boolean array (for matrices).
    /// </summary>
    /// <param name="keepIndices">2D boolean array where true means keep the weight.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a mask from a 2D grid of true/false values.
    /// True means keep the weight at that position, false means prune (remove) it.
    /// </para>
    /// </remarks>
    public PruningMask(bool[,] keepIndices)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        int rows = keepIndices.GetLength(0);
        int cols = keepIndices.GetLength(1);
        _mask = new Matrix<T>(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                _mask[i, j] = keepIndices[i, j] ? _numOps.One : _numOps.Zero;
            }
        }
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

        // Use vectorized PointwiseMultiply for SIMD acceleration
        return weights.PointwiseMultiply(_mask);
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
            int totalElements = weights.Shape[0] * weights.Shape[1];

            // Handle case where mask is 1D (1xN) but tensor is 2D (MxK) with M*K == N
            // This happens when mask was created from a flattened tensor
            if (_mask.Rows == 1 && _mask.Columns == totalElements)
            {
                var flatWeights = weights.ToVector();
                var flatMask = _mask.GetRow(0);
                var flatResult = flatWeights.PointwiseMultiply(flatMask);
                return Tensor<T>.FromVector(flatResult, (int[])weights.Shape.Clone());
            }

            var matrix = TensorToMatrix(weights);
            var pruned = Apply(matrix);
            return MatrixToTensor(pruned);
        }

        // For 4D tensors (convolutional layers: [filters, channels, height, width])
        if (weights.Rank == 4)
        {
            int filters = weights.Shape[0];
            int channels = weights.Shape[1];
            int height = weights.Shape[2];
            int width = weights.Shape[3];
            int spatialSize = height * width;

            // Check if mask dimensions match filter/channel for structured pruning
            if (_mask.Rows == filters && _mask.Columns == channels)
            {
                // Apply filter/channel-level pruning using vectorized operations
                // Broadcast mask [filters, channels] to [filters, channels, height, width]
                int totalElements = filters * channels * spatialSize;
                var maskData = new T[totalElements];

                // Fill mask data by broadcasting each mask value across spatial dimensions
                int idx = 0;
                for (int f = 0; f < filters; f++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        T maskValue = _mask[f, c];
                        // Fill spatial elements with the same mask value
                        for (int s = 0; s < spatialSize; s++)
                        {
                            maskData[idx++] = maskValue;
                        }
                    }
                }

                var maskTensor = Tensor<T>.FromVector(new Vector<T>(maskData), new int[] { filters, channels, height, width });

                // Use vectorized PointwiseMultiply for SIMD acceleration
                return weights.PointwiseMultiply(maskTensor);
            }
            else
            {
                // Unstructured pruning: apply mask element-by-element
                var flatWeights = weights.ToVector();
                int totalElements = flatWeights.Length;

                if (_mask.Rows * _mask.Columns != totalElements)
                {
                    throw new ArgumentException(
                        $"Mask shape [{_mask.Rows}, {_mask.Columns}] does not match 4D tensor total elements ({totalElements}) " +
                        $"or filter/channel dimensions [{filters}, {channels}]");
                }

                // Convert mask to flat vector and use vectorized PointwiseMultiply
                var flatMask = new T[totalElements];
                int idx = 0;
                for (int i = 0; i < _mask.Rows; i++)
                {
                    for (int j = 0; j < _mask.Columns; j++)
                    {
                        flatMask[idx++] = _mask[i, j];
                    }
                }

                var flatMaskVector = new Vector<T>(flatMask);
                var flatResult = flatWeights.PointwiseMultiply(flatMaskVector);

                return Tensor<T>.FromVector(flatResult, (int[])weights.Shape.Clone());
            }
        }

        throw new NotSupportedException($"Tensor rank {weights.Rank} not supported for pruning");
    }

    /// <summary>
    /// Applies the mask to a weight vector.
    /// </summary>
    /// <param name="weights">Weight vector to prune</param>
    /// <returns>Pruned weight vector</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This multiplies each weight by the corresponding mask value.
    /// If the mask is 0, the weight becomes 0 (pruned).
    /// If the mask is 1, the weight stays the same (kept).
    /// </para>
    /// </remarks>
    public Vector<T> Apply(Vector<T> weights)
    {
        // For 1D vectors, the mask is stored as a single-row matrix
        if (weights.Length != _mask.Columns)
            throw new ArgumentException($"Weight vector length ({weights.Length}) must match mask columns ({_mask.Columns})");

        // Extract mask row as vector and use vectorized PointwiseMultiply
        var maskVector = _mask.GetRow(0);
        return weights.PointwiseMultiply(maskVector);
    }

    /// <summary>
    /// Updates the mask with new keep/prune decisions for 2D masks.
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
    /// Updates the mask with new keep/prune decisions for 1D masks.
    /// </summary>
    /// <param name="keepIndices">Boolean array indicating which weights to keep (true) or prune (false)</param>
    /// <exception cref="ArgumentException">Thrown when keepIndices length doesn't match mask columns</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This updates which weights should be kept or removed.
    /// Pass true for weights to keep, false for weights to prune.
    /// </para>
    /// </remarks>
    public void UpdateMask(bool[] keepIndices)
    {
        if (keepIndices.Length != _mask.Columns)
            throw new ArgumentException("keepIndices length must match mask columns");

        for (int i = 0; i < keepIndices.Length; i++)
        {
            _mask[0, i] = keepIndices[i] ? _numOps.One : _numOps.Zero;
        }
    }

    /// <summary>
    /// Updates the mask from an Array (supports both 1D and 2D arrays).
    /// </summary>
    /// <param name="keepIndices">Array of boolean indices.</param>
    public void UpdateMask(Array keepIndices)
    {
        if (keepIndices is bool[] boolArray1D)
        {
            UpdateMask(boolArray1D);
        }
        else if (keepIndices is bool[,] boolArray2D)
        {
            UpdateMask(boolArray2D);
        }
        else
        {
            throw new ArgumentException("keepIndices must be bool[] or bool[,]");
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

        // Use interface method to get mask data instead of unsafe casting
        var otherData = otherMask.GetMaskData();

        int idx = 0;
        for (int i = 0; i < _mask.Rows; i++)
        {
            for (int j = 0; j < _mask.Columns; j++)
            {
                // Logical AND: both must be non-zero to keep
                bool keep = !_numOps.Equals(_mask[i, j], _numOps.Zero) &&
                            !_numOps.Equals(otherData[idx], _numOps.Zero);
                combined[i, j] = keep ? _numOps.One : _numOps.Zero;
                idx++;
            }
        }

        return new PruningMask<T>(combined);
    }

    /// <summary>
    /// Gets the sparsity pattern type (unstructured for this implementation).
    /// </summary>
    public SparsityPattern Pattern => SparsityPattern.Unstructured;

    /// <summary>
    /// Gets the raw mask data as a flat array.
    /// </summary>
    /// <returns>Flattened mask values.</returns>
    public T[] GetMaskData()
    {
        var data = new T[_mask.Rows * _mask.Columns];
        int idx = 0;
        for (int i = 0; i < _mask.Rows; i++)
        {
            for (int j = 0; j < _mask.Columns; j++)
            {
                data[idx++] = _mask[i, j];
            }
        }
        return data;
    }

    /// <summary>
    /// Gets indices of non-zero (kept) elements.
    /// </summary>
    /// <returns>Array of indices where mask is non-zero.</returns>
    public int[] GetKeptIndices()
    {
        var indices = new List<int>();
        int idx = 0;
        for (int i = 0; i < _mask.Rows; i++)
        {
            for (int j = 0; j < _mask.Columns; j++)
            {
                if (!_numOps.Equals(_mask[i, j], _numOps.Zero))
                {
                    indices.Add(idx);
                }
                idx++;
            }
        }
        return indices.ToArray();
    }

    /// <summary>
    /// Gets indices of zero (pruned) elements.
    /// </summary>
    /// <returns>Array of indices where mask is zero.</returns>
    public int[] GetPrunedIndices()
    {
        var indices = new List<int>();
        int idx = 0;
        for (int i = 0; i < _mask.Rows; i++)
        {
            for (int j = 0; j < _mask.Columns; j++)
            {
                if (_numOps.Equals(_mask[i, j], _numOps.Zero))
                {
                    indices.Add(idx);
                }
                idx++;
            }
        }
        return indices.ToArray();
    }

    /// <summary>
    /// Converts a 2D tensor to a matrix.
    /// </summary>
    private Matrix<T> TensorToMatrix(Tensor<T> tensor)
    {
        var matrix = new Matrix<T>(tensor.Shape[0], tensor.Shape[1]);
        for (int i = 0; i < tensor.Shape[0]; i++)
            for (int j = 0; j < tensor.Shape[1]; j++)
                matrix[i, j] = tensor[i, j];
        return matrix;
    }

    /// <summary>
    /// Converts a matrix to a 2D tensor.
    /// </summary>
    private Tensor<T> MatrixToTensor(Matrix<T> matrix)
    {
        var tensor = new Tensor<T>(new int[] { matrix.Rows, matrix.Columns });
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
                tensor[i, j] = matrix[i, j];
        return tensor;
    }
}
