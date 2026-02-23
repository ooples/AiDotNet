using AiDotNet.Augmentation.Tabular;
using AiDotNet.Helpers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Preprocessing.DataPreparation;

/// <summary>
/// A row operation that applies data augmentation to increase dataset size.
/// </summary>
/// <remarks>
/// <para>
/// This operation wraps tabular augmenters (like SMOTE) to generate synthetic samples.
/// Both features (X) and labels (y) are augmented together to maintain alignment.
/// </para>
/// <para>
/// <b>For Beginners:</b> Data augmentation creates new synthetic data points based on
/// your existing data. This is especially useful when:
/// - You have imbalanced classes (one class has way more samples than another)
/// - You have limited training data
/// - You want to reduce overfitting
/// </para>
/// <para>
/// <b>Common Use Case - SMOTE:</b> If you're predicting fraud (rare) vs normal (common),
/// SMOTE creates synthetic fraud examples so your model learns to recognize fraud better.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class AugmentationOperation<T> : IRowOperation<T>
{
    private readonly TabularAugmenterBase<T> _augmenter;
    private readonly Vector<T>? _targetLabels;
    private bool _isFitted;

    /// <inheritdoc/>
    public bool IsFitted => _isFitted;

    /// <inheritdoc/>
    public string Description => $"Data augmentation using {_augmenter.GetType().Name}";

    /// <summary>
    /// Gets the underlying augmenter.
    /// </summary>
    public TabularAugmenterBase<T> Augmenter => _augmenter;

    /// <summary>
    /// Creates a new augmentation operation.
    /// </summary>
    /// <param name="augmenter">
    /// The tabular augmenter to use (e.g., SmoteAugmenter).
    /// </param>
    /// <param name="targetLabels">
    /// Optional: specific labels to augment. If null, augments minority classes automatically.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when augmenter is null.</exception>
    public AugmentationOperation(
        TabularAugmenterBase<T> augmenter,
        Vector<T>? targetLabels = null)
    {
        Guard.NotNull(augmenter);
        _augmenter = augmenter;
        _targetLabels = targetLabels;
    }

    /// <inheritdoc/>
    public (Matrix<T> X, Vector<T> y) FitResample(Matrix<T> X, Vector<T> y)
    {
        if (X is null) throw new ArgumentNullException(nameof(X));
        if (y is null) throw new ArgumentNullException(nameof(y));

        if (X.Rows != y.Length)
        {
            throw new ArgumentException(
                $"X has {X.Rows} rows but y has {y.Length} elements. They must match.",
                nameof(y));
        }

        // Apply augmentation to generate synthetic samples
        var context = new Augmentation.AugmentationContext<T>(isTraining: true);
        var augmentedX = _augmenter.Apply(X, context);

        // For augmenters that don't modify row count, return original y
        if (augmentedX.Rows == X.Rows)
        {
            _isFitted = true;
            return (augmentedX, y);
        }

        // For SMOTE-like augmenters, we need to generate corresponding labels
        // The augmenter should have added samples for minority classes
        var augmentedY = GenerateAugmentedLabels(X, y, augmentedX);

        _isFitted = true;
        return (augmentedX, augmentedY);
    }

    /// <inheritdoc/>
    public (Tensor<T> X, Tensor<T> y) FitResampleTensor(Tensor<T> X, Tensor<T> y)
    {
        if (X is null) throw new ArgumentNullException(nameof(X));
        if (y is null) throw new ArgumentNullException(nameof(y));

        if (X.Shape[0] != y.Shape[0])
        {
            throw new ArgumentException(
                $"X has {X.Shape[0]} samples but y has {y.Shape[0]} samples. They must match.",
                nameof(y));
        }

        // Convert tensor to matrix for tabular augmentation (flatten non-batch dimensions)
        int nSamples = X.Shape[0];
        int nFeatures = 1;
        for (int i = 1; i < X.Rank; i++)
        {
            nFeatures *= X.Shape[i];
        }

        var matrixX = new Matrix<T>(nSamples, nFeatures);
        for (int i = 0; i < nSamples; i++)
        {
            int flatIdx = 0;
            FlattenSampleToRow(X, i, matrixX, i, ref flatIdx);
        }

        // Extract labels from tensor (assuming 1D or taking first element)
        var vectorY = new Vector<T>(nSamples);
        for (int i = 0; i < nSamples; i++)
        {
            vectorY[i] = y.Rank == 1 ? y[i] : y[new int[y.Rank].Select((_, idx) => idx == 0 ? i : 0).ToArray()];
        }

        // Apply augmentation
        var context = new Augmentation.AugmentationContext<T>(isTraining: true);
        var augmentedMatrix = _augmenter.Apply(matrixX, context);

        // If no new samples were generated, return original tensors
        if (augmentedMatrix.Rows == nSamples)
        {
            _isFitted = true;
            return (X, y);
        }

        int newSamples = augmentedMatrix.Rows;

        // Convert back to tensor shape
        int[] newXShape = (int[])X.Shape.Clone();
        newXShape[0] = newSamples;
        var augmentedX = new Tensor<T>(newXShape);

        // Copy original samples
        for (int i = 0; i < nSamples; i++)
        {
            CopyTensorSample(X, augmentedX, i, i);
        }

        // Convert augmented matrix rows back to tensor samples
        for (int i = nSamples; i < newSamples; i++)
        {
            int flatIdx = 0;
            UnflattenRowToSample(augmentedMatrix, i, augmentedX, i, ref flatIdx);
        }

        // Generate augmented labels
        var augmentedVectorY = GenerateAugmentedLabels(matrixX, vectorY, augmentedMatrix);

        // Convert vector back to tensor
        int[] newYShape = (int[])y.Shape.Clone();
        newYShape[0] = newSamples;
        var augmentedY = new Tensor<T>(newYShape);

        for (int i = 0; i < newSamples; i++)
        {
            if (y.Rank == 1)
            {
                augmentedY[i] = augmentedVectorY[i];
            }
            else
            {
                int[] indices = new int[y.Rank];
                indices[0] = i;
                augmentedY[indices] = augmentedVectorY[i];
            }
        }

        _isFitted = true;
        return (augmentedX, augmentedY);
    }

    private void FlattenSampleToRow(Tensor<T> tensor, int sampleIdx, Matrix<T> matrix, int rowIdx, ref int flatIdx)
    {
        if (tensor.Rank == 1)
        {
            matrix[rowIdx, flatIdx++] = tensor[sampleIdx];
            return;
        }

        int[] indices = new int[tensor.Rank];
        indices[0] = sampleIdx;
        FlattenRecursive(tensor, matrix, rowIdx, indices, 1, ref flatIdx);
    }

    private void FlattenRecursive(Tensor<T> tensor, Matrix<T> matrix, int rowIdx, int[] indices, int dim, ref int flatIdx)
    {
        if (dim == tensor.Rank)
        {
            matrix[rowIdx, flatIdx++] = tensor[indices];
            return;
        }

        for (int i = 0; i < tensor.Shape[dim]; i++)
        {
            indices[dim] = i;
            FlattenRecursive(tensor, matrix, rowIdx, indices, dim + 1, ref flatIdx);
        }
    }

    private void UnflattenRowToSample(Matrix<T> matrix, int rowIdx, Tensor<T> tensor, int sampleIdx, ref int flatIdx)
    {
        if (tensor.Rank == 1)
        {
            tensor[sampleIdx] = matrix[rowIdx, flatIdx++];
            return;
        }

        int[] indices = new int[tensor.Rank];
        indices[0] = sampleIdx;
        UnflattenRecursive(matrix, rowIdx, tensor, indices, 1, ref flatIdx);
    }

    private void UnflattenRecursive(Matrix<T> matrix, int rowIdx, Tensor<T> tensor, int[] indices, int dim, ref int flatIdx)
    {
        if (dim == tensor.Rank)
        {
            tensor[indices] = matrix[rowIdx, flatIdx++];
            return;
        }

        for (int i = 0; i < tensor.Shape[dim]; i++)
        {
            indices[dim] = i;
            UnflattenRecursive(matrix, rowIdx, tensor, indices, dim + 1, ref flatIdx);
        }
    }

    private void CopyTensorSample(Tensor<T> source, Tensor<T> dest, int srcIdx, int destIdx)
    {
        if (source.Rank == 1)
        {
            dest[destIdx] = source[srcIdx];
            return;
        }

        int[] srcIndices = new int[source.Rank];
        int[] destIndices = new int[dest.Rank];
        srcIndices[0] = srcIdx;
        destIndices[0] = destIdx;
        CopyRecursive(source, dest, srcIndices, destIndices, 1);
    }

    private void CopyRecursive(Tensor<T> source, Tensor<T> dest, int[] srcIndices, int[] destIndices, int dim)
    {
        if (dim == source.Rank)
        {
            dest[destIndices] = source[srcIndices];
            return;
        }

        for (int i = 0; i < source.Shape[dim]; i++)
        {
            srcIndices[dim] = i;
            destIndices[dim] = i;
            CopyRecursive(source, dest, srcIndices, destIndices, dim + 1);
        }
    }

    private Vector<T> GenerateAugmentedLabels(Matrix<T> originalX, Vector<T> originalY, Matrix<T> augmentedX)
    {
        int originalRows = originalX.Rows;
        int augmentedRows = augmentedX.Rows;
        int newSamples = augmentedRows - originalRows;

        if (newSamples <= 0)
        {
            return originalY;
        }

        // Create new label vector
        var newY = new T[augmentedRows];

        // Copy original labels
        for (int i = 0; i < originalRows; i++)
        {
            newY[i] = originalY[i];
        }

        // For new samples, we need to determine their labels
        // SMOTE generates samples for minority class, so we need to identify which class
        // This is a simplified approach - in practice, the augmenter should provide this info
        if (_targetLabels != null && _targetLabels.Length > 0)
        {
            // Use the first target label for all new samples
            for (int i = originalRows; i < augmentedRows; i++)
            {
                newY[i] = _targetLabels[0];
            }
        }
        else
        {
            // Find minority class (most common use case for augmentation)
            var minorityLabel = FindMinorityClass(originalY);
            for (int i = originalRows; i < augmentedRows; i++)
            {
                newY[i] = minorityLabel;
            }
        }

        return new Vector<T>(newY);
    }

    private T FindMinorityClass(Vector<T> y)
    {
        var counts = new Dictionary<double, (T label, int count)>();

        for (int i = 0; i < y.Length; i++)
        {
            double key = Convert.ToDouble(y[i]);
            if (counts.TryGetValue(key, out var existing))
            {
                counts[key] = (existing.label, existing.count + 1);
            }
            else
            {
                counts[key] = (y[i], 1);
            }
        }

        // Return the label with minimum count
        T minorityLabel = y[0];
        int minCount = int.MaxValue;

        foreach (var kvp in counts)
        {
            if (kvp.Value.count < minCount)
            {
                minCount = kvp.Value.count;
                minorityLabel = kvp.Value.label;
            }
        }

        return minorityLabel;
    }
}
