using AiDotNet.Augmentation.Tabular;
using AiDotNet.Tensors.LinearAlgebra;

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
        _augmenter = augmenter ?? throw new ArgumentNullException(nameof(augmenter));
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
