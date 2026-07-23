using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular;

/// <summary>
/// Base class for tabular data augmentations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Tabular augmentation transforms structured data (like spreadsheets)
/// to improve model generalization. Unlike image augmentation which uses geometric transforms,
/// tabular augmentation focuses on:
/// <list type="bullet">
/// <item>Adding noise to numerical features</item>
/// <item>Mixing samples together (MixUp)</item>
/// <item>Dropping features for regularization</item>
/// <item>Synthetic sample generation (SMOTE)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public abstract class TabularAugmenterBase<T> : AugmentationBase<T, Matrix<T>>
{
    /// <summary>
    /// Initializes a new tabular augmentation.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation (0.0 to 1.0).</param>
    protected TabularAugmenterBase(double probability = 1.0) : base(probability)
    {
    }

    /// <summary>
    /// Gets the number of features in the input data.
    /// </summary>
    /// <param name="data">The input matrix (rows=samples, columns=features).</param>
    /// <returns>The number of features.</returns>
    protected int GetFeatureCount(Matrix<T> data)
    {
        return data.Columns;
    }

    /// <summary>
    /// Gets the number of samples in the input data.
    /// </summary>
    /// <param name="data">The input matrix (rows=samples, columns=features).</param>
    /// <returns>The number of samples.</returns>
    protected int GetSampleCount(Matrix<T> data)
    {
        return data.Rows;
    }
}

/// <summary>
/// Base class for tabular augmentations that mix multiple samples together.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public abstract class TabularMixingAugmenterBase<T> : LabelMixingAugmentationBase<T, Matrix<T>>
{
    /// <summary>
    /// Initializes a new tabular mixing augmentation.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation.</param>
    /// <param name="alpha">The alpha parameter for Beta distribution sampling.</param>
    protected TabularMixingAugmenterBase(double probability = 1.0, double alpha = 0.2)
        : base(probability, alpha)
    {
    }
}
