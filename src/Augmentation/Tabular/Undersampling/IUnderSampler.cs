using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular.Undersampling;

/// <summary>
/// Interface for undersampling techniques that reduce the majority class.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Undersampling is a technique to handle imbalanced datasets by
/// removing samples from the majority class. This helps classifiers focus on the minority class
/// without being overwhelmed by majority samples.</para>
///
/// <para><b>Common undersampling methods:</b>
/// <list type="bullet">
/// <item>Random: Randomly remove majority samples</item>
/// <item>NearMiss: Remove majority samples far from minority class</item>
/// <item>Tomek Links: Remove majority samples forming Tomek links</item>
/// <item>Edited Nearest Neighbors: Remove misclassified samples</item>
/// </list>
/// </para>
///
/// <para><b>Trade-offs:</b> Undersampling may discard useful information, but reduces
/// training time and can prevent majority class bias. Often combined with oversampling
/// (e.g., SMOTE-Tomek, SMOTE-ENN).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("UnderSampler")]
public interface IUnderSampler<T>
{
    /// <summary>
    /// Performs undersampling on the dataset.
    /// </summary>
    /// <param name="data">The full dataset with both classes.</param>
    /// <param name="labels">Class labels for each sample.</param>
    /// <param name="minorityLabel">The label value for the minority class.</param>
    /// <returns>Tuple of (undersampled data, undersampled labels).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method reduces the majority class while keeping all
    /// minority samples. The result has a more balanced class distribution.</para>
    /// </remarks>
    (Matrix<T> Data, Vector<T> Labels) Undersample(
        Matrix<T> data,
        Vector<T> labels,
        T minorityLabel);
}
