using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation;

/// <summary>
/// Defines an operation that can change the number of rows in a dataset.
/// </summary>
/// <remarks>
/// <para>
/// Row operations are fundamentally different from standard transformations because they
/// modify both features (X) and labels (y) together to maintain alignment. Examples include:
/// - Outlier removal (reduces rows)
/// - SMOTE oversampling (adds rows)
/// - Undersampling (reduces rows)
/// </para>
/// <para>
/// <b>Critical:</b> Row operations are only applied during training (Fit), never during
/// prediction. This follows the industry standard established by imbalanced-learn.
/// </para>
/// <para>
/// <b>For Beginners:</b> Some preprocessing steps need to add or remove entire data points
/// (rows) from your dataset. When you remove an outlier or create a synthetic sample,
/// you need to update both the input features AND the corresponding labels together,
/// otherwise they would become misaligned.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public interface IRowOperation<T>
{
    /// <summary>
    /// Gets whether this operation has been fitted to data.
    /// </summary>
    bool IsFitted { get; }

    /// <summary>
    /// Fits the operation to data and applies the row modification for Matrix/Vector data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method learns any necessary parameters from the data (e.g., outlier thresholds,
    /// class distributions) and then applies the row operation.
    /// </para>
    /// <para>
    /// <b>Important:</b> This is the ONLY time the row operation is applied. During prediction,
    /// the data passes through unchanged because you cannot remove or add samples when making
    /// predictions on new data.
    /// </para>
    /// </remarks>
    /// <param name="X">The feature matrix where each row is a sample.</param>
    /// <param name="y">The label vector where each element corresponds to a row in X.</param>
    /// <returns>A tuple containing the modified (X, y) with rows added or removed.</returns>
    (Matrix<T> X, Vector<T> y) FitResample(Matrix<T> X, Vector<T> y);

    /// <summary>
    /// Fits the operation to data and applies the row modification for Tensor data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method learns any necessary parameters from the data and applies the row operation
    /// to tensor data (e.g., images, sequences, or other multi-dimensional structures).
    /// </para>
    /// <para>
    /// <b>Important:</b> The first dimension of the tensor is assumed to be the sample/batch dimension.
    /// </para>
    /// </remarks>
    /// <param name="X">The feature tensor where the first dimension is samples.</param>
    /// <param name="y">The label tensor where the first dimension corresponds to samples in X.</param>
    /// <returns>A tuple containing the modified (X, y) with samples added or removed.</returns>
    (Tensor<T> X, Tensor<T> y) FitResampleTensor(Tensor<T> X, Tensor<T> y);

    /// <summary>
    /// Gets a description of what this operation does.
    /// </summary>
    string Description { get; }
}
