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
    /// Fits the operation to data and applies the row modification.
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
    /// Gets a description of what this operation does.
    /// </summary>
    string Description { get; }
}
