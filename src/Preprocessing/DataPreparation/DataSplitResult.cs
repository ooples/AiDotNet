using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation;

/// <summary>
/// Contains the results of a data split operation for Matrix/Vector data.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> After splitting your data, this class holds all the pieces:
/// - Training data (XTrain, yTrain): What your model learns from
/// - Test data (XTest, yTest): What you evaluate your model on
/// - Validation data (XValidation, yValidation): Optional, for hyperparameter tuning
/// - Indices: Which rows from the original data ended up in each set
/// </para>
/// <para>
/// <b>Why Track Indices?</b>
/// Knowing which original samples are in each set is useful for:
/// - Debugging: "Why did my model get sample #42 wrong?"
/// - Analysis: Comparing predictions to original data
/// - Reproducibility: Recording exactly how data was split
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class DataSplitResult<T>
{
    /// <summary>
    /// Gets the feature matrix for training.
    /// </summary>
    public required Matrix<T> XTrain { get; init; }

    /// <summary>
    /// Gets the feature matrix for testing.
    /// </summary>
    public required Matrix<T> XTest { get; init; }

    /// <summary>
    /// Gets the target vector for training (null for unsupervised learning).
    /// </summary>
    public Vector<T>? yTrain { get; init; }

    /// <summary>
    /// Gets the target vector for testing (null for unsupervised learning).
    /// </summary>
    public Vector<T>? yTest { get; init; }

    /// <summary>
    /// Gets the feature matrix for validation (optional three-way split).
    /// </summary>
    public Matrix<T>? XValidation { get; init; }

    /// <summary>
    /// Gets the target vector for validation (optional three-way split).
    /// </summary>
    public Vector<T>? yValidation { get; init; }

    /// <summary>
    /// Gets the indices of rows from the original data that are in the training set.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Example:</b> If TrainIndices = [0, 2, 5, 7], then XTrain contains
    /// rows 0, 2, 5, and 7 from the original data.
    /// </para>
    /// </remarks>
    public required int[] TrainIndices { get; init; }

    /// <summary>
    /// Gets the indices of rows from the original data that are in the test set.
    /// </summary>
    public required int[] TestIndices { get; init; }

    /// <summary>
    /// Gets the indices of rows from the original data that are in the validation set (optional).
    /// </summary>
    public int[]? ValidationIndices { get; init; }

    /// <summary>
    /// Gets the fold index for cross-validation methods (0-based).
    /// </summary>
    /// <remarks>
    /// <para>
    /// For single-split methods, this is null or 0.
    /// For k-fold, this indicates which fold (0 to k-1) this result represents.
    /// </para>
    /// </remarks>
    public int? FoldIndex { get; init; }

    /// <summary>
    /// Gets the total number of folds for cross-validation methods.
    /// </summary>
    public int? TotalFolds { get; init; }

    /// <summary>
    /// Gets the repeat index for repeated cross-validation methods (0-based).
    /// </summary>
    public int? RepeatIndex { get; init; }

    /// <summary>
    /// Gets the total number of repeats for repeated cross-validation methods.
    /// </summary>
    public int? TotalRepeats { get; init; }

    /// <summary>
    /// Gets whether this result includes a validation set.
    /// </summary>
    public bool HasValidationSet => XValidation != null;

    /// <summary>
    /// Gets a human-readable summary of this split result.
    /// </summary>
    /// <returns>A string describing the split sizes.</returns>
    public string GetSummary()
    {
        var parts = new List<string>
        {
            $"Train: {XTrain.Rows} samples",
            $"Test: {XTest.Rows} samples"
        };

        if (XValidation != null)
        {
            parts.Add($"Validation: {XValidation.Rows} samples");
        }

        if (FoldIndex.HasValue && TotalFolds.HasValue)
        {
            parts.Add($"Fold {FoldIndex.Value + 1}/{TotalFolds.Value}");
        }

        if (RepeatIndex.HasValue && TotalRepeats.HasValue)
        {
            parts.Add($"Repeat {RepeatIndex.Value + 1}/{TotalRepeats.Value}");
        }

        return string.Join(", ", parts);
    }
}

/// <summary>
/// Contains the results of a data split operation for Tensor data.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is the same as DataSplitResult, but for tensor (multi-dimensional) data
/// like images, sequences, or other complex structures.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class TensorSplitResult<T>
{
    /// <summary>
    /// Gets the feature tensor for training.
    /// </summary>
    public required Tensor<T> XTrain { get; init; }

    /// <summary>
    /// Gets the feature tensor for testing.
    /// </summary>
    public required Tensor<T> XTest { get; init; }

    /// <summary>
    /// Gets the target tensor for training (null for unsupervised learning).
    /// </summary>
    public Tensor<T>? yTrain { get; init; }

    /// <summary>
    /// Gets the target tensor for testing (null for unsupervised learning).
    /// </summary>
    public Tensor<T>? yTest { get; init; }

    /// <summary>
    /// Gets the feature tensor for validation (optional three-way split).
    /// </summary>
    public Tensor<T>? XValidation { get; init; }

    /// <summary>
    /// Gets the target tensor for validation (optional three-way split).
    /// </summary>
    public Tensor<T>? yValidation { get; init; }

    /// <summary>
    /// Gets the indices of samples from the original data that are in the training set.
    /// </summary>
    public required int[] TrainIndices { get; init; }

    /// <summary>
    /// Gets the indices of samples from the original data that are in the test set.
    /// </summary>
    public required int[] TestIndices { get; init; }

    /// <summary>
    /// Gets the indices of samples from the original data that are in the validation set (optional).
    /// </summary>
    public int[]? ValidationIndices { get; init; }

    /// <summary>
    /// Gets the fold index for cross-validation methods (0-based).
    /// </summary>
    public int? FoldIndex { get; init; }

    /// <summary>
    /// Gets the total number of folds for cross-validation methods.
    /// </summary>
    public int? TotalFolds { get; init; }

    /// <summary>
    /// Gets the repeat index for repeated cross-validation methods (0-based).
    /// </summary>
    public int? RepeatIndex { get; init; }

    /// <summary>
    /// Gets the total number of repeats for repeated cross-validation methods.
    /// </summary>
    public int? TotalRepeats { get; init; }

    /// <summary>
    /// Gets whether this result includes a validation set.
    /// </summary>
    public bool HasValidationSet => XValidation != null;

    /// <summary>
    /// Gets a human-readable summary of this split result.
    /// </summary>
    /// <returns>A string describing the split sizes.</returns>
    public string GetSummary()
    {
        var parts = new List<string>
        {
            $"Train: {XTrain.Shape[0]} samples",
            $"Test: {XTest.Shape[0]} samples"
        };

        if (XValidation != null)
        {
            parts.Add($"Validation: {XValidation.Shape[0]} samples");
        }

        if (FoldIndex.HasValue && TotalFolds.HasValue)
        {
            parts.Add($"Fold {FoldIndex.Value + 1}/{TotalFolds.Value}");
        }

        if (RepeatIndex.HasValue && TotalRepeats.HasValue)
        {
            parts.Add($"Repeat {RepeatIndex.Value + 1}/{TotalRepeats.Value}");
        }

        return string.Join(", ", parts);
    }
}
