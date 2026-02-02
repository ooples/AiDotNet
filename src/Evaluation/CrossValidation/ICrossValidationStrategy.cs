using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.CrossValidation;

/// <summary>
/// Defines a cross-validation splitting strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Cross-validation strategies determine how to split your data into
/// training and validation sets. Different strategies are appropriate for different data types:
/// <list type="bullet">
/// <item><b>K-Fold:</b> Standard approach for most data</item>
/// <item><b>Stratified K-Fold:</b> Preserves class distribution (for classification)</item>
/// <item><b>Time Series Split:</b> Respects temporal order (for time series)</item>
/// <item><b>Leave-One-Out:</b> Maximum data usage but computationally expensive</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for data elements.</typeparam>
public interface ICrossValidationStrategy<T>
{
    /// <summary>
    /// Gets the name of this cross-validation strategy.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the number of splits (folds) this strategy will generate.
    /// </summary>
    int NumSplits { get; }

    /// <summary>
    /// Gets a description of this strategy suitable for documentation.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Generates train/validation index splits for the given data size.
    /// </summary>
    /// <param name="dataSize">Number of samples in the dataset.</param>
    /// <param name="labels">Optional labels for stratified splitting.</param>
    /// <returns>Enumerable of (trainIndices, validationIndices) tuples.</returns>
    IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> Split(int dataSize, ReadOnlySpan<T> labels = default);
}

/// <summary>
/// Represents a single cross-validation fold with train and validation data.
/// </summary>
/// <typeparam name="T">The numeric type for data elements.</typeparam>
public readonly struct CVFold<T>
{
    /// <summary>
    /// The fold number (0-indexed).
    /// </summary>
    public int FoldIndex { get; init; }

    /// <summary>
    /// Indices for training samples.
    /// </summary>
    public int[] TrainIndices { get; init; }

    /// <summary>
    /// Indices for validation samples.
    /// </summary>
    public int[] ValidationIndices { get; init; }

    /// <summary>
    /// Number of training samples in this fold.
    /// </summary>
    public int TrainSize => TrainIndices?.Length ?? 0;

    /// <summary>
    /// Number of validation samples in this fold.
    /// </summary>
    public int ValidationSize => ValidationIndices?.Length ?? 0;
}
