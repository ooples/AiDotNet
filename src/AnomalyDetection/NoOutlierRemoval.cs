using AiDotNet.Interfaces;

namespace AiDotNet.AnomalyDetection;

/// <summary>
/// A no-operation outlier removal implementation that preserves all data without modification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class is used when you don't want to remove any outliers from your data.
/// It simply returns the original data unchanged, which is useful as a default option or when
/// you've already cleaned your data.
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When your data has already been preprocessed for outliers
/// - When you want to compare model performance with and without outlier removal
/// - When your domain requires keeping all data points (e.g., fraud detection)
/// </para>
/// </remarks>
public class NoOutlierRemoval<T, TInput, TOutput> : IOutlierRemoval<T, TInput, TOutput>
{
    /// <summary>
    /// Returns the input data unchanged without removing any outliers.
    /// </summary>
    /// <param name="inputs">The input feature data.</param>
    /// <param name="outputs">The target values.</param>
    /// <returns>A tuple containing the original inputs and outputs unchanged.</returns>
    public (TInput CleanedInputs, TOutput CleanedOutputs) RemoveOutliers(TInput inputs, TOutput outputs)
    {
        return (inputs, outputs);
    }
}
