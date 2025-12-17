namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for detecting and removing outliers from datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Outliers are unusual data points that differ significantly from most of your data.
/// These unusual values can negatively impact machine learning models by skewing results.
/// 
/// This interface provides a standard way to implement different outlier detection and removal
/// techniques. By removing outliers, you can often improve the accuracy and reliability of your
/// machine learning models.
/// </remarks>
public interface IOutlierRemoval<T, TInput, TOutput>
{
    /// <summary>
    /// Removes outliers from the input data and returns the cleaned dataset.
    /// </summary>
    /// <param name="inputs">
    /// The input feature matrix where each row represents a data point and each column represents a feature.
    /// </param>
    /// <param name="outputs">
    /// The target values vector where each element corresponds to a row in the input matrix.
    /// </param>
    /// <returns>
    /// A tuple containing:
    /// - CleanedInputs: A matrix of input features with outliers removed
    /// - CleanedOutputs: A vector of output values corresponding to the cleaned inputs
    /// </returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method analyzes your dataset to find unusual values (outliers) and removes them.
    /// 
    /// The inputs parameter is a matrix where:
    /// - Each row represents one sample or data point
    /// - Each column represents one feature or attribute
    /// 
    /// The outputs parameter is a vector where:
    /// - Each element is the target value or label for the corresponding row in the inputs matrix
    /// 
    /// After processing, the method returns both the cleaned inputs and outputs with outliers removed,
    /// maintaining the relationship between input features and their corresponding output values.
    /// </remarks>
    (TInput CleanedInputs, TOutput CleanedOutputs) RemoveOutliers(TInput inputs, TOutput outputs);
}
