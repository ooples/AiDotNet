namespace AiDotNet.OutlierRemoval;

/// <summary>
/// Implements a pass-through outlier removal strategy that does not remove any data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides a null implementation of the <see cref="IOutlierRemoval{T}"/> interface.
/// It acts as a no-operation (no-op) implementation that simply returns the original inputs and outputs
/// without performing any outlier detection or removal. This can be useful as a baseline for comparison
/// with other outlier removal strategies, or in situations where you want to maintain the interface
/// consistency but explicitly choose not to perform outlier removal.
/// </para>
/// <para><b>For Beginners:</b> This class is a "do nothing" version of outlier removal.
/// 
/// While other outlier removal classes actually look for and remove unusual data points,
/// this class simply returns all your data exactly as it was given, without removing anything.
/// 
/// You might use this class when:
/// - You want to compare results with and without outlier removal
/// - You're testing code that expects an outlier removal component, but you don't actually want to remove outliers
/// - You believe all your data points are valid and don't want any removed
/// 
/// Think of it like a placebo pill in medicine - it maintains the same interface as active treatments
/// but doesn't actually do anything.
/// </para>
/// </remarks>
public class NoOutlierRemoval<T, TInput, TOutput> : IOutlierRemoval<T, TInput, TOutput>
{
    /// <summary>
    /// Returns the original inputs and outputs without removing any data points.
    /// </summary>
    /// <param name="inputs">The matrix of input data, where each row represents a data point and each column represents a feature.</param>
    /// <param name="outputs">The vector of output values corresponding to each row in the inputs matrix.</param>
    /// <returns>A tuple containing the original inputs and outputs with no changes.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the <see cref="IOutlierRemoval{T}.RemoveOutliers"/> method, but instead of
    /// performing any outlier detection or removal, it simply returns the original inputs and outputs
    /// unchanged. This maintains the interface contract while effectively skipping the outlier removal step.
    /// </para>
    /// <para><b>For Beginners:</b> This method just passes your data through without changing anything.
    /// 
    /// When you call this method:
    /// - It takes your input data (features) and output data (labels/values)
    /// - It immediately returns exactly the same data without analyzing or modifying it
    /// 
    /// It's like having a filter that doesn't actually filter anything out - everything passes through unchanged.
    /// This is useful when you want to maintain the same code structure but skip the outlier removal step.
    /// </para>
    /// </remarks>
    public (TInput CleanedInputs, TOutput CleanedOutputs) RemoveOutliers(TInput inputs, TOutput outputs)
    {
        return (inputs, outputs);
    }
}
