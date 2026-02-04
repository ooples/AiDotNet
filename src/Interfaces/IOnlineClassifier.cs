namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for online (incremental) classification models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Online classifiers can learn from data one sample at a time,
/// without needing to retrain from scratch. This is useful for:
/// <list type="bullet">
/// <item>Streaming data where samples arrive continuously</item>
/// <item>Large datasets that don't fit in memory</item>
/// <item>Applications requiring real-time adaptation</item>
/// </list>
/// </para>
///
/// <para><b>Key differences from batch learning:</b>
/// <list type="bullet">
/// <item>No need to store all training data</item>
/// <item>Model updates incrementally with each sample</item>
/// <item>Can adapt to concept drift</item>
/// <item>Trade-off: may not achieve same accuracy as batch training</item>
/// </list>
/// </para>
///
/// <para><b>Common online learning algorithms:</b>
/// <list type="bullet">
/// <item>Hoeffding Tree (Very Fast Decision Tree)</item>
/// <item>Online Naive Bayes</item>
/// <item>Stochastic Gradient Descent</item>
/// <item>Perceptron</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface IOnlineClassifier<T> : IClassifier<T>
{
    /// <summary>
    /// Updates the model with a single training sample.
    /// </summary>
    /// <param name="features">Feature vector for the sample.</param>
    /// <param name="label">Class label for the sample.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the key method for online learning.
    /// Call it once for each new sample to incrementally update the model.</para>
    /// </remarks>
    void PartialFit(Vector<T> features, T label);

    /// <summary>
    /// Updates the model with a batch of training samples.
    /// </summary>
    /// <param name="features">Feature matrix [n_samples, n_features].</param>
    /// <param name="labels">Class labels for each sample.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Convenience method to update with multiple samples at once.
    /// Internally calls PartialFit for each sample.</para>
    /// </remarks>
    void PartialFit(Matrix<T> features, Vector<T> labels);

    /// <summary>
    /// Gets the total number of samples the model has seen.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Useful for tracking how much data has been used
    /// to train the model incrementally.</para>
    /// </remarks>
    long SamplesSeen { get; }

    /// <summary>
    /// Gets whether the model is warm (has seen at least one sample).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some online models need at least one sample before
    /// they can make predictions. Check this property to know if the model is ready.</para>
    /// </remarks>
    bool IsWarm { get; }
}
