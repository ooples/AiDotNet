using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for algorithmic outlier/anomaly detection using a fit-predict pattern.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This interface provides a machine learning-style approach to outlier detection.
/// Unlike simple statistical methods (like Z-score or IQR), algorithmic detectors learn patterns
/// from your data and can detect more complex types of anomalies.
/// </para>
/// <para>
/// The typical workflow is:
/// 1. Create a detector with your desired parameters
/// 2. Call <see cref="Fit"/> to train the detector on "normal" data
/// 3. Call <see cref="Predict"/> to identify outliers in new data
/// 4. Optionally use <see cref="DecisionFunction"/> to get anomaly scores
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("OutlierDetector")]
public interface IOutlierDetector<T>
{
    /// <summary>
    /// Trains the outlier detector on the provided data.
    /// </summary>
    /// <param name="X">
    /// The training data matrix where each row is a sample and each column is a feature.
    /// For novelty detection, this should contain only "normal" (non-outlier) samples.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method teaches the detector what "normal" data looks like.
    /// After fitting, the detector can identify data points that don't fit the learned pattern.
    /// </para>
    /// </remarks>
    void Fit(Matrix<T> X);

    /// <summary>
    /// Predicts whether each sample is an inlier or outlier.
    /// </summary>
    /// <param name="X">
    /// The data matrix to predict on, where each row is a sample.
    /// </param>
    /// <returns>
    /// A vector where each element is:
    /// <list type="bullet">
    /// <item><description>1 for inliers (normal points)</description></item>
    /// <item><description>-1 for outliers (anomalies)</description></item>
    /// </list>
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After training with <see cref="Fit"/>, use this method to check
    /// if new data points are normal (1) or outliers (-1). Think of it as asking
    /// "does this data point look similar to what I learned?"
    /// </para>
    /// </remarks>
    Vector<T> Predict(Matrix<T> X);

    /// <summary>
    /// Computes the anomaly score for each sample.
    /// </summary>
    /// <param name="X">
    /// The data matrix to score, where each row is a sample.
    /// </param>
    /// <returns>
    /// A vector of anomaly scores where higher values indicate more anomalous samples.
    /// The interpretation of scores depends on the specific algorithm.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> While <see cref="Predict"/> gives a binary answer (inlier/outlier),
    /// this method gives you a "how anomalous" score for each point. Points with higher scores
    /// are more likely to be outliers. This is useful when you want to:
    /// - Rank points by how unusual they are
    /// - Use a custom threshold for classification
    /// - Investigate borderline cases more closely
    /// </para>
    /// </remarks>
    Vector<T> DecisionFunction(Matrix<T> X);

    /// <summary>
    /// Gets the threshold used to classify samples as inliers or outliers.
    /// </summary>
    /// <remarks>
    /// Samples with decision function values below this threshold are classified as outliers.
    /// </remarks>
    T Threshold { get; }

    /// <summary>
    /// Gets a value indicating whether the detector has been fitted to data.
    /// </summary>
    bool IsFitted { get; }
}
