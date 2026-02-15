using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the interface for online (incremental) learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Online learning models can update their parameters incrementally as new data arrives,
/// without needing to retrain from scratch on all data. This is essential for streaming
/// data and large-scale machine learning.
/// </para>
/// <para>
/// <b>For Beginners:</b> Online learning is like learning continuously from experience:
///
/// Traditional (Batch) Learning:
/// - Collect ALL the data first
/// - Train the model once on everything
/// - If new data arrives, retrain from scratch
///
/// Online (Incremental) Learning:
/// - Start with minimal or no data
/// - Learn from each new example as it arrives
/// - Continuously adapt to new patterns
///
/// Why use online learning?
/// - Streaming data: Data arrives continuously (e.g., stock prices, web clicks)
/// - Large datasets: Too big to fit in memory all at once
/// - Changing patterns: Data distribution shifts over time (concept drift)
/// - Real-time adaptation: Need to respond quickly to new information
///
/// Common applications:
/// - Spam filtering (adapt to new spam patterns)
/// - Recommendation systems (adapt to user preferences)
/// - Fraud detection (adapt to new fraud patterns)
/// - Stock trading (adapt to market conditions)
///
/// References:
/// - Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent"
/// - Domingos &amp; Hulten (2000). "Mining High-Speed Data Streams"
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("OnlineLearningModel")]
public interface IOnlineLearningModel<T> : IFullModel<T, Matrix<T>, Vector<T>>
{
    /// <summary>
    /// Updates the model with a single training example.
    /// </summary>
    /// <param name="x">The feature vector for one sample.</param>
    /// <param name="y">The target value for the sample.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the core of online learning - updating the model
    /// one example at a time. Each call adjusts the model slightly based on the new example.
    ///
    /// Unlike batch training which sees all data multiple times, online learning typically
    /// sees each example only once (single-pass learning).
    /// </para>
    /// </remarks>
    void PartialFit(Vector<T> x, T y);

    /// <summary>
    /// Updates the model with a mini-batch of training examples.
    /// </summary>
    /// <param name="x">The feature matrix (rows = samples).</param>
    /// <param name="y">The target values.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mini-batch learning is a compromise between:
    /// - Pure online (one example at a time): Most adaptive but noisy
    /// - Batch (all data at once): Most stable but slow to adapt
    ///
    /// Mini-batches (e.g., 32-128 examples) provide good balance between
    /// adaptation speed and stability.
    /// </para>
    /// </remarks>
    void PartialFit(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Predicts the target value for a single sample.
    /// </summary>
    /// <param name="x">The feature vector.</param>
    /// <returns>The predicted value.</returns>
    T PredictSingle(Vector<T> x);

    /// <summary>
    /// Gets the number of samples the model has seen.
    /// </summary>
    /// <returns>Total number of training samples processed.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tracks how much "experience" the model has.
    /// More samples generally means more reliable predictions, but the model
    /// should still adapt to new patterns.
    /// </para>
    /// </remarks>
    long GetSampleCount();

    /// <summary>
    /// Resets the model to its initial state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this to start fresh, forgetting all learned patterns.
    /// Useful when you know the underlying data distribution has completely changed.
    /// </para>
    /// </remarks>
    void Reset();

    /// <summary>
    /// Gets the current learning rate.
    /// </summary>
    /// <returns>The learning rate.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Learning rate controls how much the model changes with each update:
    /// - High learning rate: Fast adaptation, but can overshoot and be unstable
    /// - Low learning rate: Slow adaptation, but more stable and precise
    ///
    /// Many online algorithms decrease the learning rate over time for convergence.
    /// </para>
    /// </remarks>
    T GetLearningRate();
}

/// <summary>
/// Defines the interface for concept drift detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Concept drift detectors monitor the data stream for changes in the underlying
/// data distribution, signaling when the model may need to adapt or retrain.
/// </para>
/// <para>
/// <b>For Beginners:</b> Concept drift happens when the patterns in your data change over time:
///
/// Examples of concept drift:
/// - Spam: New types of spam emails emerge
/// - Shopping: Customer preferences change seasonally
/// - Fraud: Fraudsters develop new techniques
/// - Weather: Climate patterns shift over years
///
/// Types of drift:
/// - Sudden drift: Abrupt change (e.g., policy change)
/// - Gradual drift: Slow transition between concepts
/// - Recurring drift: Patterns come back (e.g., seasonal)
/// - Incremental drift: Small, continuous changes
///
/// Without drift detection, your model's accuracy will silently degrade as
/// the data it was trained on becomes less relevant.
///
/// References:
/// - Gama et al. (2004). "Learning with Drift Detection"
/// - Bifet &amp; Gavaldà (2007). "Learning from Time-Changing Data with Adaptive Windowing"
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("DriftDetector")]
public interface IDriftDetector<T>
{
    /// <summary>
    /// Updates the detector with a new observation.
    /// </summary>
    /// <param name="value">The new value (typically prediction error or loss).</param>
    /// <returns>The current drift status.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Feed the detector your model's prediction errors.
    /// When errors start increasing, it indicates potential drift.
    ///
    /// The detector returns:
    /// - NoDrift: Everything is normal
    /// - Warning: Performance degrading, might need attention
    /// - Drift: Significant change detected, model needs updating
    /// </para>
    /// </remarks>
    DriftStatus Update(T value);

    /// <summary>
    /// Checks if drift has been detected.
    /// </summary>
    bool IsDriftDetected { get; }

    /// <summary>
    /// Checks if a warning (potential drift) has been detected.
    /// </summary>
    bool IsWarning { get; }

    /// <summary>
    /// Resets the detector to its initial state.
    /// </summary>
    void Reset();

    /// <summary>
    /// Gets the estimated change point (when drift started).
    /// </summary>
    /// <returns>Index of estimated change point, or -1 if not available.</returns>
    long GetChangePoint();

    /// <summary>
    /// Gets the current detection statistics.
    /// </summary>
    /// <returns>Dictionary of statistic name to value.</returns>
    Dictionary<string, T> GetStatistics();
}

/// <summary>
/// Represents the status of drift detection.
/// </summary>
public enum DriftStatus
{
    /// <summary>
    /// No drift detected - model is performing as expected.
    /// </summary>
    NoDrift,

    /// <summary>
    /// Warning level - performance may be degrading, monitor closely.
    /// </summary>
    Warning,

    /// <summary>
    /// Drift detected - significant change in data distribution.
    /// </summary>
    Drift
}
