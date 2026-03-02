namespace AiDotNet.Enums;

/// <summary>
/// Defines the tasks that a time series foundation model can perform.
/// </summary>
/// <remarks>
/// <para>
/// Time series foundation models are versatile architectures that can handle multiple
/// downstream tasks beyond simple forecasting. This enum provides type-safe task selection
/// instead of error-prone string-based approaches.
/// </para>
/// <para>
/// <b>For Beginners:</b> Traditional time series models are built for a single purpose
/// (e.g., only forecasting). Foundation models are more flexible — the same model can
/// forecast future values, detect anomalies, classify patterns, fill in missing data,
/// or produce embeddings. This enum lets you tell the model which task you want it to perform.
/// </para>
/// </remarks>
public enum TimeSeriesFoundationModelTask
{
    /// <summary>
    /// Predict future values of a time series given historical context.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the most common time series task. Given past data
    /// (e.g., the last 512 stock prices), the model predicts the next N values
    /// (e.g., the next 96 prices). The prediction can be a single point estimate
    /// or a probabilistic distribution.
    /// </para>
    /// </remarks>
    Forecasting,

    /// <summary>
    /// Detect unusual patterns or outliers in a time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Anomaly detection identifies data points that deviate
    /// significantly from expected behavior. For example, a sudden spike in server
    /// response time or an unusual drop in sales revenue. The model outputs an
    /// anomaly score for each time step — higher scores indicate more unusual behavior.
    /// </para>
    /// </remarks>
    AnomalyDetection,

    /// <summary>
    /// Classify an entire time series into one of several categories.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Classification assigns a label to a whole time series segment.
    /// For example, classifying an ECG signal as "normal" or "arrhythmia", or categorizing
    /// a stock's behavior as "trending", "mean-reverting", or "volatile". The model outputs
    /// probabilities for each class.
    /// </para>
    /// </remarks>
    Classification,

    /// <summary>
    /// Fill in missing values within a time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Real-world time series data often has gaps — sensor failures,
    /// network outages, or recording errors can cause missing values. Imputation uses the
    /// surrounding context to intelligently fill in these gaps. The model looks at the
    /// available data before and after each gap to estimate what the missing values should be.
    /// </para>
    /// </remarks>
    Imputation,

    /// <summary>
    /// Generate a fixed-size vector representation (embedding) of a time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An embedding converts a variable-length time series into a
    /// fixed-size vector that captures its essential characteristics. These embeddings can
    /// then be used for downstream tasks like clustering similar time series, computing
    /// similarity between series, or as features for other machine learning models.
    /// Think of it as a "fingerprint" that summarizes the time series' key patterns.
    /// </para>
    /// </remarks>
    Embedding
}
