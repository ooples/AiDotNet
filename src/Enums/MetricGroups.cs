namespace AiDotNet.Enums;

/// <summary>
/// Defines the types of metrics that can be used to evaluate model performance.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This enum groups different ways to measure how well a model is performing.
/// Different types of machine learning problems need different metrics to evaluate them properly.
/// For example, you wouldn't use the same scoring method for predicting house prices as you
/// would for classifying emails as spam or not spam.
/// </para>
/// </remarks>
public enum MetricGroups
{
    /// <summary>
    /// Metrics for evaluating regression model performance, like MAE, MSE, RMSE.
    /// </summary>
    Regression,

    /// <summary>
    /// Metrics for evaluating binary classification performance, like accuracy, precision, recall.
    /// </summary>
    BinaryClassification,

    /// <summary>
    /// Metrics for evaluating multi-class classification performance, like accuracy, F1-score.
    /// </summary>
    MulticlassClassification,

    /// <summary>
    /// Metrics specific to time series forecasting, like MAPE, Theil's U, Durbin-Watson.
    /// </summary>
    TimeSeries,

    /// <summary>
    /// Metrics for evaluating clustering quality, like silhouette score, inertia.
    /// </summary>
    Clustering,

    /// <summary>
    /// Metrics for evaluating neural network and deep learning model performance.
    /// </summary>
    NeuralNetwork,

    /// <summary>
    /// Metrics that apply to all model types, like training time, memory usage.
    /// </summary>
    General
}