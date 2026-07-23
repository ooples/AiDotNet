using System;
using System.Collections.Generic;
using AiDotNet.Enums;

namespace AiDotNet.Finance.Interfaces;

/// <summary>
/// Interface for multi-task time series foundation models that support forecasting,
/// anomaly detection, classification, imputation, and embedding generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Time series foundation models are large pretrained models that can handle multiple
/// downstream tasks using a single architecture. Unlike traditional models that are
/// purpose-built for one task (e.g., only forecasting), foundation models learn general
/// representations of time series data during pretraining and can be applied to various
/// tasks with minimal or no fine-tuning.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of foundation models as "Swiss Army knives" for time series:
///
/// <b>Traditional approach:</b> Train separate models for each task
/// - Model A: Forecasting
/// - Model B: Anomaly detection
/// - Model C: Classification
/// - Model D: Imputation
///
/// <b>Foundation model approach:</b> One pretrained model, many tasks
/// - Same model handles forecasting, anomaly detection, classification, etc.
/// - Works out of the box on new data (zero-shot capability)
/// - Can be fine-tuned for better performance on specific datasets
///
/// <b>Key Benefits:</b>
/// <list type="bullet">
/// <item>Reduced development time — no need to train multiple models</item>
/// <item>Better generalization — pretrained on diverse data</item>
/// <item>Consistent feature extraction — shared representations across tasks</item>
/// <item>Lower total compute — one model instead of many</item>
/// </list>
/// </para>
/// <para>
/// <b>Reference:</b> This interface is inspired by models such as MOMENT (CMU, ICML 2024),
/// Chronos-2 (Amazon, 2025), Moirai 2.0 (Salesforce, 2025), TimesFM 2.5 (Google, 2025),
/// and Tiny Time Mixers (IBM, NeurIPS 2024).
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("TimeSeriesFoundationModel")]
public interface ITimeSeriesFoundationModel<T> : IForecastingModel<T>
{
    /// <summary>
    /// Gets the list of tasks this foundation model supports.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Not every foundation model supports every task. Check this
    /// property before calling a task-specific method to avoid <see cref="NotSupportedException"/>.
    /// For example, a forecasting-only model like TimesFM will only list
    /// <see cref="TimeSeriesFoundationModelTask.Forecasting"/>.
    /// </para>
    /// </remarks>
    IReadOnlyList<TimeSeriesFoundationModelTask> SupportedTasks { get; }

    /// <summary>
    /// Gets the task this model is currently configured to perform.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When a model supports multiple tasks, this property indicates
    /// which task is currently active. The default is typically
    /// <see cref="TimeSeriesFoundationModelTask.Forecasting"/>.
    /// </para>
    /// </remarks>
    TimeSeriesFoundationModelTask CurrentTask { get; }

    /// <summary>
    /// Gets the size variant of this foundation model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Foundation models come in different sizes (Tiny, Small, Base,
    /// Large, etc.). Larger models are generally more accurate but require more memory
    /// and computation time.
    /// </para>
    /// </remarks>
    FoundationModelSize ModelSize { get; }

    /// <summary>
    /// Gets the maximum context length (number of input time steps) the model can process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the maximum number of past time steps you can feed
    /// to the model. Longer context lets the model see more history, which can improve
    /// accuracy for patterns with long-term dependencies. Modern foundation models
    /// support contexts from 512 to 16,384 time steps.
    /// </para>
    /// </remarks>
    int MaxContextLength { get; }

    /// <summary>
    /// Gets the maximum prediction horizon the model can produce in a single forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the furthest into the future the model can predict
    /// without using autoregressive (iterative) generation. For longer horizons, use
    /// <see cref="IForecastingModel{T}.AutoregressiveForecast"/>.
    /// </para>
    /// </remarks>
    int MaxPredictionHorizon { get; }

    /// <summary>
    /// Detects anomalies in the input time series.
    /// </summary>
    /// <param name="series">Input time series tensor of shape [batch_size, sequence_length, num_features].</param>
    /// <param name="threshold">
    /// Optional anomaly threshold. If null, the model uses a default threshold based on
    /// reconstruction error statistics. Values closer to 0 detect more anomalies (higher sensitivity).
    /// </param>
    /// <returns>
    /// Anomaly score tensor of shape [batch_size, sequence_length, 1] where higher values
    /// indicate more anomalous time steps.
    /// </returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when the model does not support anomaly detection
    /// (i.e., <see cref="TimeSeriesFoundationModelTask.AnomalyDetection"/> is not in <see cref="SupportedTasks"/>).
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method examines each time step and assigns an anomaly score.
    /// Higher scores mean the model considers that time step more unusual. You can then apply
    /// a threshold to decide which points are anomalies:
    /// <code>
    /// var scores = model.DetectAnomalies(data);
    /// // Points where score > threshold are considered anomalies
    /// </code>
    /// </para>
    /// </remarks>
    Tensor<T> DetectAnomalies(Tensor<T> series, double? threshold = null);

    /// <summary>
    /// Classifies the input time series into one of several categories.
    /// </summary>
    /// <param name="series">Input time series tensor of shape [batch_size, sequence_length, num_features].</param>
    /// <param name="numClasses">The number of output classes.</param>
    /// <returns>
    /// Classification logits tensor of shape [batch_size, numClasses]. Apply softmax
    /// to convert to probabilities.
    /// </returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when the model does not support classification
    /// (i.e., <see cref="TimeSeriesFoundationModelTask.Classification"/> is not in <see cref="SupportedTasks"/>).
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This assigns a label to an entire time series. For example,
    /// classifying a sensor reading as "normal operation" vs "equipment failure":
    /// <code>
    /// var logits = model.Classify(sensorData, numClasses: 3);
    /// // logits[0] = score for class 0, logits[1] = score for class 1, etc.
    /// </code>
    /// </para>
    /// </remarks>
    Tensor<T> Classify(Tensor<T> series, int numClasses);

    /// <summary>
    /// Fills in missing values in a time series using the surrounding context.
    /// </summary>
    /// <param name="series">
    /// Input time series tensor of shape [batch_size, sequence_length, num_features].
    /// Missing values should be represented as NaN or zero (depending on the model).
    /// </param>
    /// <param name="mask">
    /// Binary mask tensor of shape [batch_size, sequence_length, num_features] where
    /// 1 indicates observed values and 0 indicates missing values to impute.
    /// </param>
    /// <returns>
    /// Imputed time series tensor of the same shape as the input, with missing values filled in.
    /// </returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when the model does not support imputation
    /// (i.e., <see cref="TimeSeriesFoundationModelTask.Imputation"/> is not in <see cref="SupportedTasks"/>).
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When your data has gaps (missing sensor readings, network outages),
    /// imputation fills them in using the available context:
    /// <code>
    /// // mask: 1 = data exists, 0 = data is missing
    /// var filled = model.Impute(dataWithGaps, mask);
    /// </code>
    /// </para>
    /// </remarks>
    Tensor<T> Impute(Tensor<T> series, Tensor<T> mask);

    /// <summary>
    /// Generates a fixed-size embedding vector for the input time series.
    /// </summary>
    /// <param name="series">Input time series tensor of shape [batch_size, sequence_length, num_features].</param>
    /// <returns>
    /// Embedding tensor of shape [batch_size, embedding_dimension] where embedding_dimension
    /// is determined by the model architecture (typically the hidden dimension).
    /// </returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when the model does not support embedding generation
    /// (i.e., <see cref="TimeSeriesFoundationModelTask.Embedding"/> is not in <see cref="SupportedTasks"/>).
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Embeddings are compact representations of time series that
    /// capture their essential characteristics. Use them for:
    /// <list type="bullet">
    /// <item>Clustering similar time series together</item>
    /// <item>Computing similarity between time series</item>
    /// <item>As features for other machine learning models</item>
    /// </list>
    /// <code>
    /// var embeddings = model.Embed(timeSeriesData);
    /// // embeddings can be used for clustering, similarity search, etc.
    /// </code>
    /// </para>
    /// </remarks>
    Tensor<T> Embed(Tensor<T> series);
}
