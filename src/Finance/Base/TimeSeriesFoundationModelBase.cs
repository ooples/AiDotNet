using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Finance.Base;

/// <summary>
/// Abstract base class for time series foundation models that support multiple downstream tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This base class extends <see cref="ForecastingModelBase{T}"/> with multi-task capabilities
/// defined by <see cref="ITimeSeriesFoundationModel{T}"/>. It provides default implementations
/// that throw <see cref="NotSupportedException"/> for optional tasks, allowing single-task models
/// (e.g., forecasting-only) to inherit without implementing every method.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the foundation that all time series foundation models build upon.
/// It provides:
/// <list type="bullet">
/// <item>Common infrastructure for both ONNX and native mode operation</item>
/// <item>Default "not supported" implementations for optional tasks</item>
/// <item>A <c>ValidateTaskSupported</c> helper to check task compatibility</item>
/// <item>Standard properties for model size, parameter count, and context limits</item>
/// </list>
///
/// Models that support only forecasting (like TimesFM) can inherit this class and only
/// override the forecasting-related methods. Multi-task models (like MOMENT) override
/// the additional task methods they support.
/// </para>
/// </remarks>
public abstract class TimeSeriesFoundationModelBase<T> : ForecastingModelBase<T>, ITimeSeriesFoundationModel<T>
{
    #region Constructors

    /// <summary>
    /// Initializes a new foundation model with deferred configuration.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="lossFunction">Optional loss function override.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor keeps the classic Finance model pattern
    /// where derived classes fill in sequence length and other settings afterward.
    /// </para>
    /// </remarks>
    protected TimeSeriesFoundationModelBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction, maxGradNorm)
    {
    }

    /// <summary>
    /// Initializes a new foundation model in native mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">Input sequence length.</param>
    /// <param name="predictionHorizon">Prediction horizon (future steps to forecast).</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="lossFunction">Optional loss function override.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when training a foundation model from scratch
    /// or fine-tuning with native C# layers.
    /// </para>
    /// </remarks>
    protected TimeSeriesFoundationModelBase(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength,
        int predictionHorizon,
        int numFeatures,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, sequenceLength, predictionHorizon, numFeatures, lossFunction)
    {
    }

    /// <summary>
    /// Initializes a new foundation model in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="sequenceLength">Input sequence length expected by the ONNX model.</param>
    /// <param name="predictionHorizon">Prediction horizon expected by the ONNX model.</param>
    /// <param name="numFeatures">Number of input features expected by the ONNX model.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you have a pretrained ONNX model
    /// and only need fast inference.
    /// </para>
    /// </remarks>
    protected TimeSeriesFoundationModelBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int sequenceLength,
        int predictionHorizon,
        int numFeatures)
        : base(architecture, onnxModelPath, sequenceLength, predictionHorizon, numFeatures)
    {
    }

    #endregion

    #region ITimeSeriesFoundationModel Properties

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Override this in derived classes to list all tasks the model supports.
    /// The default implementation returns only <see cref="TimeSeriesFoundationModelTask.Forecasting"/>.
    /// </para>
    /// </remarks>
    public virtual IReadOnlyList<TimeSeriesFoundationModelTask> SupportedTasks { get; } =
        new[] { TimeSeriesFoundationModelTask.Forecasting };

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Override this in derived classes if the model supports switching
    /// between tasks at runtime.
    /// </para>
    /// </remarks>
    public virtual TimeSeriesFoundationModelTask CurrentTask { get; } = TimeSeriesFoundationModelTask.Forecasting;

    /// <inheritdoc/>
    public abstract FoundationModelSize ModelSize { get; }

    /// <inheritdoc/>
    public abstract int MaxContextLength { get; }

    /// <inheritdoc/>
    public abstract int MaxPredictionHorizon { get; }

    #endregion

    #region Multi-Task Methods (Default NotSupportedException)

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Default implementation throws <see cref="NotSupportedException"/>.
    /// Override in derived classes that support anomaly detection.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> DetectAnomalies(Tensor<T> series, double? threshold = null)
    {
        ValidateTaskSupported(TimeSeriesFoundationModelTask.AnomalyDetection);
        throw new NotSupportedException($"Override {nameof(DetectAnomalies)} in a derived class to provide an implementation.");
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Default implementation throws <see cref="NotSupportedException"/>.
    /// Override in derived classes that support classification.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> Classify(Tensor<T> series, int numClasses)
    {
        ValidateTaskSupported(TimeSeriesFoundationModelTask.Classification);
        throw new NotSupportedException($"Override {nameof(Classify)} in a derived class to provide an implementation.");
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Default implementation throws <see cref="NotSupportedException"/>.
    /// Override in derived classes that support imputation.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> Impute(Tensor<T> series, Tensor<T> mask)
    {
        ValidateTaskSupported(TimeSeriesFoundationModelTask.Imputation);
        throw new NotSupportedException($"Override {nameof(Impute)} in a derived class to provide an implementation.");
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Default implementation throws <see cref="NotSupportedException"/>.
    /// Override in derived classes that support embedding generation.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> Embed(Tensor<T> series)
    {
        ValidateTaskSupported(TimeSeriesFoundationModelTask.Embedding);
        throw new NotSupportedException($"Override {nameof(Embed)} in a derived class to provide an implementation.");
    }

    #endregion

    #region Helpers

    /// <summary>
    /// Validates that the specified task is supported by this model.
    /// </summary>
    /// <param name="task">The task to validate.</param>
    /// <exception cref="NotSupportedException">
    /// Thrown when the task is not in <see cref="SupportedTasks"/>.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Call this at the start of any task-specific method to give
    /// users a clear error message if they try to use an unsupported task.
    /// </para>
    /// </remarks>
    protected void ValidateTaskSupported(TimeSeriesFoundationModelTask task)
    {
        if (!SupportedTasks.Contains(task))
        {
            throw new NotSupportedException(
                $"{GetType().Name} does not support the '{task}' task. " +
                $"Supported tasks: {string.Join(", ", SupportedTasks)}.");
        }
    }

    #endregion
}
