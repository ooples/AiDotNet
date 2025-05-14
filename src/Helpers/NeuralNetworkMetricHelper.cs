using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Enums;

/// <summary>
/// Provides helper methods for determining appropriate metrics for neural network task types.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps you figure out which measurements (metrics) make sense 
/// for evaluating different types of neural networks. For example, accuracy is a good way to 
/// evaluate a network that categorizes data, but wouldn't make sense for a network that predicts 
/// house prices.
/// </para>
/// </remarks>
public static class NeuralNetworkMetricHelper
{
    /// <summary>
    /// Gets metric types valid for a specific neural network task type.
    /// </summary>
    /// <param name="taskType">The neural network task type.</param>
    /// <returns>A set of metric types valid for the specified neural network task.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method tells you which metrics make sense for evaluating
    /// a neural network performing a specific task. For example, accuracy is appropriate for
    /// classification tasks but not for regression tasks.
    /// </para>
    /// </remarks>
    public static HashSet<MetricType> GetValidMetricTypes(NeuralNetworkTaskType taskType)
    {
        // Get the corresponding model type for this task
        ModelType correspondingModelType = MapTaskTypeToModelType(taskType);

        // Get supported metric groups for this model type
        var supportedGroups = ModelTypeHelper.GetSupportedMetricGroups(correspondingModelType);

        // Add neural network specific groups if needed
        var taskGroups = new List<MetricGroups>(supportedGroups);
        if (!taskGroups.Contains(MetricGroups.NeuralNetwork))
        {
            taskGroups.Add(MetricGroups.NeuralNetwork);
        }

        // Create a hashset to store valid metrics
        var validMetrics = new HashSet<MetricType>();

        // Get all metric types
        foreach (MetricType metricType in Enum.GetValues(typeof(MetricType)))
        {
            // Get the groups for this metric
            var metricGroups = ModelTypeHelper.GetMetricGroups(metricType);

            // If any group matches supported groups, add it
            if (metricGroups.Any(group => taskGroups.Contains(group)))
            {
                validMetrics.Add(metricType);
            }
        }

        // Add task-specific metrics
        AddTaskSpecificMetrics(validMetrics, taskType);

        return validMetrics;
    }

    /// <summary>
    /// Maps a neural network task type to a corresponding standard model type.
    /// </summary>
    /// <param name="taskType">The neural network task type.</param>
    /// <returns>The corresponding model type.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method translates a neural network task to a standard model
    /// type with similar characteristics. For example, a neural network doing regression
    /// gets mapped to LinearRegression for metric selection purposes.
    /// </para>
    /// </remarks>
    public static ModelType MapTaskTypeToModelType(NeuralNetworkTaskType taskType)
    {
        return taskType switch
        {
            NeuralNetworkTaskType.Regression => ModelType.LinearRegression,
            NeuralNetworkTaskType.BinaryClassification => ModelType.LogisticRegression,
            NeuralNetworkTaskType.MultiClassClassification => ModelType.MultinomialLogisticRegression,
            NeuralNetworkTaskType.MultiLabelClassification => ModelType.MultinomialLogisticRegression,
            NeuralNetworkTaskType.Clustering => ModelType.NeuralNetwork,
            NeuralNetworkTaskType.TimeSeriesForecasting => ModelType.TimeSeriesRegression,
            NeuralNetworkTaskType.ImageClassification => ModelType.ConvolutionalNeuralNetwork,
            NeuralNetworkTaskType.ObjectDetection => ModelType.ConvolutionalNeuralNetwork,
            NeuralNetworkTaskType.ImageSegmentation => ModelType.ConvolutionalNeuralNetwork,
            NeuralNetworkTaskType.NaturalLanguageProcessing => ModelType.RecurrentNeuralNetwork,
            NeuralNetworkTaskType.TextGeneration => ModelType.RecurrentNeuralNetwork,
            NeuralNetworkTaskType.ReinforcementLearning => ModelType.DeepQNetwork,
            NeuralNetworkTaskType.Generative => ModelType.GenerativeAdversarialNetwork,
            NeuralNetworkTaskType.SpeechRecognition => ModelType.RecurrentNeuralNetwork,
            NeuralNetworkTaskType.SequenceToSequence => ModelType.LSTMNeuralNetwork,
            NeuralNetworkTaskType.SequenceClassification => ModelType.RecurrentNeuralNetwork,
            NeuralNetworkTaskType.AnomalyDetection => ModelType.Autoencoder,
            NeuralNetworkTaskType.Recommendation => ModelType.NeuralNetwork,
            NeuralNetworkTaskType.DimensionalityReduction => ModelType.Autoencoder,
            NeuralNetworkTaskType.AudioProcessing => ModelType.ConvolutionalNeuralNetwork,
            NeuralNetworkTaskType.Translation => ModelType.Transformer,
            _ => ModelType.NeuralNetwork
        };
    }

    /// <summary>
    /// Adds metrics that are specifically relevant to a particular neural network task type.
    /// </summary>
    /// <param name="metrics">The set of metrics to add to.</param>
    /// <param name="taskType">The neural network task type.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This internal method adds specialized metrics that are particularly
    /// important for specific neural network tasks, like perplexity for language models.
    /// </para>
    /// </remarks>
    private static void AddTaskSpecificMetrics(HashSet<MetricType> metrics, NeuralNetworkTaskType taskType)
    {
        switch (taskType)
        {
            case NeuralNetworkTaskType.NaturalLanguageProcessing:
            case NeuralNetworkTaskType.TextGeneration:
            case NeuralNetworkTaskType.Translation:
                metrics.Add(MetricType.Perplexity);
                metrics.Add(MetricType.LogLikelihood);
                break;

            case NeuralNetworkTaskType.Generative:
                metrics.Add(MetricType.KLDivergence);
                metrics.Add(MetricType.LogLikelihood);
                break;

            case NeuralNetworkTaskType.ReinforcementLearning:
                // Add reinforcement learning specific metrics if available
                break;

            case NeuralNetworkTaskType.BinaryClassification:
                // Ensure these important classification metrics are included
                metrics.Add(MetricType.Accuracy);
                metrics.Add(MetricType.Precision);
                metrics.Add(MetricType.Recall);
                metrics.Add(MetricType.F1Score);
                metrics.Add(MetricType.AUCROC);
                metrics.Add(MetricType.AUCPR);
                break;

            case NeuralNetworkTaskType.MultiClassClassification:
            case NeuralNetworkTaskType.MultiLabelClassification:
                // Ensure these important classification metrics are included
                metrics.Add(MetricType.Accuracy);
                metrics.Add(MetricType.Precision);
                metrics.Add(MetricType.Recall);
                metrics.Add(MetricType.F1Score);
                metrics.Add(MetricType.CrossEntropyLoss);
                break;

            case NeuralNetworkTaskType.Regression:
                // Ensure these important regression metrics are included
                metrics.Add(MetricType.MSE);
                metrics.Add(MetricType.RMSE);
                metrics.Add(MetricType.MAE);
                metrics.Add(MetricType.R2);
                break;

            case NeuralNetworkTaskType.TimeSeriesForecasting:
                metrics.Add(MetricType.MSE);
                metrics.Add(MetricType.RMSE);
                metrics.Add(MetricType.MAE);
                metrics.Add(MetricType.MAPE);
                metrics.Add(MetricType.AutoCorrelationFunction);
                metrics.Add(MetricType.DynamicTimeWarping);
                break;

            case NeuralNetworkTaskType.AnomalyDetection:
                metrics.Add(MetricType.Precision);
                metrics.Add(MetricType.Recall);
                metrics.Add(MetricType.F1Score);
                metrics.Add(MetricType.AUCROC);
                break;

            case NeuralNetworkTaskType.Clustering:
                metrics.Add(MetricType.SilhouetteScore);
                metrics.Add(MetricType.DaviesBouldinIndex);
                metrics.Add(MetricType.CalinskiHarabaszIndex);
                metrics.Add(MetricType.VariationOfInformation);
                break;

            case NeuralNetworkTaskType.ImageClassification:
            case NeuralNetworkTaskType.ObjectDetection:
            case NeuralNetworkTaskType.ImageSegmentation:
                metrics.Add(MetricType.Accuracy);
                metrics.Add(MetricType.Precision);
                metrics.Add(MetricType.Recall);
                metrics.Add(MetricType.F1Score);
                metrics.Add(MetricType.CrossEntropyLoss);
                break;
        }

        // Add LogLikelihood for all neural network tasks as it's generally applicable
        metrics.Add(MetricType.LogLikelihood);
    }

    /// <summary>
    /// Checks if a specific metric is valid for a neural network task type.
    /// </summary>
    /// <param name="metricType">The metric type to check.</param>
    /// <param name="taskType">The neural network task type.</param>
    /// <returns>True if the metric is valid for the specified task type; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method checks whether a specific metric makes sense for a
    /// particular neural network task. For example, it will tell you that accuracy is valid for
    /// classification but not for regression.
    /// </para>
    /// </remarks>
    public static bool IsValidMetricForTask(MetricType metricType, NeuralNetworkTaskType taskType)
    {
        var validMetrics = GetValidMetricTypes(taskType);
        return validMetrics.Contains(metricType);
    }

    /// <summary>
    /// Gets the metric groups appropriate for a neural network task type.
    /// </summary>
    /// <param name="taskType">The neural network task type.</param>
    /// <returns>An array of metric groups appropriate for the specified task.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method tells you which categories of metrics are suitable
    /// for different neural network tasks. For example, regression metrics for neural networks
    /// doing regression, classification metrics for those doing classification, etc.
    /// </para>
    /// </remarks>
    public static MetricGroups[] GetMetricGroupsForTask(NeuralNetworkTaskType taskType)
    {
        var correspondingModelType = MapTaskTypeToModelType(taskType);
        var baseGroups = ModelTypeHelper.GetSupportedMetricGroups(correspondingModelType).ToList();

        // Always include neural network metric group 
        if (!baseGroups.Contains(MetricGroups.NeuralNetwork))
        {
            baseGroups.Add(MetricGroups.NeuralNetwork);
        }

        // Always include general metrics
        if (!baseGroups.Contains(MetricGroups.General))
        {
            baseGroups.Add(MetricGroups.General);
        }

        return baseGroups.ToArray();
    }

    /// <summary>
    /// Gets the recommended top metrics for evaluating a neural network performing a specific task.
    /// </summary>
    /// <param name="taskType">The neural network task type.</param>
    /// <returns>An array of the most important metrics for evaluating the specified task.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method gives you a focused list of the most important metrics for
    /// evaluating your neural network for a specific task. These are the metrics you should pay
    /// the most attention to when assessing model performance.
    /// </para>
    /// </remarks>
    public static MetricType[] GetPrimaryMetrics(NeuralNetworkTaskType taskType)
    {
        return taskType switch
        {
            NeuralNetworkTaskType.Regression => new[]
            {
                MetricType.RMSE,
                MetricType.MAE,
                MetricType.R2,
                MetricType.MSE
            },

            NeuralNetworkTaskType.BinaryClassification => new[]
            {
                MetricType.AUCROC,
                MetricType.F1Score,
                MetricType.Accuracy,
                MetricType.Precision,
                MetricType.Recall
            },

            NeuralNetworkTaskType.MultiClassClassification => new[]
            {
                MetricType.Accuracy,
                MetricType.F1Score,
                MetricType.Precision,
                MetricType.Recall,
                MetricType.CrossEntropyLoss
            },

            NeuralNetworkTaskType.MultiLabelClassification => new[]
            {
                MetricType.F1Score,
                MetricType.Precision,
                MetricType.Recall,
                MetricType.AveragePrecision,
                MetricType.CrossEntropyLoss
            },

            NeuralNetworkTaskType.Clustering => new[]
            {
                MetricType.SilhouetteScore,
                MetricType.DaviesBouldinIndex,
                MetricType.CalinskiHarabaszIndex
            },

            NeuralNetworkTaskType.NaturalLanguageProcessing => new[]
            {
                MetricType.Perplexity,
                MetricType.CrossEntropyLoss,
                MetricType.LogLikelihood
            },

            NeuralNetworkTaskType.TextGeneration => new[]
            {
                MetricType.Perplexity,
                MetricType.CrossEntropyLoss,
                MetricType.LogLikelihood
            },

            NeuralNetworkTaskType.ImageClassification => new[]
            {
                MetricType.Accuracy,
                MetricType.F1Score,
                MetricType.CrossEntropyLoss
            },

            NeuralNetworkTaskType.ObjectDetection => new[]
            {
                MetricType.Precision,
                MetricType.Recall,
                MetricType.F1Score,
                MetricType.MeanAveragePrecision
            },

            NeuralNetworkTaskType.TimeSeriesForecasting => new[]
            {
                MetricType.RMSE,
                MetricType.MAE,
                MetricType.MAPE,
                MetricType.DynamicTimeWarping
            },

            NeuralNetworkTaskType.AnomalyDetection => new[]
            {
                MetricType.Precision,
                MetricType.Recall,
                MetricType.F1Score,
                MetricType.AUCROC
            },

            NeuralNetworkTaskType.Generative => new[]
            {
                MetricType.KLDivergence,
                MetricType.LogLikelihood
            },

            NeuralNetworkTaskType.ReinforcementLearning => new[]
            {
                MetricType.LogLikelihood
            },

            NeuralNetworkTaskType.Translation => new[]
            {
                MetricType.Perplexity,
                MetricType.LogLikelihood,
                MetricType.LevenshteinDistance
            },

            NeuralNetworkTaskType.SpeechRecognition => new[]
            {
                MetricType.LevenshteinDistance,
                MetricType.LogLikelihood
            },

            _ => Array.Empty<MetricType>()
        };
    }

    /// <summary>
    /// Gets a descriptive name for a neural network task type.
    /// </summary>
    /// <param name="taskType">The neural network task type.</param>
    /// <returns>A user-friendly name for the task type.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides a more readable, descriptive name for
    /// each neural network task type, which can be useful in user interfaces or reports.
    /// </para>
    /// </remarks>
    public static string GetTaskName(NeuralNetworkTaskType taskType)
    {
        return taskType switch
        {
            NeuralNetworkTaskType.Regression => "Neural Network Regression",
            NeuralNetworkTaskType.BinaryClassification => "Neural Network Binary Classification",
            NeuralNetworkTaskType.MultiClassClassification => "Neural Network Multiclass Classification",
            NeuralNetworkTaskType.MultiLabelClassification => "Neural Network Multilabel Classification",
            NeuralNetworkTaskType.Clustering => "Neural Network Clustering",
            NeuralNetworkTaskType.NaturalLanguageProcessing => "Natural Language Processing",
            NeuralNetworkTaskType.TextGeneration => "Text Generation Neural Network",
            NeuralNetworkTaskType.ImageClassification => "Image Classification Neural Network",
            NeuralNetworkTaskType.ObjectDetection => "Object Detection Neural Network",
            NeuralNetworkTaskType.ImageSegmentation => "Image Segmentation Neural Network",
            NeuralNetworkTaskType.Generative => "Generative Neural Network",
            NeuralNetworkTaskType.ReinforcementLearning => "Reinforcement Learning Neural Network",
            NeuralNetworkTaskType.TimeSeriesForecasting => "Time Series Forecasting Neural Network",
            NeuralNetworkTaskType.SequenceToSequence => "Sequence-to-Sequence Neural Network",
            NeuralNetworkTaskType.SequenceClassification => "Sequence Classification Neural Network",
            NeuralNetworkTaskType.AnomalyDetection => "Anomaly Detection Neural Network",
            NeuralNetworkTaskType.Recommendation => "Recommendation System Neural Network",
            NeuralNetworkTaskType.DimensionalityReduction => "Dimensionality Reduction Neural Network",
            NeuralNetworkTaskType.SpeechRecognition => "Speech Recognition Neural Network",
            NeuralNetworkTaskType.AudioProcessing => "Audio Processing Neural Network",
            NeuralNetworkTaskType.Translation => "Neural Machine Translation",
            NeuralNetworkTaskType.Custom => "Custom Neural Network",
            _ => "Neural Network"
        };
    }
}