using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.NeuralNetworks.Metrics;

/// <summary>
/// Provides metrics for evaluating segmentation and classification tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// These metrics are essential for evaluating the quality of segmentation models
/// on 3D data like point cloud segmentation, mesh segmentation, and voxel classification.
/// </para>
/// <para><b>For Beginners:</b> When a model predicts labels for different parts of a 3D shape
/// (like "leg", "seat", "back" for a chair), these metrics tell us how accurate those
/// predictions are compared to the ground truth labels.
/// </para>
/// </remarks>
public static class SegmentationMetrics<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes Mean Intersection over Union (mIoU) for segmentation.
    /// </summary>
    /// <param name="predictions">The predicted class labels as integers.</param>
    /// <param name="groundTruth">The ground truth class labels as integers.</param>
    /// <param name="numClasses">The total number of classes.</param>
    /// <param name="ignoreIndex">Optional class index to ignore (e.g., for unlabeled regions). Default is -1 (none).</param>
    /// <returns>The mIoU value in range [0, 1]. Higher is better (1 = perfect).</returns>
    /// <remarks>
    /// <para>
    /// mIoU computes IoU (intersection/union) for each class and averages them. It's the
    /// standard metric for semantic segmentation tasks.
    /// </para>
    /// <para><b>For Beginners:</b> IoU measures how much the predicted region overlaps with
    /// the true region. mIoU averages this across all classes:
    /// - mIoU = 1.0: Perfect segmentation
    /// - mIoU &gt; 0.7: Good segmentation
    /// - mIoU 0.5-0.7: Acceptable segmentation
    /// - mIoU &lt; 0.5: Poor segmentation
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when arrays have different lengths.</exception>
    public static double MeanIoU(int[] predictions, int[] groundTruth, int numClasses, int ignoreIndex = -1)
    {
        if (predictions.Length != groundTruth.Length)
        {
            throw new ArgumentException(
                $"Predictions and ground truth must have the same length. Got {predictions.Length} and {groundTruth.Length}.");
        }

        var confusionMatrix = ComputeConfusionMatrix(predictions, groundTruth, numClasses, ignoreIndex);

        double sumIoU = 0.0;
        int validClasses = 0;

        for (int c = 0; c < numClasses; c++)
        {
            if (c == ignoreIndex) continue;

            int truePositives = confusionMatrix[c, c];
            int falsePositives = 0;
            int falseNegatives = 0;

            for (int i = 0; i < numClasses; i++)
            {
                if (i != c)
                {
                    falsePositives += confusionMatrix[i, c];
                    falseNegatives += confusionMatrix[c, i];
                }
            }

            int union = truePositives + falsePositives + falseNegatives;
            if (union > 0)
            {
                sumIoU += (double)truePositives / union;
                validClasses++;
            }
        }

        return validClasses > 0 ? sumIoU / validClasses : 0.0;
    }

    /// <summary>
    /// Computes per-class IoU for segmentation.
    /// </summary>
    /// <param name="predictions">The predicted class labels.</param>
    /// <param name="groundTruth">The ground truth class labels.</param>
    /// <param name="numClasses">The total number of classes.</param>
    /// <param name="ignoreIndex">Optional class index to ignore. Default is -1 (none).</param>
    /// <returns>Array of IoU values for each class.</returns>
    /// <exception cref="ArgumentException">Thrown when arrays have different lengths.</exception>
    public static double[] PerClassIoU(int[] predictions, int[] groundTruth, int numClasses, int ignoreIndex = -1)
    {
        if (predictions.Length != groundTruth.Length)
        {
            throw new ArgumentException(
                $"Predictions and ground truth must have the same length. Got {predictions.Length} and {groundTruth.Length}.");
        }

        var confusionMatrix = ComputeConfusionMatrix(predictions, groundTruth, numClasses, ignoreIndex);
        var iouValues = new double[numClasses];

        for (int c = 0; c < numClasses; c++)
        {
            if (c == ignoreIndex)
            {
                iouValues[c] = double.NaN;
                continue;
            }

            int truePositives = confusionMatrix[c, c];
            int falsePositives = 0;
            int falseNegatives = 0;

            for (int i = 0; i < numClasses; i++)
            {
                if (i != c)
                {
                    falsePositives += confusionMatrix[i, c];
                    falseNegatives += confusionMatrix[c, i];
                }
            }

            int union = truePositives + falsePositives + falseNegatives;
            iouValues[c] = union > 0 ? (double)truePositives / union : 0.0;
        }

        return iouValues;
    }

    /// <summary>
    /// Computes overall accuracy for classification/segmentation.
    /// </summary>
    /// <param name="predictions">The predicted class labels.</param>
    /// <param name="groundTruth">The ground truth class labels.</param>
    /// <param name="ignoreIndex">Optional class index to ignore. Default is -1 (none).</param>
    /// <returns>The accuracy value in range [0, 1]. Higher is better.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Accuracy is simply the percentage of correct predictions.
    /// If 80 out of 100 points are labeled correctly, accuracy = 0.8.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when arrays have different lengths.</exception>
    public static double Accuracy(int[] predictions, int[] groundTruth, int ignoreIndex = -1)
    {
        if (predictions.Length != groundTruth.Length)
        {
            throw new ArgumentException(
                $"Predictions and ground truth must have the same length. Got {predictions.Length} and {groundTruth.Length}.");
        }

        int correct = 0;
        int total = 0;

        for (int i = 0; i < predictions.Length; i++)
        {
            if (groundTruth[i] == ignoreIndex) continue;

            if (predictions[i] == groundTruth[i])
            {
                correct++;
            }
            total++;
        }

        return total > 0 ? (double)correct / total : 0.0;
    }

    /// <summary>
    /// Computes per-class accuracy.
    /// </summary>
    /// <param name="predictions">The predicted class labels.</param>
    /// <param name="groundTruth">The ground truth class labels.</param>
    /// <param name="numClasses">The total number of classes.</param>
    /// <param name="ignoreIndex">Optional class index to ignore. Default is -1 (none).</param>
    /// <returns>Array of accuracy values for each class.</returns>
    /// <exception cref="ArgumentException">Thrown when arrays have different lengths.</exception>
    public static double[] PerClassAccuracy(int[] predictions, int[] groundTruth, int numClasses, int ignoreIndex = -1)
    {
        if (predictions.Length != groundTruth.Length)
        {
            throw new ArgumentException(
                $"Predictions and ground truth must have the same length. Got {predictions.Length} and {groundTruth.Length}.");
        }

        var confusionMatrix = ComputeConfusionMatrix(predictions, groundTruth, numClasses, ignoreIndex);
        var accuracies = new double[numClasses];

        for (int c = 0; c < numClasses; c++)
        {
            if (c == ignoreIndex)
            {
                accuracies[c] = double.NaN;
                continue;
            }

            int truePositives = confusionMatrix[c, c];
            int classTotal = 0;
            for (int i = 0; i < numClasses; i++)
            {
                classTotal += confusionMatrix[c, i];
            }

            accuracies[c] = classTotal > 0 ? (double)truePositives / classTotal : 0.0;
        }

        return accuracies;
    }

    /// <summary>
    /// Computes precision for each class.
    /// </summary>
    /// <param name="predictions">The predicted class labels.</param>
    /// <param name="groundTruth">The ground truth class labels.</param>
    /// <param name="numClasses">The total number of classes.</param>
    /// <param name="ignoreIndex">Optional class index to ignore. Default is -1 (none).</param>
    /// <returns>Array of precision values for each class.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Precision answers: "Of all the points I labeled as class X,
    /// how many are actually class X?" High precision = few false positives.
    /// </para>
    /// </remarks>
    public static double[] Precision(int[] predictions, int[] groundTruth, int numClasses, int ignoreIndex = -1)
    {
        if (predictions.Length != groundTruth.Length)
        {
            throw new ArgumentException(
                $"Predictions and ground truth must have the same length. Got {predictions.Length} and {groundTruth.Length}.");
        }

        var confusionMatrix = ComputeConfusionMatrix(predictions, groundTruth, numClasses, ignoreIndex);
        var precisions = new double[numClasses];

        for (int c = 0; c < numClasses; c++)
        {
            if (c == ignoreIndex)
            {
                precisions[c] = double.NaN;
                continue;
            }

            int truePositives = confusionMatrix[c, c];
            int predictedPositives = 0;
            for (int i = 0; i < numClasses; i++)
            {
                predictedPositives += confusionMatrix[i, c];
            }

            precisions[c] = predictedPositives > 0 ? (double)truePositives / predictedPositives : 0.0;
        }

        return precisions;
    }

    /// <summary>
    /// Computes recall for each class.
    /// </summary>
    /// <param name="predictions">The predicted class labels.</param>
    /// <param name="groundTruth">The ground truth class labels.</param>
    /// <param name="numClasses">The total number of classes.</param>
    /// <param name="ignoreIndex">Optional class index to ignore. Default is -1 (none).</param>
    /// <returns>Array of recall values for each class.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Recall answers: "Of all the actual class X points,
    /// how many did I correctly identify?" High recall = few false negatives.
    /// </para>
    /// </remarks>
    public static double[] Recall(int[] predictions, int[] groundTruth, int numClasses, int ignoreIndex = -1)
    {
        return PerClassAccuracy(predictions, groundTruth, numClasses, ignoreIndex);
    }

    /// <summary>
    /// Computes F1 score for each class.
    /// </summary>
    /// <param name="predictions">The predicted class labels.</param>
    /// <param name="groundTruth">The ground truth class labels.</param>
    /// <param name="numClasses">The total number of classes.</param>
    /// <param name="ignoreIndex">Optional class index to ignore. Default is -1 (none).</param>
    /// <returns>Array of F1 scores for each class.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> F1 is the harmonic mean of precision and recall.
    /// It balances both concerns: F1 = 2 * (precision * recall) / (precision + recall)
    /// </para>
    /// </remarks>
    public static double[] F1Score(int[] predictions, int[] groundTruth, int numClasses, int ignoreIndex = -1)
    {
        var precision = Precision(predictions, groundTruth, numClasses, ignoreIndex);
        var recall = Recall(predictions, groundTruth, numClasses, ignoreIndex);
        var f1Scores = new double[numClasses];

        for (int c = 0; c < numClasses; c++)
        {
            if (c == ignoreIndex || double.IsNaN(precision[c]) || double.IsNaN(recall[c]))
            {
                f1Scores[c] = double.NaN;
                continue;
            }

            double sum = precision[c] + recall[c];
            f1Scores[c] = sum > 0 ? 2 * precision[c] * recall[c] / sum : 0.0;
        }

        return f1Scores;
    }

    /// <summary>
    /// Computes the mean F1 score across all classes.
    /// </summary>
    /// <param name="predictions">The predicted class labels.</param>
    /// <param name="groundTruth">The ground truth class labels.</param>
    /// <param name="numClasses">The total number of classes.</param>
    /// <param name="ignoreIndex">Optional class index to ignore. Default is -1 (none).</param>
    /// <returns>The mean F1 score.</returns>
    public static double MeanF1Score(int[] predictions, int[] groundTruth, int numClasses, int ignoreIndex = -1)
    {
        var f1Scores = F1Score(predictions, groundTruth, numClasses, ignoreIndex);

        double sum = 0;
        int count = 0;

        for (int c = 0; c < numClasses; c++)
        {
            if (!double.IsNaN(f1Scores[c]))
            {
                sum += f1Scores[c];
                count++;
            }
        }

        return count > 0 ? sum / count : 0.0;
    }

    /// <summary>
    /// Computes the confusion matrix.
    /// </summary>
    /// <param name="predictions">The predicted class labels.</param>
    /// <param name="groundTruth">The ground truth class labels.</param>
    /// <param name="numClasses">The total number of classes.</param>
    /// <param name="ignoreIndex">Optional class index to ignore. Default is -1 (none).</param>
    /// <returns>
    /// A confusion matrix where [i, j] is the count of samples with true label i predicted as j.
    /// </returns>
    public static int[,] ConfusionMatrix(int[] predictions, int[] groundTruth, int numClasses, int ignoreIndex = -1)
    {
        if (predictions.Length != groundTruth.Length)
        {
            throw new ArgumentException(
                $"Predictions and ground truth must have the same length. Got {predictions.Length} and {groundTruth.Length}.");
        }

        return ComputeConfusionMatrix(predictions, groundTruth, numClasses, ignoreIndex);
    }

    #region Private Methods

    /// <summary>
    /// Computes the confusion matrix internally.
    /// </summary>
    private static int[,] ComputeConfusionMatrix(int[] predictions, int[] groundTruth, int numClasses, int ignoreIndex)
    {
        var matrix = new int[numClasses, numClasses];

        for (int i = 0; i < predictions.Length; i++)
        {
            int pred = predictions[i];
            int gt = groundTruth[i];

            if (gt == ignoreIndex || gt < 0 || gt >= numClasses) continue;
            if (pred < 0 || pred >= numClasses) continue;

            matrix[gt, pred]++;
        }

        return matrix;
    }

    #endregion
}
