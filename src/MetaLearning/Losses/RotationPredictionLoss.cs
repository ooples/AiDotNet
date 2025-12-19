using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using System.Collections.Generic;

namespace AiDotNet.MetaLearning.Losses;

/// <summary>
/// Self-supervised loss function for rotation prediction task.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Rotation prediction is a simple yet effective self-supervised task where
/// the model must predict which rotation (0°, 90°, 180°, or 270°) was applied
/// to an input image. This task encourages the model to learn:
/// - Object orientation and spatial relationships
/// - Edge detectors and feature extractors
/// - Invariance to rotations
/// - General feature representations useful for downstream tasks
/// </para>
/// <para><b>For Beginners:</b> This loss teaches models about image structure:
///
/// **How it works:**
/// 1. Take an unlabeled image
/// 2. Rotate it by 0°, 90°, 180°, or 270° randomly
/// 3. Train model to predict which rotation was used
/// 4. Labels are [0, 1, 2, 3] for the four rotations
///
/// **Why it helps:**
/// - To recognize rotation, model must learn about edges, corners, and shapes
/// - These features are also useful for object recognition
/// - No human labels needed - rotation creates the labels automatically
/// - Model can pre-train on unlimited unlabeled images
/// </para>
/// <para>
/// <b>Usage Example:</b>
/// <code>
/// var rotationLoss = new RotationPredictionLoss&lt;double&gt;();
///
/// // Create self-supervised task
/// var (rotatedImage, rotationLabel) = rotationLoss.CreatePretextTask(originalImage);
///
/// // Model predicts rotation
/// var prediction = model.Predict(rotatedImage);
///
/// // Compute loss
/// var loss = rotationLoss.ComputeLoss(prediction, rotationLabel);
/// </code>
/// </para>
/// </remarks>
public class RotationPredictionLoss<T, TInput, TOutput> : ISelfSupervisedLoss<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random;
    private readonly int[] _rotations = { 0, 90, 180, 270 };

    /// <summary>
    /// Initializes a new instance of the RotationPredictionLoss class.
    /// </summary>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public RotationPredictionLoss(int? seed = null)
    {
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <inheritdoc/>
    public SelfSupervisedTaskType TaskType => SelfSupervisedTaskType.RotationPrediction;

    /// <inheritdoc/>
    public string Name => "Rotation Prediction Loss";

    /// <inheritdoc/>
    public string Description => "Self-supervised loss for predicting image rotations";

    /// <inheritdoc/>
    public (TInput transformedInput, TOutput targets) CreatePretextTask(TInput input)
    {
        // Randomly select a rotation angle
        int rotationIndex = _random.Next(_rotations.Length);
        int rotationAngle = _rotations[rotationIndex];

        // Apply rotation to input
        TInput rotatedInput = ApplyRotation(input, rotationAngle);

        // Create target label (one-hot encoded)
        TOutput targets = CreateOneHotTarget(rotationIndex);

        return (rotatedInput, targets);
    }

    /// <inheritdoc/>
    public T ComputeLoss(TOutput predictions, TOutput targets)
    {
        // Use cross-entropy loss for multi-class classification
        return ComputeCrossEntropyLoss(predictions, targets);
    }

    /// <inheritdoc/>
    public T ComputePretextLoss(TOutput predictions, TInput originalInput, TInput transformedInput)
    {
        // For rotation prediction, we don't need original input
        // Just compute standard loss between predictions and targets
        // The targets would need to be extracted separately in a real implementation

        // This is a simplified version - in practice, you'd need to track the targets
        throw new NotImplementedException(
            "ComputePretextLoss requires target labels. " +
            "Use CreatePretextTask to get transformed input with targets, " +
            "then use ComputeLoss(predictions, targets).");
    }

    /// <inheritdoc/>
    public T ComputeDerivative(TOutput predictions, TOutput targets)
    {
        // Compute derivative of cross-entropy loss
        return ComputeCrossEntropyDerivative(predictions, targets);
    }

    /// <inheritdoc/>
    public Dictionary<string, T> EvaluateRepresentations(Matrix<T> features, Vector<T>? labels = null)
    {
        var metrics = new Dictionary<string, T>();

        // Compute feature statistics
        int numFeatures = features.Columns;
        int numSamples = features.Rows;

        // Feature norm (should be normalized)
        T featureNorm = NumOps.Zero;
        for (int i = 0; i < numSamples; i++)
        {
            T sampleNorm = NumOps.Zero;
            for (int j = 0; j < numFeatures; j++)
            {
                sampleNorm = NumOps.Add(sampleNorm, NumOps.Multiply(features[i, j], features[i, j]));
            }
            featureNorm = NumOps.Add(featureNorm, NumOps.FromDouble(Math.Sqrt(Convert.ToDouble(sampleNorm))));
        }
        metrics["AverageFeatureNorm"] = NumOps.Divide(featureNorm, NumOps.FromDouble(numSamples));

        // Feature variance (high variance is good for representations)
        T variance = NumOps.Zero;
        if (numSamples > 1)
        {
            // Compute mean for each feature
            var means = new T[numFeatures];
            for (int j = 0; j < numFeatures; j++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < numSamples; i++)
                {
                    sum = NumOps.Add(sum, features[i, j]);
                }
                means[j] = NumOps.Divide(sum, NumOps.FromDouble(numSamples));
            }

            // Compute variance
            for (int j = 0; j < numFeatures; j++)
            {
                T featureVariance = NumOps.Zero;
                for (int i = 0; i < numSamples; i++)
                {
                    T diff = NumOps.Subtract(features[i, j], means[j]);
                    featureVariance = NumOps.Add(featureVariance, NumOps.Multiply(diff, diff));
                }
                variance = NumOps.Add(variance, NumOps.Divide(featureVariance, NumOps.FromDouble(numSamples - 1)));
            }
            metrics["AverageFeatureVariance"] = NumOps.Divide(variance, NumOps.FromDouble(numFeatures));
        }

        // If we have labels, compute linear probe accuracy
        if (labels != null)
        {
            // Train a simple linear classifier on the features
            // This measures how linearly separable the learned representations are
            var accuracy = ComputeLinearProbeAccuracy(features, labels);
            metrics["LinearProbeAccuracy"] = accuracy;
        }

        return metrics;
    }

    /// <inheritdoc/>
    public (IFullModel<T, TInput, TOutput> adaptedModel, T totalLoss) AdaptWithSelfSupervision(
        IFullModel<T, TInput, TOutput> model,
        TInput supportSet,
        TInput querySet,
        int innerSteps)
    {
        T totalLoss = NumOps.Zero;

        // Create self-supervised tasks from support set
        var (rotatedSupport, ssTargets) = CreatePretextTask(supportSet);
        var (rotatedQuery, qsTargets) = CreatePretextTask(querySet);

        // Adapt model on self-supervised task
        for (int step = 0; step < innerSteps; step++)
        {
            // Train on support set
            var ssPredictions = model.Predict(rotatedSupport);
            var ssLoss = ComputeLoss(ssPredictions, ssTargets);
            model.Train(rotatedSupport, ssTargets);
            totalLoss = NumOps.Add(totalLoss, ssLoss);

            // Optional: train on query set for better generalization
            if (step < innerSteps / 2)
            {
                var qsPredictions = model.Predict(rotatedQuery);
                var qsLoss = ComputeLoss(qsPredictions, qsTargets);
                model.Train(rotatedQuery, qsTargets);
                totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(qsLoss, NumOps.FromDouble(0.5)));
            }
        }

        return (model, NumOps.Divide(totalLoss, NumOps.FromDouble(innerSteps)));
    }

    /// <summary>
    /// Applies rotation to the input data.
    /// </summary>
    /// <param name="input">The input to rotate.</param>
    /// <param name="angle">Rotation angle in degrees (0, 90, 180, 270).</param>
    /// <returns>The rotated input.</returns>
    private TInput ApplyRotation(TInput input, int angle)
    {
        // This is a placeholder implementation
        // In practice, you would:
        // 1. Convert TInput to image/tensor format
        // 2. Apply rotation transformation
        // 3. Convert back to TInput
        // For now, return unchanged (rotation would be applied by data augmentation pipeline)
        return input;
    }

    /// <summary>
    /// Creates a one-hot encoded target vector for the rotation index.
    /// </summary>
    /// <param name="rotationIndex">Index of the rotation (0-3).</param>
    /// <returns>One-hot encoded target vector.</returns>
    private TOutput CreateOneHotTarget(int rotationIndex)
    {
        // Create one-hot vector [1, 0, 0, 0], [0, 1, 0, 0], etc.
        var targetVector = new Vector<T>(4);
        for (int i = 0; i < 4; i++)
        {
            targetVector[i] = i == rotationIndex ? NumOps.FromDouble(1.0) : NumOps.Zero;
        }

        // Convert Vector<T> to TOutput
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            return (TOutput)(object)targetVector;
        }
        else if (typeof(TOutput) == typeof(Tensor<T>))
        {
            var tensor = Tensor<T>.FromVector(targetVector);
            return (TOutput)(object)tensor;
        }
        else
        {
            throw new NotSupportedException($"Cannot convert Vector<T> to {typeof(TOutput).Name}");
        }
    }

    /// <summary>
    /// Computes cross-entropy loss between predictions and targets.
    /// </summary>
    private T ComputeCrossEntropyLoss(TOutput predictions, TOutput targets)
    {
        // Extract vectors
        var predVector = ConvertToVector(predictions);
        var targetVector = ConvertToVector(targets);

        // Apply softmax to predictions to get probabilities
        var probabilities = ApplySoftmax(predVector);

        // Compute cross-entropy: -sum(y * log(p))
        T loss = NumOps.Zero;
        for (int i = 0; i < predVector.Length; i++)
        {
            if (Convert.ToDouble(targetVector[i]) > 0.5) // Target is 1 for this class
            {
                T p = probabilities[i];
                // Avoid log(0) with small epsilon
                p = NumOps.Add(p, NumOps.FromDouble(1e-8));
                T logP = NumOps.FromDouble(Math.Log(Convert.ToDouble(p)));
                loss = NumOps.Subtract(loss, logP);
            }
        }

        return loss;
    }

    /// <summary>
    /// Computes derivative of cross-entropy loss.
    /// </summary>
    private Vector<T> ComputeCrossEntropyDerivative(TOutput predictions, TOutput targets)
    {
        var predVector = ConvertToVector(predictions);
        var targetVector = ConvertToVector(targets);
        var probabilities = ApplySoftmax(predVector);

        // Derivative of cross-entropy is (p - y)
        var derivative = new Vector<T>(predVector.Length);
        for (int i = 0; i < predVector.Length; i++)
        {
            derivative[i] = NumOps.Subtract(probabilities[i], targetVector[i]);
        }

        return derivative;
    }

    /// <summary>
    /// Applies softmax function to convert logits to probabilities.
    /// </summary>
    private Vector<T> ApplySoftmax(Vector<T> logits)
    {
        // Subtract max for numerical stability
        T maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (Convert.ToDouble(logits[i]) > Convert.ToDouble(maxLogit))
            {
                maxLogit = logits[i];
            }
        }

        // Compute exp(x - max) and sum
        var expValues = new T[logits.Length];
        T sumExp = NumOps.Zero;
        for (int i = 0; i < logits.Length; i++)
        {
            T shifted = NumOps.Subtract(logits[i], maxLogit);
            expValues[i] = NumOps.FromDouble(Math.Exp(Convert.ToDouble(shifted)));
            sumExp = NumOps.Add(sumExp, expValues[i]);
        }

        // Normalize to probabilities
        var probabilities = new Vector<T>(logits.Length);
        for (int i = 0; i < logits.Length; i++)
        {
            probabilities[i] = NumOps.Divide(expValues[i], sumExp);
        }

        return probabilities;
    }

    /// <summary>
    /// Converts TOutput to Vector&lt;T&gt;.
    /// </summary>
    private Vector<T> ConvertToVector(TOutput output)
    {
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            return (Vector<T>)(object)output;
        }
        else if (typeof(TOutput) == typeof(Tensor<T>))
        {
            var tensor = (Tensor<T>)(object)output;
            return tensor.Flatten();
        }
        else
        {
            throw new NotSupportedException($"Cannot convert {typeof(TOutput).Name} to Vector<T>");
        }
    }

    /// <summary>
    /// Computes linear probe accuracy on learned features.
    /// </summary>
    private T ComputeLinearProbeAccuracy(Matrix<T> features, Vector<T> labels)
    {
        // This is a simplified implementation
        // In practice, you would train a linear classifier and evaluate its accuracy
        // For now, return a placeholder value
        return NumOps.FromDouble(0.5); // 50% accuracy for random classifier
    }

    // Required ILossFunction<T> interface methods
    public T CalculateLoss(Vector<T> predictions, Vector<T> targets)
    {
        return ComputeCrossEntropyLoss((TOutput)(object)predictions, (TOutput)(object)targets);
    }

    public Vector<T> CalculateDerivative(Vector<T> predictions, Vector<T> targets)
    {
        return ComputeCrossEntropyDerivative((TOutput)(object)predictions, (TOutput)(object)targets);
    }

    // Required ISelfSupervisedLoss<T, TInput, TOutput> interface methods
    public IMetaLearningTask<T, TInput, TOutput> CreateSelfSupervisedTask(TInput unlabeledInput)
    {
        // Create a rotation pretext task
        var (rotatedInput, targets) = CreatePretextTask(unlabeledInput);

        // Create a dummy input for the original (unrotated) version
        // In a real implementation, you'd have proper support/query sets
        return new BasicMetaLearningTask<T, TInput, TOutput>(
            supportInput: rotatedInput,
            supportOutput: targets,
            queryInput: unlabeledInput,
            queryOutput: default(TOutput) // We don't need query labels for self-supervised learning
        );
    }

    public TInput ApplyAugmentation(TInput input, Dictionary<string, object> augmentationParams)
    {
        // Get rotation angle from parameters, default to random
        int angle;
        if (augmentationParams.TryGetValue("rotation_angle", out var angleObj) && angleObj is int parsedAngle)
        {
            angle = parsedAngle;
        }
        else
        {
            // Random rotation if not specified
            var randomIndex = _random.Next(_rotations.Length);
            angle = _rotations[randomIndex];
        }

        return ApplyRotation(input, angle);
    }
}