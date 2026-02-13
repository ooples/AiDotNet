
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels;

/// <summary>
/// Fine-tuner for sentence transformer embedding models on domain-specific training data using triplet loss.
/// </summary>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
/// <remarks>
/// <para>
/// Fine-tuning adapts pre-trained embedding models to perform better on specific domains or tasks by
/// training on custom (anchor, positive, negative) triplets. This improves embedding quality for
/// specialized use cases like legal documents, medical terminology, or company-specific content.
/// </para>
/// <para><b>For Beginners:</b> Think of fine-tuning like teaching a translator specialized vocabulary.
/// 
/// Pre-trained model (general knowledge):
/// - "bank" → embedding that works for both "river bank" and "financial bank"
/// - Problem: Not precise for your specific domain!
/// 
/// Fine-tuned model (specialized):
/// - If you're building a financial app, train it with examples:
///   - Anchor: "bank account"
///   - Positive: "savings account" (similar in YOUR domain)
///   - Negative: "river bank" (different in YOUR domain)
/// - Result: Model learns "bank" means "financial institution" in your context
/// 
/// Real-world example:
/// Medical domain:
/// - General model: "cold" could mean temperature or illness
/// - After fine-tuning with medical data:
///   - Anchor: "patient has cold"
///   - Positive: "patient has flu" (similar symptoms)
///   - Negative: "cold weather" (unrelated)
/// - Model now correctly groups medical conditions together
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// // Initialize fine-tuner
/// var fineTuner = new SentenceTransformersFineTuner&lt;double&gt;(
///     baseModelPath: "models/all-MiniLM-L6-v2.onnx",
///     outputModelPath: "models/my-domain-model.onnx",
///     epochs: 10,
///     learningRate: 2e-5,
///     dimension: 384
/// );
/// 
/// // Prepare training data (anchor, positive, negative)
/// var trainingData = new List&lt;(string, string, string)&gt;
/// {
///     ("fraud detection", "fraudulent transaction", "legitimate payment"),
///     ("credit card", "debit card", "business card"),
///     ("interest rate", "APR", "laptop battery"),
///     // ... more examples
/// };
/// 
/// // Fine-tune the model
/// fineTuner.FineTune(trainingData);
/// 
/// // Use fine-tuned model
/// var embedding = fineTuner.Embed("detect fraudulent activity");
/// // Now produces embeddings optimized for your financial domain!
/// </code>
/// </para>
/// <para><b>How It Works:</b>
/// Training process:
/// 
/// 1. Triplet Loss Function:
///    - Anchor embedding (A): Embed("fraud detection")
///    - Positive embedding (P): Embed("fraudulent transaction") - should be similar
///    - Negative embedding (N): Embed("legitimate payment") - should be different
///    - Loss = max(0, distance(A,P) - distance(A,N) + margin)
///    - Goal: Make distance(A,P) small and distance(A,N) large
/// 
/// 2. Training Loop:
///    - For each epoch (10 iterations):
///      * For each training triplet:
///        - Generate embeddings for anchor, positive, negative
///        - Calculate triplet loss
///        - Update model weights to minimize loss
///        - Cache updated embeddings
/// 
/// 3. Result:
///    - Model learns to embed domain-specific texts closer together
///    - Generic texts pushed further apart
///    - Improved retrieval accuracy for your specific use case
/// 
/// Current implementation simulates training with embedding caching.
/// Real training requires gradient descent and backpropagation through the neural network.
/// </para>
/// <para><b>Benefits:</b>
/// - Domain adaptation - Customize embeddings for specific industry/task
/// - Improved accuracy - Better retrieval performance on your data
/// - Less training data - Fine-tuning needs 100-10,000 examples vs millions for pre-training
/// - Transfer learning - Leverages existing knowledge from pre-trained model
/// - Cost-effective - Faster and cheaper than training from scratch
/// </para>
/// <para><b>Limitations:</b>
/// - Requires quality training data (good triplets are crucial)
/// - Can overfit with too few examples (aim for 1,000+ triplets minimum)
/// - Needs domain expertise to create meaningful triplets
/// - Current implementation simulates training (real training requires ML framework)
/// - Training time increases with model size and dataset size
/// </para>
/// </remarks>
public class SentenceTransformersFineTuner<T> : EmbeddingModelBase<T>
{
    private readonly string _baseModelPath;
    private readonly string _outputModelPath;
    private readonly int _epochs;
    private readonly double _learningRate;
    private readonly int _dimension;

    private ONNXSentenceTransformer<T> _baseModel;
    private bool _isFineTuned;
    private Dictionary<string, Vector<T>> _fineTunedEmbeddingsCache;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="SentenceTransformersFineTuner{T}"/> class.
    /// </summary>
    /// <param name="baseModelPath">Path to the base model to fine-tune.</param>
    /// <param name="outputModelPath">Path where fine-tuned model will be saved.</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <param name="learningRate">Learning rate for fine-tuning.</param>
    /// <param name="dimension">The embedding dimension.</param>
    public SentenceTransformersFineTuner(
        string baseModelPath,
        string outputModelPath,
        int epochs,
        T learningRate,
        int dimension)
    {
        Guard.NotNull(baseModelPath);
        _baseModelPath = baseModelPath;
        Guard.NotNull(outputModelPath);
        _outputModelPath = outputModelPath;

        if (epochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(epochs), "Epochs must be positive");

        _epochs = epochs;
        _learningRate = Convert.ToDouble(learningRate);
        _dimension = dimension;

        _baseModel = new ONNXSentenceTransformer<T>(_baseModelPath, _dimension, MaxTokens);
        _isFineTuned = false;
        _fineTunedEmbeddingsCache = new Dictionary<string, Vector<T>>();
    }

    /// <inheritdoc />
    public override int EmbeddingDimension => _dimension;

    /// <inheritdoc />
    public override int MaxTokens => 512;

    /// <summary>
    /// Fine-tunes the model on provided training data.
    /// </summary>
    /// <param name="trainingPairs">Training pairs of (anchor, positive, negative) texts.</param>
    public void FineTune(IEnumerable<(string anchor, string positive, string negative)> trainingPairs)
    {
        if (trainingPairs == null)
            throw new ArgumentNullException(nameof(trainingPairs));

        var pairs = trainingPairs.ToList();
        if (pairs.Count == 0)
        {
            throw new ArgumentException("Training pairs cannot be empty", nameof(trainingPairs));
        }

        // Simulate fine-tuning process
        // In production, this would use actual neural network training with:
        // 1. Triplet loss or contrastive loss
        // 2. Backpropagation through sentence transformer layers
        // 3. Optimizer (Adam/SGD) from src/Optimizers/
        // 4. Save updated model weights to ONNX format

        Console.WriteLine($"Fine-tuning model on {pairs.Count} training pairs for {_epochs} epochs...");

        // For this implementation, we'll create an adjustment layer
        // that modifies base embeddings based on training data patterns
        var adjustmentVectors = new Dictionary<string, Vector<T>>();

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            foreach (var (anchor, positive, negative) in pairs)
            {
                // Get base embeddings
                var anchorEmb = _baseModel.Embed(anchor);
                var positiveEmb = _baseModel.Embed(positive);
                var negativeEmb = _baseModel.Embed(negative);

                // Calculate triplet loss components
                var posDistance = CalculateDistance(anchorEmb, positiveEmb);
                var negDistance = CalculateDistance(anchorEmb, negativeEmb);

                // If positive is not closer than negative, apply adjustment
                if (posDistance >= negDistance)
                {
                    // Create adjustment to move anchor closer to positive
                    var adjustment = CreateAdjustmentVector(anchorEmb, positiveEmb, _learningRate);

                    // Store adjustment (simplified - in production would update model weights)
                    if (!adjustmentVectors.ContainsKey(anchor))
                    {
                        adjustmentVectors[anchor] = adjustment;
                    }
                    else
                    {
                        // Average with existing adjustment
                        adjustmentVectors[anchor] = AverageVectors(adjustmentVectors[anchor], adjustment);
                    }
                }
            }
        }

        // Apply adjustments to cache
        foreach (var kvp in adjustmentVectors)
        {
            var baseEmb = _baseModel.Embed(kvp.Key);
            _fineTunedEmbeddingsCache[kvp.Key] = ApplyAdjustment(baseEmb, kvp.Value);
        }

        _isFineTuned = true;
        Console.WriteLine($"Fine-tuning complete. Adjusted {adjustmentVectors.Count} embeddings.");
    }

    /// <inheritdoc />
    protected override Vector<T> EmbedCore(string text)
    {
        // Check if we have a fine-tuned version
        if (_isFineTuned && _fineTunedEmbeddingsCache.ContainsKey(text))
        {
            return _fineTunedEmbeddingsCache[text];
        }

        // Fall back to base model
        return _baseModel.Embed(text);
    }

    private double CalculateDistance(Vector<T> v1, Vector<T> v2)
    {
        // Euclidean distance
        var sum = 0.0;
        for (int i = 0; i < v1.Length; i++)
        {
            var diff = Convert.ToDouble(NumOps.Subtract(v1[i], v2[i]));
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    private Vector<T> CreateAdjustmentVector(Vector<T> anchor, Vector<T> positive, double learningRate)
    {
        var values = new T[anchor.Length];
        for (int i = 0; i < anchor.Length; i++)
        {
            var diff = NumOps.Subtract(positive[i], anchor[i]);
            var adjustment = NumOps.Multiply(diff, NumOps.FromDouble(learningRate));
            values[i] = adjustment;
        }
        return new Vector<T>(values);
    }

    private Vector<T> AverageVectors(Vector<T> v1, Vector<T> v2)
    {
        var values = new T[v1.Length];
        for (int i = 0; i < v1.Length; i++)
        {
            var sum = NumOps.Add(v1[i], v2[i]);
            values[i] = NumOps.Divide(sum, NumOps.FromDouble(2.0));
        }
        return new Vector<T>(values);
    }

    private Vector<T> ApplyAdjustment(Vector<T> baseEmb, Vector<T> adjustment)
    {
        var values = new T[baseEmb.Length];
        for (int i = 0; i < baseEmb.Length; i++)
        {
            values[i] = NumOps.Add(baseEmb[i], adjustment[i]);
        }
        return new Vector<T>(values).Normalize();
    }

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _baseModel?.Dispose();
            }

            _disposed = true;
        }

        base.Dispose(disposing);
    }
}


