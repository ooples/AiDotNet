using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Prototypical Networks (ProtoNets) algorithm for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Prototypical Networks learn a metric space where classification can be performed by computing
/// distances to prototype representations of each class. Each prototype is the mean vector of
/// the support set examples for that class.
/// </para>
/// <para><b>For Beginners:</b> ProtoNets learns to recognize new classes from just a few examples:
///
/// **How it works:**
/// 1. For each new class, create a "prototype" (average of all examples)
/// 2. To classify a new example, find which prototype is closest
/// 3. Distance is measured in a learned feature space
/// 4. Uses soft nearest neighbor with learnable distance metric
///
/// **Simple example:**
/// - Support set: 3 images each of 5 different animal species (15 images total)
/// - Create prototype for each species by averaging their features
/// - Query image: classify by finding nearest animal prototype
/// - Learning: train encoder to make same-species images cluster together
/// </para>
/// <para><b>Algorithm - Prototypical Networks:</b>
/// <code>
/// # Encoding phase (learnable)
/// feature_encoder = NeuralNetwork()  # Maps x -> embedding(x)
///
/// # Episode training
/// for each episode:
///     # Sample N-way K-shot task
///     support_set = {examples_from_N_classes, K_examples_each}
///     query_set = {examples_from_same_N_classes}
///
///     # Compute class prototypes (non-parametric)
///     for each class c:
///         prototype_c = mean(embedding(x) for x in support_examples_of_class_c)
///
///     # Classification by distance
///     for each query example x:
///         distances = [distance(embedding(x), prototype_c) for c in classes]
///         probabilities = softmax(-distances)
///         loss = cross_entropy(probabilities, true_label)
///
///     # Update encoder (no prototypes to store!)
///     backpropagate(loss)
///     update(feature_encoder.parameters)
/// </code>
/// </para>
/// <para><b>Key Insights:</b>
///
/// 1. **Non-parametric Classification**: No classifier parameters to learn,
///    just need a good feature encoder. Prototypes are computed on-the-fly.
///
/// 2. **Metric Learning**: The encoder learns to cluster same-class examples
///    and separate different classes in the feature space.
///
/// 3. **Efficient Adaptation**: To adapt to new classes, just compute new
///    prototypes - no gradient updates needed!
///
/// 4. **Interpretable**: Prototypes provide an intuitive representation of each
///    class as the "average example".
/// </para>
/// <para>
/// Reference: Snell, J., Swersky, K., &amp; Zemel, R. (2017).
/// Prototypical Networks for Few-shot Learning.
/// </para>
/// </remarks>
public class ProtoNetsAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly ProtoNetsOptions<T, TInput, TOutput> _protoNetsOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _protoNetsOptions;

    /// <summary>
    /// Attention weights for prototype enhancement (if enabled).
    /// </summary>
    private Matrix<T>? _attentionWeights;

    /// <summary>
    /// Class-specific scaling factors for adaptive distance computation.
    /// </summary>
    private Dictionary<int, T>? _classScalingFactors;

    /// <summary>
    /// Initializes a new instance of the ProtoNetsAlgorithm class.
    /// </summary>
    /// <param name="options">The configuration options for ProtoNets.</param>
    /// <exception cref="ArgumentNullException">Thrown when options or required components are null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a ProtoNets model ready for few-shot learning.
    ///
    /// <b>What ProtoNets needs:</b>
    /// - <b>MetaModel:</b> Neural network that maps inputs to features (e.g., CNN for images)
    /// - <b>DistanceFunction:</b> How to measure similarity (Euclidean, Cosine, etc.)
    ///
    /// <b>What happens during training:</b>
    /// 1. Sample episodes with N classes, K examples each
    /// 2. Compute prototypes by averaging features
    /// 3. Train encoder to make same-class features close
    /// 4. Test on query set from same classes
    ///
    /// <b>What happens during testing:</b>
    /// 1. Get K examples of each new class
    /// 2. Compute prototypes (no training needed!)
    /// 3. Classify new examples by nearest prototype
    /// </para>
    /// </remarks>
    public ProtoNetsAlgorithm(ProtoNetsOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            null) // ProtoNets doesn't use inner optimizer (non-parametric adaptation)
    {
        _protoNetsOptions = options;

        // Validate configuration
        if (!_protoNetsOptions.IsValid())
        {
            throw new ArgumentException("ProtoNets configuration is invalid. Check all parameters.", nameof(options));
        }

        // Initialize attention weights if using attention mechanism
        if (_protoNetsOptions.UseAttentionMechanism)
        {
            _attentionWeights = new Matrix<T>(0, 0);
        }

        // Initialize class-specific scaling factors if using adaptive scaling
        if (_protoNetsOptions.UseAdaptiveClassScaling)
        {
            _classScalingFactors = new Dictionary<int, T>();
        }
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.ProtoNets"/>.</value>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ProtoNets;

    /// <summary>
    /// Performs one meta-training step using ProtoNets' episodic training.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on.</param>
    /// <returns>The average meta-loss across all tasks in the batch.</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <remarks>
    /// <para>
    /// ProtoNets training is simpler than MAML because there's no inner loop gradient computation:
    /// </para>
    /// <para>
    /// <b>For each task in the batch:</b>
    /// 1. Encode support set examples to get feature embeddings
    /// 2. Compute class prototypes (mean of each class's embeddings)
    /// 3. Encode query set examples
    /// 4. Compute distances from query embeddings to prototypes
    /// 5. Apply softmax to get class probabilities
    /// 6. Compute cross-entropy loss
    /// </para>
    /// <para>
    /// <b>Meta-update:</b>
    /// Average losses across all tasks and backpropagate to update the feature encoder.
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        T totalLoss = NumOps.Zero;
        Vector<T>? accumulatedGradients = null;

        foreach (var task in taskBatch.Tasks)
        {
            // Compute episode loss and gradients
            var (episodeLoss, episodeGradients) = TrainEpisode(task);
            totalLoss = NumOps.Add(totalLoss, episodeLoss);

            // Accumulate gradients
            if (accumulatedGradients == null)
            {
                accumulatedGradients = episodeGradients;
            }
            else
            {
                for (int i = 0; i < accumulatedGradients.Length; i++)
                {
                    accumulatedGradients[i] = NumOps.Add(accumulatedGradients[i], episodeGradients[i]);
                }
            }
        }

        if (accumulatedGradients != null)
        {
            // Average gradients
            T batchSizeT = NumOps.FromDouble(taskBatch.BatchSize);
            for (int i = 0; i < accumulatedGradients.Length; i++)
            {
                accumulatedGradients[i] = NumOps.Divide(accumulatedGradients[i], batchSizeT);
            }

            // Apply gradient clipping if configured
            if (_protoNetsOptions.GradientClipThreshold.HasValue && _protoNetsOptions.GradientClipThreshold.Value > 0)
            {
                accumulatedGradients = ClipGradients(accumulatedGradients, _protoNetsOptions.GradientClipThreshold.Value);
            }

            // Update feature encoder parameters
            var currentParams = MetaModel.GetParameters();
            var updatedParams = ApplyGradients(currentParams, accumulatedGradients, _protoNetsOptions.OuterLearningRate);
            MetaModel.SetParameters(updatedParams);
        }

        // Return average loss
        return NumOps.Divide(totalLoss, NumOps.FromDouble(taskBatch.BatchSize));
    }

    /// <summary>
    /// Adapts to a new task by computing class prototypes from the support set.
    /// </summary>
    /// <param name="task">The new task containing support set examples.</param>
    /// <returns>A PrototypicalModel that classifies by nearest prototype.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para>
    /// This is where ProtoNets shines - adaptation is instantaneous! Unlike MAML which requires
    /// gradient descent steps, ProtoNets just computes prototypes from the support set.
    /// </para>
    /// <para>
    /// <b>Adaptation Process:</b>
    /// 1. Encode all support set examples using the trained feature encoder
    /// 2. Group embeddings by class
    /// 3. Compute prototype for each class (mean of class embeddings)
    /// 4. Return a model that classifies by distance to prototypes
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After meta-training, when you have a new task with labeled
    /// examples, call this method. The returned model can immediately classify new examples
    /// by finding the nearest class prototype - no additional training needed!
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        // For ProtoNets, adaptation means computing prototypes from support set
        // No gradient updates needed - that's the beauty of non-parametric methods!
        return new PrototypicalModel<T, TInput, TOutput>(
            MetaModel,
            task.SupportInput,
            task.SupportOutput,
            _protoNetsOptions,
            NumOps);
    }

    /// <summary>
    /// Trains the feature encoder on a single episode.
    /// </summary>
    /// <param name="task">The meta-learning task containing support and query sets.</param>
    /// <returns>Tuple of (episode loss, gradients).</returns>
    private (T loss, Vector<T> gradients) TrainEpisode(IMetaLearningTask<T, TInput, TOutput> task)
    {
        // Step 1: Encode support set examples to feature space
        var supportFeatures = EncodeExamples(task.SupportInput);

        // Step 2: Encode query set examples to feature space
        var queryFeatures = EncodeExamples(task.QueryInput);

        // Step 3: Compute class prototypes by averaging support features
        var classPrototypes = ComputeClassPrototypes(supportFeatures, task.SupportOutput);

        // Step 4: Compute distances from query features to class prototypes
        var distances = ComputeDistances(queryFeatures, classPrototypes);

        // Step 5: Apply temperature scaling and softmax to get probabilities
        var probabilities = ApplySoftmaxToDistances(distances);

        // Step 6: Compute cross-entropy loss
        var loss = ComputeCrossEntropyLoss(probabilities, task.QueryOutput);

        // Step 7: Compute gradients for encoder update
        var gradients = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);

        return (loss, gradients);
    }

    /// <summary>
    /// Encodes input examples to feature space using the feature encoder.
    /// </summary>
    /// <param name="inputs">The input examples to encode.</param>
    /// <returns>The encoded feature representations as a matrix (rows = examples, cols = features).</returns>
    private Matrix<T> EncodeExamples(TInput inputs)
    {
        // Get predictions from the model (which acts as the encoder)
        var encoded = MetaModel.Predict(inputs);

        // Convert to matrix format
        Matrix<T> featureMatrix = ConvertToMatrix(encoded);

        // Normalize features if configured
        if (_protoNetsOptions.NormalizeFeatures)
        {
            NormalizeFeatures(featureMatrix);
        }

        return featureMatrix;
    }

    /// <summary>
    /// Converts the output to a matrix format.
    /// </summary>
    private Matrix<T> ConvertToMatrix(TOutput output)
    {
        if (output is Matrix<T> matrix)
        {
            return matrix;
        }

        if (output is Tensor<T> tensor)
        {
            return TensorToMatrix(tensor);
        }

        if (output is Vector<T> vector)
        {
            // N scalar predictions (one per example) - create N-row, 1-column matrix
            // so each example has a 1-dimensional embedding
            var result = new Matrix<T>(vector.Length, 1);
            for (int i = 0; i < vector.Length; i++)
            {
                result[i, 0] = vector[i];
            }
            return result;
        }

        throw new NotSupportedException($"Output type {typeof(TOutput).Name} cannot be converted to Matrix<T>.");
    }

    /// <summary>
    /// Converts a tensor to a matrix.
    /// </summary>
    private Matrix<T> TensorToMatrix(Tensor<T> tensor)
    {
        if (tensor.Shape.Length == 1)
        {
            // 1D tensor - single row
            var result = new Matrix<T>(1, tensor.Shape[0]);
            for (int j = 0; j < tensor.Shape[0]; j++)
            {
                result[0, j] = tensor[new int[] { j }];
            }
            return result;
        }
        else if (tensor.Shape.Length == 2)
        {
            // 2D tensor - direct conversion
            int rows = tensor.Shape[0];
            int cols = tensor.Shape[1];
            var result = new Matrix<T>(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = tensor[new int[] { i, j }];
                }
            }
            return result;
        }
        else
        {
            // Higher dimensional - flatten to 2D (batch x features)
            int batchSize = tensor.Shape[0];
            int featureSize = tensor.Length / batchSize;
            var result = new Matrix<T>(batchSize, featureSize);

            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < featureSize; f++)
                {
                    int flatIndex = b * featureSize + f;
                    var multiDimIndex = ComputeMultiDimIndex(flatIndex, tensor.Shape, 1);
                    multiDimIndex[0] = b;
                    result[b, f] = tensor[multiDimIndex];
                }
            }
            return result;
        }
    }

    /// <summary>
    /// Converts a flat index to multi-dimensional tensor indices.
    /// </summary>
    private int[] ComputeMultiDimIndex(int flatIndex, int[] shape, int startDim)
    {
        var indices = new int[shape.Length];
        int remaining = flatIndex;

        for (int i = shape.Length - 1; i >= startDim; i--)
        {
            indices[i] = remaining % shape[i];
            remaining /= shape[i];
        }

        return indices;
    }

    /// <summary>
    /// Computes class prototypes by averaging features of examples from the same class.
    /// </summary>
    private Dictionary<int, Vector<T>> ComputeClassPrototypes(Matrix<T> supportFeatures, TOutput supportLabels)
    {
        var prototypes = new Dictionary<int, Vector<T>>();
        var classFeatures = new Dictionary<int, List<Vector<T>>>();

        // Group features by class
        for (int i = 0; i < supportFeatures.Rows; i++)
        {
            var feature = GetRow(supportFeatures, i);
            int classLabel = GetClassLabel(supportLabels, i);

            if (!classFeatures.ContainsKey(classLabel))
            {
                classFeatures[classLabel] = new List<Vector<T>>();
            }
            classFeatures[classLabel].Add(feature);
        }

        // Compute prototype for each class (mean of features)
        foreach (var kvp in classFeatures)
        {
            int classLabel = kvp.Key;
            var features = kvp.Value;

            // Compute mean of all features for this class
            var prototype = ComputeMeanVector(features);

            // Apply attention weighting if enabled
            if (_protoNetsOptions.UseAttentionMechanism && _attentionWeights != null)
            {
                prototype = ApplyAttentionWeights(prototype, classLabel);
            }

            prototypes[classLabel] = prototype;
        }

        return prototypes;
    }

    /// <summary>
    /// Computes distances between query features and class prototypes.
    /// </summary>
    private Matrix<T> ComputeDistances(Matrix<T> queryFeatures, Dictionary<int, Vector<T>> classPrototypes)
    {
        int numQueries = queryFeatures.Rows;
        int numClasses = classPrototypes.Count;
        var distances = new Matrix<T>(numQueries, numClasses);

        // Get sorted class labels for consistent column ordering
        var classLabels = classPrototypes.Keys.ToList();
        classLabels.Sort();

        // Compute distance from each query to each class prototype
        for (int q = 0; q < numQueries; q++)
        {
            var queryFeature = GetRow(queryFeatures, q);

            for (int c = 0; c < numClasses; c++)
            {
                int classLabel = classLabels[c];
                var prototype = classPrototypes[classLabel];

                T distance = _protoNetsOptions.DistanceFunction switch
                {
                    ProtoNetsDistanceFunction.Euclidean => ComputeEuclideanDistance(queryFeature, prototype),
                    ProtoNetsDistanceFunction.Cosine => ComputeCosineDistance(queryFeature, prototype),
                    ProtoNetsDistanceFunction.Mahalanobis => ComputeMahalanobisDistance(queryFeature, prototype),
                    _ => ComputeEuclideanDistance(queryFeature, prototype)
                };

                // Apply class-specific scaling if enabled
                if (_protoNetsOptions.UseAdaptiveClassScaling && _classScalingFactors != null)
                {
                    distance = ApplyClassScaling(distance, classLabel);
                }

                distances[q, c] = distance;
            }
        }

        return distances;
    }

    /// <summary>
    /// Applies softmax to distances to convert them to class probabilities.
    /// </summary>
    private Matrix<T> ApplySoftmaxToDistances(Matrix<T> distances)
    {
        int numQueries = distances.Rows;
        int numClasses = distances.Columns;
        var probabilities = new Matrix<T>(numQueries, numClasses);

        // Apply temperature scaling
        var scaledDistances = ApplyTemperatureScaling(distances);

        for (int q = 0; q < numQueries; q++)
        {
            // Find maximum distance for numerical stability
            T maxDistance = scaledDistances[q, 0];
            for (int c = 1; c < numClasses; c++)
            {
                if (NumOps.ToDouble(scaledDistances[q, c]) > NumOps.ToDouble(maxDistance))
                {
                    maxDistance = scaledDistances[q, c];
                }
            }

            // Compute exp(-distance + max) and sum
            var expValues = new T[numClasses];
            T sumExp = NumOps.Zero;

            for (int c = 0; c < numClasses; c++)
            {
                T negDistance = NumOps.Negate(scaledDistances[q, c]);
                T shifted = NumOps.Add(negDistance, maxDistance);
                expValues[c] = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(shifted)));
                sumExp = NumOps.Add(sumExp, expValues[c]);
            }

            // Normalize to probabilities
            for (int c = 0; c < numClasses; c++)
            {
                probabilities[q, c] = NumOps.Divide(expValues[c], sumExp);
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Applies temperature scaling to distances.
    /// </summary>
    private Matrix<T> ApplyTemperatureScaling(Matrix<T> distances)
    {
        if (Math.Abs(_protoNetsOptions.Temperature - 1.0) < 1e-10)
        {
            return distances; // No scaling needed
        }

        int rows = distances.Rows;
        int cols = distances.Columns;
        var scaled = new Matrix<T>(rows, cols);

        T temperature = NumOps.FromDouble(_protoNetsOptions.Temperature);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                scaled[i, j] = NumOps.Divide(distances[i, j], temperature);
            }
        }

        return scaled;
    }

    /// <summary>
    /// Computes cross-entropy loss between predicted probabilities and true labels.
    /// </summary>
    private T ComputeCrossEntropyLoss(Matrix<T> probabilities, TOutput trueLabels)
    {
        T totalLoss = NumOps.Zero;
        int numExamples = probabilities.Rows;

        for (int i = 0; i < numExamples; i++)
        {
            int trueClass = GetClassLabel(trueLabels, i);

            // Ensure class index is within bounds
            if (trueClass >= 0 && trueClass < probabilities.Columns)
            {
                T predictedProb = probabilities[i, trueClass];

                // Add small epsilon to avoid log(0)
                predictedProb = NumOps.Add(predictedProb, NumOps.FromDouble(1e-8));

                T logProb = NumOps.FromDouble(Math.Log(NumOps.ToDouble(predictedProb)));
                T exampleLoss = NumOps.Negate(logProb);

                totalLoss = NumOps.Add(totalLoss, exampleLoss);
            }
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(numExamples));
    }

    /// <summary>
    /// Computes Euclidean distance between two feature vectors using IEngine for vectorization.
    /// </summary>
    private T ComputeEuclideanDistance(Vector<T> a, Vector<T> b)
    {
        // diff = a - b
        var diff = Engine.Subtract(a, b);

        // squared = diff * diff (element-wise)
        var squared = Engine.Multiply(diff, diff);

        // Sum all squared elements
        T sumSquares = NumOps.Zero;
        for (int i = 0; i < squared.Length; i++)
        {
            sumSquares = NumOps.Add(sumSquares, squared[i]);
        }

        return NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(sumSquares)));
    }

    /// <summary>
    /// Computes cosine distance between two feature vectors.
    /// </summary>
    private T ComputeCosineDistance(Vector<T> a, Vector<T> b)
    {
        // Compute dot product using IEngine
        var elementProduct = Engine.Multiply(a, b);
        T dotProduct = NumOps.Zero;
        for (int i = 0; i < elementProduct.Length; i++)
        {
            dotProduct = NumOps.Add(dotProduct, elementProduct[i]);
        }

        // Compute norms using IEngine
        var aSquared = Engine.Multiply(a, a);
        var bSquared = Engine.Multiply(b, b);

        T normASq = NumOps.Zero;
        T normBSq = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            normASq = NumOps.Add(normASq, aSquared[i]);
            normBSq = NumOps.Add(normBSq, bSquared[i]);
        }

        T normA = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(normASq)));
        T normB = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(normBSq)));

        // Avoid division by zero
        T denominator = NumOps.Multiply(normA, normB);
        if (NumOps.ToDouble(denominator) < 1e-8)
        {
            return NumOps.One; // Maximum distance
        }

        // Cosine similarity = dot / (||a|| * ||b||)
        T cosineSimilarity = NumOps.Divide(dotProduct, denominator);

        // Cosine distance = 1 - cosine similarity
        return NumOps.Subtract(NumOps.One, cosineSimilarity);
    }

    /// <summary>
    /// Computes Mahalanobis distance using learned covariance scaling.
    /// </summary>
    private T ComputeMahalanobisDistance(Vector<T> a, Vector<T> b)
    {
        // Simplified Mahalanobis distance with scalar scaling
        // Full implementation would use a learned covariance matrix
        var diff = Engine.Subtract(a, b);
        var squared = Engine.Multiply(diff, diff);

        T sumSquares = NumOps.Zero;
        for (int i = 0; i < squared.Length; i++)
        {
            sumSquares = NumOps.Add(sumSquares, squared[i]);
        }

        // Apply Mahalanobis scaling factor
        return NumOps.Multiply(sumSquares, NumOps.FromDouble(_protoNetsOptions.MahalanobisScaling));
    }

    /// <summary>
    /// Applies feature normalization (L2 normalization).
    /// </summary>
    private void NormalizeFeatures(Matrix<T> features)
    {
        for (int i = 0; i < features.Rows; i++)
        {
            // Compute L2 norm
            T sumSquares = NumOps.Zero;
            for (int j = 0; j < features.Columns; j++)
            {
                T squared = NumOps.Multiply(features[i, j], features[i, j]);
                sumSquares = NumOps.Add(sumSquares, squared);
            }
            T norm = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(sumSquares)));

            // Normalize if norm is not zero
            if (NumOps.ToDouble(norm) > 1e-8)
            {
                for (int j = 0; j < features.Columns; j++)
                {
                    features[i, j] = NumOps.Divide(features[i, j], norm);
                }
            }
        }
    }

    /// <summary>
    /// Computes the mean of a list of vectors.
    /// </summary>
    private Vector<T> ComputeMeanVector(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0)
            throw new ArgumentException("Cannot compute mean of empty vector list.");

        int dimension = vectors[0].Length;
        var mean = new Vector<T>(dimension);

        // Sum all vectors
        foreach (var vector in vectors)
        {
            if (vector.Length != dimension)
                throw new ArgumentException("All vectors must have the same dimension.");

            for (int i = 0; i < dimension; i++)
            {
                mean[i] = NumOps.Add(mean[i], vector[i]);
            }
        }

        // Divide by count to get mean
        T divisor = NumOps.FromDouble(vectors.Count);
        for (int i = 0; i < dimension; i++)
        {
            mean[i] = NumOps.Divide(mean[i], divisor);
        }

        return mean;
    }

    /// <summary>
    /// Gets a row from a matrix as a vector.
    /// </summary>
    private Vector<T> GetRow(Matrix<T> matrix, int rowIndex)
    {
        var row = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
        {
            row[j] = matrix[rowIndex, j];
        }
        return row;
    }

    /// <summary>
    /// Extracts class label from output at specified index.
    /// </summary>
    private int GetClassLabel(TOutput output, int index)
    {
        if (output is Vector<T> vector)
        {
            if (index < vector.Length)
            {
                // Get class label at the specified index
                return (int)NumOps.ToDouble(vector[index]);
            }
            else
            {
                throw new ArgumentOutOfRangeException(nameof(index),
                    $"Index {index} is out of range for vector of length {vector.Length}");
            }
        }
        else if (output is Tensor<T> tensor)
        {
            if (tensor.Shape.Length == 1)
            {
                // 1D tensor - class indices
                if (index < tensor.Shape[0])
                {
                    return (int)NumOps.ToDouble(tensor[new int[] { index }]);
                }
            }
            else if (tensor.Shape.Length == 2)
            {
                // 2D tensor - batch x one-hot or batch x 1
                if (tensor.Shape[1] == 1)
                {
                    // Class indices
                    return (int)NumOps.ToDouble(tensor[new int[] { index, 0 }]);
                }
                else
                {
                    // One-hot - find argmax
                    int maxIdx = 0;
                    T maxVal = tensor[new int[] { index, 0 }];
                    for (int c = 1; c < tensor.Shape[1]; c++)
                    {
                        T val = tensor[new int[] { index, c }];
                        if (NumOps.ToDouble(val) > NumOps.ToDouble(maxVal))
                        {
                            maxVal = val;
                            maxIdx = c;
                        }
                    }
                    return maxIdx;
                }
            }
        }
        else if (output is Matrix<T> matrix)
        {
            if (matrix.Columns == 1)
            {
                // Class indices
                return (int)NumOps.ToDouble(matrix[index, 0]);
            }
            else
            {
                // One-hot - find argmax
                int maxIdx = 0;
                T maxVal = matrix[index, 0];
                for (int c = 1; c < matrix.Columns; c++)
                {
                    if (NumOps.ToDouble(matrix[index, c]) > NumOps.ToDouble(maxVal))
                    {
                        maxVal = matrix[index, c];
                        maxIdx = c;
                    }
                }
                return maxIdx;
            }
        }

        return 0; // Default
    }

    /// <summary>
    /// Applies attention weights to enhance prototype computation.
    /// </summary>
    private Vector<T> ApplyAttentionWeights(Vector<T> prototype, int classLabel)
    {
        // Placeholder implementation
        // Full implementation would learn attention mechanism to weight important features
        return prototype;
    }

    /// <summary>
    /// Applies class-specific scaling to distance computation.
    /// </summary>
    private T ApplyClassScaling(T distance, int classLabel)
    {
        if (_classScalingFactors != null && _classScalingFactors.TryGetValue(classLabel, out var scaling))
        {
            return NumOps.Multiply(distance, scaling);
        }
        return distance;
    }
}

/// <summary>
/// Prototypical model for few-shot classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This model encapsulates the ProtoNets inference mechanism with pre-computed prototypes.
/// It is returned by <see cref="ProtoNetsAlgorithm{T, TInput, TOutput}.Adapt"/> and provides
/// fast classification without any gradient computation.
/// </para>
/// <para><b>For Beginners:</b> After adapting ProtoNets to a new task, you get this model.
/// It can classify new examples instantly by finding the nearest class prototype.
/// </para>
/// </remarks>
public class PrototypicalModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _featureEncoder;
    private readonly Dictionary<int, Vector<T>> _classPrototypes;
    private readonly ProtoNetsOptions<T, TInput, TOutput> _options;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the PrototypicalModel.
    /// </summary>
    /// <param name="featureEncoder">The trained feature encoder.</param>
    /// <param name="supportInputs">Support set inputs for computing prototypes.</param>
    /// <param name="supportOutputs">Support set outputs (labels).</param>
    /// <param name="options">ProtoNets configuration options.</param>
    /// <param name="numOps">Numeric operations for type T.</param>
    public PrototypicalModel(
        IFullModel<T, TInput, TOutput> featureEncoder,
        TInput supportInputs,
        TOutput supportOutputs,
        ProtoNetsOptions<T, TInput, TOutput> options,
        INumericOperations<T> numOps)
    {
        Guard.NotNull(featureEncoder);
        _featureEncoder = featureEncoder;
        Guard.NotNull(options);
        _options = options;
        Guard.NotNull(numOps);
        _numOps = numOps;
        _classPrototypes = new Dictionary<int, Vector<T>>();

        // Compute prototypes from support set
        ComputePrototypes(supportInputs, supportOutputs);
    }

    /// <summary>
    /// Gets the model metadata.
    /// </summary>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Makes predictions using prototype-based classification.
    /// </summary>
    /// <param name="input">The input to classify.</param>
    /// <returns>Predicted class probabilities.</returns>
    public TOutput Predict(TInput input)
    {
        // Encode input to feature space
        var features = _featureEncoder.Predict(input);

        if (_classPrototypes.Count == 0)
        {
            return features; // Return encoder output if no prototypes
        }

        // Convert to matrix to handle batch inputs properly
        var featureMatrix = ConvertToMatrix(features);
        if (featureMatrix == null || featureMatrix.Rows == 0)
        {
            return features;
        }

        var classLabels = _classPrototypes.Keys.ToList();
        classLabels.Sort();
        int numClasses = classLabels.Count;
        int batchSize = featureMatrix.Rows;

        // Create output tensor for all predictions [batchSize] or [batchSize, numClasses]
        var allProbabilities = new List<List<T>>();

        // Process each sample in the batch
        for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
        {
            // Extract feature vector for this sample
            var featureVector = GetRow(featureMatrix, sampleIdx);

            // Normalize if configured
            if (_options.NormalizeFeatures)
            {
                featureVector = NormalizeVector(featureVector);
            }

            // Compute distances to all prototypes
            var distances = new List<T>();
            foreach (var label in classLabels)
            {
                T distance = ComputeDistance(featureVector, _classPrototypes[label]);
                distances.Add(distance);
            }

            // Apply softmax to distances to get probabilities
            var probabilities = ApplySoftmax(distances);
            allProbabilities.Add(probabilities);
        }

        // Convert to appropriate output format
        return ConvertBatchToOutput(allProbabilities, classLabels, batchSize);
    }

    /// <summary>
    /// Trains the model (not applicable for prototype-based models).
    /// </summary>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Prototype models don't support training. Compute new prototypes instead.");
    }

    /// <summary>
    /// Updates model parameters (not applicable for prototype-based models).
    /// </summary>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("Prototype models don't have trainable parameters.");
    }

    /// <summary>
    /// Gets model parameters (not applicable for prototype-based models).
    /// </summary>
    public Vector<T> GetParameters()
    {
        throw new NotSupportedException("Prototype models don't have trainable parameters.");
    }

    /// <summary>
    /// Computes class prototypes from support set.
    /// </summary>
    private void ComputePrototypes(TInput supportInputs, TOutput supportOutputs)
    {
        // Encode support set
        var encodedSupport = _featureEncoder.Predict(supportInputs);

        // Convert to matrix
        var featureMatrix = ConvertToMatrix(encodedSupport);
        if (featureMatrix == null)
        {
            return;
        }

        // Normalize if configured
        if (_options.NormalizeFeatures)
        {
            NormalizeMatrix(featureMatrix);
        }

        // Group by class and compute means
        var classFeatures = new Dictionary<int, List<Vector<T>>>();

        for (int i = 0; i < featureMatrix.Rows; i++)
        {
            var feature = GetRow(featureMatrix, i);
            int classLabel = GetClassLabel(supportOutputs, i);

            if (!classFeatures.ContainsKey(classLabel))
            {
                classFeatures[classLabel] = new List<Vector<T>>();
            }
            classFeatures[classLabel].Add(feature);
        }

        // Compute prototype for each class
        foreach (var kvp in classFeatures)
        {
            var prototype = ComputeMeanVector(kvp.Value);
            _classPrototypes[kvp.Key] = prototype;
        }
    }

    private Matrix<T>? ConvertToMatrix(TOutput output)
    {
        if (output is Matrix<T> matrix)
        {
            return matrix;
        }

        if (output is Tensor<T> tensor)
        {
            return TensorToMatrix(tensor);
        }

        if (output is Vector<T> vector)
        {
            // N scalar predictions (one per example) - create N-row, 1-column matrix
            var result = new Matrix<T>(vector.Length, 1);
            for (int i = 0; i < vector.Length; i++)
            {
                result[i, 0] = vector[i];
            }
            return result;
        }

        return null;
    }

    private Matrix<T> TensorToMatrix(Tensor<T> tensor)
    {
        if (tensor.Shape.Length == 2)
        {
            int rows = tensor.Shape[0];
            int cols = tensor.Shape[1];
            var result = new Matrix<T>(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = tensor[new int[] { i, j }];
                }
            }
            return result;
        }

        // Flatten higher dimensions
        int batchSize = tensor.Shape[0];
        int featureSize = tensor.Length / batchSize;
        var matrix = new Matrix<T>(batchSize, featureSize);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < featureSize; f++)
            {
                int flatIdx = b * featureSize + f;
                matrix[b, f] = tensor.GetFlat(flatIdx);
            }
        }

        return matrix;
    }

    private Vector<T>? ConvertToVector(TOutput output)
    {
        if (output is Vector<T> vector)
        {
            return vector;
        }

        if (output is Tensor<T> tensor)
        {
            // Convert tensor to matrix format, then extract first row as vector
            // This ensures the vector dimension matches prototypes computed from rows
            var matrix = TensorToMatrix(tensor);
            return GetRow(matrix, 0);
        }

        if (output is Matrix<T> matrix2)
        {
            // Extract first row as vector
            return GetRow(matrix2, 0);
        }

        return null;
    }

    private Vector<T> NormalizeVector(Vector<T> vector)
    {
        T sumSquares = _numOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sumSquares = _numOps.Add(sumSquares, _numOps.Multiply(vector[i], vector[i]));
        }

        T norm = _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(sumSquares)));

        if (_numOps.ToDouble(norm) < 1e-8)
        {
            return vector;
        }

        var normalized = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            normalized[i] = _numOps.Divide(vector[i], norm);
        }
        return normalized;
    }

    private void NormalizeMatrix(Matrix<T> matrix)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            T sumSquares = _numOps.Zero;
            for (int j = 0; j < matrix.Columns; j++)
            {
                sumSquares = _numOps.Add(sumSquares, _numOps.Multiply(matrix[i, j], matrix[i, j]));
            }

            T norm = _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(sumSquares)));

            if (_numOps.ToDouble(norm) > 1e-8)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    matrix[i, j] = _numOps.Divide(matrix[i, j], norm);
                }
            }
        }
    }

    private T ComputeDistance(Vector<T> a, Vector<T> b)
    {
        return _options.DistanceFunction switch
        {
            ProtoNetsDistanceFunction.Euclidean => ComputeEuclideanDistance(a, b),
            ProtoNetsDistanceFunction.Cosine => ComputeCosineDistance(a, b),
            ProtoNetsDistanceFunction.Mahalanobis => ComputeMahalanobisDistance(a, b),
            _ => ComputeEuclideanDistance(a, b)
        };
    }

    private T ComputeEuclideanDistance(Vector<T> a, Vector<T> b)
    {
        T sumSquares = _numOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = _numOps.Subtract(a[i], b[i]);
            sumSquares = _numOps.Add(sumSquares, _numOps.Multiply(diff, diff));
        }
        return _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(sumSquares)));
    }

    private T ComputeCosineDistance(Vector<T> a, Vector<T> b)
    {
        T dotProduct = _numOps.Zero;
        T normASq = _numOps.Zero;
        T normBSq = _numOps.Zero;

        for (int i = 0; i < a.Length; i++)
        {
            dotProduct = _numOps.Add(dotProduct, _numOps.Multiply(a[i], b[i]));
            normASq = _numOps.Add(normASq, _numOps.Multiply(a[i], a[i]));
            normBSq = _numOps.Add(normBSq, _numOps.Multiply(b[i], b[i]));
        }

        T normA = _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(normASq)));
        T normB = _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(normBSq)));

        T denominator = _numOps.Multiply(normA, normB);
        if (_numOps.ToDouble(denominator) < 1e-8)
        {
            return _numOps.One;
        }

        T cosineSimilarity = _numOps.Divide(dotProduct, denominator);
        return _numOps.Subtract(_numOps.One, cosineSimilarity);
    }

    private T ComputeMahalanobisDistance(Vector<T> a, Vector<T> b)
    {
        T sumSquares = _numOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = _numOps.Subtract(a[i], b[i]);
            sumSquares = _numOps.Add(sumSquares, _numOps.Multiply(diff, diff));
        }
        return _numOps.Multiply(sumSquares, _numOps.FromDouble(_options.MahalanobisScaling));
    }

    private List<T> ApplySoftmax(List<T> distances)
    {
        if (distances.Count == 0)
            return new List<T>();

        // Apply negative sign (smaller distance = higher probability)
        var negDistances = distances.Select(d => _numOps.Negate(d)).ToList();

        // Apply temperature scaling
        if (Math.Abs(_options.Temperature - 1.0) >= 1e-10)
        {
            T temp = _numOps.FromDouble(_options.Temperature);
            negDistances = negDistances.Select(d => _numOps.Divide(d, temp)).ToList();
        }

        // Find max for numerical stability
        T maxDist = negDistances[0];
        foreach (var d in negDistances)
        {
            if (_numOps.ToDouble(d) > _numOps.ToDouble(maxDist))
                maxDist = d;
        }

        // Compute exp values
        var expValues = negDistances.Select(d =>
        {
            T shifted = _numOps.Subtract(d, maxDist);
            return _numOps.FromDouble(Math.Exp(_numOps.ToDouble(shifted)));
        }).ToList();

        // Sum exp values
        T sumExp = _numOps.Zero;
        foreach (var e in expValues)
        {
            sumExp = _numOps.Add(sumExp, e);
        }

        // Normalize
        return expValues.Select(e => _numOps.Divide(e, sumExp)).ToList();
    }

    private TOutput ConvertToOutput(List<T> probabilities, List<int> classLabels)
    {
        // Create output as tensor
        var tensor = new Tensor<T>(new int[] { probabilities.Count });
        for (int i = 0; i < probabilities.Count; i++)
        {
            tensor[new int[] { i }] = probabilities[i];
        }

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            return (TOutput)(object)tensor;
        }
        else if (typeof(TOutput) == typeof(Vector<T>))
        {
            return (TOutput)(object)tensor.ToVector();
        }
        else
        {
            throw new NotSupportedException($"Output type {typeof(TOutput).Name} is not supported.");
        }
    }

    private TOutput ConvertBatchToOutput(List<List<T>> allProbabilities, List<int> classLabels, int batchSize)
    {
        int numClasses = classLabels.Count;

        // Create output as 2D tensor [batchSize, numClasses]
        var tensor = new Tensor<T>(new int[] { batchSize, numClasses });
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < numClasses; j++)
            {
                tensor[new int[] { i, j }] = allProbabilities[i][j];
            }
        }

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            return (TOutput)(object)tensor;
        }
        else if (typeof(TOutput) == typeof(Matrix<T>))
        {
            var matrix = new Matrix<T>(batchSize, numClasses);
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < numClasses; j++)
                {
                    matrix[i, j] = allProbabilities[i][j];
                }
            }
            return (TOutput)(object)matrix;
        }
        else if (typeof(TOutput) == typeof(Vector<T>))
        {
            // Flatten to vector if single sample (return per-class probabilities)
            if (batchSize == 1)
            {
                return (TOutput)(object)new Vector<T>(allProbabilities[0].ToArray());
            }
            // For multiple samples, return argmax class prediction per sample
            // This matches the input label format (one scalar label per sample)
            var predictions = new Vector<T>(batchSize);
            for (int i = 0; i < batchSize; i++)
            {
                int maxIdx = 0;
                T maxVal = allProbabilities[i][0];
                for (int j = 1; j < numClasses; j++)
                {
                    if (_numOps.ToDouble(allProbabilities[i][j]) > _numOps.ToDouble(maxVal))
                    {
                        maxVal = allProbabilities[i][j];
                        maxIdx = j;
                    }
                }
                predictions[i] = _numOps.FromDouble(classLabels[maxIdx]);
            }
            return (TOutput)(object)predictions;
        }
        else
        {
            throw new NotSupportedException($"Output type {typeof(TOutput).Name} is not supported.");
        }
    }

    private Vector<T> GetRow(Matrix<T> matrix, int rowIndex)
    {
        var row = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
        {
            row[j] = matrix[rowIndex, j];
        }
        return row;
    }

    private Vector<T> ComputeMeanVector(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0)
            throw new ArgumentException("Cannot compute mean of empty vector list.");

        int dimension = vectors[0].Length;
        var mean = new Vector<T>(dimension);

        foreach (var vector in vectors)
        {
            for (int i = 0; i < dimension; i++)
            {
                mean[i] = _numOps.Add(mean[i], vector[i]);
            }
        }

        T divisor = _numOps.FromDouble(vectors.Count);
        for (int i = 0; i < dimension; i++)
        {
            mean[i] = _numOps.Divide(mean[i], divisor);
        }

        return mean;
    }

    private int GetClassLabel(TOutput output, int index)
    {
        if (output is Vector<T> vector)
        {
            if (index < vector.Length)
            {
                return (int)_numOps.ToDouble(vector[index]);
            }
        }
        else if (output is Tensor<T> tensor)
        {
            if (tensor.Shape.Length >= 1 && index < tensor.Shape[0])
            {
                return (int)_numOps.ToDouble(tensor[new int[] { index }]);
            }
        }

        return 0;
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>Model metadata including information about prototypes.</returns>
    public ModelMetadata<T> GetModelMetadata()
    {
        return Metadata;
    }
}
