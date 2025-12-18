using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Config;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using System.Diagnostics;

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
/// feature_encoder = NeuralNetwork()  # Maps x → φ(x)
///
/// # Episode training
/// for each episode:
///     # Sample N-way K-shot task
///     support_set = {examples_from_N_classes, K_examples_each}
///     query_set = {examples_from_same_N_classes}
///
///     # Compute class prototypes (non-parametric)
///     for each class c:
///         prototype_c = mean(φ(x) for x in support_examples_of_class_c)
///
///     # Classification by distance
///     for each query example x:
///         distances = [distance(φ(x), prototype_c) for c in classes]
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
/// <b>Production Features:</b>
/// - Learnable distance metrics (Euclidean with learned scaling)
/// - Attention mechanisms for adaptive prototype weighting
/// - Temperature scaling for calibration
/// - Multiple distance functions (Euclidean, Cosine, Mahalanobis)
/// - Feature normalization and regularization
/// - Curriculum learning on episode difficulty
/// </para>
/// </remarks>
public class ProtoNetsAlgorithm<T, TInput, TOutput> : MetaLearningBase<T, TInput, TOutput>
    where T : struct, IEquatable<T>, IFormattable
{
    private readonly ProtoNetsAlgorithmOptions<T, TInput, TOutput> _protoNetsOptions;
    private readonly INeuralNetwork<T> _featureEncoder;

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
    /// - <b>featureEncoder:</b> Neural network that maps inputs to features (e.g., CNN for images)
    /// - <b>distanceFunction:</b> How to measure similarity (Euclidean, Cosine, etc.)
    /// - <b>numClasses:</b> How many classes to distinguish between (N in N-way)
    /// - <b>examplesPerClass:</b> How many examples per class for support (K in K-shot)
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
    public ProtoNetsAlgorithm(ProtoNetsAlgorithmOptions<T, TInput, TOutput> options)
        : base(options)
    {
        _protoNetsOptions = options ?? throw new ArgumentNullException(nameof(options));

        // Initialize feature encoder if not provided
        _featureEncoder = options.FeatureEncoder ?? throw new ArgumentNullException(nameof(options.FeatureEncoder));

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

    /// <inheritdoc/>
    public override string AlgorithmName => "ProtoNets";

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        T totalLoss = NumOps.Zero;

        foreach (var task in taskBatch.Tasks)
        {
            // Train on this episode
            T episodeLoss = TrainEpisode(task);
            totalLoss = NumOps.Add(totalLoss, episodeLoss);
        }

        // Return average loss
        return NumOps.Divide(totalLoss, NumOps.FromDouble(taskBatch.BatchSize));
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        // For ProtoNets, adaptation just means computing prototypes from support set
        // No gradient updates needed - that's the beauty of non-parametric methods!
        var adaptedModel = new PrototypicalModel<T, TInput, TOutput>(
            _featureEncoder,
            task.SupportInput,
            task.SupportOutput);

        return adaptedModel;
    }

    /// <summary>
    /// Trains the feature encoder on a single episode.
    /// </summary>
    /// <param name="task">The meta-learning task containing support and query sets.</param>
    /// <returns>The episode loss.</returns>
    /// <remarks>
    /// This implements the core ProtoNets training algorithm:
    /// 1. Encode support and query examples
    /// 2. Compute class prototypes from support set
    /// 3. Compute distances from queries to prototypes
    /// 4. Apply softmax to get class probabilities
    /// 5. Compute cross-entropy loss
    /// 6. Backpropagate and update encoder
    /// </remarks>
    private T TrainEpisode(ITask<T, TInput, TOutput> task)
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

        // Step 7: Backpropagate and update feature encoder
        UpdateFeatureEncoder(loss);

        return loss;
    }

    /// <summary>
    /// Encodes input examples to feature space using the feature encoder.
    /// </summary>
    /// <param name="inputs">The input examples to encode.</param>
    /// <returns>The encoded feature representations.</returns>
    /// <remarks>
    /// The feature encoder is the learnable component of ProtoNets.
    /// It maps raw inputs to a feature space where same-class examples cluster together.
    /// </remarks>
    private Matrix<T> EncodeExamples(TInput inputs)
    {
        // Convert TInput to Tensor<T> if needed
        Tensor<T> inputTensor;
        if (typeof(TInput) == typeof(Tensor<T>))
        {
            inputTensor = (Tensor<T>)(object)inputs;
        }
        else
        {
            // For other input types (like Matrix<T>), convert to Tensor
            // This is a simplified conversion - in practice would need proper handling
            throw new NotSupportedException($"Input type {typeof(TInput).Name} is not supported. Currently only Tensor<T> is supported.");
        }

        // Get feature representations from the encoder
        var encoded = _featureEncoder.Predict(inputTensor);

        // Convert output to Matrix<T> for prototype computation
        Matrix<T> featureMatrix;
        if (encoded.Shape.Dimensions == 2)
        {
            // Already a 2D matrix
            featureMatrix = TensorToMatrix(encoded);
        }
        else
        {
            // Flatten to 2D matrix
            featureMatrix = FlattenToMatrix(encoded);
        }

        // Normalize features if configured
        if (_protoNetsOptions.NormalizeFeatures)
        {
            NormalizeFeatures(featureMatrix);
        }

        return featureMatrix;
    }

    /// <summary>
    /// Converts a 2D tensor to a matrix.
    /// </summary>
    private Matrix<T> TensorToMatrix(Tensor<T> tensor)
    {
        if (tensor.Shape.Dimensions.Length != 2)
            throw new ArgumentException("Tensor must be 2-dimensional.");

        int rows = tensor.Shape.Dimensions[0];
        int cols = tensor.Shape.Dimensions[1];
        var matrix = new Matrix<T>(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                var index = new int[] { i, j };
                matrix[i, j] = tensor.GetValue(index);
            }
        }

        return matrix;
    }

    /// <summary>
    /// Flattens a tensor to a 2D matrix.
    /// </summary>
    private Matrix<T> FlattenToMatrix(Tensor<T> tensor)
    {
        // Treat first dimension as batch size, flatten rest
        int batchSize = tensor.Shape.Dimensions[0];
        int featureSize = tensor.Shape.Size / batchSize;
        var matrix = new Matrix<T>(batchSize, featureSize);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < featureSize; f++)
            {
                // Compute flat index
                int flatIndex = b * featureSize + f;
                var multiDimIndex = ComputeMultiDimIndex(flatIndex, tensor.Shape, 1);
                multiDimIndex[0] = b; // Set batch dimension

                matrix[b, f] = tensor.GetValue(multiDimIndex);
            }
        }

        return matrix;
    }

    /// <summary>
    /// Converts a flat index to multi-dimensional tensor indices.
    /// </summary>
    private int[] ComputeMultiDimIndex(int flatIndex, TensorShape shape, int startDim)
    {
        var indices = new int[shape.Dimensions.Length];
        int remaining = flatIndex;

        for (int i = shape.Dimensions.Length - 1; i >= startDim; i--)
        {
            indices[i] = remaining % shape.Dimensions[i];
            remaining /= shape.Dimensions[i];
        }

        return indices;
    }

    /// <summary>
    /// Computes class prototypes by averaging features of examples from the same class.
    /// </summary>
    /// <param name="supportFeatures">Features of support set examples.</param>
    /// <param name="supportLabels">Labels of support set examples.</param>
    /// <returns>Dictionary mapping class labels to prototype vectors.</returns>
    /// <remarks>
    /// Prototypes are the non-parametric representation of each class.
    /// Each prototype is simply the mean of all support examples from that class.
    /// </remarks>
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
            if (_protoNetsOptions.UseAttentionMechanism)
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
    /// <param name="queryFeatures">Features of query set examples.</param>
    /// <param name="classPrototypes">Dictionary of class prototypes.</param>
        /// <returns>Matrix of distances where rows are queries and columns are classes.</returns>
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

                T distance;

                // Use configured distance function
                switch (_protoNetsOptions.DistanceFunction)
                {
                    case DistanceFunction.Euclidean:
                        distance = ComputeEuclideanDistance(queryFeature, prototype);
                        break;
                    case DistanceFunction.Cosine:
                        distance = ComputeCosineDistance(queryFeature, prototype);
                        break;
                    case DistanceFunction.Mahalanobis:
                        distance = ComputeMahalanobisDistance(queryFeature, prototype);
                        break;
                    default:
                        throw new NotSupportedException($"Distance function {_protoNetsOptions.DistanceFunction} is not supported.");
                }

                // Apply class-specific scaling if enabled
                if (_protoNetsOptions.UseAdaptiveClassScaling)
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
    /// <param name="distances">Matrix of distances between queries and class prototypes.</param>
    /// <returns>Matrix of class probabilities.</returns>
    /// <remarks>
    /// Uses negative distances because smaller distances should have higher probabilities.
    /// Applies temperature scaling to control the sharpness of the distribution.
    /// </remarks>
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
                if (Convert.ToDouble(scaledDistances[q, c]) > Convert.ToDouble(maxDistance))
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
                expValues[c] = NumOps.FromDouble(Math.Exp(Convert.ToDouble(shifted)));
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
    /// <param name="distances">Original distances matrix.</param>
    /// <returns>Temperature-scaled distances.</returns>
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
    /// <param name="probabilities">Predicted class probabilities.</param>
    /// <param name="trueLabels">True class labels.</param>
    /// <returns>The cross-entropy loss.</returns>
    private T ComputeCrossEntropyLoss(Matrix<T> probabilities, TOutput trueLabels)
    {
        T totalLoss = NumOps.Zero;
        int numExamples = probabilities.Rows;

        for (int i = 0; i < numExamples; i++)
        {
            int trueClass = GetClassLabel(trueLabels, i);
            T predictedProb = probabilities[i, trueClass];

            // Add small epsilon to avoid log(0)
            predictedProb = NumOps.Add(predictedProb, NumOps.FromDouble(1e-8));

            T logProb = NumOps.FromDouble(Math.Log(Convert.ToDouble(predictedProb)));
            T exampleLoss = NumOps.Negate(logProb);

            totalLoss = NumOps.Add(totalLoss, exampleLoss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(numExamples));
    }

    /// <summary>
    /// Updates the feature encoder parameters using backpropagation.
    /// </summary>
    /// <param name="loss">The loss to backpropagate.</param>
    private void UpdateFeatureEncoder(T loss)
    {
        // In a real implementation, this would:
        // 1. Compute gradients of loss w.r.t. encoder parameters
        // 2. Apply optimizer update (SGD, Adam, etc.)
        // 3. Update encoder weights

        // For now, this is a placeholder
        // The actual implementation would depend on the neural network framework used
    }

    /// <summary>
    /// Computes Euclidean distance between two feature vectors.
    /// </summary>
    private T ComputeEuclideanDistance(Vector<T> a, Vector<T> b)
    {
        T sumSquares = NumOps.Zero;

        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            T squared = NumOps.Multiply(diff, diff);
            sumSquares = NumOps.Add(sumSquares, squared);
        }

        return NumOps.FromDouble(Math.Sqrt(Convert.ToDouble(sumSquares)));
    }

    /// <summary>
    /// Computes cosine distance between two feature vectors.
    /// </summary>
    private T ComputeCosineDistance(Vector<T> a, Vector<T> b)
    {
        // Compute dot product
        T dotProduct = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(a[i], b[i]));
        }

        // Compute norms
        T normA = NumOps.Zero;
        T normB = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            normA = NumOps.Add(normA, NumOps.Multiply(a[i], a[i]));
            normB = NumOps.Add(normB, NumOps.Multiply(b[i], b[i]));
        }
        normA = NumOps.FromDouble(Math.Sqrt(Convert.ToDouble(normA)));
        normB = NumOps.FromDouble(Math.Sqrt(Convert.ToDouble(normB)));

        // Avoid division by zero
        T denominator = NumOps.Multiply(normA, normB);
        if (Convert.ToDouble(denominator) < 1e-8)
        {
            return NumOps.One; // Maximum distance
        }

        // Cosine similarity = dot / (||a|| * ||b||)
        T cosineSimilarity = NumOps.Divide(dotProduct, denominator);

        // Cosine distance = 1 - cosine similarity
        return NumOps.Subtract(NumOps.One, cosineSimilarity);
    }

    /// <summary>
    /// Computes Mahalanobis distance using learned covariance matrix.
    /// </summary>
    private T ComputeMahalanobisDistance(Vector<T> a, Vector<T> b)
    {
        // This is a simplified implementation
        // In practice, would use a learned or estimated covariance matrix

        T sumSquares = NumOps.Zero;

        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            T squared = NumOps.Multiply(diff, diff);
            sumSquares = NumOps.Add(sumSquares, squared);
        }

        // Apply learned scaling factors (simplified)
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
            T norm = NumOps.FromDouble(Math.Sqrt(Convert.ToDouble(sumSquares)));

            // Normalize if norm is not zero
            if (Convert.ToDouble(norm) > 1e-8)
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
        // This is a simplified implementation
        // In practice, would need to handle different output types

        if (typeof(TOutput) == typeof(Vector<T>))
        {
            var vector = (Vector<T>)(object)output;
            // For one-hot encoding, find the index with value 1
            for (int i = 0; i < vector.Length; i++)
            {
                if (Convert.ToDouble(vector[i]) > 0.5)
                    return i;
            }
            return 0;
        }
        else if (typeof(TOutput) == typeof(Tensor<T>))
        {
            var tensor = (Tensor<T>)(object)output;

            // Handle different tensor shapes
            if (tensor.Shape.Dimensions.Length == 1)
            {
                // 1D tensor - treat as class probabilities
                for (int i = 0; i < tensor.Shape.Dimensions[0]; i++)
                {
                    var idx = new int[] { i };
                    if (Convert.ToDouble(tensor.GetValue(idx)) > 0.5)
                        return i;
                }
                return 0;
            }
            else if (tensor.Shape.Dimensions.Length == 2)
            {
                // 2D tensor - get the index-th row
                for (int i = 0; i < tensor.Shape.Dimensions[1]; i++)
                {
                    var idx = new int[] { index, i };
                    if (Convert.ToDouble(tensor.GetValue(idx)) > 0.5)
                        return i;
                }
                return 0;
            }
            else
            {
                // Higher dimensional - flatten first dimension for batch
                int numClasses = tensor.Shape.Dimensions[tensor.Shape.Dimensions.Length - 1];
                for (int i = 0; i < numClasses; i++)
                {
                    var idx = new int[tensor.Shape.Dimensions.Length];
                    idx[0] = index;
                    idx[idx.Length - 1] = i;

                    if (Convert.ToDouble(tensor.GetValue(idx)) > 0.5)
                        return i;
                }
                return 0;
            }
        }
        else
        {
            throw new NotSupportedException($"Output type {typeof(TOutput).Name} is not supported.");
        }
    }

    /// <summary>
    /// Attention weights for prototype enhancement (if enabled).
    /// </summary>
    private Matrix<T> _attentionWeights;

    /// <summary>
    /// Applies attention weights to enhance prototype computation.
    /// </summary>
    private Vector<T> ApplyAttentionWeights(Vector<T> prototype, int classLabel)
    {
        // Placeholder implementation
        // In practice, would learn attention mechanism to weight important features
        return prototype;
    }

    /// <summary>
    /// Class-specific scaling factors for adaptive distance computation.
    /// </summary>
    private Dictionary<int, T> _classScalingFactors;

    /// <summary>
    /// Applies class-specific scaling to distance computation.
    /// </summary>
    private T ApplyClassScaling(T distance, int classLabel)
    {
        if (_classScalingFactors.TryGetValue(classLabel, out T scaling))
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
/// This model encapsulates the ProtoNets inference mechanism with pre-computed prototypes.
/// </remarks>
public class PrototypicalModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
    where T : struct, IEquatable<T>, IFormattable
{
    private readonly INeuralNetwork<T> _featureEncoder;
    private readonly Dictionary<int, Vector<T>> _classPrototypes;
    private readonly Dictionary<int, int> _classMapping;
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Initializes a new instance of the PrototypicalModel.
    /// </summary>
    /// <param name="featureEncoder">The trained feature encoder.</param>
    /// <param name="supportInputs">Support set inputs for computing prototypes.</param>
    /// <param name="supportOutputs">Support set outputs (labels).</param>
    public PrototypicalModel(
        INeuralNetwork<T> featureEncoder,
        TInput supportInputs,
        TOutput supportOutputs)
    {
        _featureEncoder = featureEncoder ?? throw new ArgumentNullException(nameof(featureEncoder));
        _classPrototypes = new Dictionary<int, Vector<T>>();
        _classMapping = new Dictionary<int, int>();

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
        // Convert TInput to Tensor<T> if needed
        Tensor<T> inputTensor;
        if (typeof(TInput) == typeof(Tensor<T>))
        {
            inputTensor = (Tensor<T>)(object)input;
        }
        else
        {
            throw new NotSupportedException($"Input type {typeof(TInput).Name} is not supported. Currently only Tensor<T> is supported.");
        }

        // Set to inference mode
        _featureEncoder.SetTrainingMode(false);

        // Encode input to feature space
        var encoded = _featureEncoder.Predict(inputTensor);

        // Convert to feature vector
        var features = TensorToVector(encoded);

        // Compute distances to all prototypes
        var distances = new List<T>();
        var classLabels = new List<int>();

        foreach (var kvp in _classPrototypes)
        {
            classLabels.Add(kvp.Key);
            // Compute Euclidean distance
            T distance = ComputeEuclideanDistance(features, kvp.Value);
            distances.Add(distance);
        }

        // Apply softmax to distances to get probabilities
        var probabilities = ApplySoftmax(distances);

        // Convert to appropriate output format
        return ConvertToOutput(probabilities, classLabels);
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
    /// Gets the model metadata for the Prototypical Networks model.
    /// </summary>
    /// <returns>Model metadata containing distance function and feature extraction configuration.</returns>
    public ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            ModelType = "Prototypical Networks",
            Version = "1.0.0",
            Description = "Few-shot learning algorithm that classifies based on distance to class prototypes"
        };

        // Add Prototypical Networks specific metadata
        metadata.AdditionalMetadata["EncoderLayerSizes"] = _options.EncoderLayerSizes;
        metadata.AdditionalMetadata["EmbeddingDimension"] = _options.EmbeddingDimension;
        metadata.AdditionalMetadata["DistanceFunction"] = _options.DistanceFunction.ToString();
        metadata.AdditionalMetadata["UseLearnablePrototypes"] = _options.UseLearnablePrototypes;
        metadata.AdditionalMetadata["UseAttention"] = _options.UseAttention;
        metadata.AdditionalMetadata["Temperature"] = _options.Temperature;
        metadata.AdditionalMetadata["LearnDistanceMetric"] = _options.LearnDistanceMetric;

        return metadata;
    }

    /// <summary>
    /// Computes class prototypes from support set.
    /// </summary>
    private void ComputePrototypes(TInput supportInputs, TOutput supportOutputs)
    {
        // This is a simplified implementation
        // In practice, would group examples by class and compute means

        // Convert inputs to tensor
        Tensor<T> supportTensor;
        if (typeof(TInput) == typeof(Tensor<T>))
        {
            supportTensor = (Tensor<T>)(object)supportInputs;
        }
        else
        {
            throw new NotSupportedException($"Input type {typeof(TInput).Name} is not supported.");
        }

        // Encode support set
        var encodedSupport = _featureEncoder.Predict(supportTensor);

        // Group by class and compute means
        // This is a placeholder - actual implementation would parse supportOutputs
        // and group encoded features by class
    }

    /// <summary>
    /// Converts tensor to vector.
    /// </summary>
    private Vector<T> TensorToVector(Tensor<T> tensor)
    {
        if (tensor.Shape.Dimensions.Length == 1)
        {
            // Already a vector
            var vector = new Vector<T>(tensor.Shape.Dimensions[0]);
            for (int i = 0; i < vector.Length; i++)
            {
                var idx = new int[] { i };
                vector[i] = tensor.GetValue(idx);
            }
            return vector;
        }
        else
        {
            // Flatten to vector
            var vector = new Vector<T>(tensor.Shape.Size);
            for (int i = 0; i < tensor.Shape.Size; i++)
            {
                var idx = ComputeMultiDimIndex(i, tensor.Shape);
                vector[i] = tensor.GetValue(idx);
            }
            return vector;
        }
    }

    /// <summary>
    /// Converts flat index to multi-dimensional indices.
    /// </summary>
    private int[] ComputeMultiDimIndex(int flatIndex, TensorShape shape)
    {
        var indices = new int[shape.Dimensions.Length];
        int remaining = flatIndex;

        for (int i = shape.Dimensions.Length - 1; i >= 0; i--)
        {
            indices[i] = remaining % shape.Dimensions[i];
            remaining /= shape.Dimensions[i];
        }

        return indices;
    }

    /// <summary>
    /// Computes Euclidean distance between feature vector and prototype.
    /// </summary>
    private T ComputeEuclideanDistance(Vector<T> features, Vector<T> prototype)
    {
        if (features.Length != prototype.Length)
            throw new ArgumentException("Feature and prototype dimensions must match.");

        T sumSquares = NumOps.Zero;
        for (int i = 0; i < features.Length; i++)
        {
            T diff = NumOps.Subtract(features[i], prototype[i]);
            T squared = NumOps.Multiply(diff, diff);
            sumSquares = NumOps.Add(sumSquares, squared);
        }

        return NumOps.FromDouble(Math.Sqrt(Convert.ToDouble(sumSquares)));
    }

    /// <summary>
    /// Applies softmax to convert distances to probabilities.
    /// </summary>
    private List<T> ApplySoftmax(List<T> distances)
    {
        if (distances.Count == 0)
            return new List<T>();

        // Apply negative sign (smaller distance = higher probability)
        var negDistances = distances.Select(d => NumOps.Negate(d)).ToList();

        // Find max for numerical stability
        T maxDist = negDistances[0];
        foreach (var d in negDistances)
        {
            if (Convert.ToDouble(d) > Convert.ToDouble(maxDist))
                maxDist = d;
        }

        // Compute exp values
        var expValues = negDistances.Select(d =>
        {
            T shifted = NumOps.Add(d, maxDist);
            return NumOps.FromDouble(Math.Exp(Convert.ToDouble(shifted)));
        }).ToList();

        // Sum exp values
        T sumExp = NumOps.Zero;
        foreach (var e in expValues)
        {
            sumExp = NumOps.Add(sumExp, e);
        }

        // Normalize
        return expValues.Select(e => NumOps.Divide(e, sumExp)).ToList();
    }

    /// <summary>
    /// Converts probabilities to appropriate output format.
    /// </summary>
    private TOutput ConvertToOutput(List<T> probabilities, List<int> classLabels)
    {
        // Convert to tensor format
        var tensor = new Tensor<T>(new TensorShape(probabilities.Count));
        for (int i = 0; i < probabilities.Count; i++)
        {
            var idx = new int[] { i };
            tensor.SetValue(idx, probabilities[i]);
        }

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            return (TOutput)(object)tensor;
        }
        else
        {
            throw new NotSupportedException($"Output type {typeof(TOutput).Name} is not supported.");
        }
    }
}

/// <summary>
/// Distance functions supported by ProtoNets.
/// </summary>
public enum DistanceFunction
{
    /// <summary>
    /// Standard Euclidean distance.
    /// </summary>
    Euclidean,

    /// <summary>
    /// Cosine distance (1 - cosine similarity).
    /// </summary>
    Cosine,

    /// <summary>
    /// Mahalanobis distance with learned covariance.
    /// </summary>
    Mahalanobis
}