using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Matching Networks for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Matching Networks use attention mechanisms over the support set to classify
/// query examples. It computes a weighted sum of support labels where weights are
/// determined by an attention function that measures similarity between examples.
/// </para>
/// <para><b>For Beginners:</b> Matching Networks learn to pay attention to similar examples:
///
/// **How it works:**
/// 1. Encode all examples (support and query) with a shared encoder
/// 2. For each query, compute attention weights with all support examples
/// 3. Use cosine similarity or learned attention for weights
/// 4. Predict weighted sum of support labels (soft nearest neighbor)
///
/// **Key insight:** The network learns how to compare examples during encoding,
/// making the similarity measure task-aware.
/// </para>
/// <para><b>Algorithm - Matching Networks:</b>
/// <code>
/// # Shared encoder that learns to produce comparable embeddings
/// encoder = NeuralNetwork()  # Maps x -> embedding(x)
/// attention_function = cosine_similarity or learned_attention
///
/// # Episode training
/// for each episode:
///     # Sample N-way K-shot task
///     support_set = {examples_from_N_classes, K_examples_each}
///     query_set = {examples_from_same_N_classes}
///
///     # Encode all examples
///     support_embeddings = encoder(support_set)
///     query_embeddings = encoder(query_set)
///
///     # For each query:
///     for each query q:
///         # Compute attention with all support examples
///         scores = [attention(embedding(q), embedding(s)) for s in support]
///         weights = softmax(scores)
///         prediction = sum(weight_i * label_i for support examples)
///
///     # Train with cross-entropy loss
/// </code>
/// </para>
/// <para><b>Key Insights:</b>
///
/// 1. **Task-Aware Embeddings**: The encoder learns to produce embeddings that
///    are meaningful for the specific classification task at hand.
///
/// 2. **Differentiable Attention**: The attention mechanism is fully differentiable,
///    allowing end-to-end training of the encoder.
///
/// 3. **No Adaptation Needed**: At test time, simply encode new examples and
///    apply the same attention mechanism - no gradient updates required.
/// </para>
/// <para>
/// Reference: Vinyals, O., Blundell, C., Lillicrap, T., Kavukcuoglu, K., &amp; Wierstra, D. (2016).
/// Matching Networks for One Shot Learning. NeurIPS.
/// </para>
/// </remarks>
public class MatchingNetworksAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MatchingNetworksOptions<T, TInput, TOutput> _matchingOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _matchingOptions;

    /// <summary>
    /// Initializes a new instance of the MatchingNetworksAlgorithm class.
    /// </summary>
    /// <param name="options">The configuration options for Matching Networks.</param>
    /// <exception cref="ArgumentNullException">Thrown when options or required components are null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a Matching Network ready for few-shot learning.
    ///
    /// <b>What Matching Networks need:</b>
    /// - <b>encoder:</b> Neural network that embeds examples into comparable space
    /// - <b>attentionFunction:</b> How to measure similarity (cosine, learned)
    /// - <b>useFullContext:</b> Whether to use all examples when encoding each example
    ///
    /// <b>What makes it different from ProtoNets:</b>
    /// - ProtoNets: Uses fixed distance (Euclidean) to class prototypes (mean)
    /// - Matching Nets: Uses learnable attention to individual support examples
    /// - No explicit class representatives - each example votes
    /// </para>
    /// </remarks>
    public MatchingNetworksAlgorithm(MatchingNetworksOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            null) // Matching Networks doesn't use inner optimizer
    {
        _matchingOptions = options;

        // Validate configuration
        if (!_matchingOptions.IsValid())
        {
            throw new ArgumentException("Matching Networks configuration is invalid. Check all parameters.", nameof(options));
        }
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.MatchingNetworks"/>.</value>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MatchingNetworks;

    /// <summary>
    /// Performs one meta-training step using Matching Networks' episodic training.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on.</param>
    /// <returns>The average meta-loss across all tasks in the batch.</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <remarks>
    /// <para>
    /// Matching Networks training computes attention-based predictions:
    /// </para>
    /// <para>
    /// <b>For each task in the batch:</b>
    /// 1. Encode all support and query examples
    /// 2. For each query, compute attention weights with all support examples
    /// 3. Predict weighted sum of support labels
    /// 4. Compute cross-entropy loss
    /// 5. Update encoder with averaged gradients
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
            if (_matchingOptions.GradientClipThreshold.HasValue && _matchingOptions.GradientClipThreshold.Value > 0)
            {
                accumulatedGradients = ClipGradients(accumulatedGradients, _matchingOptions.GradientClipThreshold.Value);
            }

            // Add L2 regularization if configured
            if (_matchingOptions.L2Regularization > 0.0)
            {
                var paramsVec = MetaModel.GetParameters();
                for (int i = 0; i < accumulatedGradients.Length && i < paramsVec.Length; i++)
                {
                    T regGrad = NumOps.Multiply(paramsVec[i], NumOps.FromDouble(2 * _matchingOptions.L2Regularization));
                    accumulatedGradients[i] = NumOps.Add(accumulatedGradients[i], regGrad);
                }
            }

            // Update encoder parameters
            var currentParams = MetaModel.GetParameters();
            var updatedParams = ApplyGradients(currentParams, accumulatedGradients, _matchingOptions.OuterLearningRate);
            MetaModel.SetParameters(updatedParams);
        }

        // Return average loss
        return NumOps.Divide(totalLoss, NumOps.FromDouble(taskBatch.BatchSize));
    }

    /// <summary>
    /// Adapts to a new task by caching support embeddings.
    /// </summary>
    /// <param name="task">The new task containing support set examples.</param>
    /// <returns>A MatchingNetworksModel that classifies using attention over support examples.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para>
    /// Matching Networks adaptation is very fast - just encode support examples and cache them.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After meta-training, when you have a new task with labeled
    /// examples, call this method. The returned model can classify new examples by
    /// comparing them to all support examples using learned attention.
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        // Return adapted model with cached support embeddings
        return new MatchingNetworksModel<T, TInput, TOutput>(
            MetaModel,
            task.SupportInput,
            task.SupportOutput,
            _matchingOptions,
            NumOps);
    }

    /// <summary>
    /// Trains the encoder on a single episode.
    /// </summary>
    private (T loss, Vector<T> gradients) TrainEpisode(IMetaLearningTask<T, TInput, TOutput> task)
    {
        // Step 1: Encode support set
        var supportEmbeddings = EncodeExamples(task.SupportInput);

        // Step 2: Encode query set
        var queryEmbeddings = EncodeExamples(task.QueryInput);

        // Step 3: Compute predictions using attention
        var predictions = ComputePredictions(queryEmbeddings, supportEmbeddings, task.SupportOutput);

        // Step 4: Compute loss
        T episodeLoss = ComputeCrossEntropyLoss(predictions, task.QueryOutput);

        // Step 5: Compute gradients through the attention mechanism using finite differences
        // This ensures gradients reflect the attention-based predictions, not a separate forward pass
        var gradients = ComputeAttentionGradients(task, episodeLoss);

        return (episodeLoss, gradients);
    }

    /// <summary>
    /// Computes gradients with respect to the attention-based loss using finite differences.
    /// This ensures the gradients properly account for the attention mechanism.
    /// </summary>
    private Vector<T> ComputeAttentionGradients(IMetaLearningTask<T, TInput, TOutput> task, T currentLoss)
    {
        var parameters = MetaModel.GetParameters();
        var gradients = new Vector<T>(parameters.Length);
        double epsilon = 1e-5;
        double currentLossVal = NumOps.ToDouble(currentLoss);

        // Sample parameters for efficiency (scale for unbiased estimation)
        int sampleCount = Math.Min(parameters.Length, 100);
        double scaleFactor = (double)parameters.Length / sampleCount;

        for (int s = 0; s < sampleCount; s++)
        {
            int i = (int)(s * parameters.Length / (double)sampleCount);

            // Perturb parameter
            T original = parameters[i];
            parameters[i] = NumOps.Add(original, NumOps.FromDouble(epsilon));
            MetaModel.SetParameters(parameters);

            // Recompute loss through the attention mechanism
            var supportEmbeddings = EncodeExamples(task.SupportInput);
            var queryEmbeddings = EncodeExamples(task.QueryInput);
            var predictions = ComputePredictions(queryEmbeddings, supportEmbeddings, task.SupportOutput);
            T perturbedLoss = ComputeCrossEntropyLoss(predictions, task.QueryOutput);

            // Compute scaled gradient
            double grad = (NumOps.ToDouble(perturbedLoss) - currentLossVal) / epsilon;
            gradients[i] = NumOps.FromDouble(grad * scaleFactor);

            // Restore parameter
            parameters[i] = original;
        }

        // Restore original parameters
        MetaModel.SetParameters(parameters);

        return gradients;
    }

    /// <summary>
    /// Encodes input examples to feature space.
    /// </summary>
    private Matrix<T> EncodeExamples(TInput inputs)
    {
        // Get predictions from the model (which acts as the encoder)
        var encoded = MetaModel.Predict(inputs);

        // Convert to matrix format
        return ConvertToMatrix(encoded);
    }

    /// <summary>
    /// Converts output to matrix format.
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
            var result = new Matrix<T>(1, vector.Length);
            for (int j = 0; j < vector.Length; j++)
            {
                result[0, j] = vector[j];
            }
            return result;
        }

        throw new NotSupportedException($"Output type {typeof(TOutput).Name} cannot be converted to Matrix<T>.");
    }

    /// <summary>
    /// Converts tensor to matrix.
    /// </summary>
    private Matrix<T> TensorToMatrix(Tensor<T> tensor)
    {
        if (tensor.Shape.Length == 1)
        {
            var result = new Matrix<T>(1, tensor.Shape[0]);
            for (int j = 0; j < tensor.Shape[0]; j++)
            {
                result[0, j] = tensor[new int[] { j }];
            }
            return result;
        }
        else if (tensor.Shape.Length >= 2)
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

        return new Matrix<T>(1, 1);
    }

    /// <summary>
    /// Computes predictions using attention mechanism.
    /// </summary>
    private Matrix<T> ComputePredictions(Matrix<T> queryEmbeddings, Matrix<T> supportEmbeddings, TOutput supportLabels)
    {
        int numQueries = queryEmbeddings.Rows;
        int numSupport = supportEmbeddings.Rows;
        int numClasses = _matchingOptions.NumClasses;

        var predictions = new Matrix<T>(numQueries, numClasses);

        // Get support label matrix (one-hot encoded)
        var supportLabelMatrix = ConvertLabelsToOneHot(supportLabels, numSupport);

        for (int q = 0; q < numQueries; q++)
        {
            var queryEmbedding = GetRow(queryEmbeddings, q);

            // Compute attention weights with all support examples
            var attentionWeights = ComputeAttentionWeights(queryEmbedding, supportEmbeddings);

            // Weighted sum of support labels
            for (int c = 0; c < numClasses; c++)
            {
                T classScore = NumOps.Zero;
                for (int s = 0; s < numSupport; s++)
                {
                    T labelValue = supportLabelMatrix[s, c];
                    T weight = attentionWeights[s];
                    classScore = NumOps.Add(classScore, NumOps.Multiply(labelValue, weight));
                }
                predictions[q, c] = classScore;
            }
        }

        return predictions;
    }

    /// <summary>
    /// Computes attention weights between query and all support examples.
    /// </summary>
    private Vector<T> ComputeAttentionWeights(Vector<T> queryEmbedding, Matrix<T> supportEmbeddings)
    {
        int numSupport = supportEmbeddings.Rows;
        var weights = new Vector<T>(numSupport);

        for (int s = 0; s < numSupport; s++)
        {
            var supportEmbedding = GetRow(supportEmbeddings, s);

            T similarity = _matchingOptions.AttentionFunction switch
            {
                MatchingNetworksAttentionFunction.Cosine => ComputeCosineSimilarity(queryEmbedding, supportEmbedding),
                MatchingNetworksAttentionFunction.DotProduct => ComputeDotProduct(queryEmbedding, supportEmbedding),
                MatchingNetworksAttentionFunction.Euclidean => ComputeNegativeEuclideanDistance(queryEmbedding, supportEmbedding),
                _ => ComputeCosineSimilarity(queryEmbedding, supportEmbedding)
            };

            // Apply temperature scaling
            if (Math.Abs(_matchingOptions.Temperature - 1.0) >= 1e-10)
            {
                similarity = NumOps.Divide(similarity, NumOps.FromDouble(_matchingOptions.Temperature));
            }

            weights[s] = similarity;
        }

        // Apply softmax to get normalized attention weights
        return ApplySoftmax(weights);
    }

    /// <summary>
    /// Computes cosine similarity between two vectors.
    /// </summary>
    private T ComputeCosineSimilarity(Vector<T> a, Vector<T> b)
    {
        T dotProduct = NumOps.Zero;
        T normASq = NumOps.Zero;
        T normBSq = NumOps.Zero;

        int minLen = Math.Min(a.Length, b.Length);
        for (int i = 0; i < minLen; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(a[i], b[i]));
            normASq = NumOps.Add(normASq, NumOps.Multiply(a[i], a[i]));
            normBSq = NumOps.Add(normBSq, NumOps.Multiply(b[i], b[i]));
        }

        T normA = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(normASq)));
        T normB = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(normBSq)));

        T denominator = NumOps.Multiply(normA, normB);
        if (NumOps.ToDouble(denominator) < 1e-8)
        {
            return NumOps.Zero;
        }

        return NumOps.Divide(dotProduct, denominator);
    }

    /// <summary>
    /// Computes dot product between two vectors.
    /// </summary>
    private T ComputeDotProduct(Vector<T> a, Vector<T> b)
    {
        T dotProduct = NumOps.Zero;
        int minLen = Math.Min(a.Length, b.Length);
        for (int i = 0; i < minLen; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(a[i], b[i]));
        }
        return dotProduct;
    }

    /// <summary>
    /// Computes negative Euclidean distance (higher = more similar).
    /// </summary>
    private T ComputeNegativeEuclideanDistance(Vector<T> a, Vector<T> b)
    {
        T sumSquares = NumOps.Zero;
        int minLen = Math.Min(a.Length, b.Length);
        for (int i = 0; i < minLen; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(diff, diff));
        }
        T distance = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(sumSquares)));
        return NumOps.Negate(distance);
    }

    /// <summary>
    /// Applies softmax to a vector.
    /// </summary>
    private Vector<T> ApplySoftmax(Vector<T> values)
    {
        var result = new Vector<T>(values.Length);

        // Find max for numerical stability
        T maxVal = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (NumOps.ToDouble(values[i]) > NumOps.ToDouble(maxVal))
            {
                maxVal = values[i];
            }
        }

        // Compute exp values and sum
        T sumExp = NumOps.Zero;
        var expValues = new T[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            T shifted = NumOps.Subtract(values[i], maxVal);
            expValues[i] = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(shifted)));
            sumExp = NumOps.Add(sumExp, expValues[i]);
        }

        // Normalize
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = NumOps.Divide(expValues[i], sumExp);
        }

        return result;
    }

    /// <summary>
    /// Computes cross-entropy loss.
    /// </summary>
    private T ComputeCrossEntropyLoss(Matrix<T> predictions, TOutput trueLabels)
    {
        T totalLoss = NumOps.Zero;
        int numExamples = predictions.Rows;

        for (int i = 0; i < numExamples; i++)
        {
            int trueClass = GetClassLabel(trueLabels, i);
            if (trueClass >= 0 && trueClass < predictions.Columns)
            {
                T predictedProb = predictions[i, trueClass];
                predictedProb = NumOps.Add(predictedProb, NumOps.FromDouble(1e-8));
                T logProb = NumOps.FromDouble(Math.Log(NumOps.ToDouble(predictedProb)));
                totalLoss = NumOps.Subtract(totalLoss, logProb);
            }
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(numExamples));
    }

    /// <summary>
    /// Converts labels to one-hot encoded matrix.
    /// </summary>
    private Matrix<T> ConvertLabelsToOneHot(TOutput labels, int numExamples)
    {
        int numClasses = _matchingOptions.NumClasses;
        var matrix = new Matrix<T>(numExamples, numClasses);

        for (int i = 0; i < numExamples; i++)
        {
            int classLabel = GetClassLabel(labels, i);
            if (classLabel >= 0 && classLabel < numClasses)
            {
                matrix[i, classLabel] = NumOps.One;
            }
        }

        return matrix;
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
    /// Gets class label from output at specified index.
    /// </summary>
    private int GetClassLabel(TOutput output, int index)
    {
        if (output is Vector<T> vector)
        {
            if (index < vector.Length)
                return (int)NumOps.ToDouble(vector[index]);
            return 0;
        }

        if (output is Matrix<T> matrix)
        {
            if (matrix.Columns == 1)
                return (int)NumOps.ToDouble(matrix[index % matrix.Rows, 0]);

            // One-hot: find argmax
            int maxIdx = 0;
            T maxVal = matrix[index % matrix.Rows, 0];
            for (int c = 1; c < matrix.Columns; c++)
            {
                if (NumOps.ToDouble(matrix[index % matrix.Rows, c]) > NumOps.ToDouble(maxVal))
                {
                    maxVal = matrix[index % matrix.Rows, c];
                    maxIdx = c;
                }
            }
            return maxIdx;
        }

        if (output is Tensor<T> tensor)
        {
            if (tensor.Shape.Length >= 1 && index < tensor.Shape[0])
            {
                return (int)NumOps.ToDouble(tensor[new int[] { index }]);
            }
        }

        return 0;
    }
}

/// <summary>
/// Matching Networks model for inference.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This model encapsulates the Matching Networks inference mechanism with pre-computed
/// support embeddings. It is returned by <see cref="MatchingNetworksAlgorithm{T, TInput, TOutput}.Adapt"/>
/// and provides fast classification using attention over support examples.
/// </para>
/// </remarks>
public class MatchingNetworksModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _encoder;
    private readonly Matrix<T> _supportEmbeddings;
    private readonly Matrix<T> _supportLabelsOneHot;
    private readonly MatchingNetworksOptions<T, TInput, TOutput> _options;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the MatchingNetworksModel.
    /// </summary>
    public MatchingNetworksModel(
        IFullModel<T, TInput, TOutput> encoder,
        TInput supportInputs,
        TOutput supportLabels,
        MatchingNetworksOptions<T, TInput, TOutput> options,
        INumericOperations<T> numOps)
    {
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _numOps = numOps ?? throw new ArgumentNullException(nameof(numOps));

        // Pre-compute support embeddings
        var encodedOutput = _encoder.Predict(supportInputs);
        _supportEmbeddings = ConvertToMatrix(encodedOutput);

        // Pre-compute one-hot labels
        _supportLabelsOneHot = ConvertLabelsToOneHot(supportLabels, _supportEmbeddings.Rows);
    }

    /// <summary>Gets the model metadata.</summary>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Makes predictions using attention over support examples.
    /// </summary>
    public TOutput Predict(TInput input)
    {
        // Encode query
        var encodedOutput = _encoder.Predict(input);
        var queryEmbeddings = ConvertToMatrix(encodedOutput);

        int numQueries = queryEmbeddings.Rows;
        int numClasses = _options.NumClasses;

        var predictions = new Matrix<T>(numQueries, numClasses);

        for (int q = 0; q < numQueries; q++)
        {
            var queryEmbedding = GetRow(queryEmbeddings, q);
            var attentionWeights = ComputeAttentionWeights(queryEmbedding);

            // Weighted sum of support labels
            for (int c = 0; c < numClasses; c++)
            {
                T classScore = _numOps.Zero;
                for (int s = 0; s < _supportEmbeddings.Rows; s++)
                {
                    T labelValue = _supportLabelsOneHot[s, c];
                    T weight = attentionWeights[s];
                    classScore = _numOps.Add(classScore, _numOps.Multiply(labelValue, weight));
                }
                predictions[q, c] = classScore;
            }
        }

        return ConvertToOutput(predictions);
    }

    /// <summary>
    /// Training is not supported for inference models.
    /// </summary>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Use the Matching Networks algorithm to train.");
    }

    /// <summary>
    /// Parameter updates are not supported for inference models.
    /// </summary>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("Matching Networks parameters are updated during training.");
    }

    /// <summary>
    /// Gets encoder parameters.
    /// </summary>
    public Vector<T> GetParameters() => _encoder.GetParameters();

    /// <summary>
    /// Gets model metadata.
    /// </summary>
    public ModelMetadata<T> GetModelMetadata() => Metadata;

    private Vector<T> ComputeAttentionWeights(Vector<T> queryEmbedding)
    {
        int numSupport = _supportEmbeddings.Rows;
        var weights = new Vector<T>(numSupport);

        for (int s = 0; s < numSupport; s++)
        {
            var supportEmbedding = GetRow(_supportEmbeddings, s);

            // Respect the AttentionFunction option (consistent with algorithm's ComputeAttentionWeights)
            T similarity = _options.AttentionFunction switch
            {
                MatchingNetworksAttentionFunction.Cosine => ComputeCosineSimilarity(queryEmbedding, supportEmbedding),
                MatchingNetworksAttentionFunction.DotProduct => ComputeDotProduct(queryEmbedding, supportEmbedding),
                MatchingNetworksAttentionFunction.Euclidean => ComputeNegativeEuclideanDistance(queryEmbedding, supportEmbedding),
                _ => ComputeCosineSimilarity(queryEmbedding, supportEmbedding)
            };

            if (Math.Abs(_options.Temperature - 1.0) >= 1e-10)
            {
                similarity = _numOps.Divide(similarity, _numOps.FromDouble(_options.Temperature));
            }

            weights[s] = similarity;
        }

        return ApplySoftmax(weights);
    }

    private T ComputeDotProduct(Vector<T> a, Vector<T> b)
    {
        T dotProduct = _numOps.Zero;
        int minLen = Math.Min(a.Length, b.Length);
        for (int i = 0; i < minLen; i++)
        {
            dotProduct = _numOps.Add(dotProduct, _numOps.Multiply(a[i], b[i]));
        }
        return dotProduct;
    }

    private T ComputeNegativeEuclideanDistance(Vector<T> a, Vector<T> b)
    {
        T sumSq = _numOps.Zero;
        int minLen = Math.Min(a.Length, b.Length);
        for (int i = 0; i < minLen; i++)
        {
            T diff = _numOps.Subtract(a[i], b[i]);
            sumSq = _numOps.Add(sumSq, _numOps.Multiply(diff, diff));
        }
        // Return negative distance so larger (less negative) = more similar
        return _numOps.Negate(_numOps.Sqrt(sumSq));
    }

    private T ComputeCosineSimilarity(Vector<T> a, Vector<T> b)
    {
        T dotProduct = _numOps.Zero;
        T normASq = _numOps.Zero;
        T normBSq = _numOps.Zero;

        int minLen = Math.Min(a.Length, b.Length);
        for (int i = 0; i < minLen; i++)
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
            return _numOps.Zero;
        }

        return _numOps.Divide(dotProduct, denominator);
    }

    private Vector<T> ApplySoftmax(Vector<T> values)
    {
        var result = new Vector<T>(values.Length);

        T maxVal = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (_numOps.ToDouble(values[i]) > _numOps.ToDouble(maxVal))
            {
                maxVal = values[i];
            }
        }

        T sumExp = _numOps.Zero;
        var expValues = new T[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            T shifted = _numOps.Subtract(values[i], maxVal);
            expValues[i] = _numOps.FromDouble(Math.Exp(_numOps.ToDouble(shifted)));
            sumExp = _numOps.Add(sumExp, expValues[i]);
        }

        for (int i = 0; i < values.Length; i++)
        {
            result[i] = _numOps.Divide(expValues[i], sumExp);
        }

        return result;
    }

    private Matrix<T> ConvertToMatrix(TOutput output)
    {
        if (output is Matrix<T> matrix)
            return matrix;

        if (output is Tensor<T> tensor)
        {
            if (tensor.Shape.Length >= 2)
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
            else if (tensor.Shape.Length == 1)
            {
                var result = new Matrix<T>(1, tensor.Shape[0]);
                for (int j = 0; j < tensor.Shape[0]; j++)
                {
                    result[0, j] = tensor[new int[] { j }];
                }
                return result;
            }
        }

        if (output is Vector<T> vector)
        {
            var result = new Matrix<T>(1, vector.Length);
            for (int j = 0; j < vector.Length; j++)
            {
                result[0, j] = vector[j];
            }
            return result;
        }

        return new Matrix<T>(1, 1);
    }

    private Matrix<T> ConvertLabelsToOneHot(TOutput labels, int numExamples)
    {
        int numClasses = _options.NumClasses;
        var matrix = new Matrix<T>(numExamples, numClasses);

        for (int i = 0; i < numExamples; i++)
        {
            int classLabel = GetClassLabel(labels, i);
            if (classLabel >= 0 && classLabel < numClasses)
            {
                matrix[i, classLabel] = _numOps.One;
            }
        }

        return matrix;
    }

    private int GetClassLabel(TOutput output, int index)
    {
        if (output is Vector<T> vector && index < vector.Length)
            return (int)_numOps.ToDouble(vector[index]);

        if (output is Tensor<T> tensor && tensor.Shape.Length >= 1 && index < tensor.Shape[0])
            return (int)_numOps.ToDouble(tensor[new int[] { index }]);

        return 0;
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

    private TOutput ConvertToOutput(Matrix<T> predictions)
    {
        if (typeof(TOutput) == typeof(Matrix<T>))
        {
            return (TOutput)(object)predictions;
        }

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            var tensor = new Tensor<T>(new int[] { predictions.Rows, predictions.Columns });
            for (int i = 0; i < predictions.Rows; i++)
            {
                for (int j = 0; j < predictions.Columns; j++)
                {
                    tensor[new int[] { i, j }] = predictions[i, j];
                }
            }
            return (TOutput)(object)tensor;
        }

        if (typeof(TOutput) == typeof(Vector<T>) && predictions.Rows == 1)
        {
            var vector = new Vector<T>(predictions.Columns);
            for (int j = 0; j < predictions.Columns; j++)
            {
                vector[j] = predictions[0, j];
            }
            return (TOutput)(object)vector;
        }

        throw new NotSupportedException($"Output type {typeof(TOutput).Name} is not supported.");
    }
}
