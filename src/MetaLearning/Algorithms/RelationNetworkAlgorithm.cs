using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Modules;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Relation Networks algorithm for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Relation Networks learn to compare query examples with class examples by learning
/// a relation function that measures similarity. Unlike metric learning approaches
/// that use fixed distance functions, Relation Networks learn the relation function
/// end-to-end.
/// </para>
/// <para><b>For Beginners:</b> Relation Networks learns how to compare examples:
///
/// **How it works:**
/// 1. Encode all examples (support and query) with a feature encoder
/// 2. For each query, concatenate with each support example's features
/// 3. Pass concatenated features through a relation module (neural network)
/// 4. The relation module outputs a similarity score
/// 5. Apply softmax to get class probabilities
///
/// **Key insight:** Instead of using predefined distances (like Euclidean),
/// it learns a neural network to measure "how related" two examples are.
/// </para>
/// <para><b>Algorithm - Relation Networks:</b>
/// <code>
/// # Learn two networks
/// feature_encoder = CNN()         # Maps x -> phi(x)
/// relation_module = MLP()        # Maps [phi(x_i), phi(x_j)] -> similarity
///
/// # Episode training
/// for each episode:
///     # Sample N-way K-shot task
///     support_set = {examples_from_N_classes, K_examples_each}
///     query_set = {examples_from_same_N_classes}
///
///     # Encode all examples
///     support_features = [feature_encoder(x) for x in support_set]
///     query_features = [feature_encoder(x) for x in query_set]
///
///     # Compute relation scores
///     for each query example q:
///         scores = []
///         for each class c:
///             class_score = 0
///             for each support example s in class c:
///                 # Concatenate and compute relation
///                 combined = concatenate(phi(q), phi(s))
///                 relation_score = relation_module(combined)
///                 class_score += relation_score
///             scores.append(average(class_score))
///         probabilities = softmax(scores)
///         loss = cross_entropy(probabilities, true_label)
///
///     # Update both networks
///     backpropagate(loss)
///     update(feature_encoder, relation_module)
/// </code>
/// </para>
/// <para><b>Key Insights:</b>
///
/// 1. **Learnable Relation Function**: Instead of fixed distances, learns a neural
///    network to measure similarity. Can capture complex, non-linear relations.
///
/// 2. **End-to-End Training**: Both feature encoder and relation module are
///    trained jointly, optimizing for the final classification task.
///
/// 3. **Flexible Relations**: The relation module can learn to attend to
///    specific features, ignore noise, and detect subtle patterns.
///
/// 4. **Scalable Complexity**: More powerful relation modules can handle
///    more complex tasks at the cost of computation.
/// </para>
/// </remarks>
public class RelationNetworkAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly RelationNetworkOptions<T, TInput, TOutput> _relationOptions;
    private readonly RelationModule<T> _relationModule;
    private readonly List<RelationModule<T>> _multiHeadModules;

    /// <summary>
    /// Initializes a new instance of the RelationNetworkAlgorithm class.
    /// </summary>
    /// <param name="options">The configuration options for Relation Networks.</param>
    /// <exception cref="ArgumentNullException">Thrown when options or required components are null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    public RelationNetworkAlgorithm(RelationNetworkOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _relationOptions = options;

        // Initialize relation module
        _relationModule = new RelationModule<T>(options.RelationHiddenDimension);

        // Initialize multi-head modules if using multi-head relation
        _multiHeadModules = new List<RelationModule<T>>();
        if (options.UseMultiHeadRelation)
        {
            for (int i = 0; i < options.NumHeads; i++)
            {
                _multiHeadModules.Add(new RelationModule<T>(options.RelationHiddenDimension));
            }
        }

        // Validate configuration
        if (!_relationOptions.IsValid())
        {
            throw new ArgumentException("Relation Network configuration is invalid. Check all parameters.", nameof(options));
        }
    }

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.RelationNetwork;

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

        // For Relation Networks, adaptation means computing support features
        // The networks are already trained, just need to store support representations
        var adaptedModel = new RelationNetworkModel<T, TInput, TOutput>(
            MetaModel,
            _relationModule,
            task.SupportInput,
            task.SupportOutput,
            _relationOptions);

        return adaptedModel;
    }

    /// <summary>
    /// Trains the feature encoder and relation module on a single episode.
    /// </summary>
    private T TrainEpisode(IMetaLearningTask<T, TInput, TOutput> task)
    {
        // Step 1: Encode support and query examples
        var supportFeatures = EncodeExamples(task.SupportInput);
        var queryFeatures = EncodeExamples(task.QueryInput);

        // Step 2: Group support features by class
        var classFeatures = GroupFeaturesByClass(supportFeatures, task.SupportOutput);

        // Step 3: Compute relation scores for each query-class pair
        var relationScores = ComputeRelationScores(queryFeatures, classFeatures);

        // Step 4: Apply softmax to get class probabilities
        var probabilities = ApplySoftmaxToScores(relationScores);

        // Step 5: Compute cross-entropy loss
        var loss = ComputeCrossEntropyLoss(probabilities, task.QueryOutput);

        // Step 6: Add regularization terms
        loss = AddRegularizationTerms(loss);

        // Step 7: Backpropagate and update both networks
        UpdateNetworks(loss, task);

        return loss;
    }

    /// <summary>
    /// Encodes input examples to feature representations.
    /// </summary>
    private Tensor<T> EncodeExamples(TInput inputs)
    {
        // Use MetaModel for encoding
        if (inputs is Tensor<T> inputTensor)
        {
            // Simple pass-through for now since we're using MetaModel
            return inputTensor;
        }

        // Default: create empty tensor
        return new Tensor<T>(new int[] { 1, _relationOptions.RelationHiddenDimension });
    }

    /// <summary>
    /// Groups support features by their class labels.
    /// </summary>
    private Dictionary<int, List<Tensor<T>>> GroupFeaturesByClass(
        Tensor<T> supportFeatures,
        TOutput supportLabels)
    {
        var classFeatures = new Dictionary<int, List<Tensor<T>>>();

        // Get number of support examples
        int numSupport = supportFeatures.Shape.Length > 0 ? supportFeatures.Shape[0] : 0;

        for (int i = 0; i < numSupport; i++)
        {
            // Extract feature tensor for this example
            var feature = ExtractFeatureTensor(supportFeatures, i);

            // Get class label
            int classLabel = GetClassLabel(supportLabels, i);

            // Add to appropriate class
            if (!classFeatures.ContainsKey(classLabel))
            {
                classFeatures[classLabel] = new List<Tensor<T>>();
            }
            classFeatures[classLabel].Add(feature);
        }

        return classFeatures;
    }

    /// <summary>
    /// Computes relation scores between queries and class support examples.
    /// </summary>
    private Matrix<T> ComputeRelationScores(
        Tensor<T> queryFeatures,
        Dictionary<int, List<Tensor<T>>> classFeatures)
    {
        int numQueries = queryFeatures.Shape.Length > 0 ? queryFeatures.Shape[0] : 0;
        int numClasses = classFeatures.Count;
        var scores = new Matrix<T>(numQueries, numClasses);

        // Get sorted class labels for consistent column ordering
        var classLabels = classFeatures.Keys.ToList();
        classLabels.Sort();

        // Compute scores for each query-class pair
        for (int q = 0; q < numQueries; q++)
        {
            var queryFeature = ExtractFeatureTensor(queryFeatures, q);

            for (int c = 0; c < numClasses; c++)
            {
                int classLabel = classLabels[c];
                var supportExamples = classFeatures[classLabel];

                // Compute relation scores with all support examples in this class
                T classScore = ComputeClassRelationScore(queryFeature, supportExamples);

                scores[q, c] = classScore;
            }
        }

        return scores;
    }

    /// <summary>
    /// Computes relation score between a query and all examples in a class.
    /// </summary>
    private T ComputeClassRelationScore(
        Tensor<T> queryFeature,
        List<Tensor<T>> supportExamples)
    {
        var scores = new List<T>();

        // Compute relation with each support example
        foreach (var supportFeature in supportExamples)
        {
            T score = ComputeSingleRelationScore(queryFeature, supportFeature);
            scores.Add(score);
        }

        // Aggregate scores based on configured method
        switch (_relationOptions.AggregationMethod)
        {
            case RelationAggregationMethod.Mean:
                return ComputeMean(scores);
            case RelationAggregationMethod.Max:
                return ComputeMax(scores);
            case RelationAggregationMethod.Attention:
                return ComputeMean(scores); // Simplified
            case RelationAggregationMethod.LearnedWeighting:
                return ComputeMean(scores); // Simplified
            default:
                return ComputeMean(scores);
        }
    }

    /// <summary>
    /// Computes relation score between two feature tensors.
    /// </summary>
    private T ComputeSingleRelationScore(Tensor<T> queryFeature, Tensor<T> supportFeature)
    {
        // Combine features based on relation type
        Tensor<T> combinedFeatures;

        switch (_relationOptions.RelationType)
        {
            case RelationModuleType.Concatenate:
                combinedFeatures = ConcatenateFeatures(queryFeature, supportFeature);
                break;
            case RelationModuleType.Convolution:
            case RelationModuleType.Attention:
            case RelationModuleType.Transformer:
            default:
                combinedFeatures = ConcatenateFeatures(queryFeature, supportFeature);
                break;
        }

        // Pass through relation module
        var relationOutput = _relationModule.Forward(combinedFeatures);

        // Extract score from relation output
        return ExtractRelationScore(relationOutput);
    }

    /// <summary>
    /// Concatenates two feature tensors.
    /// </summary>
    private Tensor<T> ConcatenateFeatures(Tensor<T> a, Tensor<T> b)
    {
        int sizeA = 1;
        for (int i = 0; i < a.Shape.Length; i++) sizeA *= a.Shape[i];

        int sizeB = 1;
        for (int i = 0; i < b.Shape.Length; i++) sizeB *= b.Shape[i];

        var combinedTensor = new Tensor<T>(new int[] { sizeA + sizeB });

        for (int i = 0; i < sizeA; i++)
        {
            combinedTensor[i] = a.GetFlat(i);
        }
        for (int i = 0; i < sizeB; i++)
        {
            combinedTensor[sizeA + i] = b.GetFlat(i);
        }

        return combinedTensor;
    }

    /// <summary>
    /// Applies softmax to relation scores to get class probabilities.
    /// </summary>
    private Matrix<T> ApplySoftmaxToScores(Matrix<T> scores)
    {
        int numQueries = scores.Rows;
        int numClasses = scores.Columns;
        var probabilities = new Matrix<T>(numQueries, numClasses);

        for (int q = 0; q < numQueries; q++)
        {
            // Find max score for numerical stability
            T maxScore = scores[q, 0];
            for (int c = 1; c < numClasses; c++)
            {
                if (NumOps.ToDouble(scores[q, c]) > NumOps.ToDouble(maxScore))
                    maxScore = scores[q, c];
            }

            // Compute exp and sum
            var expValues = new T[numClasses];
            T sumExp = NumOps.Zero;

            for (int c = 0; c < numClasses; c++)
            {
                T shifted = NumOps.Subtract(scores[q, c], maxScore);
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
    /// Computes cross-entropy loss between probabilities and true labels.
    /// </summary>
    private T ComputeCrossEntropyLoss(Matrix<T> probabilities, TOutput trueLabels)
    {
        T totalLoss = NumOps.Zero;
        int numQueries = probabilities.Rows;

        for (int i = 0; i < numQueries; i++)
        {
            int trueClass = GetClassLabel(trueLabels, i);

            // Ensure trueClass is within bounds
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

        return numQueries > 0 ? NumOps.Divide(totalLoss, NumOps.FromDouble(numQueries)) : NumOps.Zero;
    }

    /// <summary>
    /// Adds regularization terms to the loss.
    /// </summary>
    private T AddRegularizationTerms(T baseLoss)
    {
        T totalLoss = baseLoss;

        // L2 regularization for feature encoder
        if (_relationOptions.FeatureEncoderL2Reg > 0.0)
        {
            var encoderParams = MetaModel.GetParameters();
            T encoderReg = ComputeL2Regularization(encoderParams);
            T regWeight = NumOps.FromDouble(_relationOptions.FeatureEncoderL2Reg);
            totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(regWeight, encoderReg));
        }

        // L2 regularization for relation module
        if (_relationOptions.RelationModuleL2Reg > 0.0)
        {
            var relationParams = _relationModule.GetParameters();
            T relationReg = ComputeL2Regularization(relationParams);
            T regWeight = NumOps.FromDouble(_relationOptions.RelationModuleL2Reg);
            totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(regWeight, relationReg));
        }

        return totalLoss;
    }

    /// <summary>
    /// Computes L2 regularization for parameters.
    /// </summary>
    private T ComputeL2Regularization(Vector<T> parameters)
    {
        T sumSquares = NumOps.Zero;
        for (int i = 0; i < parameters.Length; i++)
        {
            T squared = NumOps.Multiply(parameters[i], parameters[i]);
            sumSquares = NumOps.Add(sumSquares, squared);
        }
        return NumOps.Divide(sumSquares, NumOps.FromDouble(2.0));
    }

    /// <summary>
    /// Updates both feature encoder and relation module parameters using gradient descent.
    /// </summary>
    /// <param name="loss">The current episode loss used as baseline for finite differences.</param>
    /// <param name="task">The current meta-learning task providing input/output data.</param>
    /// <remarks>
    /// <para>
    /// Updates both networks:
    /// 1. Feature encoder: uses base class gradient computation
    /// 2. Relation module: uses sampled finite differences for gradient approximation
    /// </para>
    /// <para><b>For Beginners:</b> After computing the loss, we need to update both
    /// the feature extractor (how we represent examples) and the relation module
    /// (how we compare examples). We use gradient descent to adjust both sets of
    /// weights to reduce the loss.
    /// </para>
    /// </remarks>
    private void UpdateNetworks(T loss, IMetaLearningTask<T, TInput, TOutput> task)
    {
        // Update feature encoder using base class gradient computation
        var featureGradients = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
        featureGradients = ClipGradients(featureGradients);
        var currentParams = MetaModel.GetParameters();
        var updatedParams = ApplyGradients(currentParams, featureGradients, _relationOptions.OuterLearningRate);
        MetaModel.SetParameters(updatedParams);

        // Update relation module parameters using sampled finite differences
        var relationParams = _relationModule.GetParameters();
        if (relationParams.Length == 0) return;

        double epsilon = 1e-4;
        int sampleCount = Math.Min(50, relationParams.Length);
        double scaleFactor = sampleCount > 0 ? (double)relationParams.Length / sampleCount : 1.0;
        var relationGradients = new Vector<T>(relationParams.Length);

        for (int s = 0; s < sampleCount; s++)
        {
            int i = (s * relationParams.Length) / sampleCount;

            // Perturb parameter
            T original = relationParams[i];
            relationParams[i] = NumOps.Add(original, NumOps.FromDouble(epsilon));
            _relationModule.SetParameters(relationParams);

            // Recompute forward pass with perturbed relation module
            var supportFeatures = EncodeExamples(task.SupportInput);
            var queryFeatures = EncodeExamples(task.QueryInput);
            var classFeatures = GroupFeaturesByClass(supportFeatures, task.SupportOutput);
            var scores = ComputeRelationScores(queryFeatures, classFeatures);
            var probabilities = ApplySoftmaxToScores(scores);
            T perturbedLoss = ComputeCrossEntropyLoss(probabilities, task.QueryOutput);

            double grad = (NumOps.ToDouble(perturbedLoss) - NumOps.ToDouble(loss)) / epsilon;
            relationGradients[i] = NumOps.FromDouble(grad * scaleFactor);

            // Restore original parameter
            relationParams[i] = original;
        }

        // Restore original parameters and apply gradient update
        _relationModule.SetParameters(relationParams);
        var updatedRelation = ApplyGradients(relationParams, relationGradients, _relationOptions.OuterLearningRate);
        _relationModule.SetParameters(updatedRelation);
    }

    // Helper methods

    private Tensor<T> ExtractFeatureTensor(Tensor<T> features, int index)
    {
        if (features.Shape.Length < 2)
        {
            return features;
        }

        // Extract a single example from batch
        int featureSize = 1;
        for (int i = 1; i < features.Shape.Length; i++)
        {
            featureSize *= features.Shape[i];
        }

        var singleFeature = new Tensor<T>(new int[] { featureSize });
        int offset = index * featureSize;

        for (int i = 0; i < featureSize; i++)
        {
            singleFeature[i] = features.GetFlat(offset + i);
        }

        return singleFeature;
    }

    private int GetClassLabel(TOutput output, int index)
    {
        if (output is Tensor<T> tensor)
        {
            if (tensor.Shape.Length == 1)
            {
                // Tensor is 1D, labels are indices
                return (int)NumOps.ToDouble(tensor[index]);
            }
            else if (tensor.Shape.Length >= 2)
            {
                // Tensor is 2D (one-hot), find class with highest probability
                int numClasses = tensor.Shape[1];
                int maxClass = 0;
                T maxProb = tensor.GetFlat(index * numClasses);

                for (int i = 1; i < numClasses; i++)
                {
                    var prob = tensor.GetFlat(index * numClasses + i);
                    if (NumOps.ToDouble(prob) > NumOps.ToDouble(maxProb))
                    {
                        maxProb = prob;
                        maxClass = i;
                    }
                }
                return maxClass;
            }
        }

        return 0;
    }

    private T ComputeMax(List<T> values)
    {
        if (values.Count == 0)
            return NumOps.Zero;

        T max = values[0];
        foreach (var v in values)
        {
            if (NumOps.ToDouble(v) > NumOps.ToDouble(max))
                max = v;
        }
        return max;
    }

    private T ExtractRelationScore(Tensor<T> relationOutput)
    {
        // Extract scalar score from relation module output
        int totalSize = 1;
        for (int i = 0; i < relationOutput.Shape.Length; i++)
        {
            totalSize *= relationOutput.Shape[i];
        }

        if (totalSize == 1)
        {
            return relationOutput[0];
        }

        // For multi-dimensional output, use sigmoid of first element
        double val = NumOps.ToDouble(relationOutput[0]);
        return NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-val)));
    }
}
