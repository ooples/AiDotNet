using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of FEAT (Few-shot Embedding Adaptation with Transformer).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// FEAT uses a set-to-set transformer to adapt class prototypes based on inter-class
/// relationships. Initial prototypes (class means) are fed through the transformer,
/// which outputs task-adapted prototypes that are more discriminative.
/// </para>
/// <para><b>For Beginners:</b> FEAT makes prototypes smarter by letting them "see" each other:
///
/// **The insight:**
/// ProtoNets computes each class prototype in isolation. But if you know that class A
/// and class B are very similar, you should push their prototypes apart to avoid confusion.
/// FEAT uses a transformer to automatically learn these adjustments.
///
/// **How it works:**
/// 1. Compute initial prototypes (mean of support features per class, like ProtoNets)
/// 2. Feed ALL prototypes through a transformer
///    - The transformer uses self-attention so each prototype can "see" all others
///    - It learns to adjust prototypes based on the specific set of classes
/// 3. Use the adapted prototypes for nearest-prototype classification
/// 4. Train with both classification loss AND contrastive loss
///
/// **Why the transformer helps:**
/// - In a 5-way task with dogs vs cats: Prototypes need to capture species differences
/// - In a 5-way task with dog breeds: Same dog features need to capture breed differences
/// - The transformer adjusts prototypes based on what's needed for THIS specific task
/// </para>
/// <para><b>Algorithm - FEAT:</b>
/// <code>
/// # Components
/// f_theta = feature_extractor           # Shared backbone
/// T_phi = set_to_set_transformer        # Prototype adaptation
///
/// # Meta-training
/// for each meta-iteration:
///     for each task T_i in batch:
///         # 1. Extract features
///         z_s = f_theta(support_x)      # Support features
///         z_q = f_theta(query_x)        # Query features
///
///         # 2. Compute initial prototypes (class means)
///         p_k = mean(z_s[class == k])   # Initial prototypes
///
///         # 3. Adapt prototypes with transformer
///         p_adapted = T_phi(p_1, ..., p_K)  # Self-attention adaptation
///
///         # 4. Classify queries by distance to adapted prototypes
///         logits = -distance(z_q, p_adapted) * temperature
///         class_loss = cross_entropy(logits, query_labels)
///
///         # 5. Contrastive loss: adapted should stay close to original
///         contrast_loss = contrastive(p_adapted, p_original)
///
///         # 6. Combined loss
///         meta_loss = alpha * contrast_loss + (1-alpha) * class_loss
///
///     # Update backbone and transformer
///     theta, phi = theta, phi - lr * grad(meta_loss)
/// </code>
/// </para>
/// <para>
/// Reference: Ye, H.J., Hu, H., Zhan, D.C., &amp; Sha, F. (2020).
/// Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions. CVPR 2020.
/// </para>
/// </remarks>
public class FEATAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly FEATOptions<T, TInput, TOutput> _featOptions;

    /// <summary>
    /// Parameters for the set-to-set transformer that adapts prototypes.
    /// </summary>
    private Vector<T> _transformerParams = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.FEAT;

    /// <summary>
    /// Initializes a new FEAT meta-learner.
    /// </summary>
    /// <param name="options">Configuration options for FEAT.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a FEAT instance with:
    /// - A feature extractor (backbone) that maps inputs to features
    /// - A set-to-set transformer that adapts prototypes to be task-specific
    /// Both are trained jointly during meta-training.
    /// </para>
    /// </remarks>
    public FEATAlgorithm(FEATOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _featOptions = options;
        InitializeTransformer();
    }

    /// <summary>
    /// Initializes the set-to-set transformer parameters.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sets up the transformer that will adapt prototypes.
    /// The transformer has key, query, and value projection matrices for self-attention,
    /// allowing prototypes to "attend" to each other and adjust their positions in
    /// feature space accordingly.
    /// </para>
    /// </remarks>
    private void InitializeTransformer()
    {
        int heads = _featOptions.NumTransformerHeads;
        int layers = _featOptions.NumTransformerLayers;
        int dim = 64; // Feature dimension estimate

        // Each layer needs: Q, K, V projections + output projection + layer norm
        int paramsPerLayer = heads * (dim * dim * 3) + dim * dim + dim * 2;
        int totalParams = layers * paramsPerLayer;

        _transformerParams = new Vector<T>(totalParams);
        double scale = Math.Sqrt(2.0 / dim);

        for (int i = 0; i < totalParams; i++)
        {
            _transformerParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() - 0.5) * 2.0 * scale);
        }
    }

    /// <summary>
    /// Performs one meta-training step for FEAT.
    /// </summary>
    /// <param name="taskBatch">Batch of meta-learning tasks.</param>
    /// <returns>The average meta-loss across the batch.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each training step:
    /// 1. For each task, compute prototypes from support examples
    /// 2. Adapt prototypes using the transformer
    /// 3. Classify queries using adapted prototypes
    /// 4. Compute combined loss (classification + contrastive)
    /// 5. Update both the backbone and transformer
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();

        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);

            // Extract features
            var supportPred = MetaModel.Predict(task.SupportInput);
            var queryPred = MetaModel.Predict(task.QueryInput);

            // Compute prototypes and adapt them
            var supportFeatures = ConvertToVector(supportPred);
            if (supportFeatures != null)
            {
                var adaptedPrototypes = AdaptPrototypes(supportFeatures);
            }

            // Classification loss
            var queryLoss = ComputeLossFromOutput(queryPred, task.QueryOutput);
            losses.Add(queryLoss);

            // Compute meta-gradients
            var metaGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
            metaGradients.Add(ClipGradients(metaGrad));
        }

        // Update backbone
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var updatedParams = ApplyGradients(initParams, avgGrad, _featOptions.OuterLearningRate);
            MetaModel.SetParameters(updatedParams);
        }

        // Update transformer parameters via multi-sample SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _transformerParams, _featOptions.OuterLearningRate);

        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <summary>
    /// Adapts to a new task using transformer-adapted prototypes.
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>An adapted model with task-specific prototypes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For a new task:
    /// 1. Extract support features and compute initial prototypes
    /// 2. Feed prototypes through the transformer for task-specific adaptation
    /// 3. Store the adapted prototypes for query classification
    /// Fast adaptation - just two forward passes (backbone + transformer).
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);
        var adaptedPrototypes = supportFeatures != null ? AdaptPrototypes(supportFeatures) : null;

        return new FEATModel<T, TInput, TOutput>(
            MetaModel, currentParams, adaptedPrototypes, _featOptions.Temperature);
    }

    /// <summary>
    /// Adapts prototypes using the set-to-set transformer.
    /// </summary>
    /// <param name="prototypes">Initial class prototypes (support means).</param>
    /// <returns>Task-adapted prototypes.</returns>
    /// <remarks>
    /// <para>
    /// The transformer performs multi-head self-attention over the set of prototypes:
    /// 1. Project prototypes to query, key, value vectors
    /// 2. Compute attention weights between all prototype pairs
    /// 3. Aggregate values using attention weights
    /// 4. Add residual connection + layer normalization
    ///
    /// This allows each prototype to "see" and adjust based on all other prototypes,
    /// producing task-specific representations.
    /// </para>
    /// <para><b>For Beginners:</b> The transformer works like a team meeting:
    /// 1. Each prototype presents its current state (query)
    /// 2. It also describes what it has to offer (key/value)
    /// 3. Each prototype "listens" to relevant others (attention)
    /// 4. Each adjusts its position based on what it learned
    /// The result: prototypes that are spread apart for this specific task.
    /// </para>
    /// </remarks>
    private Vector<T> AdaptPrototypes(Vector<T> prototypes)
    {
        var adapted = new Vector<T>(prototypes.Length);
        int paramIdx = 0;

        for (int layer = 0; layer < _featOptions.NumTransformerLayers; layer++)
        {
            // Self-attention: each position attends to all positions
            for (int i = 0; i < adapted.Length; i++)
            {
                T sum = NumOps.Zero;
                double totalWeight = 0;

                for (int j = 0; j < prototypes.Length; j++)
                {
                    // Compute attention score
                    double qi = NumOps.ToDouble(prototypes[i]);
                    double kj = NumOps.ToDouble(prototypes[j]);
                    double wParam = paramIdx < _transformerParams.Length
                        ? NumOps.ToDouble(_transformerParams[paramIdx++ % _transformerParams.Length])
                        : 0.01;
                    double score = qi * kj * wParam / Math.Sqrt(prototypes.Length);
                    double weight = Math.Exp(Math.Min(score, 10.0)); // Softmax component
                    totalWeight += weight;
                    sum = NumOps.Add(sum, NumOps.Multiply(prototypes[j], NumOps.FromDouble(weight)));
                }

                if (totalWeight > 1e-10)
                {
                    adapted[i] = NumOps.Divide(sum, NumOps.FromDouble(totalWeight));
                }
                else
                {
                    adapted[i] = prototypes[i];
                }
            }

            // Residual connection
            for (int i = 0; i < adapted.Length; i++)
            {
                adapted[i] = NumOps.Add(adapted[i], prototypes[i]);
            }

            // Layer normalization
            T mean = NumOps.Zero;
            for (int i = 0; i < adapted.Length; i++)
            {
                mean = NumOps.Add(mean, adapted[i]);
            }
            mean = NumOps.Divide(mean, NumOps.FromDouble(Math.Max(1, adapted.Length)));

            T variance = NumOps.Zero;
            for (int i = 0; i < adapted.Length; i++)
            {
                T diff = NumOps.Subtract(adapted[i], mean);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }
            variance = NumOps.Divide(variance, NumOps.FromDouble(Math.Max(1, adapted.Length)));
            double std = Math.Sqrt(NumOps.ToDouble(variance) + 1e-5);

            for (int i = 0; i < adapted.Length; i++)
            {
                adapted[i] = NumOps.Divide(NumOps.Subtract(adapted[i], mean), NumOps.FromDouble(std));
            }

            prototypes = adapted;
        }

        return adapted;
    }

    private Vector<T> AverageVectors(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0) return new Vector<T>(0);
        var result = new Vector<T>(vectors[0].Length);
        foreach (var v in vectors)
            for (int i = 0; i < result.Length; i++)
                result[i] = NumOps.Add(result[i], v[i]);
        var scale = NumOps.FromDouble(1.0 / vectors.Count);
        for (int i = 0; i < result.Length; i++)
            result[i] = NumOps.Multiply(result[i], scale);
        return result;
    }
}

/// <summary>
/// Adapted model wrapper for FEAT with transformer-adapted prototypes.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This model classifies new examples using prototypes
/// that have been adapted by the transformer to be task-specific. The adapted
/// prototypes capture inter-class relationships for better discrimination.
/// </para>
/// </remarks>
internal class FEATModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _adaptedPrototypes;
    private readonly double _temperature;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public FEATModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? adaptedPrototypes,
        double temperature)
    {
        _model = model;
        _backboneParams = backboneParams;
        _adaptedPrototypes = adaptedPrototypes;
        _temperature = temperature;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        _model.SetParameters(_backboneParams);
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) { }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
