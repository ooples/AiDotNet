using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions",
    "https://arxiv.org/abs/1812.03664",
    Year = 2020,
    Authors = "Han-Jia Ye, Hexiang Hu, De-Chuan Zhan, Fei Sha")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class FEATAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly FEATOptions<T, TInput, TOutput> _featOptions;

    /// <summary>
    /// Parameters for the set-to-set transformer that adapts prototypes.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
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
            options.InnerOptimizer,
            optimizerUse: MetaLearnerOptimizerUse.Outer)
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
        int dim = Math.Max(_featOptions.TransformerDim, 1);

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

        var initParams = ParamModel.GetParameters();
        T contrastiveWeight = NumOps.FromDouble(_featOptions.ContrastiveWeight);

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(initParams);

            // Extract features
            var supportPred = MetaModel.Predict(task.SupportInput);

            // Compute prototypes and adapt them
            var supportFeatures = ConvertToVector(supportPred);
            Vector<T>? adaptedPrototypes = null;
            if (supportFeatures != null)
            {
                adaptedPrototypes = AdaptPrototypes(supportFeatures);
            }

            // Apply adaptation-derived modulation to backbone before computing query loss
            // This connects the transformer adaptation to the training loss
            var modFactor = ComputeModulationFactor(supportFeatures, adaptedPrototypes);
            if (modFactor is not null)
            {
                var currentParams = ParamModel.GetParameters();
                ParamModel.SetParameters(ParameterModulation.Apply(Engine, currentParams, modFactor));
            }

            // Classification loss on modulated backbone (transformer affects loss)
            var queryPred = MetaModel.Predict(task.QueryInput);
            var queryLoss = ComputeLossFromOutput(queryPred, task.QueryOutput);

            // Contrastive loss: adapted prototypes should stay close to original
            // L_contrastive = ||adapted - original||^2 / dim
            T contrastiveLoss = NumOps.Zero;
            if (_featOptions.ContrastiveWeight > 0 && adaptedPrototypes != null && supportFeatures != null)
            {
                int minLen = Math.Min(supportFeatures.Length, adaptedPrototypes.Length);
                if (minLen > 0)
                {
                    var diff = Engine.Subtract(
                        adaptedPrototypes.GetSubVector(0, minLen), supportFeatures.GetSubVector(0, minLen));
                    contrastiveLoss = NumOps.Divide(Engine.DotProduct(diff, diff), NumOps.FromDouble(minLen));
                }
            }

            // Combined loss = classification + contrastive
            T totalLoss = NumOps.Add(queryLoss,
                NumOps.Multiply(contrastiveWeight, contrastiveLoss));
            losses.Add(totalLoss);

            // Compute meta-gradients
            var metaGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
            metaGradients.Add(ClipGradients(metaGrad));
        }

        // Update backbone
        ApplyOuterUpdate(initParams, metaGradients, _featOptions.OuterLearningRate);

        // Update transformer parameters via multi-sample SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _transformerParams, _featOptions.OuterLearningRate, ComputeAuxLoss);

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
        var currentParams = ParamModel.GetParameters();
        var supportPred = MetaModel.Predict(task.SupportInput);
        var supportFeatures = ConvertToVector(supportPred);
        var adaptedPrototypes = supportFeatures != null ? AdaptPrototypes(supportFeatures) : null;

        // Compute modulation factors from adapted vs raw prototypes
        Vector<T>? modulationFactors = ComputeModulationFactor(supportFeatures, adaptedPrototypes);

        return new FEATModel<T, TInput, TOutput>(
            MetaModel, currentParams, adaptedPrototypes, _featOptions.Temperature, modulationFactors);
    }

    /// <summary>
    /// Computes the average loss over a task batch using the transformer-adapted prototypes.
    /// Called by SPSA to measure how perturbed transformer params affect loss.
    /// </summary>
    private double ComputeAuxLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var initParams = ParamModel.GetParameters();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(initParams);

            var supportPred = MetaModel.Predict(task.SupportInput);
            var supportFeatures = ConvertToVector(supportPred);
            Vector<T>? adaptedPrototypes = null;
            if (supportFeatures != null)
                adaptedPrototypes = AdaptPrototypes(supportFeatures);

            var auxModFactor = ComputeModulationFactor(supportFeatures, adaptedPrototypes);
            if (auxModFactor is not null)
            {
                var currentParams = ParamModel.GetParameters();
                ParamModel.SetParameters(ParameterModulation.Apply(Engine, currentParams, auxModFactor));
            }

            var queryPred = MetaModel.Predict(task.QueryInput);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(queryPred, task.QueryOutput));
        }

        ParamModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }

    /// <summary>
    /// Computes a scalar modulation factor by comparing adapted vs raw feature vectors.
    /// Returns the clamped average ratio of adapted/raw values, or null if no valid ratios.
    /// </summary>
    private Vector<T>? ComputeModulationFactor(Vector<T>? rawFeatures, Vector<T>? adaptedFeatures)
    {
        if (rawFeatures == null || adaptedFeatures == null) return null;
        return ParameterModulation.MeanClampedRatio(Engine, rawFeatures, adaptedFeatures, 0.5, 2.0, 1e-10);
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
    internal Vector<T> AdaptPrototypes(Vector<T> prototypes)
    {
        // A stack of zero set-to-set layers is the identity map: the prototypes come back unadapted.
        int n = prototypes.Length;
        var adapted = prototypes;
        if (n == 0 || _featOptions.NumTransformerLayers <= 0)
        {
            return adapted;
        }

        // Attention weights are read from the transformer parameters in (layer, i, j) order; once they
        // run out, every remaining pair uses the constant weight 0.01.
        int paramIdx = 0;
        int pairCount = n * n;
        T invSqrtN = NumOps.FromDouble(1.0 / Math.Sqrt(n));
        T maxScore = NumOps.FromDouble(10.0);
        T normCount = NumOps.FromDouble(n);
        T layerNormEps = NumOps.FromDouble(1e-5);
        T fallbackWeight = NumOps.FromDouble(0.01);
        T minTotalWeight = NumOps.FromDouble(1e-10);

        for (int layer = 0; layer < _featOptions.NumTransformerLayers; layer++)
        {
            // Frozen layer input (avoid in-place mutation)
            var x = Tensor<T>.FromVector(prototypes);

            int available = Math.Max(0, Math.Min(pairCount, _transformerParams.Length - paramIdx));
            var layerWeights = available == pairCount
                ? _transformerParams.GetSubVector(paramIdx, pairCount)
                : Vector<T>.Concatenate(
                    available > 0 ? _transformerParams.GetSubVector(paramIdx, available) : new Vector<T>(0),
                    Engine.Fill(pairCount - available, fallbackWeight));
            paramIdx += available;

            // score[i, j] = x_i * x_j * w_ij / sqrt(n); weight = exp(min(score, 10))
            var pairWeights = Engine.Reshape(Tensor<T>.FromVector(layerWeights), new[] { n, n });
            var scores = Engine.TensorMultiplyScalar(
                Engine.TensorMultiply(Engine.TensorOuter(x, x), pairWeights), invSqrtN);
            var weights = Engine.TensorExp(Engine.TensorMin(scores, maxScore));
            var totalWeight = Engine.ReduceSum(weights, new[] { 1 }, false);
            var weightedSum = Engine.Reshape(
                Engine.TensorMatMul(weights, Engine.Reshape(x, new[] { n, 1 })), new[] { n });

            // Rows whose total weight underflows keep their input value.
            var valid = Engine.TensorGreaterThan(totalWeight, minTotalWeight);
            var safeTotal = Engine.TensorWhere(valid, totalWeight, Engine.TensorOnesLike(totalWeight));
            var attended = Engine.TensorWhere(valid, Engine.TensorDivide(weightedSum, safeTotal), x);

            // Residual connection (uses frozen layer input)
            var residual = Engine.TensorAdd(attended, x).ToVector();

            // Layer normalization (population variance, eps = 1e-5)
            T mean = NumOps.Divide(Engine.Sum(residual), normCount);
            var centered = Engine.Subtract(residual, Engine.Fill(n, mean));
            T variance = NumOps.Divide(Engine.DotProduct(centered, centered), normCount);
            T std = NumOps.Sqrt(NumOps.Add(variance, layerNormEps));
            adapted = Engine.Divide(centered, std);

            prototypes = adapted;
        }

        return adapted;
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
/// <para><b>Thread Safety:</b> This class is NOT thread-safe. Predict mutates the shared
/// model's parameters via SetParameters. Callers must ensure single-threaded access.</para>
/// </remarks>
internal class FEATModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>, IAdaptedMetaModel<T>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _backboneParams;
    private readonly Vector<T>? _adaptedPrototypes;
    private readonly double _temperature;
    private readonly Vector<T>? _modulationFactors;

    /// <inheritdoc/>
    public Vector<T>? AdaptedSupportFeatures => _adaptedPrototypes;

    /// <inheritdoc/>
    public Vector<T>? ParameterModulationFactors => _modulationFactors;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    public FEATModel(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> backboneParams,
        Vector<T>? adaptedPrototypes,
        double temperature,
        Vector<T>? modulationFactors)
    {
        // Predict pushes the adapted parameters into this model, so it is a copy: holding the meta-learner's
        // MetaModel would let every prediction overwrite the meta-model.
        _model = (model ?? throw new ArgumentNullException(nameof(model))).Clone();
        _backboneParams = backboneParams;
        _adaptedPrototypes = adaptedPrototypes;
        _temperature = temperature;
        _modulationFactors = modulationFactors;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        if (_modulationFactors != null && _modulationFactors.Length > 0)
        {
            var modulated = ParameterModulation.Apply(AiDotNetEngine.Current, _backboneParams, _modulationFactors);
            InterfaceGuard.Parameterizable(_model).SetParameters(modulated);
        }
        else
        {
            InterfaceGuard.Parameterizable(_model).SetParameters(_backboneParams);
        }
        return _model.Predict(input);
    }

    /// <summary>Training not supported on adapted models.</summary>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException("Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
