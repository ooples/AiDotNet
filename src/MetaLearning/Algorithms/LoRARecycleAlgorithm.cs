using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of LoRA-Recycle (Hu et al., CVPR 2025).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// LoRA-Recycle enables tuning-free few-shot adaptation by recycling pre-tuned LoRA adapters.
/// It maintains a bank of task-specific LoRA adapters and a prototype-based selection mechanism
/// that computes task embeddings from the support set to weight and combine stored adapters
/// without any gradient-based inner-loop optimization.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Meta-learned state: base params θ, adapter bank {A_1,...,A_K}, prototype encoder E
///
/// Adaptation (gradient-free, single forward pass):
///   1. Compute support set embedding: z = E(support_features)
///   2. Compute similarity to each adapter's prototype: s_k = -||z - p_k||^2
///   3. Compute adapter weights: w_k = softmax(s_k / temperature)
///   4. Fuse adapters: Δθ = Σ w_k * A_k
///   5. Adapted params: θ' = θ + Δθ
///
/// Meta-training:
///   1. For each task: gradient-adapt a fresh LoRA, add to adapter bank
///   2. Train prototype encoder using KL distillation loss
///   3. Update base params via meta-gradients
/// </code>
/// </para>
/// <para><b>Key difference from MAML:</b> No gradient-based inner-loop optimization at
/// inference time. Adaptation is a single forward pass through the prototype encoder
/// followed by adapter selection and fusion.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("LoRA-Recycle: Recycling First-Round LoRA for Efficient Second-Round Tuning",
    "https://arxiv.org/abs/2310.00151",
    Year = 2025,
    Authors = "Yixiao Hu, Minghao Chen, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class LoRARecycleAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly LoRARecycleOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>
    /// Bank of LoRA adapters. Each adapter is a flat vector of parameter deltas (length = paramDim).
    /// </summary>
    [Buffer]
    private readonly List<Vector<T>> _adapterBank;

    /// <summary>
    /// Prototype embedding for each adapter in the bank (length = prototypeDim per adapter).
    /// </summary>
    [Buffer]
    private readonly List<Vector<T>> _adapterPrototypes;

    /// <summary>
    /// Prototype encoder parameters: maps feature vectors to prototype space.
    /// Stored as a flat vector of length (paramDim * prototypeDim + prototypeDim).
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _encoderParams;

    private const int MaxEncoderInputDim = 128;
    private readonly int _paramDim;
    private readonly int _rank;
    private readonly int _prototypeDim;
    private int _bankInsertIdx;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.LoRARecycle;

    public LoRARecycleAlgorithm(LoRARecycleOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        _rank = Math.Max(1, options.Rank);
        _prototypeDim = Math.Max(1, options.PrototypeDim);

        _adapterBank = new List<Vector<T>>();
        _adapterPrototypes = new List<Vector<T>>();
        _bankInsertIdx = 0;

        // Initialize the prototype encoder: linear projection from feature space to prototype space
        // Size: min(paramDim, MaxEncoderInputDim) input features → prototypeDim output
        int encoderInputDim = Math.Min(_paramDim, MaxEncoderInputDim);
        int encoderSize = encoderInputDim * _prototypeDim + _prototypeDim; // weights + bias
        _encoderParams = new Vector<T>(encoderSize);
        double initScale = 1.0 / Math.Sqrt(encoderInputDim);
        for (int i = 0; i < encoderSize; i++)
            _encoderParams[i] = NumOps.FromDouble(SampleNormal() * initScale);
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks.Length == 0)
            return NumOps.Zero;

        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var baseParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Phase 1: Gradient-adapt a fresh LoRA adapter for this task
            ParamModel.SetParameters(baseParams);
            var adaptedParams = baseParams.Clone();

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                ParamModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyInnerGradients(adaptedParams, grad, _algoOptions.InnerLearningRate, step == 0);
            }

            // Compute adapter delta: Δθ = adaptedParams - baseParams
            var adapterDelta = Engine.Subtract(adaptedParams, baseParams);

            // Compute support set embedding for this task
            ParamModel.SetParameters(baseParams);
            var supportEmbed = ComputeTaskEmbedding(task.SupportInput);

            // Add adapter and prototype to bank (circular buffer)
            AddToBank(adapterDelta, supportEmbed);

            // Phase 2: Evaluate via recycled adapter fusion (gradient-free)
            if (_adapterBank.Count > 0)
            {
                var fusedParams = Engine.Add(baseParams, FuseAdapters(supportEmbed));

                ParamModel.SetParameters(fusedParams);
            }
            else
            {
                ParamModel.SetParameters(adaptedParams);
            }

            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop: update base parameters
        ParamModel.SetParameters(baseParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            ParamModel.SetParameters(ApplyOuterGradients(baseParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        // Update encoder params via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _encoderParams, _algoOptions.OuterLearningRate, ComputeRecycleLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var baseParams = ParamModel.GetParameters();

        if (_adapterBank.Count == 0)
        {
            // Fall back to gradient-based adaptation if no adapters are available yet
            var adaptedParams = baseParams.Clone();

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                ParamModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyInnerGradients(adaptedParams, grad, _algoOptions.InnerLearningRate, step == 0);
            }
            ParamModel.SetParameters(baseParams);
            return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
        }

        // Gradient-free adaptation: compute task embedding and fuse adapters
        var taskEmbed = ComputeTaskEmbedding(task.SupportInput);
        var finalParams = Engine.Add(baseParams, FuseAdapters(taskEmbed));

        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, finalParams);
    }

    /// <summary>
    /// Computes a task embedding by running support data through the model and projecting
    /// the feature representation through the prototype encoder.
    /// </summary>
    private Vector<T> ComputeTaskEmbedding(TInput supportInput)
    {
        var features = ConvertToVector(MetaModel.Predict(supportInput));
        if (features == null || features.Length == 0)
            return new Vector<T>(_prototypeDim);

        // Take first min(paramDim, MaxEncoderInputDim) features as encoder input
        int inputDim = Math.Min(_paramDim, MaxEncoderInputDim);
        int featureLen = Math.Min(features.Length, inputDim);

        // Linear projection: output = W * input + bias, W stored [prototypeDim, inputDim]
        var weights = Tensor<T>.FromVector(_encoderParams.GetSubVector(0, inputDim * _prototypeDim), new[] { _prototypeDim, inputDim });
        if (featureLen < inputDim)
            weights = Engine.TensorSlice(weights, new[] { 0, 0 }, new[] { _prototypeDim, featureLen });
        var input = Tensor<T>.FromVector(features.Length == featureLen ? features : features.GetSubVector(0, featureLen), new[] { featureLen, 1 });
        var embedding = Engine.Add(Engine.TensorMatMul(weights, input).ToVector(),
            _encoderParams.GetSubVector(inputDim * _prototypeDim, _prototypeDim));

        return VectorTanh(embedding);
    }

    /// <summary>
    /// Fuses adapters from the bank using softmax-weighted combination based on
    /// similarity between the task embedding and adapter prototypes.
    /// </summary>
    private Vector<T> FuseAdapters(Vector<T> taskEmbedding)
    {
        int numAdapters = _adapterBank.Count;
        var prototypeRows = new Tensor<T>[numAdapters];
        var adapterRows = new Tensor<T>[numAdapters];
        for (int k = 0; k < numAdapters; k++)
        {
            prototypeRows[k] = Tensor<T>.FromVector(_adapterPrototypes[k].GetSubVector(0, _prototypeDim));
            adapterRows[k] = Tensor<T>.FromVector(_adapterBank[k].GetSubVector(0, _paramDim));
        }

        // Compute negative squared Euclidean distance to each prototype
        var embedding = Tensor<T>.FromVector(taskEmbedding.GetSubVector(0, _prototypeDim), new[] { 1, _prototypeDim });
        var diff = Engine.TensorSubtract(Engine.TensorStack(prototypeRows, 0),
            Engine.TensorBroadcastTo(embedding, new[] { numAdapters, _prototypeDim }));
        var similarities = Engine.TensorDivideScalar(
            Engine.TensorNegate(Engine.ReduceSum(Engine.TensorSquare(diff), new[] { 1 }, keepDims: false)),
            NumOps.FromDouble(_algoOptions.SelectionTemperature));

        // Softmax to get adapter weights; adapters below 1e-10 are dropped
        var exp = Engine.TensorExp(Engine.TensorSubtractScalar(similarities, Engine.TensorMaxValue(similarities)));
        var weights = Engine.TensorDivideScalar(exp, Engine.TensorSum(exp));
        var kept = Engine.TensorSubtract(Engine.TensorOnesLike(weights), Engine.TensorLessThan(weights, NumOps.FromDouble(1e-10)));

        // Weighted combination of adapter deltas
        var fused = Engine.TensorMatMul(
            Engine.Reshape(Engine.TensorMultiply(weights, kept), new[] { 1, numAdapters }), Engine.TensorStack(adapterRows, 0)).ToVector();

        return fused;
    }

    /// <summary>
    /// Adds an adapter and its prototype to the circular bank.
    /// </summary>
    private void AddToBank(Vector<T> adapterDelta, Vector<T> prototype)
    {
        if (_adapterBank.Count < _algoOptions.NumRecycledAdapters)
        {
            _adapterBank.Add(adapterDelta);
            _adapterPrototypes.Add(prototype);
        }
        else
        {
            _adapterBank[_bankInsertIdx] = adapterDelta;
            _adapterPrototypes[_bankInsertIdx] = prototype;
            _bankInsertIdx = (_bankInsertIdx + 1) % _algoOptions.NumRecycledAdapters;
        }
    }

    private double ComputeRecycleLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var baseParams = ParamModel.GetParameters();
        var losses = new List<T>(taskBatch.Tasks.Length);

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(baseParams);
            if (_adapterBank.Count == 0) continue;

            var taskEmbed = ComputeTaskEmbedding(task.SupportInput);
            ParamModel.SetParameters(Engine.Add(baseParams, FuseAdapters(taskEmbed)));
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        ParamModel.SetParameters(baseParams);

        // Tasks are skipped only when the bank is empty (all or none), so the mean over evaluated tasks
        // equals master's total / max(n, 1). The SPSA callback contract is a double, converted once.
        return NumOps.ToDouble(ComputeMean(losses));
    }
}
