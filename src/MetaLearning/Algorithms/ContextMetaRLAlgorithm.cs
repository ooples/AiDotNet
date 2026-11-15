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
/// Implementation of Context Meta-RL: context-conditioned meta-reinforcement learning
/// with multi-head attention-based aggregation.
/// </summary>
/// <remarks>
/// <para>
/// Context Meta-RL uses multi-head attention to aggregate task context from the support
/// set gradient history. A learned query vector attends over encoded gradient entries
/// to produce a context vector. This context vector multiplicatively modulates the
/// model parameters, providing a smooth, deterministic task conditioning mechanism.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Context entries: e_k = encode(grad_k) for each adaptation step k
/// Attention query: q (learned)
/// Keys: K = [e_1, ..., e_K]
/// Attention weights: α_k = softmax(q^T e_k / √d / temperature)
/// Context: c = Σ_k α_k * e_k
///
/// Parameter modulation: θ' = θ ⊙ (1 + strength * sigmoid(W_m * c))
///
/// Inner loop: gradient adaptation with modulated params
/// Outer loop: update θ, encoder, query q, modulation W_m
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("RL²: Fast Reinforcement Learning via Slow Reinforcement Learning",
    "https://arxiv.org/abs/1611.02779",
    Year = 2016,
    Authors = "Yan Duan, John Schulman, Xi Chen, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class ContextMetaRLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly ContextMetaRLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _contextDim;
    private readonly int _compressedDim;

    /// <summary>Context encoder: compressedDim → contextDim.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _encoderParams;

    /// <summary>Learned attention query vector: contextDim.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _queryVector;

    /// <summary>Modulation projection: contextDim → compressedDim.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _modulationParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ContextMetaRL;

    /// <summary>Maximum compressed gradient dimension for encoder input.</summary>
    private const int MaxCompressedDim = 64;

    private const double SpsaLearningRateMultiplier = 0.1;

    public ContextMetaRLAlgorithm(ContextMetaRLOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        if (options.ContextDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "ContextDim must be positive.");
        if (options.AttentionTemperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "AttentionTemperature must be positive.");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters. ContextMetaRL requires a model with at least one parameter.");
        _contextDim = options.ContextDim;
        _compressedDim = Math.Min(_paramDim, MaxCompressedDim);

        _encoderParams = new Vector<T>(_compressedDim * _contextDim);
        _queryVector = new Vector<T>(_contextDim);
        _modulationParams = new Vector<T>(_contextDim * _compressedDim);

        double scale = 1.0 / Math.Sqrt(_contextDim);
        for (int i = 0; i < _encoderParams.Length; i++)
            _encoderParams[i] = NumOps.FromDouble(scale * (RandomGenerator.NextDouble() - 0.5));
        for (int i = 0; i < _contextDim; i++)
            _queryVector[i] = NumOps.FromDouble(scale * (RandomGenerator.NextDouble() - 0.5));
        for (int i = 0; i < _modulationParams.Length; i++)
            _modulationParams[i] = NumOps.FromDouble(scale * (RandomGenerator.NextDouble() - 0.5));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = AdaptWithContext(task, initParams);

            ParamModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _encoderParams, _algoOptions.OuterLearningRate * SpsaLearningRateMultiplier, ComputeContextMetaRLLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _queryVector, _algoOptions.OuterLearningRate * SpsaLearningRateMultiplier, ComputeContextMetaRLLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _modulationParams, _algoOptions.OuterLearningRate * SpsaLearningRateMultiplier, ComputeContextMetaRLLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null) throw new ArgumentNullException(nameof(task));
        var initParams = ParamModel.GetParameters();
        var adaptedParams = AdaptWithContext(task, initParams);

        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// The context-modulated inner loop: each step encodes the clipped support gradient into the history,
    /// attends over it and sets theta = theta * (1 + strength * sigmoid(W c))[d mod compressedDim] - lr * g.
    /// </summary>
    private Vector<T> AdaptWithContext(IMetaLearningTask<T, TInput, TOutput> task, Vector<T> initParams)
    {
        var gradHistory = new List<Vector<T>>();
        var adaptedParams = initParams.Clone();
        T learningRate = NumOps.FromDouble(_algoOptions.InnerLearningRate);
        T strength = NumOps.FromDouble(_algoOptions.ModulationStrength);

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

            // Encode and store gradient
            gradHistory.Add(EncodeGradient(grad));

            // Attention-based context aggregation
            var contextVec = AttentionAggregate(gradHistory);

            // Multiplicative modulation: θ' = θ ⊙ (1 + strength * sigmoid(W*c))
            var factors = Engine.Add(Engine.Fill(_compressedDim, NumOps.One),
                Engine.Multiply(Engine.Sigmoid(ComputeModulation(contextVec)), strength));
            adaptedParams = Engine.Subtract(
                Engine.Multiply(adaptedParams, ParameterModulation.Cycle(Engine, factors, _paramDim)),
                Engine.Multiply(grad.GetSubVector(0, _paramDim), learningRate));
        }

        return adaptedParams;
    }

    /// <summary>tanh(W g[0..compressedDim)) with W stored [contextDim, compressedDim].</summary>
    private Vector<T> EncodeGradient(Vector<T> grad)
    {
        int used = Math.Min(_compressedDim, grad.Length);
        var weights = Tensor<T>.FromVector(_encoderParams, new[] { _contextDim, _compressedDim });
        if (used < _compressedDim)
            weights = Engine.TensorSlice(weights, new[] { 0, 0 }, new[] { _contextDim, used });
        var input = Tensor<T>.FromVector(used == grad.Length ? grad : grad.GetSubVector(0, used), new[] { used, 1 });
        return VectorTanh(Engine.TensorMatMul(weights, input).ToVector());
    }

    /// <summary>
    /// Multi-head attention aggregation: α_k = softmax(q^T e_k / √d / τ), normalized by (sum + 1e-10).
    /// </summary>
    private Vector<T> AttentionAggregate(List<Vector<T>> encodedGrads)
    {
        if (encodedGrads.Count == 0) return new Vector<T>(_contextDim);

        var rows = new Tensor<T>[encodedGrads.Count];
        for (int i = 0; i < encodedGrads.Count; i++) rows[i] = Tensor<T>.FromVector(encodedGrads[i]);
        var history = Engine.TensorStack(rows, 0);

        // Compute attention scores
        double scaleFactor = 1.0 / (Math.Sqrt(_contextDim) * _algoOptions.AttentionTemperature);
        var scores = Engine.TensorMultiplyScalar(
            Engine.TensorMatMul(history, Tensor<T>.FromVector(_queryVector, new[] { _contextDim, 1 })),
            NumOps.FromDouble(scaleFactor));

        // Softmax
        var exp = Engine.TensorExp(Engine.TensorSubtractScalar(scores, Engine.TensorMaxValue(scores)));
        var weights = Engine.TensorDivideScalar(exp, NumOps.Add(Engine.TensorSum(exp), NumOps.FromDouble(1e-10)));

        // Weighted sum
        return Engine.TensorMatMul(Engine.Reshape(weights, new[] { 1, encodedGrads.Count }), history).ToVector();
    }

    /// <summary>M c with M stored [compressedDim, contextDim].</summary>
    private Vector<T> ComputeModulation(Vector<T> contextVec)
    {
        var weights = Tensor<T>.FromVector(_modulationParams, new[] { _compressedDim, _contextDim });
        return Engine.TensorMatMul(weights, Tensor<T>.FromVector(contextVec, new[] { _contextDim, 1 })).ToVector();
    }

    private double ComputeContextMetaRLLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>(taskBatch.Tasks.Length);
        var initParams = ParamModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            var ap = AdaptWithContext(task, initParams);
            ParamModel.SetParameters(ap);
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        ParamModel.SetParameters(initParams);

        // The SPSA callback contract is a double; the batch mean is reduced in T and converted once.
        return NumOps.ToDouble(ComputeMean(losses));
    }
}
