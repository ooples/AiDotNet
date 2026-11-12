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
/// Implementation of In-Context RL: meta-RL via in-context learning without
/// explicit gradient updates at test time.
/// </summary>
/// <remarks>
/// <para>
/// In-Context RL trains a model to adapt through its forward pass by conditioning on a
/// growing context buffer. The context buffer stores compressed representations of past
/// (input, prediction, loss) triplets. A context aggregator (learned attention-like
/// mechanism) combines the buffer entries into a context vector that modulates the
/// model's parameters. At test time, no gradients are needed — the model improves
/// purely by observing more support examples in its context.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Context buffer: B = [(h_1, l_1), ..., (h_t, l_t)]
///   h_i = hash(input_i, prediction_i), l_i = loss_i
///
/// Context aggregation: c = Σ_i softmax(-l_i / τ) * h_i  (loss-weighted average)
/// Parameter modulation: θ' = θ + W_c * c
///
/// Meta-training: build context sequentially
///   For step k:
///     Predict on support, record (h, loss) in buffer
///     Aggregate context → modulate params
///     Compute loss with modulated params
///
/// Outer loop: update θ, W_c, context encoder
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
[ResearchPaper("Transformers are Sample Efficient World Models",
    "https://arxiv.org/abs/2209.00588",
    Year = 2022,
    Authors = "Vincent Micheli, Eloi Alonso, Francois Fleuret")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class InContextRLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly InContextRLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _contextDim;
    private readonly int _compressedDim;

    /// <summary>Context encoder parameters: gradient → context entry.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _contextEncoderParams;

    /// <summary>Context-to-parameter modulation projection.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _modulationParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.InContextRL;

    public InContextRLAlgorithm(InContextRLOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        if (options.ContextEmbeddingDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "ContextEmbeddingDim must be positive.");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        _contextDim = options.ContextEmbeddingDim;
        _compressedDim = Math.Min(_paramDim, 64);

        // Context encoder: compressedDim → contextDim
        _contextEncoderParams = new Vector<T>(_compressedDim * _contextDim);
        for (int i = 0; i < _contextEncoderParams.Length; i++)
            _contextEncoderParams[i] = NumOps.FromDouble(0.01 * (RandomGenerator.NextDouble() - 0.5));

        // Modulation: contextDim → compressedDim
        _modulationParams = new Vector<T>(_contextDim * _compressedDim);
        for (int i = 0; i < _modulationParams.Length; i++)
            _modulationParams[i] = NumOps.FromDouble(0.01 * (RandomGenerator.NextDouble() - 0.5));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();
        T contextWeight = NumOps.FromDouble(_algoOptions.ContextPredictionWeight);

        foreach (var task in taskBatch.Tasks)
        {
            // Build context buffer sequentially
            var currentParams = AdaptInContext(task, initParams, out var contextLosses);

            ParamModel.SetParameters(currentParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Context prediction auxiliary loss
            var totalLoss = NumOps.Add(queryLoss, NumOps.Multiply(contextWeight, BufferVariance(contextLosses)));
            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _contextEncoderParams, _algoOptions.OuterLearningRate * 0.1, ComputeInContextLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _modulationParams, _algoOptions.OuterLearningRate * 0.1, ComputeInContextLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = ParamModel.GetParameters();
        var currentParams = AdaptInContext(task, initParams, out _);

        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, currentParams);
    }

    /// <summary>
    /// The in-context loop: each step records (support loss, encoded gradient) in a buffer of at most
    /// ContextBufferSize entries and sets theta = theta_0 + modulation(aggregated context), cycled over the
    /// parameters. <paramref name="contextLosses"/> receives the final buffer's losses.
    /// </summary>
    private Vector<T> AdaptInContext(IMetaLearningTask<T, TInput, TOutput> task, Vector<T> initParams, out List<T> contextLosses)
    {
        var contextEntries = new List<Vector<T>>();
        contextLosses = new List<T>();
        var currentParams = initParams.Clone();

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            // Predict and record context
            ParamModel.SetParameters(currentParams);
            T stepLoss = ComputeLossFromOutput(MetaModel.Predict(task.SupportInput), task.SupportOutput);

            // Encode context entry from gradient
            var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            contextEntries.Add(EncodeContextEntry(grad));
            contextLosses.Add(stepLoss);

            // Trim buffer to max size
            while (contextEntries.Count > _algoOptions.ContextBufferSize)
            {
                contextEntries.RemoveAt(0);
                contextLosses.RemoveAt(0);
            }

            // Aggregate context and modulate parameters
            var modulation = ComputeModulation(AggregateContext(contextEntries, contextLosses));
            currentParams = Engine.Add(initParams.GetSubVector(0, _paramDim), ParameterModulation.Cycle(Engine, modulation, _paramDim));
        }

        return currentParams;
    }

    /// <summary>Population variance of the buffered support losses; 0 for fewer than two.</summary>
    private T BufferVariance(List<T> contextLosses)
    {
        if (contextLosses.Count <= 1) return NumOps.Zero;

        var bufferLosses = new Vector<T>(contextLosses);
        T count = NumOps.FromDouble(bufferLosses.Length);
        var centered = Engine.Subtract(bufferLosses, Engine.Fill(bufferLosses.Length, NumOps.Divide(Engine.Sum(bufferLosses), count)));
        return NumOps.Divide(Engine.DotProduct(centered, centered), count);
    }

    /// <summary>tanh(W g[0..compressedDim)) with W stored [contextDim, compressedDim].</summary>
    private Vector<T> EncodeContextEntry(Vector<T> grad)
    {
        int used = Math.Min(_compressedDim, grad.Length);
        var weights = Tensor<T>.FromVector(_contextEncoderParams, new[] { _contextDim, _compressedDim });
        if (used < _compressedDim)
            weights = Engine.TensorSlice(weights, new[] { 0, 0 }, new[] { _contextDim, used });
        var input = Tensor<T>.FromVector(used == grad.Length ? grad : grad.GetSubVector(0, used), new[] { used, 1 });
        return VectorTanh(Engine.TensorMatMul(weights, input).ToVector());
    }

    /// <summary>
    /// Loss-weighted attention (lower loss, higher weight): w_i = exp(-l_i - max_j(-l_j)) / (sum + 1e-10),
    /// result = sum_i w_i e_i.
    /// </summary>
    private Vector<T> AggregateContext(List<Vector<T>> entries, List<T> entryLosses)
    {
        if (entries.Count == 0) return new Vector<T>(_contextDim);

        var negLoss = Tensor<T>.FromVector(Engine.Negate(new Vector<T>(entryLosses)));
        var exp = Engine.TensorExp(Engine.TensorSubtractScalar(negLoss, Engine.TensorMaxValue(negLoss)));
        var weights = Engine.TensorDivideScalar(exp, NumOps.Add(Engine.TensorSum(exp), NumOps.FromDouble(1e-10)));

        var rows = new Tensor<T>[entries.Count];
        for (int i = 0; i < entries.Count; i++) rows[i] = Tensor<T>.FromVector(entries[i]);
        return Engine.TensorMatMul(Engine.Reshape(weights, new[] { 1, entries.Count }), Engine.TensorStack(rows, 0)).ToVector();
    }

    /// <summary>M c with M stored [compressedDim, contextDim].</summary>
    private Vector<T> ComputeModulation(Vector<T> contextVec)
    {
        var weights = Tensor<T>.FromVector(_modulationParams, new[] { _compressedDim, _contextDim });
        return Engine.TensorMatMul(weights, Tensor<T>.FromVector(contextVec, new[] { _contextDim, 1 })).ToVector();
    }

    private double ComputeInContextLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>(taskBatch.Tasks.Length);
        var initParams = ParamModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            var cp = AdaptInContext(task, initParams, out _);
            ParamModel.SetParameters(cp);
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        ParamModel.SetParameters(initParams);

        // The SPSA callback contract is a double; the batch mean is reduced in T and converted once.
        return NumOps.ToDouble(ComputeMean(losses));
    }
}
