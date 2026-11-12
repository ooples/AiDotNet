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
/// Implementation of HyperNet Meta-RL: hypernetwork-based task-specific parameter generation.
/// </summary>
/// <remarks>
/// <para>
/// A hypernetwork takes a task embedding (computed from the support gradient) and generates
/// the full policy parameter vector in a single forward pass. The task embedding is computed
/// by compressing the initial support gradient through a learned encoder. The hypernetwork
/// is a 2-layer MLP that maps the embedding to a parameter delta vector.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Task embedding: e = encoder(compress(∇L_support))
/// Hypernetwork: Δθ = HyperNet(e) = W₂ * ReLU(W₁ * e)
/// Effective params: θ_task = θ_base + Δθ
///
/// Inner loop: fine-tune θ_task on support (optional)
/// Outer loop: update θ_base, encoder, HyperNet params
/// Regularization: ||Δθ||² to prevent large parameter shifts
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
[ResearchPaper("HyperNetworks",
    "https://arxiv.org/abs/1609.09106",
    Year = 2016,
    Authors = "David Ha, Andrew Dai, Quoc V. Le")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class HyperNetMetaRLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly HyperNetMetaRLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _embDim;
    private readonly int _hiddenDim;
    private readonly int _compressedDim;

    private const int MaxCompressedDim = 128;

    /// <summary>Task encoder: compressedDim → embDim.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _encoderParams;

    /// <summary>Hypernetwork layer 1: embDim → hiddenDim.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _hyperLayer1;

    /// <summary>Hypernetwork layer 2: hiddenDim → compressedDim.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _hyperLayer2;

    /// <summary>SPSA learning rate multiplier for auxiliary parameter updates.</summary>
    private const double SpsaLearningRateMultiplier = 0.1;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.HyperNetMetaRL;

    public HyperNetMetaRLAlgorithm(HyperNetMetaRLOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        if (options.TaskEmbeddingDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "TaskEmbeddingDim must be positive.");
        if (options.HyperNetHiddenDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "HyperNetHiddenDim must be positive.");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        _embDim = options.TaskEmbeddingDim;
        _hiddenDim = options.HyperNetHiddenDim;
        _compressedDim = Math.Min(_paramDim, MaxCompressedDim);

        _encoderParams = InitRandom(_compressedDim * _embDim);
        _hyperLayer1 = InitRandom(_embDim * _hiddenDim);
        _hyperLayer2 = InitRandom(_hiddenDim * _compressedDim);
    }

    private Vector<T> InitRandom(int size)
    {
        var v = new Vector<T>(size);
        double scale = 1.0 / Math.Sqrt(size);
        for (int i = 0; i < size; i++)
            v[i] = NumOps.FromDouble(scale * SampleNormal());
        return v;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks.Length == 0) return NumOps.Zero;

        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Compute task embedding from support gradient
            ParamModel.SetParameters(initParams);
            var supportGrad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            var embedding = ComputeTaskEmbedding(supportGrad);

            // Generate parameter delta via hypernetwork
            var paramDelta = HyperNetForward(embedding);

            // Apply parameter delta
            var taskParams = ApplyDelta(initParams, paramDelta);

            // Optional fine-tuning
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                ParamModel.SetParameters(taskParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                taskParams = ApplyInnerGradients(taskParams, grad, _algoOptions.InnerLearningRate, step == 0);
            }

            ParamModel.SetParameters(taskParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Parameter regularization: ||Δθ||²
            var totalLoss = NumOps.Add(queryLoss, ParamRegularization(paramDelta));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _encoderParams, _algoOptions.OuterLearningRate * SpsaLearningRateMultiplier, ComputeHyperNetLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _hyperLayer1, _algoOptions.OuterLearningRate * SpsaLearningRateMultiplier, ComputeHyperNetLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _hyperLayer2, _algoOptions.OuterLearningRate * SpsaLearningRateMultiplier, ComputeHyperNetLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null) throw new ArgumentNullException(nameof(task));
        var initParams = ParamModel.GetParameters();
        var supportGrad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
        var embedding = ComputeTaskEmbedding(supportGrad);
        var paramDelta = HyperNetForward(embedding);

        var taskParams = ApplyDelta(initParams, paramDelta);

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(taskParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            taskParams = ApplyInnerGradients(taskParams, grad, _algoOptions.InnerLearningRate, step == 0);
        }

        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, taskParams);
    }

    /// <summary>theta_d + delta[d mod compressedDim]: the compressed delta tiled over the parameters.</summary>
    private Vector<T> ApplyDelta(Vector<T> initParams, Vector<T> paramDelta)
        => Engine.Add(initParams.GetSubVector(0, _paramDim), ParameterModulation.Cycle(Engine, paramDelta, _paramDim));

    /// <summary>Parameter regularization ParamRegWeight * ||delta||^2.</summary>
    private T ParamRegularization(Vector<T> paramDelta)
        => NumOps.Multiply(NumOps.FromDouble(_algoOptions.ParamRegWeight), Engine.DotProduct(paramDelta, paramDelta));

    /// <summary>tanh(W_enc g) over the first compressedDim gradient entries (zero-padded when shorter).</summary>
    private Vector<T> ComputeTaskEmbedding(Vector<T> grad)
    {
        int used = Math.Min(_compressedDim, grad.Length);
        var g = used == _compressedDim ? grad.GetSubVector(0, _compressedDim)
            : Vector<T>.Concatenate(used > 0 ? grad.GetSubVector(0, used) : new Vector<T>(0), new Vector<T>(_compressedDim - used));
        var encoder = Tensor<T>.FromVector(_encoderParams, new[] { _embDim, _compressedDim });
        var preAct = Engine.TensorMatMul(encoder, Tensor<T>.FromVector(g, new[] { _compressedDim, 1 })).ToVector();
        return VectorTanh(preAct);
    }

    private Vector<T> HyperNetForward(Vector<T> embedding)
    {
        // Layer 1: embedding -> hidden (ReLU)
        var layer1 = Tensor<T>.FromVector(_hyperLayer1, new[] { _hiddenDim, _embDim });
        var hidden = Engine.ReLU(Engine.TensorMatMul(layer1, Tensor<T>.FromVector(embedding, new[] { _embDim, 1 })));

        // Layer 2: hidden -> param delta
        var layer2 = Tensor<T>.FromVector(_hyperLayer2, new[] { _compressedDim, _hiddenDim });
        return Engine.TensorMatMul(layer2, hidden).ToVector();
    }

    private double ComputeHyperNetLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>(taskBatch.Tasks.Length);
        var initParams = ParamModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(initParams);
            var sg = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            var emb = ComputeTaskEmbedding(sg);
            var pd = HyperNetForward(emb);
            var tp = ApplyDelta(initParams, pd);
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                ParamModel.SetParameters(tp);
                var g = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                tp = ApplyInnerGradients(tp, g, _algoOptions.InnerLearningRate, step == 0);
            }
            ParamModel.SetParameters(tp);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Include parameter regularization to match MetaTrain objective
            losses.Add(NumOps.Add(queryLoss, ParamRegularization(pd)));
        }
        ParamModel.SetParameters(initParams);

        // The SPSA callback contract is a double; the batch mean is reduced in T and converted once.
        return NumOps.ToDouble(ComputeMean(losses));
    }
}
