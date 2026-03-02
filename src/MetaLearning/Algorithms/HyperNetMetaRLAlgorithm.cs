using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

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
public class HyperNetMetaRLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly HyperNetMetaRLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _embDim;
    private readonly int _hiddenDim;
    private readonly int _compressedDim;

    private const int MaxCompressedDim = 128;

    /// <summary>Task encoder: compressedDim → embDim.</summary>
    private Vector<T> _encoderParams;

    /// <summary>Hypernetwork layer 1: embDim → hiddenDim.</summary>
    private Vector<T> _hyperLayer1;

    /// <summary>Hypernetwork layer 2: hiddenDim → compressedDim.</summary>
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
        _paramDim = options.MetaModel.GetParameters().Length;
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
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Compute task embedding from support gradient
            MetaModel.SetParameters(initParams);
            var supportGrad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            var embedding = ComputeTaskEmbedding(supportGrad);

            // Generate parameter delta via hypernetwork
            var paramDelta = HyperNetForward(embedding);

            // Apply parameter delta
            var taskParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++)
                taskParams[d] = NumOps.Add(initParams[d], paramDelta[d % _compressedDim]);

            // Optional fine-tuning
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(taskParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                taskParams = ApplyGradients(taskParams, grad, _algoOptions.InnerLearningRate);
            }

            MetaModel.SetParameters(taskParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Parameter regularization: ||Δθ||²
            double paramReg = 0;
            for (int d = 0; d < _compressedDim; d++) { double pd = NumOps.ToDouble(paramDelta[d]); paramReg += pd * pd; }
            var totalLoss = NumOps.Add(queryLoss, NumOps.FromDouble(_algoOptions.ParamRegWeight * paramReg));

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
        var initParams = MetaModel.GetParameters();
        var supportGrad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
        var embedding = ComputeTaskEmbedding(supportGrad);
        var paramDelta = HyperNetForward(embedding);

        var taskParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
            taskParams[d] = NumOps.Add(initParams[d], paramDelta[d % _compressedDim]);

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(taskParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            taskParams = ApplyGradients(taskParams, grad, _algoOptions.InnerLearningRate);
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, taskParams);
    }

    private Vector<T> ComputeTaskEmbedding(Vector<T> grad)
    {
        var emb = new Vector<T>(_embDim);
        for (int e = 0; e < _embDim; e++)
        {
            double sum = 0;
            for (int d = 0; d < _compressedDim && d < grad.Length; d++)
                sum += NumOps.ToDouble(grad[d]) * NumOps.ToDouble(_encoderParams[e * _compressedDim + d]);
            emb[e] = NumOps.FromDouble(Math.Tanh(sum));
        }
        return emb;
    }

    private Vector<T> HyperNetForward(Vector<T> embedding)
    {
        // Layer 1: embedding → hidden (ReLU)
        var hidden = new double[_hiddenDim];
        for (int h = 0; h < _hiddenDim; h++)
        {
            double sum = 0;
            for (int e = 0; e < _embDim; e++)
                sum += NumOps.ToDouble(embedding[e]) * NumOps.ToDouble(_hyperLayer1[h * _embDim + e]);
            hidden[h] = Math.Max(0, sum);
        }

        // Layer 2: hidden → param delta
        var delta = new Vector<T>(_compressedDim);
        for (int d = 0; d < _compressedDim; d++)
        {
            double sum = 0;
            for (int h = 0; h < _hiddenDim; h++)
                sum += hidden[h] * NumOps.ToDouble(_hyperLayer2[d * _hiddenDim + h]);
            delta[d] = NumOps.FromDouble(sum);
        }
        return delta;
    }

    private double ComputeHyperNetLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);
            var sg = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            var emb = ComputeTaskEmbedding(sg);
            var pd = HyperNetForward(emb);
            var tp = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) tp[d] = NumOps.Add(initParams[d], pd[d % _compressedDim]);
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(tp);
                var g = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                tp = ApplyGradients(tp, g, _algoOptions.InnerLearningRate);
            }
            MetaModel.SetParameters(tp);
            double queryLoss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));

            // Include parameter regularization to match MetaTrain objective
            double paramReg = 0;
            for (int d = 0; d < _compressedDim; d++) { double pdv = NumOps.ToDouble(pd[d]); paramReg += pdv * pdv; }
            totalLoss += queryLoss + _algoOptions.ParamRegWeight * paramReg;
        }
        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
