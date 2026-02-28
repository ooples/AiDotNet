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

    /// <summary>Task encoder: compressedDim → embDim.</summary>
    private Vector<T> _encoderParams;

    /// <summary>Hypernetwork layer 1: embDim → hiddenDim.</summary>
    private Vector<T> _hyperLayer1;

    /// <summary>Hypernetwork layer 2: hiddenDim → compressedDim.</summary>
    private Vector<T> _hyperLayer2;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.HyperNetMetaRL;

    public HyperNetMetaRLAlgorithm(HyperNetMetaRLOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _embDim = options.TaskEmbeddingDim;
        _hiddenDim = options.HyperNetHiddenDim;
        _compressedDim = Math.Min(_paramDim, 128);

        _encoderParams = InitRandom(_compressedDim * _embDim);
        _hyperLayer1 = InitRandom(_embDim * _hiddenDim);
        _hyperLayer2 = InitRandom(_hiddenDim * _compressedDim);
    }

    private Vector<T> InitRandom(int size)
    {
        var v = new Vector<T>(size);
        double scale = 1.0 / Math.Sqrt(size);
        for (int i = 0; i < size; i++)
        {
            double u1 = Math.Max(1e-10, RandomGenerator.NextDouble());
            double u2 = RandomGenerator.NextDouble();
            v[i] = NumOps.FromDouble(scale * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return v;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
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
                taskParams[d] = NumOps.Add(initParams[d], NumOps.FromDouble(paramDelta[d % _compressedDim]));

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
            for (int d = 0; d < _compressedDim; d++) paramReg += paramDelta[d] * paramDelta[d];
            var totalLoss = NumOps.Add(queryLoss, NumOps.FromDouble(_algoOptions.ParamRegWeight * paramReg));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _encoderParams, _algoOptions.OuterLearningRate * 0.1, ComputeHyperNetLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _hyperLayer1, _algoOptions.OuterLearningRate * 0.1, ComputeHyperNetLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _hyperLayer2, _algoOptions.OuterLearningRate * 0.1, ComputeHyperNetLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        MetaModel.SetParameters(initParams);
        var supportGrad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
        var embedding = ComputeTaskEmbedding(supportGrad);
        var paramDelta = HyperNetForward(embedding);

        var taskParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
            taskParams[d] = NumOps.Add(initParams[d], NumOps.FromDouble(paramDelta[d % _compressedDim]));

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(taskParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            taskParams = ApplyGradients(taskParams, grad, _algoOptions.InnerLearningRate);
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, taskParams);
    }

    private double[] ComputeTaskEmbedding(Vector<T> grad)
    {
        var emb = new double[_embDim];
        for (int e = 0; e < _embDim; e++)
        {
            double sum = 0;
            for (int d = 0; d < _compressedDim && d < grad.Length; d++)
                sum += NumOps.ToDouble(grad[d]) * NumOps.ToDouble(_encoderParams[e * _compressedDim + d]);
            emb[e] = Math.Tanh(sum);
        }
        return emb;
    }

    private double[] HyperNetForward(double[] embedding)
    {
        // Layer 1: embedding → hidden (ReLU)
        var hidden = new double[_hiddenDim];
        for (int h = 0; h < _hiddenDim; h++)
        {
            double sum = 0;
            for (int e = 0; e < _embDim; e++)
                sum += embedding[e] * NumOps.ToDouble(_hyperLayer1[h * _embDim + e]);
            hidden[h] = Math.Max(0, sum);
        }

        // Layer 2: hidden → param delta
        var delta = new double[_compressedDim];
        for (int d = 0; d < _compressedDim; d++)
        {
            double sum = 0;
            for (int h = 0; h < _hiddenDim; h++)
                sum += hidden[h] * NumOps.ToDouble(_hyperLayer2[d * _hiddenDim + h]);
            delta[d] = sum;
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
            for (int d = 0; d < _paramDim; d++) tp[d] = NumOps.Add(initParams[d], NumOps.FromDouble(pd[d % _compressedDim]));
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(tp);
                var g = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                tp = ApplyGradients(tp, g, _algoOptions.InnerLearningRate);
            }
            MetaModel.SetParameters(tp);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
