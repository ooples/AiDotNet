using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of ETPN: Embedding-Transformed Prototypical Networks.
/// </summary>
/// <remarks>
/// <para>
/// ETPN learns a task-specific embedding transformation applied transductively. The
/// transformation is computed from both support and query gradient information, enabling
/// the adapted parameter space to be more discriminative for the specific task.
/// A learned projection maps combined gradient features to per-parameter scaling factors.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Support embedding: e_s = compress(grad_support)
/// Query embedding: e_q = compress(grad_query)
/// Combined: e = e_s + QueryInfluenceWeight * e_q
///
/// Transform: t_d = sigmoid(W_transform · e)  (per compressed dim)
/// Transformed gradient: grad'_d = t[d % compressedDim] * grad_d
///
/// Transductive iterations:
///   Adapt using transformed gradients
///   Recompute query embedding with updated params
///   Update transform
///
/// L_meta = L_query + TransformRegWeight * ||W_transform||²
/// </code>
/// </para>
/// </remarks>
public class ETPNAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly ETPNOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _compressedDim;

    /// <summary>Transform projection: compressedDim × transformDim.</summary>
    private Vector<T> _transformParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ETPN;

    public ETPNAlgorithm(ETPNOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _compressedDim = Math.Min(_paramDim, 64);

        _transformParams = new Vector<T>(_compressedDim * options.TransformDim);
        double scale = 1.0 / Math.Sqrt(options.TransformDim);
        for (int i = 0; i < _transformParams.Length; i++)
            _transformParams[i] = NumOps.FromDouble(scale * (RandomGenerator.NextDouble() - 0.5));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            // Compute support embedding
            MetaModel.SetParameters(adaptedParams);
            var supportGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var supportEmb = CompressGradient(supportGrad);

            for (int tIter = 0; tIter < _algoOptions.TransductiveIterations; tIter++)
            {
                // Compute query embedding for transductive signal
                MetaModel.SetParameters(adaptedParams);
                var queryGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
                var queryEmb = CompressGradient(queryGrad);

                // Combine embeddings
                var combined = new double[_algoOptions.TransformDim];
                for (int e = 0; e < _algoOptions.TransformDim && e < supportEmb.Length; e++)
                    combined[e] = supportEmb[e] + _algoOptions.QueryInfluenceWeight * queryEmb[e];

                // Compute transform
                var transform = ComputeTransform(combined);

                // Adapt with transformed gradients
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                for (int d = 0; d < _paramDim; d++)
                {
                    int cd = d % _compressedDim;
                    adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * transform[cd] * NumOps.ToDouble(grad[d])));
                }
            }

            // Remaining adaptation steps (non-transductive)
            int transSteps = _algoOptions.TransductiveIterations;
            for (int step = transSteps; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Transform regularization
            double transformReg = 0;
            for (int i = 0; i < _transformParams.Length; i++)
                transformReg += NumOps.ToDouble(_transformParams[i]) * NumOps.ToDouble(_transformParams[i]);

            var totalLoss = NumOps.Add(queryLoss,
                NumOps.FromDouble(_algoOptions.TransformRegWeight * transformReg / _transformParams.Length));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _transformParams, _algoOptions.OuterLearningRate * 0.1, ComputeETPNLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

        MetaModel.SetParameters(adaptedParams);
        var supportGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
        var supportEmb = CompressGradient(supportGrad);

        var combined = new double[_algoOptions.TransformDim];
        for (int e = 0; e < _algoOptions.TransformDim && e < supportEmb.Length; e++)
            combined[e] = supportEmb[e];
        var transform = ComputeTransform(combined);

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            for (int d = 0; d < _paramDim; d++)
            {
                int cd = d % _compressedDim;
                adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                    NumOps.FromDouble(_algoOptions.InnerLearningRate * transform[cd] * NumOps.ToDouble(grad[d])));
            }
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    private double[] CompressGradient(Vector<T> grad)
    {
        var result = new double[_algoOptions.TransformDim];
        int bucketSize = Math.Max(1, _paramDim / _algoOptions.TransformDim);
        for (int e = 0; e < _algoOptions.TransformDim; e++)
        {
            double sum = 0;
            int start = e * bucketSize;
            for (int d = start; d < start + bucketSize && d < grad.Length; d++)
                sum += NumOps.ToDouble(grad[d]);
            result[e] = Math.Tanh(sum / bucketSize);
        }
        return result;
    }

    private double[] ComputeTransform(double[] combined)
    {
        var transform = new double[_compressedDim];
        for (int d = 0; d < _compressedDim; d++)
        {
            double sum = 0;
            for (int e = 0; e < _algoOptions.TransformDim; e++)
                sum += combined[e] * NumOps.ToDouble(_transformParams[d * _algoOptions.TransformDim + e]);
            transform[d] = 1.0 / (1.0 + Math.Exp(-sum)); // sigmoid → [0,1] scaling
            transform[d] = 0.5 + transform[d]; // shift to [0.5, 1.5] so it doesn't kill gradients
        }
        return transform;
    }

    private double ComputeETPNLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            var ap = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) ap[d] = initParams[d];
            MetaModel.SetParameters(ap);
            var sg = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            var se = CompressGradient(sg);
            var c = new double[_algoOptions.TransformDim];
            for (int e = 0; e < _algoOptions.TransformDim && e < se.Length; e++) c[e] = se[e];
            var t = ComputeTransform(c);
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(ap);
                var g = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                for (int d = 0; d < _paramDim; d++)
                    ap[d] = NumOps.Subtract(ap[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * t[d % _compressedDim] * NumOps.ToDouble(g[d])));
            }
            MetaModel.SetParameters(ap);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
