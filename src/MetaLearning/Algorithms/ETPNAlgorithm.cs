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
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters. ETPN requires a model with at least one parameter.");
        if (options.TransformDim <= 0)
            throw new ArgumentException("TransformDim must be positive.", nameof(options));
        if (options.TransductiveIterations < 0)
            throw new ArgumentException("TransductiveIterations must be non-negative.", nameof(options));

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
            var supportEmb = CompressVector(supportGrad, _algoOptions.TransformDim);

            int effectiveTransIter = Math.Min(_algoOptions.TransductiveIterations, _algoOptions.AdaptationSteps);
            for (int tIter = 0; tIter < effectiveTransIter; tIter++)
            {
                // Compute query embedding for transductive signal
                MetaModel.SetParameters(adaptedParams);
                var queryGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
                var queryEmb = CompressVector(queryGrad, _algoOptions.TransformDim);

                // Combine embeddings
                var combined = new Vector<T>(_algoOptions.TransformDim);
                for (int e = 0; e < _algoOptions.TransformDim && e < supportEmb.Length; e++)
                    combined[e] = NumOps.FromDouble(NumOps.ToDouble(supportEmb[e]) + _algoOptions.QueryInfluenceWeight * NumOps.ToDouble(queryEmb[e]));

                // Compute transform
                var transform = ComputeTransform(combined);

                // Adapt with transformed gradients
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                for (int d = 0; d < _paramDim; d++)
                {
                    int cd = d % _compressedDim;
                    adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * NumOps.ToDouble(transform[cd]) * NumOps.ToDouble(grad[d])));
                }
            }

            // Remaining adaptation steps (non-transductive)
            int transSteps = effectiveTransIter;
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

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

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
        var supportEmb = CompressVector(supportGrad, _algoOptions.TransformDim);

        var combined = new Vector<T>(_algoOptions.TransformDim);
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
                    NumOps.FromDouble(_algoOptions.InnerLearningRate * NumOps.ToDouble(transform[cd]) * NumOps.ToDouble(grad[d])));
            }
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    private Vector<T> ComputeTransform(Vector<T> combined)
    {
        var transform = new Vector<T>(_compressedDim);
        for (int d = 0; d < _compressedDim; d++)
        {
            double sum = 0;
            for (int e = 0; e < _algoOptions.TransformDim; e++)
                sum += NumOps.ToDouble(combined[e]) * NumOps.ToDouble(_transformParams[d * _algoOptions.TransformDim + e]);
            double sigmoid = 1.0 / (1.0 + Math.Exp(-sum));
            transform[d] = NumOps.FromDouble(0.5 + sigmoid); // shift to [0.5, 1.5] so it doesn't kill gradients
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
            var se = CompressVector(sg, _algoOptions.TransformDim);
            var t = ComputeTransform(se);
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(ap);
                var g = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                for (int d = 0; d < _paramDim; d++)
                    ap[d] = NumOps.Subtract(ap[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * NumOps.ToDouble(t[d % _compressedDim]) * NumOps.ToDouble(g[d])));
            }
            MetaModel.SetParameters(ap);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
