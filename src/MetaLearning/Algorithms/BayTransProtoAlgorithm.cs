using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of BayTransProto: Bayesian Transductive Prototypical Networks.
/// </summary>
/// <remarks>
/// <para>
/// BayTransProto combines Bayesian posterior sampling with transductive refinement.
/// After standard inner-loop adaptation, the algorithm samples multiple parameter vectors
/// from a Gaussian posterior centered on the adapted params. Transductive refinement
/// then uses query-set gradients to iteratively update the posterior mean, leveraging
/// unlabeled query data to improve adaptation.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Inner loop: θ_adapted = MAML adaptation on support
///
/// Posterior: q(θ) = N(θ_adapted, exp(log_var) * I)
/// Samples: θ_s ~ q(θ) for s = 1..S
///
/// Transductive refinement (T steps):
///   For each step:
///     Evaluate ensemble on query → ensemble loss
///     θ_adapted ← θ_adapted - TransductiveLR * ∇_query(ensemble_loss)
///
/// KL regularization: KL(q || N(θ_init, I))
/// L_meta = mean(ensemble_loss) + KLWeight * KL
/// </code>
/// </para>
/// </remarks>
public class BayTransProtoAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly BayTransProtoOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Learned log-variance for the posterior.</summary>
    private Vector<T> _logVar;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.BayTransProto;

    public BayTransProtoAlgorithm(BayTransProtoOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _logVar = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
            _logVar[d] = NumOps.FromDouble(options.InitialLogVar);
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Phase 1: Standard inner-loop adaptation
            var adaptedMean = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedMean[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedMean);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedMean = ApplyGradients(adaptedMean, grad, _algoOptions.InnerLearningRate);
            }

            // Phase 2: Transductive refinement using query gradients
            for (int tStep = 0; tStep < _algoOptions.TransductiveSteps; tStep++)
            {
                // Evaluate ensemble on query
                var ensembleLosses = new List<T>();
                for (int s = 0; s < _algoOptions.NumPosteriorSamples; s++)
                {
                    var sampled = SamplePosterior(adaptedMean);
                    MetaModel.SetParameters(sampled);
                    ensembleLosses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
                }

                // Use query gradients to refine posterior mean
                MetaModel.SetParameters(adaptedMean);
                var queryGrad = ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput));
                for (int d = 0; d < _paramDim; d++)
                    adaptedMean[d] = NumOps.Subtract(adaptedMean[d],
                        NumOps.FromDouble(_algoOptions.TransductiveLR * NumOps.ToDouble(queryGrad[d])));
            }

            // Final evaluation: ensemble loss
            var finalLosses = new List<T>();
            for (int s = 0; s < _algoOptions.NumPosteriorSamples; s++)
            {
                var sampled = SamplePosterior(adaptedMean);
                MetaModel.SetParameters(sampled);
                finalLosses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            }
            var ensembleLoss = ComputeMean(finalLosses);

            // KL divergence: KL(N(adapted, σ²) || N(init, I))
            double kl = 0;
            for (int d = 0; d < _paramDim; d++)
            {
                double logv = NumOps.ToDouble(_logVar[d]);
                double v = Math.Exp(logv);
                double meanDiff = NumOps.ToDouble(adaptedMean[d]) - NumOps.ToDouble(initParams[d]);
                kl += 0.5 * (v + meanDiff * meanDiff - 1.0 - logv);
            }

            var totalLoss = NumOps.Add(ensembleLoss, NumOps.FromDouble(_algoOptions.KLWeight * kl / _paramDim));
            losses.Add(totalLoss);

            MetaModel.SetParameters(adaptedMean);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _logVar, _algoOptions.OuterLearningRate * 0.01, ComputeBayTransLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var adaptedMean = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedMean[d] = initParams[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedMean);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedMean = ApplyGradients(adaptedMean, grad, _algoOptions.InnerLearningRate);
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedMean);
    }

    private Vector<T> SamplePosterior(Vector<T> mean)
    {
        var sample = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
        {
            double std = Math.Exp(0.5 * NumOps.ToDouble(_logVar[d]));
            double u1 = Math.Max(1e-10, RandomGenerator.NextDouble());
            double u2 = RandomGenerator.NextDouble();
            double noise = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            sample[d] = NumOps.Add(mean[d], NumOps.FromDouble(std * noise));
        }
        return sample;
    }

    private double ComputeBayTransLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            var ap = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) ap[d] = initParams[d];
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(ap);
                var g = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                ap = ApplyGradients(ap, g, _algoOptions.InnerLearningRate);
            }
            MetaModel.SetParameters(ap);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
