using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of BayProNet: Bayesian Prototypical Networks with uncertainty-aware
/// prototype distributions for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// BayProNet extends Prototypical Networks by modeling class prototypes as Gaussian
/// distributions rather than point estimates. In the parameter-space formulation,
/// adaptation produces a posterior distribution N(θ_adapted, σ²) over parameters,
/// and predictions are made by ensembling over parameter samples. The variance σ² is
/// meta-learned per parameter dimension, enabling uncertainty-aware few-shot learning.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Meta-learned state: θ_0 (meta params), log(σ²) (per-param log-variance)
///
/// Inner loop (Bayesian adaptation for each task):
///   1. θ_adapted = gradient descent on support from θ_0
///   2. Sample S parameter vectors: θ_s ~ N(θ_adapted, σ²)
///   3. Ensemble prediction: ŷ = (1/S) Σ_s f(x; θ_s)
///   4. Predictive uncertainty from ensemble variance
///
/// Outer loop:
///   L = (1/T) Σ_τ [ensemble_loss_τ + KL_weight * KL(N(θ_adapted, σ²) || N(θ_0, I))]
///   Update θ_0 via gradient descent
///   Update log(σ²) via SPSA
/// </code>
/// </para>
/// </remarks>
public class BayProNetAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly BayProNetOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Meta-learned per-parameter log-variance for the posterior distribution.</summary>
    private Vector<T> _posteriorLogVar;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.BayProNet;

    public BayProNetAlgorithm(BayProNetOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;

        // Initialize per-parameter log-variance
        _posteriorLogVar = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
            _posteriorLogVar[d] = NumOps.FromDouble(options.InitialPrototypeLogVar);
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Inner loop: adapt backbone on support set
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
            }

            // Sample S parameter vectors from N(θ_adapted, σ²) and compute ensemble loss
            int numSamples = Math.Max(1, _algoOptions.EmbeddingDim / 8); // use EmbeddingDim to control samples
            var sampleLosses = new List<T>();
            var sampleGrads = new List<Vector<T>>();

            for (int s = 0; s < numSamples; s++)
            {
                var sampledParams = SamplePosterior(adaptedParams);
                MetaModel.SetParameters(sampledParams);
                var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
                sampleLosses.Add(queryLoss);
                sampleGrads.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
            }

            var ensembleLoss = ComputeMean(sampleLosses);

            // KL regularization: KL(N(θ_adapted, σ²) || N(θ_0, I))
            double klPenalty = ComputeAdaptedKL(adaptedParams, initParams);
            var totalLoss = NumOps.Add(ensembleLoss, NumOps.FromDouble(_algoOptions.KLWeight * klPenalty));

            losses.Add(totalLoss);
            metaGradients.Add(AverageVectors(sampleGrads));
        }

        // Outer loop: update meta-parameters
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var newParams = ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate);
            MetaModel.SetParameters(newParams);
        }

        // Update posterior log-variance via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _posteriorLogVar, _algoOptions.OuterLearningRate * 0.1, ComputeBayProNetLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
        }

        MetaModel.SetParameters(initParams);

        // Encode uncertainty as modulation: high-variance dims get dampened
        var modulationFactors = new double[_paramDim];
        for (int d = 0; d < _paramDim; d++)
        {
            double var_d = Math.Exp(NumOps.ToDouble(_posteriorLogVar[d]));
            // Sigmoid-based confidence: high variance → low modulation
            modulationFactors[d] = 1.0 / (1.0 + var_d / _algoOptions.Temperature);
        }

        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams, modulationFactors: modulationFactors);
    }

    /// <summary>
    /// Samples parameters from the posterior: θ_s ~ N(θ_adapted, diag(σ²)).
    /// Uses Box-Muller transform for Gaussian noise.
    /// </summary>
    private Vector<T> SamplePosterior(Vector<T> mean)
    {
        var sample = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
        {
            double std = Math.Sqrt(Math.Exp(NumOps.ToDouble(_posteriorLogVar[d])));
            double u1 = Math.Max(1e-10, RandomGenerator.NextDouble());
            double u2 = RandomGenerator.NextDouble();
            double noise = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            sample[d] = NumOps.Add(mean[d], NumOps.FromDouble(std * noise));
        }
        return sample;
    }

    /// <summary>
    /// KL(N(μ_post, σ²_post) || N(μ_prior, I)) = 0.5 * Σ_d [σ²_d + (μ_post_d - μ_prior_d)² - 1 - log(σ²_d)]
    /// </summary>
    private double ComputeAdaptedKL(Vector<T> posterior, Vector<T> prior)
    {
        double kl = 0;
        for (int d = 0; d < _paramDim; d++)
        {
            double var_d = Math.Exp(NumOps.ToDouble(_posteriorLogVar[d]));
            double diff = NumOps.ToDouble(posterior[d]) - NumOps.ToDouble(prior[d]);
            kl += var_d + diff * diff - 1.0 - NumOps.ToDouble(_posteriorLogVar[d]);
        }
        return 0.5 * kl;
    }

    private double ComputeBayProNetLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
            }

            int numSamples = Math.Max(1, _algoOptions.EmbeddingDim / 8);
            double sampleLossSum = 0;
            for (int s = 0; s < numSamples; s++)
            {
                var sampledParams = SamplePosterior(adaptedParams);
                MetaModel.SetParameters(sampledParams);
                sampleLossSum += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            }
            double ensembleLoss = sampleLossSum / numSamples;
            double klPenalty = ComputeAdaptedKL(adaptedParams, initParams);
            totalLoss += ensembleLoss + _algoOptions.KLWeight * klPenalty;
        }

        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
