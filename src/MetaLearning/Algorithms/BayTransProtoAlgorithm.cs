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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Transductive Information Maximization for Few-Shot Learning",
    "https://arxiv.org/abs/2008.11297",
    Year = 2020,
    Authors = "Malik Boudiaf, Ziko Imtiaz Masud, Jerome Rony, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class BayTransProtoAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly BayTransProtoOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>Learned log-variance for the posterior.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _logVar;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.BayTransProto;

    public BayTransProtoAlgorithm(BayTransProtoOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        _logVar = Engine.Fill(_paramDim, NumOps.FromDouble(options.InitialLogVar));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();
        T transductiveLR = NumOps.FromDouble(_algoOptions.TransductiveLR);
        T klScale = NumOps.FromDouble(_algoOptions.KLWeight / _paramDim);

        foreach (var task in taskBatch.Tasks)
        {
            // Phase 1: Standard inner-loop adaptation
            var adaptedMean = initParams.Clone();

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                ParamModel.SetParameters(adaptedMean);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedMean = ApplyInnerGradients(adaptedMean, grad, _algoOptions.InnerLearningRate, step == 0);
            }

            // Phase 2: Transductive refinement using query gradients
            for (int tStep = 0; tStep < _algoOptions.TransductiveSteps; tStep++)
            {
                // Use query gradients to refine posterior mean
                ParamModel.SetParameters(adaptedMean);
                var queryGrad = ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput));
                adaptedMean = Engine.Subtract(adaptedMean, Engine.Multiply(queryGrad, transductiveLR));
            }

            // Final evaluation: ensemble loss
            var finalLosses = new List<T>();
            for (int s = 0; s < _algoOptions.NumPosteriorSamples; s++)
            {
                var sampled = SamplePosterior(adaptedMean);
                ParamModel.SetParameters(sampled);
                finalLosses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            }
            var ensembleLoss = ComputeMean(finalLosses);

            // KL divergence: KL(N(adapted, σ²) || N(init, I))
            T kl = DiagonalGaussianKL(adaptedMean, initParams);

            var totalLoss = NumOps.Add(ensembleLoss, NumOps.Multiply(klScale, kl));
            losses.Add(totalLoss);

            ParamModel.SetParameters(adaptedMean);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _logVar, _algoOptions.OuterLearningRate * 0.01, ComputeBayTransLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = ParamModel.GetParameters();
        var adaptedMean = initParams.Clone();

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedMean);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedMean = ApplyInnerGradients(adaptedMean, grad, _algoOptions.InnerLearningRate, step == 0);
        }

        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedMean);
    }

    private Vector<T> SamplePosterior(Vector<T> mean)
    {
        var sample = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
        {
            double std = Math.Exp(0.5 * NumOps.ToDouble(_logVar[d]));
            sample[d] = NumOps.Add(mean[d], NumOps.FromDouble(std * SampleNormal()));
        }
        return sample;
    }

    /// <summary>KL(N(mean, diag exp(logVar)) || N(prior, I)) = 0.5 * sum_d [v_d + (mean_d - prior_d)^2 - 1 - log v_d].</summary>
    private T DiagonalGaussianKL(Vector<T> mean, Vector<T> prior)
    {
        var diff = Engine.Subtract(mean, prior);
        T kl = NumOps.Subtract(
            NumOps.Add(Engine.Sum(Engine.Exp(_logVar)), Engine.DotProduct(diff, diff)),
            NumOps.Add(NumOps.FromDouble(_paramDim), Engine.Sum(_logVar)));
        return NumOps.Multiply(NumOps.FromDouble(0.5), kl);
    }

    private double ComputeBayTransLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = ParamModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            var ap = initParams.Clone();
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                ParamModel.SetParameters(ap);
                var g = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                ap = ApplyInnerGradients(ap, g, _algoOptions.InnerLearningRate, step == 0);
            }
            ParamModel.SetParameters(ap);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        ParamModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
