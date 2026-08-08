using AiDotNet.Interfaces;
using AiDotNet.MetaLearning.Components;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration for <see cref="AiDotNet.MetaLearning.Algorithms.SImPaAlgorithm{T, TInput, TOutput}"/> —
/// SImPa, statistical implicit PAC-Bayes meta-learning.
/// </summary>
/// <remarks>
/// <para>
/// Defaults are the paper's where it states them (Nguyen, Do and Carneiro, arXiv:2003.02455):
/// <c>z ~ U[0,1]^128</c>, generator hidden widths 256 and 512, <c>epsilon = 0.1</c>, one posterior sample
/// per task while training and 32 at test time, 512 Monte Carlo samples for the phi-network, and a
/// learning rate of 1e-4.
/// </para>
/// <para><b>For Beginners:</b> These settings control how the model represents its uncertainty about each
/// task (a small generator network turning random numbers into parameters), how the divergence from the
/// prior is estimated when there is no formula for it, and how confident the generalization guarantee is.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class SImPaOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    /// <inheritdoc cref="IMetaLearnerOptions{T}"/>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    /// <summary>Gets or sets the inner-loop learning rate. Default 1e-4, the paper's value.</summary>
    public double InnerLearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the outer-loop learning rate. Default 1e-4, the paper's value.</summary>
    public double OuterLearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the inner adaptation steps applied to the generator per task. Default 5.</summary>
    public int AdaptationSteps { get; set; } = 5;

    /// <summary>Gets or sets T, tasks per meta-iteration. Default 20, the paper's value.</summary>
    public int MetaBatchSize { get; set; } = 20;

    public int NumMetaIterations { get; set; } = 1000;
    public double? GradientClipThreshold { get; set; } = 10.0;
    public int? RandomSeed { get => Seed; set => Seed = value; }
    public int EvaluationTasks { get; set; } = 100;
    public int EvaluationFrequency { get; set; } = 100;
    public bool EnableCheckpointing { get; set; } = false;
    public int CheckpointFrequency { get; set; } = 500;
    public bool UseFirstOrder { get; set; } = true;
    public ILossFunction<T>? LossFunction { get; set; }
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the latent noise width for the implicit posterior generator. Default 128, the paper's
    /// value.
    /// </summary>
    public int LatentDimension { get; set; } = ImplicitPosteriorGenerator<double>.PaperLatentDimension;

    /// <summary>Gets or sets the generator's first hidden width. Default 256, the paper's value.</summary>
    public int GeneratorFirstHiddenWidth { get; set; } = ImplicitPosteriorGenerator<double>.PaperFirstHiddenWidth;

    /// <summary>Gets or sets the generator's second hidden width. Default 512, the paper's value.</summary>
    public int GeneratorSecondHiddenWidth { get; set; } = ImplicitPosteriorGenerator<double>.PaperSecondHiddenWidth;

    /// <summary>
    /// Gets or sets the posterior samples drawn per task while META-TRAINING. Default 1, the paper's value.
    /// </summary>
    /// <remarks>
    /// One is enough during training because the meta-objective is already an expectation over a batch of
    /// tasks, so the sampling noise averages out across the batch rather than needing to be averaged away
    /// within each task.
    /// </remarks>
    public int TrainingPosteriorSamples { get; set; } = 1;

    /// <summary>
    /// Gets or sets the posterior samples drawn at ADAPTATION time. Default 32, the paper's value.
    /// </summary>
    /// <remarks>
    /// Higher than the training count on purpose: adaptation faces a single task, so there is no batch to
    /// average over, and it is the number of samples here that turns the implicit posterior into the
    /// calibrated predictive distribution the paper reports results for.
    /// </remarks>
    public int AdaptationPosteriorSamples { get; set; } = 32;

    /// <summary>
    /// Gets or sets the Monte Carlo samples used to train the phi-network. Default 512, the paper's value.
    /// </summary>
    public int KLMonteCarloSamples { get; set; } = CompressionLemmaKLEstimator<double>.PaperMonteCarloSamples;

    /// <summary>Gets or sets the ascent steps taken on the compression-lemma objective. Default 32.</summary>
    public int KLEstimatorSteps { get; set; } = 32;

    /// <summary>Gets or sets the ascent step size for the phi-network. Default 1e-2.</summary>
    public double KLEstimatorLearningRate { get; set; } = 1e-2;

    /// <summary>Gets or sets the phi-network's hidden width. Default 64.</summary>
    public int KLEstimatorHiddenWidth { get; set; } = 64;

    /// <summary>
    /// Gets or sets epsilon, the bound's confidence parameter. Default 0.1, the paper's value.
    /// </summary>
    public double Epsilon { get; set; } = PacBayesMetaBound.PaperEpsilon;

    /// <summary>
    /// Gets or sets the prior standard deviation over task parameters, <c>sigma_w</c>. Default 1.0, the
    /// paper's value.
    /// </summary>
    public double PriorStdDev { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the meta-parameter posterior standard deviation, <c>sigma_0</c>. Default 1e-6, the
    /// paper's value.
    /// </summary>
    /// <remarks>
    /// Deliberately tiny: it makes <c>q(theta; psi)</c> nearly a point mass, so the META-level KL term
    /// stays small and the bound is dominated by the task-level term. That is the paper's choice, not an
    /// accident of tuning.
    /// </remarks>
    public double MetaPosteriorStdDev { get; set; } = 1e-6;

    /// <summary>Creates the options for a meta-model.</summary>
    public SImPaOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        Guard.NotNull(metaModel);
        MetaModel = metaModel;
    }

    /// <inheritdoc />
    public bool IsValid() =>
        MetaModel != null
        && OuterLearningRate > 0 && InnerLearningRate > 0
        && MetaBatchSize > 1                     // Theorem 2's meta term divides by (T - 1)
        && LatentDimension > 0
        && GeneratorFirstHiddenWidth > 0 && GeneratorSecondHiddenWidth > 0
        && TrainingPosteriorSamples > 0 && AdaptationPosteriorSamples > 0
        && KLMonteCarloSamples > 0 && KLEstimatorSteps >= 0 && KLEstimatorLearningRate > 0
        && KLEstimatorHiddenWidth > 0
        && Epsilon > 0 && Epsilon <= 1
        && PriorStdDev > 0 && MetaPosteriorStdDev > 0;

    /// <inheritdoc />
    public IMetaLearnerOptions<T> Clone() => new SImPaOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        LatentDimension = LatentDimension,
        GeneratorFirstHiddenWidth = GeneratorFirstHiddenWidth,
        GeneratorSecondHiddenWidth = GeneratorSecondHiddenWidth,
        TrainingPosteriorSamples = TrainingPosteriorSamples,
        AdaptationPosteriorSamples = AdaptationPosteriorSamples,
        KLMonteCarloSamples = KLMonteCarloSamples,
        KLEstimatorSteps = KLEstimatorSteps,
        KLEstimatorLearningRate = KLEstimatorLearningRate,
        KLEstimatorHiddenWidth = KLEstimatorHiddenWidth,
        Epsilon = Epsilon, PriorStdDev = PriorStdDev, MetaPosteriorStdDev = MetaPosteriorStdDev,
    };
}
