using System;
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Components;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// SImPa — statistical implicit PAC-Bayes meta-learning: a PAC-Bayes meta-learner whose task-specific
/// posterior is IMPLICIT, generated from noise rather than assumed to be a diagonal Gaussian.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Cuong Nguyen, Thanh-Toan Do and Gustavo Carneiro, "PAC-Bayes meta-learning with implicit task-specific
/// posteriors" (arXiv:2003.02455). The authors name the method SImPa.
/// </para>
/// <para>
/// THREE PARTS, each independently testable:
/// <see cref="Posterior"/> (<see cref="ImplicitPosteriorGenerator{T}"/>) draws task parameters as
/// <c>w = G(z; lambda)</c> with <c>z ~ U[0,1]^128</c>; <see cref="KLEstimator"/>
/// (<see cref="CompressionLemmaKLEstimator{T}"/>) recovers the KL term from SAMPLES via the compression
/// lemma, because an implicit posterior has no density to integrate; and
/// <see cref="PacBayesMetaBound"/> assembles the paper's two-level bound over both unseen samples and
/// unseen tasks.
/// </para>
/// <para>
/// WHAT THIS REPLACED, and why it was not a small correction. The previous implementation of this
/// citation used a single POINT posterior with a closed-form diagonal-Gaussian KL
/// (<c>0.5 * sum_d (theta_post_d - theta_prior_d)^2 / sigma_d^2</c>), plus a "data-dependent prior" phase
/// split and a "flex" parameter interpolating toward ERM. None of that appears in the cited paper, and the
/// diagonal Gaussian is specifically the assumption the paper's abstract names as the thing its
/// generative posterior is "more expressive" than. So the code implemented the baseline the paper was
/// written to beat, under the paper's name.
/// </para>
/// <para><b>For Beginners:</b> Ordinary meta-learning gives one best guess of a task's parameters. This
/// keeps a whole distribution of plausible parameter sets — represented by a small network that turns
/// random numbers into parameters, so the distribution can be any shape — and it comes with a
/// mathematical guarantee about performance on tasks it has never seen.</para>
/// </remarks>
/// <example>
/// <code>
/// var options = new SImPaOptions&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(model);
/// var simpa = new SImPaAlgorithm&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(options);
/// double bound = Convert.ToDouble(simpa.MetaTrain(taskBatch));
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// The citation itself was already correct (URL, title, authors, year) — it was the MECHANISM that did not
// match it. See the class remarks for what was replaced and why.
[ResearchPaper("PAC-Bayes Meta-Learning with Implicit Task-Specific Posteriors",
    "https://arxiv.org/abs/2003.02455",
    Year = 2020,
    Authors = "Cuong Nguyen, Thanh-Toan Do, Gustavo Carneiro")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public class SImPaAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private IParameterizable<T, TInput, TOutput>? _cachedParamModel;
    private IParameterizable<T, TInput, TOutput> ParamModel => _cachedParamModel ??= InterfaceGuard.Parameterizable(MetaModel);

    private readonly SImPaOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly Random _rng;

    /// <summary>Gets the implicit task-specific posterior generator.</summary>
    public ImplicitPosteriorGenerator<T> Posterior { get; }

    /// <summary>Gets the compression-lemma KL estimator.</summary>
    public CompressionLemmaKLEstimator<T> KLEstimator { get; }

    /// <summary>
    /// Gets the most recent PAC-Bayes bound value computed by <see cref="MetaTrain"/>, or NaN before the
    /// first call.
    /// </summary>
    /// <remarks>
    /// Exposed because the bound is the paper's actual output — the point of a PAC-Bayes method is the
    /// guarantee, and a training loss alone discards it.
    /// </remarks>
    public double LastBound { get; private set; } = double.NaN;

    /// <summary>Gets the most recent task-level KL estimate, or NaN before the first call.</summary>
    public double LastTaskKL { get; private set; } = double.NaN;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.SImPa;

    /// <summary>Creates the meta-learner.</summary>
    /// <param name="options">Configuration; defaults are the paper's where it states them.</param>
    public SImPaAlgorithm(SImPaOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        _rng = options.RandomSeed.HasValue ? new Random(options.RandomSeed.Value) : new Random(4242);

        Posterior = new ImplicitPosteriorGenerator<T>(
            outputDimension: _paramDim,
            latentDimension: options.LatentDimension,
            firstHiddenWidth: options.GeneratorFirstHiddenWidth,
            secondHiddenWidth: options.GeneratorSecondHiddenWidth,
            rng: new Random(_rng.Next()));

        KLEstimator = new CompressionLemmaKLEstimator<T>(
            inputDimension: _paramDim,
            hiddenWidth: options.KLEstimatorHiddenWidth,
            rng: new Random(_rng.Next()));
    }

    /// <summary>
    /// Draws samples from the PRIOR <c>p(w)</c>, a zero-mean Gaussian with standard deviation
    /// <c>sigma_w</c>.
    /// </summary>
    /// <remarks>
    /// The PRIOR stays a simple Gaussian and only the POSTERIOR is implicit — that asymmetry is the
    /// paper's, and it is what keeps the compression lemma usable: the lemma's second term needs
    /// <c>E_p[e^phi]</c>, which requires only that <c>p</c> can be sampled cheaply, while nothing
    /// anywhere needs <c>q</c>'s density.
    /// </remarks>
    private IReadOnlyList<Vector<T>> SamplePrior(int count)
    {
        var samples = new List<Vector<T>>(count);
        double sigma = _algoOptions.PriorStdDev;

        for (int s = 0; s < count; s++)
        {
            var v = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++)
            {
                // Box-Muller; a uniform prior would make the log-term in the bound meaningless.
                double u1 = Math.Max(1e-12, _rng.NextDouble());
                double u2 = _rng.NextDouble();
                double g = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                v[d] = NumOps.FromDouble(g * sigma);
            }
            samples.Add(v);
        }

        return samples;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        Guard.NotNull(taskBatch);
        if (taskBatch.Tasks.Length == 0) return NumOps.Zero;

        var metaParams = ParamModel.GetParameters();
        var lambda = Posterior.GetParameters();

        var queryLosses = new List<double>();
        var metaGradients = new List<Vector<T>>();
        int validationSampleCount = 0;

        foreach (var task in taskBatch.Tasks)
        {
            // Inner loop: adapt the GENERATOR's weights lambda, not a point parameter vector. This is the
            // structural difference from a MAML-style inner loop — what gets adapted is the whole
            // distribution over task parameters.
            var taskLambda = AdaptGeneratorToTask(lambda, task, _algoOptions.AdaptationSteps);
            Posterior.SetParameters(taskLambda);

            // Query loss under samples from the adapted implicit posterior.
            var posteriorSamples = Posterior.SampleMany(_algoOptions.TrainingPosteriorSamples, _rng);
            double taskLoss = 0.0;
            foreach (var w in posteriorSamples)
            {
                ParamModel.SetParameters(w);
                taskLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            }
            taskLoss /= posteriorSamples.Count;
            queryLosses.Add(taskLoss);

            if (validationSampleCount == 0) validationSampleCount = CountSamples(task.QueryOutput);

            // Meta-gradient from the last drawn sample's position, which is where the loss was measured.
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));

            Posterior.SetParameters(lambda);
        }

        // Task-level KL, estimated from samples because q has no density. Drawn from the meta-level
        // generator so the estimate reflects the shared posterior the bound is stated over.
        Posterior.SetParameters(lambda);
        int mc = _algoOptions.KLMonteCarloSamples;
        double taskKL = KLEstimator.EstimateKL(
            Posterior.SampleMany(mc, _rng),
            SamplePrior(mc),
            _algoOptions.KLEstimatorSteps,
            _algoOptions.KLEstimatorLearningRate,
            _rng);
        LastTaskKL = taskKL;

        // Meta-level KL for a near-point-mass q(theta; psi) against a sigma_w Gaussian prior: the closed
        // form IS available here, because q(theta) is an explicit isotropic Gaussian. Only the TASK
        // posterior is implicit, so using the closed form at this level is correct rather than a relapse.
        double metaKL = GaussianKLToZeroMeanPrior(metaParams, _algoOptions.MetaPosteriorStdDev, _algoOptions.PriorStdDev);

        double empirical = Mean(queryLosses);
        int taskCount = taskBatch.Tasks.Length;

        // The bound needs at least 2 validation samples and 2 tasks. Below that it is undefined rather
        // than large, so the empirical loss is reported and the bound left as NaN — saying "no guarantee"
        // instead of inventing one.
        if (validationSampleCount > 1 && taskCount > 1)
        {
            LastBound = PacBayesMetaBound.MetaLearning(
                empirical, taskKL, metaKL, validationSampleCount, taskCount, _algoOptions.Epsilon);
        }
        else
        {
            LastBound = double.NaN;
        }

        // Outer loop.
        ParamModel.SetParameters(metaParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            ParamModel.SetParameters(ApplyGradients(metaParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        // The bound is the training signal when it is defined; the paper minimizes the bound, not the raw
        // empirical loss, and reporting the loss instead would hide the complexity term entirely.
        return NumOps.FromDouble(double.IsNaN(LastBound) ? empirical : LastBound);
    }

    /// <summary>
    /// Adapts the generator's weights on a task's support set, so the whole posterior moves rather than a
    /// single parameter estimate.
    /// </summary>
    /// <remarks>
    /// SPSA on lambda. The generated parameters reach the loss only through the base model's forward pass,
    /// so an analytic gradient would need to be backpropagated through the generator as well — which the
    /// meta-learner's plumbing does not carry. Two loss evaluations per step is what makes adapting a
    /// distribution affordable, and it is the same technique this codebase already uses for auxiliary
    /// parameters.
    /// </remarks>
    private Vector<T> AdaptGeneratorToTask(
        Vector<T> lambda, IMetaLearningTask<T, TInput, TOutput> task, int steps)
    {
        var current = new Vector<T>(lambda.Length);
        for (int i = 0; i < lambda.Length; i++) current[i] = lambda[i];

        double best = SupportLossFor(current, task);

        for (int step = 0; step < steps; step++)
        {
            double c = Math.Max(1e-4, _algoOptions.InnerLearningRate);
            var delta = new double[current.Length];
            var plus = new Vector<T>(current.Length);
            var minus = new Vector<T>(current.Length);

            for (int i = 0; i < current.Length; i++)
            {
                delta[i] = _rng.NextDouble() < 0.5 ? -1.0 : 1.0;
                double w = NumOps.ToDouble(current[i]);
                plus[i] = NumOps.FromDouble(w + (c * delta[i]));
                minus[i] = NumOps.FromDouble(w - (c * delta[i]));
            }

            double fPlus = SupportLossFor(plus, task);
            double fMinus = SupportLossFor(minus, task);
            if (double.IsNaN(fPlus) || double.IsNaN(fMinus)) continue;

            double scale = (fPlus - fMinus) / (2.0 * c);
            var candidate = new Vector<T>(current.Length);
            for (int i = 0; i < current.Length; i++)
            {
                // DESCENT on the support loss.
                double g = scale / delta[i];
                candidate[i] = NumOps.FromDouble(NumOps.ToDouble(current[i]) - (_algoOptions.InnerLearningRate * g));
            }

            double value = SupportLossFor(candidate, task);
            if (double.IsNaN(value) || value >= best) continue;

            current = candidate;
            best = value;
        }

        return current;
    }

    private double SupportLossFor(Vector<T> lambda, IMetaLearningTask<T, TInput, TOutput> task)
    {
        Posterior.SetParameters(lambda);
        var samples = Posterior.SampleMany(_algoOptions.TrainingPosteriorSamples, _rng);

        double total = 0.0;
        foreach (var w in samples)
        {
            ParamModel.SetParameters(w);
            total += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.SupportInput), task.SupportOutput));
        }
        return total / samples.Count;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Adaptation draws <see cref="SImPaOptions{T, TInput, TOutput}.AdaptationPosteriorSamples"/> samples
    /// (32 by default, the paper's value) and returns a model at their MEAN. The mean is a summary for
    /// callers that need a single model; <see cref="SamplePosterior"/> exposes the individual particles,
    /// which is what the paper's calibration results are computed from — averaging first would discard
    /// exactly the predictive spread that makes the method well calibrated.
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        Guard.NotNull(task);

        var metaParams = ParamModel.GetParameters();
        var lambda = Posterior.GetParameters();

        var taskLambda = AdaptGeneratorToTask(lambda, task, _algoOptions.AdaptationSteps);
        Posterior.SetParameters(taskLambda);
        var samples = Posterior.SampleMany(_algoOptions.AdaptationPosteriorSamples, _rng);

        var mean = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
        {
            double sum = 0.0;
            for (int s = 0; s < samples.Count; s++) sum += NumOps.ToDouble(samples[s][d]);
            mean[d] = NumOps.FromDouble(sum / samples.Count);
        }

        Posterior.SetParameters(lambda);
        ParamModel.SetParameters(metaParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, mean);
    }

    /// <summary>
    /// Adapts to a task and returns the individual posterior PARTICLES rather than their mean.
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <param name="sampleCount">
    /// Particles to draw; defaults to the configured adaptation count (32, the paper's value).
    /// </param>
    /// <remarks>
    /// This is the honest output of a Bayesian method: the spread across particles IS the model's
    /// uncertainty, and it is what the paper's calibration claims rest on. A caller that only ever uses
    /// <see cref="Adapt"/> gets a point model and none of that information.
    /// </remarks>
    public IReadOnlyList<Vector<T>> SamplePosterior(
        IMetaLearningTask<T, TInput, TOutput> task, int? sampleCount = null)
    {
        Guard.NotNull(task);

        int count = sampleCount ?? _algoOptions.AdaptationPosteriorSamples;
        var lambda = Posterior.GetParameters();

        var taskLambda = AdaptGeneratorToTask(lambda, task, _algoOptions.AdaptationSteps);
        Posterior.SetParameters(taskLambda);
        var samples = Posterior.SampleMany(count, _rng);

        Posterior.SetParameters(lambda);
        return samples;
    }

    /// <summary>
    /// Closed-form KL between an isotropic Gaussian <c>N(mu, sigmaQ^2 I)</c> and <c>N(0, sigmaP^2 I)</c>.
    /// </summary>
    /// <remarks>
    /// Used ONLY at the meta level, where <c>q(theta; psi)</c> is an explicit isotropic Gaussian and the
    /// closed form is therefore correct. The task level cannot use this, and that distinction is the
    /// whole point of the method.
    /// </remarks>
    private double GaussianKLToZeroMeanPrior(Vector<T> mean, double sigmaQ, double sigmaP)
    {
        double vq = sigmaQ * sigmaQ;
        double vp = sigmaP * sigmaP;

        double squaredNorm = 0.0;
        for (int d = 0; d < mean.Length; d++)
        {
            double m = NumOps.ToDouble(mean[d]);
            squaredNorm += m * m;
        }

        // Per-dimension: 0.5 * (vq/vp + mu^2/vp - 1 + ln(vp/vq)).
        int k = mean.Length;
        double kl = 0.5 * ((k * vq / vp) + (squaredNorm / vp) - k + (k * Math.Log(vp / vq)));
        return Math.Max(0.0, kl);
    }

    private static double Mean(List<double> values)
    {
        if (values.Count == 0) return 0.0;
        double sum = 0.0;
        for (int i = 0; i < values.Count; i++) sum += values[i];
        return sum / values.Count;
    }

    /// <summary>
    /// Counts samples in a query target, for the <c>m^v</c> term of Theorem 2.
    /// </summary>
    private static int CountSamples(TOutput output) => output switch
    {
        Vector<T> v => v.Length,
        Matrix<T> m => m.Rows,
        Tensors.LinearAlgebra.Tensor<T> t => t.Shape.Length > 0 ? t.Shape[0] : 0,
        _ => 0,
    };
}
