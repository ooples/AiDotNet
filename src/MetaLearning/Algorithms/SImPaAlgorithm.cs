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
/// var model = new SimpleRegression&lt;double&gt;();
/// var options = new SImPaOptions&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(model);
/// var simpa = new SImPaAlgorithm&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(options);
/// var taskBatch = new TaskBatch&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(
///     new IMetaLearningTask&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;[]
///     {
///         new MetaLearningTask&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     });
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
public partial class SImPaAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly SImPaOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly Random _rng;

    /// <summary>Gets the implicit task-specific posterior generator.</summary>
    public ImplicitPosteriorGenerator<T> Posterior { get; }

    /// <summary>Gets the compression-lemma KL estimator.</summary>
    public CompressionLemmaKLEstimator<T> KLEstimator { get; }

    /// <summary>
    /// <c>psi</c>, the hyper-meta-parameter: the mean of <c>q(theta; psi) = N(theta; psi, sigma_0 I)</c>
    /// (eq. 9). It lives in the GENERATOR's parameter space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This, and not the base network's weights, is what SImPa meta-learns. Algorithm 1 line 18 starts
    /// each task's generator at a draw from this distribution (<c>lambda_i &lt;- theta</c>), and the base
    /// network never owns persistent weights at all, because every <c>w_i</c> is <c>G(z; lambda_i)</c>
    /// (eq. 10). The paper is explicit that "the meta-parameter of interest is the model initialisation";
    /// the model being initialised is the generator.
    /// </para>
    /// <para>
    /// Declared as a buffer rather than a trainable parameter because this algorithm moves it itself
    /// (line 12); no optimizer walks a gradient tape through it.
    /// </para>
    /// </remarks>
    [Buffer]
    private Vector<T> _psi;

    /// <summary>
    /// <c>phi_0</c>, the meta-level initialisation of the compression-lemma network (Algorithm 1 line 3).
    /// </summary>
    /// <remarks>
    /// The paper learns a starting point for <c>phi</c> with MAML instead of training a fresh phi-network
    /// per task, purely to make the KL estimation affordable. Every task resets to this vector (line 19),
    /// and it is then ascended on the average of the tasks' KL lower bounds (line 13).
    /// </remarks>
    [Buffer]
    private Vector<T> _phiZero;

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
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.None)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        _rng = RandomHelper.CreateSeededRandom(options.RandomSeed ?? 4242);

        Posterior = new ImplicitPosteriorGenerator<T>(
            outputDimension: _paramDim,
            latentDimension: options.LatentDimension,
            firstHiddenWidth: options.GeneratorFirstHiddenWidth,
            secondHiddenWidth: options.GeneratorSecondHiddenWidth,
            rng: RandomHelper.CreateSeededRandom(_rng.Next()));

        KLEstimator = new CompressionLemmaKLEstimator<T>(
            inputDimension: _paramDim,
            hiddenWidth: options.KLEstimatorHiddenWidth,
            rng: RandomHelper.CreateSeededRandom(_rng.Next()));

        // The two meta-parameters start where their networks were initialised, so an untrained learner and
        // a freshly constructed generator agree.
        _psi = Posterior.GetParameters();
        _phiZero = KLEstimator.GetParameters();
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
        // Box-Muller; a uniform prior would make the log-term in the bound meaningless. Two NextDouble per
        // entry (u1 then u2), sample by sample, in order; the loop only stores the draws.
        var draws = new Vector<T>(2 * count * _paramDim);
        for (int i = 0; i < draws.Length; i++)
            draws[i] = NumOps.FromDouble(_rng.NextDouble());
        var normals = Engine.Multiply(BoxMuller(draws), NumOps.FromDouble(_algoOptions.PriorStdDev));

        var samples = new List<Vector<T>>(count);
        for (int s = 0; s < count; s++)
            samples.Add(normals.GetSubVector(s * _paramDim, _paramDim));
        return samples;
    }

    /// <summary>
    /// Box-Muller over interleaved uniform pairs (u1, u2): <c>sqrt(-2 ln max(u1, 1e-12)) cos(2 pi u2)</c>,
    /// one standard normal per pair.
    /// </summary>
    private Vector<T> BoxMuller(Vector<T> interleavedUniforms)
    {
        int n = interleavedUniforms.Length / 2;
        var pairs = Tensor<T>.FromVector(interleavedUniforms, new[] { n, 2 });
        var u1 = Engine.TensorMax(Engine.TensorSlice(pairs, new[] { 0, 0 }, new[] { n, 1 }), NumOps.FromDouble(1e-12));
        var u2 = Engine.TensorSlice(pairs, new[] { 0, 1 }, new[] { n, 1 });
        var radius = Engine.TensorSqrt(Engine.TensorMultiplyScalar(Engine.TensorLog(u1), NumOps.FromDouble(-2.0)));
        var angle = Engine.TensorCos(Engine.TensorMultiplyScalar(u2, NumOps.FromDouble(2.0 * Math.PI)));
        return Engine.TensorMultiply(radius, angle).Reshape(new[] { n }).ToVector();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Algorithm 1's TRAIN procedure: one draw of <c>theta</c> (line 5), then the lower level of lines
    /// 17-24 per task, then the two meta-updates — <c>psi</c> descends the average Theorem 2 bound
    /// (line 12) and <c>phi_0</c> ascends the average KL lower bound (line 13).
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        Guard.NotNull(taskBatch);
        if (taskBatch.Tasks.Length == 0) return NumOps.Zero;

        // The base network holds no meta-learned weights. Its parameter vector is scratch for the forward
        // passes below, overwritten once per posterior sample, and restored before returning.
        var baseParams = ParamModel.GetParameters();

        // Line 5: theta ~ N(psi, sigma_0 I). With the paper's sigma_0 = 1e-6 this is very nearly psi
        // itself, which is the point: q(theta; psi) exists so KL[q(theta)||p(theta)] is defined at all
        // (a Dirac delta would make it infinite), not to inject exploration.
        var theta = SampleMetaParameter();

        var queryLosses = new List<double>();
        var klLowerBounds = new List<double>();
        var phiDirections = new List<Vector<T>>();
        var lambdaDirections = new List<Vector<T>>();
        int validationSampleCount = 0;

        try
        {
            foreach (var task in taskBatch.Tasks)
            {
                // ---- OPTIMISE LOWER-LEVEL, Algorithm 1 lines 17-24 ----

                // Line 19: phi_i starts at phi_0 for EVERY task. Carrying on from wherever the previous
                // task left the phi-network is the "train phi from scratch" approach the paper replaces
                // with a MAML initialisation, and it silently makes the KL estimate depend on task order.
                KLEstimator.SetParameters(_phiZero);

                // Line 18: lambda_i <- theta.
                Posterior.SetParameters(theta);

                // Lines 20-21: phi_i maximises the compression-lemma lower bound (11), and that maximum
                // IS the KL estimate (12). An implicit posterior has no density, so this is the only way
                // the KL term in either level can be evaluated.
                int mc = _algoOptions.KLMonteCarloSamples;
                klLowerBounds.Add(KLEstimator.EstimateKL(
                    Posterior.SampleMany(mc, _rng),
                    SamplePrior(mc),
                    _algoOptions.KLEstimatorSteps,
                    _algoOptions.KLEstimatorLearningRate,
                    _rng));
                phiDirections.Add(Difference(KLEstimator.GetParameters(), _phiZero));

                // Line 22: lambda_i minimises Theorem 1's bound on the support set. The KL term was fixed
                // at line 21 and does not move with lambda, so within this step the bound and the support
                // loss share a minimiser — which is why descending the support loss IS line 22 rather
                // than an approximation of it.
                var taskLambda = AdaptGeneratorToTask(theta, task, _algoOptions.AdaptationSteps);
                lambdaDirections.Add(Difference(taskLambda, theta));
                Posterior.SetParameters(taskLambda);

                // Step 9's empirical term: the QUERY loss under samples from the adapted posterior.
                var posteriorSamples = Posterior.SampleMany(_algoOptions.TrainingPosteriorSamples, _rng);
                queryLosses.Add(MeanLoss(posteriorSamples, task.QueryInput, task.QueryOutput));

                if (validationSampleCount == 0) validationSampleCount = CountSamples(task.QueryOutput);
            }
        }
        finally
        {
            // Both components are scratch between tasks; leaving either holding a task's adapted weights
            // would make the next call depend on the last task of the previous one.
            Posterior.SetParameters(theta);
            KLEstimator.SetParameters(_phiZero);
        }

        LastTaskKL = Mean(klLowerBounds);

        // Meta-level KL for a near-point-mass q(theta; psi) against a sigma Gaussian prior, taken in the
        // GENERATOR's parameter space because that is where theta lives. The closed form is correct here:
        // only the TASK posterior is implicit, so using it at this level is not a relapse.
        double metaKL = GaussianKLToZeroMeanPrior(_psi, _algoOptions.MetaPosteriorStdDev, _algoOptions.PriorStdDev);

        double empirical = Mean(queryLosses);
        int taskCount = taskBatch.Tasks.Length;

        // The bound needs at least 2 validation samples and 2 tasks. Below that it is undefined rather
        // than large, so the empirical loss is reported and the bound left as NaN — saying "no guarantee"
        // instead of inventing one.
        LastBound = validationSampleCount > 1 && taskCount > 1
            ? PacBayesMetaBound.MetaLearning(
                empirical, LastTaskKL, metaKL, validationSampleCount, taskCount, _algoOptions.Epsilon)
            : double.NaN;

        ApplyMetaUpdates(lambdaDirections, phiDirections, taskCount, metaKL);
        ParamModel.SetParameters(baseParams);

        // The bound is the training signal when it is defined; the paper minimizes the bound, not the raw
        // empirical loss, and reporting the loss instead would hide the complexity term entirely.
        return NumOps.FromDouble(double.IsNaN(LastBound) ? empirical : LastBound);
    }

    /// <summary>Draws <c>theta ~ N(psi, sigma_0 I)</c> — eq. 9, Algorithm 1 line 5.</summary>
    private Vector<T> SampleMetaParameter()
    {
        // Box-Muller: two NextDouble per entry (u1 then u2), in order; the loop only stores the draws.
        var draws = new Vector<T>(2 * _psi.Length);
        for (int i = 0; i < draws.Length; i++)
            draws[i] = NumOps.FromDouble(_rng.NextDouble());
        return Engine.Add(_psi, Engine.Multiply(BoxMuller(draws), NumOps.FromDouble(_algoOptions.MetaPosteriorStdDev)));
    }

    /// <summary>Elementwise <c>from - baseline</c>.</summary>
    private Vector<T> Difference(Vector<T> from, Vector<T> baseline) => Engine.Subtract(from, baseline);

    /// <summary>
    /// Algorithm 1 lines 12 and 13: <c>psi</c> by descent on the average Theorem 2 bound, <c>phi_0</c> by
    /// ascent on the average KL lower bound.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Both use the FIRST-ORDER MAML direction — the mean of where each task's lower level moved from its
    /// shared starting point — because neither the generator nor the phi-network is differentiable here:
    /// both are hand-written forward passes over <c>double</c> with no tape to backpropagate through.
    /// That is an approximation of the paper's SGD/SGA, and it is the same first-order truncation
    /// <see cref="SImPaOptions{T, TInput, TOutput}.UseFirstOrder"/> already describes.
    /// </para>
    /// <para>
    /// The one part of the bound whose dependence on <c>psi</c> IS available in closed form — the
    /// meta-level KL term — is differentiated exactly and subtracted, so <c>psi</c> is pulled back toward
    /// the prior instead of only toward the tasks. Chain rule through
    /// <c>metaTerm = sqrt((metaKL + metaLog) / (2 (T - 1)))</c> and
    /// <c>d(metaKL)/d(psi) = psi / sigma^2</c> gives the coefficient below. Without it the complexity
    /// half of the bound would be reported but never actually optimised.
    /// </para>
    /// </remarks>
    private void ApplyMetaUpdates(
        List<Vector<T>> lambdaDirections, List<Vector<T>> phiDirections, int taskCount, double metaKL)
    {
        if (lambdaDirections.Count > 0)
        {
            var direction = AverageVectors(lambdaDirections);

            double pull = 0.0;
            if (taskCount > 1)
            {
                double metaLog = taskCount * Math.Log(taskCount) / _algoOptions.Epsilon;
                double metaTerm = Math.Sqrt((metaKL + metaLog) / (2.0 * (taskCount - 1)));
                if (metaTerm > 0.0)
                {
                    double variance = _algoOptions.PriorStdDev * _algoOptions.PriorStdDev;
                    pull = 1.0 / (variance * 4.0 * (taskCount - 1) * metaTerm);
                }
            }

            // psi + rate (direction - pull psi) = (1 - rate pull) psi + rate direction.
            double rate = _algoOptions.OuterLearningRate;
            _psi = Engine.Add(
                Engine.Multiply(_psi, NumOps.FromDouble(1.0 - (rate * pull))),
                Engine.Multiply(direction, NumOps.FromDouble(rate)));
            Posterior.SetParameters(_psi);
        }

        if (phiDirections.Count > 0)
        {
            var direction = AverageVectors(phiDirections);
            // ASCENT: line 13 is SGA, because phi_0 initialises a MAXIMISATION.
            _phiZero = Engine.Add(_phiZero, Engine.Multiply(direction, NumOps.FromDouble(_algoOptions.PhiMetaLearningRate)));
            KLEstimator.SetParameters(_phiZero);
        }
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
        var current = lambda.Clone();
        double best = SupportLossFor(current, task);
        T plusOne = NumOps.One;
        T minusOne = NumOps.Negate(NumOps.One);

        for (int step = 0; step < steps; step++)
        {
            double c = Math.Max(1e-4, _algoOptions.InnerLearningRate);

            // Rademacher directions: one NextDouble per weight, in order; the loop only stores them.
            var delta = new Vector<T>(current.Length);
            for (int i = 0; i < current.Length; i++)
                delta[i] = _rng.NextDouble() < 0.5 ? minusOne : plusOne;

            double fPlus = SupportLossFor(Along(current, delta, c), task);
            double fMinus = SupportLossFor(Along(current, delta, -c), task);
            if (double.IsNaN(fPlus) || double.IsNaN(fMinus)) continue;

            // DESCENT on the support loss. With delta_i = +/-1, the SPSA gradient scale / delta_i is scale * delta_i.
            double scale = (fPlus - fMinus) / (2.0 * c);
            var candidate = Along(current, delta, -_algoOptions.InnerLearningRate * scale);

            double value = SupportLossFor(candidate, task);
            if (double.IsNaN(value) || value >= best) continue;

            current = candidate;
            best = value;
        }

        return current;
    }

    /// <summary><c>origin + coefficient * direction</c>.</summary>
    private Vector<T> Along(Vector<T> origin, Vector<T> direction, double coefficient)
        => Engine.Add(origin, Engine.Multiply(direction, NumOps.FromDouble(coefficient)));

    private double SupportLossFor(Vector<T> lambda, IMetaLearningTask<T, TInput, TOutput> task)
    {
        Posterior.SetParameters(lambda);
        var samples = Posterior.SampleMany(_algoOptions.TrainingPosteriorSamples, _rng);
        return MeanLoss(samples, task.SupportInput, task.SupportOutput);
    }

    /// <summary>The base model's loss on (x, y), averaged over the given parameter samples.</summary>
    private double MeanLoss(IReadOnlyList<Vector<T>> parameterSamples, TInput x, TOutput y)
    {
        var losses = new List<T>(parameterSamples.Count);
        foreach (var w in parameterSamples)
        {
            ParamModel.SetParameters(w);
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(x), y));
        }

        return NumOps.ToDouble(Engine.Mean(new Vector<T>(losses.ToArray())));
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

        // Algorithm 1 lines 5 and 18: a task starts from a draw of theta, not from whatever lambda the
        // generator happens to be holding after the last call.
        var theta = SampleMetaParameter();

        var taskLambda = AdaptGeneratorToTask(theta, task, _algoOptions.AdaptationSteps);
        Posterior.SetParameters(taskLambda);
        var samples = Posterior.SampleMany(_algoOptions.AdaptationPosteriorSamples, _rng);

        var stacked = Tensor<T>.FromVector(
            Vector<T>.Concatenate(new List<Vector<T>>(samples)), new[] { samples.Count, _paramDim });
        var mean = Engine.ReduceMean(stacked, new[] { 0 }, keepDims: false).ToVector();

        Posterior.SetParameters(_psi);
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
        var theta = SampleMetaParameter();

        // ParamModel IS RESTORED TOO, not just Posterior. AdaptGeneratorToTask reaches
        // SupportLossFor, which calls ParamModel.SetParameters(w) once per posterior sample -- so on
        // return the SHARED MetaModel was left holding a random generator sample instead of the
        // meta-parameters. Adapt() at the top of this file already captures and restores both;
        // this path captured only Posterior, so merely SAMPLING the posterior silently corrupted the
        // meta-model for every later caller.
        var metaParams = ParamModel.GetParameters();
        try
        {
            var taskLambda = AdaptGeneratorToTask(theta, task, _algoOptions.AdaptationSteps);
            Posterior.SetParameters(taskLambda);
            return Posterior.SampleMany(count, _rng);
        }
        finally
        {
            Posterior.SetParameters(_psi);
            ParamModel.SetParameters(metaParams);
        }
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

        double squaredNorm = NumOps.ToDouble(Engine.DotProduct(mean, mean));

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
