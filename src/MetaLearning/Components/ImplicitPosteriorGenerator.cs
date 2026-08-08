using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.MetaLearning.Components;

/// <summary>
/// SImPa's IMPLICIT task-specific posterior: task parameters are produced by pushing latent noise
/// through a generator network, <c>w = G(z; lambda)</c> with <c>z ~ U[0,1]^d</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This is the contribution of Nguyen, Do and Carneiro, "PAC-Bayes meta-learning with implicit
/// task-specific posteriors" (arXiv:2003.02455), which the paper's abstract states as a
/// "generative-based approach to estimate the posterior of task-specific model parameters more
/// expressively compared to the usual assumption based on a multivariate normal distribution with a
/// diagonal covariance matrix".
/// </para>
/// <para>
/// WHY THIS CANNOT BE A GAUSSIAN, which is the whole point: a diagonal-covariance normal forces every
/// parameter to vary independently and symmetrically about its mean, so it cannot represent a posterior
/// with correlated parameters or several modes. Few-shot tasks routinely have several distinct plausible
/// solutions, and a diagonal Gaussian must either straddle them — putting most of its mass where no
/// solution lies — or collapse onto one. A generator has no such restriction: an arbitrarily shaped
/// distribution is reachable by warping uniform noise, at the cost of having no density at all.
/// </para>
/// <para>
/// HAVING NO DENSITY IS NOT AN OVERSIGHT, it is the defining property. <c>q(w)</c> can be SAMPLED but
/// never evaluated, so the KL term in the PAC-Bayes bound cannot be written in closed form and needs
/// <see cref="CompressionLemmaKLEstimator{T}"/>. Any implementation that reaches for a closed-form
/// diagonal-Gaussian KL has, by that act, replaced this method with the baseline it was designed to beat.
/// </para>
/// <para><b>For Beginners:</b> Instead of assuming the "uncertainty" about a task's parameters is a simple
/// bell curve, this feeds random numbers into a small network that outputs a full set of parameters. Run
/// it again with different random numbers and you get a different plausible set. The collection of
/// everything it can output IS the uncertainty, and it can be any shape at all.</para>
/// </remarks>
public class ImplicitPosteriorGenerator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The paper's latent dimension: <c>z ~ U[0,1]^128</c>.</summary>
    /// <remarks>Alias for <see cref="SImPaPaperConstants.LatentDimension"/>, where the value lives.</remarks>
    public const int PaperLatentDimension = SImPaPaperConstants.LatentDimension;

    /// <summary>The paper's first hidden width.</summary>
    /// <remarks>Alias for <see cref="SImPaPaperConstants.FirstHiddenWidth"/>.</remarks>
    public const int PaperFirstHiddenWidth = SImPaPaperConstants.FirstHiddenWidth;

    /// <summary>The paper's second hidden width.</summary>
    /// <remarks>Alias for <see cref="SImPaPaperConstants.SecondHiddenWidth"/>.</remarks>
    public const int PaperSecondHiddenWidth = SImPaPaperConstants.SecondHiddenWidth;

    private readonly int _latent;
    private readonly int _hidden1;
    private readonly int _hidden2;
    private readonly int _outputDim;

    // Flat lambda layout, in this order: W1, b1, W2, b2, W3, b3.
    private readonly int _w1, _b1, _w2, _b2, _w3, _b3;

    private Vector<T> _lambda;

    /// <summary>Gets the dimension of the generated parameter vector.</summary>
    public int OutputDimension => _outputDim;

    /// <summary>Gets the latent noise dimension.</summary>
    public int LatentDimension => _latent;

    /// <summary>Gets the number of generator weights in <c>lambda</c>.</summary>
    public int ParameterCount => _lambda.Length;

    /// <summary>
    /// Creates a generator producing <paramref name="outputDimension"/>-length parameter vectors.
    /// </summary>
    /// <param name="outputDimension">Length of the parameter vector to generate.</param>
    /// <param name="latentDimension">Latent noise width; the paper's value is 128.</param>
    /// <param name="firstHiddenWidth">First hidden width; the paper's value is 256.</param>
    /// <param name="secondHiddenWidth">Second hidden width; the paper's value is 512.</param>
    /// <param name="rng">Seeded source for the initial generator weights.</param>
    /// <remarks>
    /// <para>
    /// The generator's weight count grows as <c>secondHiddenWidth * outputDimension</c>, so generating the
    /// parameters of a large network is expensive by construction — the paper applies this to the small
    /// networks few-shot benchmarks use. The widths are settable so that cost can be traded off
    /// deliberately rather than by accident.
    /// </para>
    /// <para>
    /// <b>THE MULTIPLIER IS 512x AT THE PAPER'S DEFAULTS, and it is worth stating in figures because the
    /// cost lands somewhere a caller does not look.</b> <c>lambda</c> is what the meta-optimizer perturbs,
    /// and it is not perturbed by backpropagation — SImPa's meta-update goes through SPSA, which costs
    /// <c>numSamples + 1</c> full evaluations and touches every one of <see cref="ParameterCount"/> entries
    /// on each perturbation direction. So a task model with 10,000 parameters does not give a
    /// 10,000-parameter meta-problem: it gives <c>512 * 10,000 + ...</c>, over five million, perturbed
    /// three times per meta-step by default. A caller sizing this from a task network's parameter count is
    /// choosing a number three orders of magnitude larger than the one they typed.
    /// </para>
    /// <para>
    /// <b>Not clamped, deliberately.</b> Silently capping <paramref name="outputDimension"/> would generate
    /// a parameter vector too short for the model it is meant to parameterize, which fails far away from
    /// here and much less legibly than a slow run. Capping <paramref name="secondHiddenWidth"/> instead
    /// would quietly depart from the paper — the thing this class exists to reproduce. The lever is
    /// <see cref="ParameterCount"/>: read it after construction and lower the widths if it is larger than
    /// the budget allows.
    /// </para>
    /// </remarks>
    public ImplicitPosteriorGenerator(
        int outputDimension,
        int latentDimension = PaperLatentDimension,
        int firstHiddenWidth = PaperFirstHiddenWidth,
        int secondHiddenWidth = PaperSecondHiddenWidth,
        Random? rng = null)
    {
        if (outputDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputDimension), outputDimension, "outputDimension must be positive.");
        if (latentDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(latentDimension), latentDimension, "latentDimension must be positive.");
        if (firstHiddenWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(firstHiddenWidth), firstHiddenWidth, "firstHiddenWidth must be positive.");
        if (secondHiddenWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(secondHiddenWidth), secondHiddenWidth, "secondHiddenWidth must be positive.");

        _latent = latentDimension;
        _hidden1 = firstHiddenWidth;
        _hidden2 = secondHiddenWidth;
        _outputDim = outputDimension;

        _w1 = _latent * _hidden1;
        _b1 = _hidden1;
        _w2 = _hidden1 * _hidden2;
        _b2 = _hidden2;
        _w3 = _hidden2 * _outputDim;
        _b3 = _outputDim;

        int total = _w1 + _b1 + _w2 + _b2 + _w3 + _b3;
        _lambda = new Vector<T>(total);

        var random = rng ?? RandomHelper.CreateSeededRandom(17);
        InitializeWeights(random);
    }

    /// <summary>
    /// Glorot-style initialization. Scaling by fan-in matters more than usual here: the output passes
    /// through a tanh, so oversized initial weights saturate it and every draw collapses to the same
    /// vector of +/-1 — which would look exactly like a point posterior and silently defeat the method.
    /// </summary>
    private void InitializeWeights(Random rng)
    {
        int offset = 0;
        offset = FillScaled(rng, offset, _w1, _latent);
        offset = FillZero(offset, _b1);
        offset = FillScaled(rng, offset, _w2, _hidden1);
        offset = FillZero(offset, _b2);
        offset = FillScaled(rng, offset, _w3, _hidden2);
        _ = FillZero(offset, _b3);
    }

    private int FillScaled(Random rng, int offset, int count, int fanIn)
    {
        double scale = Math.Sqrt(1.0 / Math.Max(1, fanIn));
        for (int i = 0; i < count; i++)
            _lambda[offset + i] = NumOps.FromDouble(((rng.NextDouble() * 2.0) - 1.0) * scale);
        return offset + count;
    }

    private int FillZero(int offset, int count)
    {
        for (int i = 0; i < count; i++) _lambda[offset + i] = NumOps.Zero;
        return offset + count;
    }

    /// <summary>Gets the flattened generator weights <c>lambda</c>.</summary>
    public Vector<T> GetParameters()
    {
        var copy = new Vector<T>(_lambda.Length);
        for (int i = 0; i < _lambda.Length; i++) copy[i] = _lambda[i];
        return copy;
    }

    /// <summary>Sets the flattened generator weights <c>lambda</c>.</summary>
    public void SetParameters(Vector<T> lambda)
    {
        if (lambda is null) throw new ArgumentNullException(nameof(lambda));
        if (lambda.Length != _lambda.Length)
        {
            throw new ArgumentException(
                $"Expected {_lambda.Length} generator weights; got {lambda.Length}.", nameof(lambda));
        }

        for (int i = 0; i < lambda.Length; i++) _lambda[i] = lambda[i];
    }

    /// <summary>
    /// Draws one sample <c>w = G(z; lambda)</c> with a fresh <c>z ~ U[0,1]^latent</c>.
    /// </summary>
    /// <param name="rng">The noise source. Distinct calls must advance it, or every draw is identical.</param>
    public Vector<T> Sample(Random rng)
    {
        if (rng is null) throw new ArgumentNullException(nameof(rng));

        var z = new double[_latent];
        for (int i = 0; i < _latent; i++) z[i] = rng.NextDouble();   // U[0,1], the paper's latent prior
        return Forward(z);
    }

    /// <summary>
    /// Draws <paramref name="count"/> independent samples — the particle set standing in for the
    /// posterior wherever an expectation under <c>q</c> is needed.
    /// </summary>
    /// <remarks>
    /// The paper draws ONE sample of <c>w</c> per task during training and 32 at test time. One suffices
    /// while training because the meta-objective is already an expectation over tasks, so the noise
    /// averages out across the batch; at test time there is a single task, and 32 samples are what turn
    /// the posterior into a calibrated predictive distribution.
    /// </remarks>
    public IReadOnlyList<Vector<T>> SampleMany(int count, Random rng)
    {
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count), count, "count must be positive.");
        if (rng is null) throw new ArgumentNullException(nameof(rng));

        var samples = new List<Vector<T>>(count);
        for (int i = 0; i < count; i++) samples.Add(Sample(rng));
        return samples;
    }

    /// <summary>
    /// Deterministically maps a supplied latent vector to parameters, for tests that need to separate the
    /// generator's mapping from the noise source.
    /// </summary>
    public Vector<T> Generate(IReadOnlyList<double> latent)
    {
        if (latent is null) throw new ArgumentNullException(nameof(latent));
        if (latent.Count != _latent)
            throw new ArgumentException($"Expected a latent of length {_latent}; got {latent.Count}.", nameof(latent));

        var z = new double[_latent];
        for (int i = 0; i < _latent; i++) z[i] = latent[i];
        return Forward(z);
    }

    private Vector<T> Forward(double[] z)
    {
        int o = 0;
        var h1 = new double[_hidden1];
        for (int j = 0; j < _hidden1; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < _latent; i++) sum += z[i] * NumOps.ToDouble(_lambda[o + (i * _hidden1) + j]);
            h1[j] = Math.Tanh(sum + NumOps.ToDouble(_lambda[o + _w1 + j]));
        }
        o += _w1 + _b1;

        var h2 = new double[_hidden2];
        for (int j = 0; j < _hidden2; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < _hidden1; i++) sum += h1[i] * NumOps.ToDouble(_lambda[o + (i * _hidden2) + j]);
            h2[j] = Math.Tanh(sum + NumOps.ToDouble(_lambda[o + _w2 + j]));
        }
        o += _w2 + _b2;

        // tanh on the OUTPUT is the paper's stated choice: it bounds the generated parameters, which keeps
        // an untrained generator from emitting parameters large enough to make the base model diverge on
        // its first forward pass. The hidden nonlinearity is not stated; tanh is used throughout for
        // consistency.
        var w = new Vector<T>(_outputDim);
        for (int j = 0; j < _outputDim; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < _hidden2; i++) sum += h2[i] * NumOps.ToDouble(_lambda[o + (i * _outputDim) + j]);
            w[j] = NumOps.FromDouble(Math.Tanh(sum + NumOps.ToDouble(_lambda[o + _w3 + j])));
        }

        return w;
    }
}
