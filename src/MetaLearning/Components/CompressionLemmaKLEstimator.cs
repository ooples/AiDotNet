using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.MetaLearning.Components;

/// <summary>
/// Estimates <c>KL[q || p]</c> from SAMPLES ONLY, via the compression lemma and a learned scalar
/// <c>phi</c>-network — SImPa's answer to a posterior that has no density.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Lemma 1 of Nguyen, Do and Carneiro (arXiv:2003.02455), the compression lemma:
/// <c>E_q[phi(w)] - ln E_p[e^phi(w)] &lt;= KL[q || p]</c>, with equality attainable. So maximizing the
/// left-hand side over a function class drives it UP toward the true KL, and its value at the maximum is
/// the estimate. The paper's Eq. 11:
/// <c>omega* = argmax_omega E_q[phi(w; omega)] - ln E_p[e^phi(w; omega)]</c>.
/// </para>
/// <para>
/// WHY A LEARNED ESTIMATOR IS NECESSARY rather than fussy. The task posterior is
/// <see cref="ImplicitPosteriorGenerator{T}"/>, which can be sampled but never evaluated, so
/// <c>ln q(w)</c> does not exist to be plugged into the KL integral. Every closed form — the
/// diagonal-Gaussian KL most of all — requires a density for BOTH arguments. This estimator needs only
/// the ability to draw samples, which is exactly what an implicit distribution offers.
/// </para>
/// <para>
/// IT IS A LOWER BOUND, and that direction is the safe one for a generalization bound: the KL enters the
/// PAC-Bayes bound positively, so under-estimating it makes the reported bound optimistic. That is a real
/// caveat and is why the estimate is only meaningful after <see cref="Maximize"/> has been run — an
/// untrained <c>phi</c> returns approximately zero for any pair of distributions, which would silently
/// erase the KL term altogether.
/// </para>
/// <para><b>For Beginners:</b> KL divergence normally needs a formula for how likely each outcome is. Here
/// we only have samples, no formula. So a small helper network is trained to tell the two sample sets
/// apart as sharply as it can, and how well it manages is itself a measure of how different they are.</para>
/// </remarks>
public class CompressionLemmaKLEstimator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Monte Carlo samples the paper draws when training the phi-network per task.</summary>
    /// <remarks>Alias for <see cref="SImPaPaperConstants.MonteCarloSamples"/>, where the value lives.</remarks>
    public const int PaperMonteCarloSamples = SImPaPaperConstants.MonteCarloSamples;

    private readonly int _inputDim;
    private readonly int _hidden;
    private readonly int _w1, _b1, _w2, _b2;

    private Vector<T> _omega;

    /// <summary>Gets the number of phi-network weights.</summary>
    public int ParameterCount => _omega.Length;

    /// <summary>Creates the estimator.</summary>
    /// <param name="inputDimension">Length of the parameter vectors being compared.</param>
    /// <param name="hiddenWidth">Width of the phi-network's single hidden layer.</param>
    /// <param name="rng">Seeded source for the initial weights.</param>
    public CompressionLemmaKLEstimator(int inputDimension, int hiddenWidth = 64, Random? rng = null)
    {
        if (inputDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputDimension), inputDimension, "inputDimension must be positive.");
        if (hiddenWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenWidth), hiddenWidth, "hiddenWidth must be positive.");

        _inputDim = inputDimension;
        _hidden = hiddenWidth;
        _w1 = _inputDim * _hidden;
        _b1 = _hidden;
        _w2 = _hidden;
        _b2 = 1;

        // PHI STARTS AT IDENTICALLY 0, so the initial estimate is exactly 0 - ln(1) = 0: the
        // compression lemma's trivial lower bound. Starting from the trivially valid point means the
        // estimate can only be improved by training, never start out spuriously large.
        //
        // THE MECHANISM IS THE SECOND LAYER, NOT ZERO INITIALIZATION EVERYWHERE. The first _w1
        // entries below are small RANDOM values -- only the entries from _w1 onward are zeroed. phi is
        // 0 because the second-layer weights and the output bias are zero, which annihilates whatever
        // the first layer produces. Breaking that symmetry in the first layer is deliberate: identical
        // hidden units would receive identical gradients and never differentiate. Stating the real
        // mechanism matters because a future change to the SECOND loop would silently break the
        // invariant while the first loop still looks like the reason it holds.
        var random = rng ?? RandomHelper.CreateSeededRandom(29);

        // U(-scale, scale) * 1e-3: one NextDouble per first-layer weight, in order; the loop only stores the draws.
        // (2u - 1) scale runs in double and each weight is rounded to T once: in Half, rounding u first and then
        // subtracting scale cancels the leading bits of the draw.
        var draws = new double[_w1];
        for (int i = 0; i < _w1; i++)
            draws[i] = random.NextDouble();
        double scale = Math.Sqrt(1.0 / Math.Max(1, _inputDim)) * 1e-3;
        var firstLayer = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(Engine.TensorMultiplyScalar(new Tensor<double>(draws, new[] { _w1 }), 2.0), -1.0),
            scale).Cast<T>().ToVector();
        _omega = Vector<T>.Concatenate(new[] { firstLayer, new Vector<T>(_b1 + _w2 + _b2) });
    }

    /// <summary>Gets the flattened phi-network weights.</summary>
    public Vector<T> GetParameters()
    {
        return _omega.Clone();
    }

    /// <summary>Sets the flattened phi-network weights.</summary>
    public void SetParameters(Vector<T> omega)
    {
        if (omega is null) throw new ArgumentNullException(nameof(omega));
        if (omega.Length != _omega.Length)
            throw new ArgumentException($"Expected {_omega.Length} phi weights; got {omega.Length}.", nameof(omega));
        _omega = omega.Clone();
    }

    /// <summary>Evaluates the scalar <c>phi(w; omega)</c>.</summary>
    public double Phi(Vector<T> w)
    {
        if (w is null) throw new ArgumentNullException(nameof(w));
        if (w.Length != _inputDim)
            throw new ArgumentException($"Expected an input of length {_inputDim}; got {w.Length}.", nameof(w));
        return NumOps.ToDouble(PhiWith(_omega, Tensor<T>.FromVector(w, new[] { 1, _inputDim })).GetFlat(0));
    }

    /// <summary>
    /// phi for every row of <paramref name="samples"/> [n, inputDim]:
    /// <c>tanh(X W1 + b1) w2 + b2</c>, W1 stored [inputDim, hidden] row-major. Returns [n].
    /// </summary>
    private Tensor<T> PhiWith(Vector<T> omega, Tensor<T> samples)
    {
        int n = samples.Shape[0];
        var w1 = Tensor<T>.FromVector(omega.GetSubVector(0, _w1), new[] { _inputDim, _hidden });
        var b1 = Tensor<T>.FromVector(omega.GetSubVector(_w1, _b1), new[] { 1, _hidden });
        var w2 = Tensor<T>.FromVector(omega.GetSubVector(_w1 + _b1, _w2), new[] { _hidden, 1 });
        var hidden = Engine.TensorTanh(Engine.TensorAdd(Engine.TensorMatMul(samples, w1), b1));
        var phi = Engine.TensorAddScalar(Engine.TensorMatMul(hidden, w2), omega[_w1 + _b1 + _w2]);
        return phi.Reshape(new[] { n });
    }

    /// <summary>Stacks equal-length sample vectors into an [n, inputDim] tensor.</summary>
    private Tensor<T> Stack(IReadOnlyList<Vector<T>> samples, string paramName)
    {
        if (samples is null) throw new ArgumentNullException(paramName);
        if (samples.Count == 0) throw new ArgumentException("At least one sample is required.", paramName);
        var flat = Vector<T>.Concatenate(new List<Vector<T>>(samples));
        if (flat.Length != samples.Count * _inputDim)
            throw new ArgumentException($"Every sample must have length {_inputDim}.", paramName);
        return Tensor<T>.FromVector(flat, new[] { samples.Count, _inputDim });
    }

    /// <summary>
    /// The compression-lemma objective <c>E_q[phi] - ln E_p[e^phi]</c>, which lower-bounds
    /// <c>KL[q || p]</c>.
    /// </summary>
    /// <param name="posteriorSamples">Samples from <c>q</c>.</param>
    /// <param name="priorSamples">Samples from <c>p</c>.</param>
    public double Objective(IReadOnlyList<Vector<T>> posteriorSamples, IReadOnlyList<Vector<T>> priorSamples)
        => ObjectiveWith(_omega, Stack(posteriorSamples, nameof(posteriorSamples)), Stack(priorSamples, nameof(priorSamples)));

    /// <summary>
    /// mean phi(q) - ln mean exp phi(p). The log-mean-exp goes through log-sum-exp, not a direct mean of
    /// exponentials: phi is unbounded, so exp(phi) overflows for a well-trained phi and the estimate would
    /// become NaN exactly when it started working.
    /// </summary>
    private double ObjectiveWith(Vector<T> omega, Tensor<T> posterior, Tensor<T> prior)
    {
        T meanPhiQ = Engine.TensorMean(PhiWith(omega, posterior));
        T logSumExpP = Engine.TensorLogSumExp(PhiWith(omega, prior), 0, keepDims: false).GetFlat(0);
        T logMeanExpP = NumOps.Subtract(logSumExpP, NumOps.FromDouble(Math.Log(prior.Shape[0])));
        return NumOps.ToDouble(NumOps.Subtract(meanPhiQ, logMeanExpP));
    }

    /// <summary>
    /// Ascends the compression-lemma objective over <c>omega</c> and returns the resulting KL estimate.
    /// </summary>
    /// <param name="posteriorSamples">Samples from <c>q</c>.</param>
    /// <param name="priorSamples">Samples from <c>p</c>.</param>
    /// <param name="steps">Ascent steps.</param>
    /// <param name="learningRate">Ascent step size.</param>
    /// <param name="rng">Source for the perturbation directions.</param>
    /// <remarks>
    /// <para>
    /// SPSA (simultaneous perturbation stochastic approximation) rather than analytic gradients, matching
    /// how this codebase already updates auxiliary parameters elsewhere. It costs two objective
    /// evaluations per step regardless of how many weights <c>omega</c> has, which is what keeps a
    /// per-task inner estimator affordable.
    /// </para>
    /// <para>
    /// The step is only APPLIED when it improves the objective. The compression lemma holds for any
    /// <c>phi</c>, so every value of this objective is a valid lower bound — but a rejected-uphill-only
    /// walk keeps the estimate monotone, which matters because a mid-training dip would show up as the
    /// PAC-Bayes bound briefly reporting less divergence than it has already proven.
    /// </para>
    /// </remarks>
    public double Maximize(
        IReadOnlyList<Vector<T>> posteriorSamples,
        IReadOnlyList<Vector<T>> priorSamples,
        int steps = 32,
        double learningRate = 1e-2,
        Random? rng = null)
    {
        if (steps < 0) throw new ArgumentOutOfRangeException(nameof(steps), steps, "steps cannot be negative.");
        if (learningRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(learningRate), learningRate, "learningRate must be positive.");

        var random = rng ?? RandomHelper.CreateSeededRandom(31);
        var posterior = Stack(posteriorSamples, nameof(posteriorSamples));
        var prior = Stack(priorSamples, nameof(priorSamples));
        double best = ObjectiveWith(_omega, posterior, prior);
        T plusOne = NumOps.One;
        T minusOne = NumOps.Negate(NumOps.One);

        for (int step = 0; step < steps; step++)
        {
            double c = Math.Max(1e-4, learningRate);

            // Rademacher directions, as SPSA requires: one NextDouble per weight; the loop only stores them.
            var delta = new Vector<T>(_omega.Length);
            for (int i = 0; i < _omega.Length; i++)
                delta[i] = random.NextDouble() < 0.5 ? minusOne : plusOne;

            double fPlus = ObjectiveWith(Along(delta, c), posterior, prior);
            double fMinus = ObjectiveWith(Along(delta, -c), posterior, prior);
            if (double.IsNaN(fPlus) || double.IsNaN(fMinus)) continue;

            // ASCENT: plus, because the compression lemma is maximized, not minimized. With delta_i = +/-1,
            // the SPSA gradient scale / delta_i equals scale * delta_i.
            double scale = (fPlus - fMinus) / (2.0 * c);
            var candidate = Along(delta, learningRate * scale);

            double value = ObjectiveWith(candidate, posterior, prior);
            if (double.IsNaN(value) || value <= best) continue;

            _omega = candidate;
            best = value;
        }

        return best;
    }

    /// <summary>omega + coefficient * direction.</summary>
    private Vector<T> Along(Vector<T> direction, double coefficient)
        => Engine.Add(_omega, Engine.Multiply(direction, NumOps.FromDouble(coefficient)));

    /// <summary>
    /// Estimates <c>KL[q || p]</c>: maximizes the compression-lemma objective and floors the result at 0.
    /// </summary>
    /// <remarks>
    /// Flooring at zero is legitimate rather than cosmetic: a KL divergence is non-negative by definition,
    /// so a negative value from the lower bound only ever means <c>phi</c> has not been trained far enough
    /// to certify anything, and zero is the correct claim in that case.
    /// </remarks>
    public double EstimateKL(
        IReadOnlyList<Vector<T>> posteriorSamples,
        IReadOnlyList<Vector<T>> priorSamples,
        int steps = 32,
        double learningRate = 1e-2,
        Random? rng = null)
        => Math.Max(0.0, Maximize(posteriorSamples, priorSamples, steps, learningRate, rng));

    /// <summary>Creates an independent estimator with the same architecture and the same weights.</summary>
    /// <remarks>
    /// Same reason as <see cref="ImplicitPosteriorGenerator{T}.Clone"/>: the clone engine deep-copies a
    /// field by probing for a parameterless <c>Clone()</c>, and without one this estimator stays shared
    /// between a learner and its copy, so <see cref="Maximize"/> on either would move both.
    /// </remarks>
    public CompressionLemmaKLEstimator<T> Clone()
    {
        var copy = new CompressionLemmaKLEstimator<T>(_inputDim, _hidden);
        copy.SetParameters(_omega);
        return copy;
    }
}
