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

    /// <summary>Monte Carlo samples the paper draws when training the phi-network per task.</summary>
    public const int PaperMonteCarloSamples = 512;

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

        _omega = new Vector<T>(_w1 + _b1 + _w2 + _b2);

        // ZERO initialization on purpose. It makes phi identically 0, so the initial estimate is exactly
        // 0 - ln(1) = 0: the compression lemma's trivial lower bound. Starting from the trivially valid
        // point means the estimate can only be improved by training, never start out spuriously large.
        var random = rng ?? new Random(29);
        double scale = Math.Sqrt(1.0 / Math.Max(1, _inputDim));
        for (int i = 0; i < _w1; i++)
            _omega[i] = NumOps.FromDouble(((random.NextDouble() * 2.0) - 1.0) * scale * 1e-3);
        for (int i = _w1; i < _omega.Length; i++) _omega[i] = NumOps.Zero;
    }

    /// <summary>Gets the flattened phi-network weights.</summary>
    public Vector<T> GetParameters()
    {
        var copy = new Vector<T>(_omega.Length);
        for (int i = 0; i < _omega.Length; i++) copy[i] = _omega[i];
        return copy;
    }

    /// <summary>Sets the flattened phi-network weights.</summary>
    public void SetParameters(Vector<T> omega)
    {
        if (omega is null) throw new ArgumentNullException(nameof(omega));
        if (omega.Length != _omega.Length)
            throw new ArgumentException($"Expected {_omega.Length} phi weights; got {omega.Length}.", nameof(omega));
        for (int i = 0; i < omega.Length; i++) _omega[i] = omega[i];
    }

    /// <summary>Evaluates the scalar <c>phi(w; omega)</c>.</summary>
    public double Phi(Vector<T> w)
    {
        if (w is null) throw new ArgumentNullException(nameof(w));
        if (w.Length != _inputDim)
            throw new ArgumentException($"Expected an input of length {_inputDim}; got {w.Length}.", nameof(w));
        return PhiWith(_omega, w);
    }

    private double PhiWith(Vector<T> omega, Vector<T> w)
    {
        double sum2 = NumOps.ToDouble(omega[_w1 + _b1 + _w2]);
        for (int j = 0; j < _hidden; j++)
        {
            double sum1 = 0.0;
            for (int i = 0; i < _inputDim; i++) sum1 += NumOps.ToDouble(w[i]) * NumOps.ToDouble(omega[(i * _hidden) + j]);
            double h = Math.Tanh(sum1 + NumOps.ToDouble(omega[_w1 + j]));
            sum2 += h * NumOps.ToDouble(omega[_w1 + _b1 + j]);
        }
        return sum2;
    }

    /// <summary>
    /// The compression-lemma objective <c>E_q[phi] - ln E_p[e^phi]</c>, which lower-bounds
    /// <c>KL[q || p]</c>.
    /// </summary>
    /// <param name="posteriorSamples">Samples from <c>q</c>.</param>
    /// <param name="priorSamples">Samples from <c>p</c>.</param>
    public double Objective(IReadOnlyList<Vector<T>> posteriorSamples, IReadOnlyList<Vector<T>> priorSamples)
        => ObjectiveWith(_omega, posteriorSamples, priorSamples);

    private double ObjectiveWith(
        Vector<T> omega, IReadOnlyList<Vector<T>> posteriorSamples, IReadOnlyList<Vector<T>> priorSamples)
    {
        if (posteriorSamples is null) throw new ArgumentNullException(nameof(posteriorSamples));
        if (priorSamples is null) throw new ArgumentNullException(nameof(priorSamples));
        if (posteriorSamples.Count == 0) throw new ArgumentException("At least one posterior sample is required.", nameof(posteriorSamples));
        if (priorSamples.Count == 0) throw new ArgumentException("At least one prior sample is required.", nameof(priorSamples));

        double meanPhiQ = 0.0;
        for (int i = 0; i < posteriorSamples.Count; i++) meanPhiQ += PhiWith(omega, posteriorSamples[i]);
        meanPhiQ /= posteriorSamples.Count;

        // log-sum-exp, not a direct mean of exponentials. phi is unbounded, so exp(phi) overflows to
        // infinity for a well-trained phi and the estimate would become NaN exactly when it started
        // working. Subtracting the max first keeps every exponent at or below zero.
        double maxPhiP = double.NegativeInfinity;
        var phiP = new double[priorSamples.Count];
        for (int i = 0; i < priorSamples.Count; i++)
        {
            phiP[i] = PhiWith(omega, priorSamples[i]);
            if (phiP[i] > maxPhiP) maxPhiP = phiP[i];
        }

        double sumExp = 0.0;
        for (int i = 0; i < phiP.Length; i++) sumExp += Math.Exp(phiP[i] - maxPhiP);
        double logMeanExpP = maxPhiP + Math.Log(sumExp / phiP.Length);

        return meanPhiQ - logMeanExpP;
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

        var random = rng ?? new Random(31);
        double best = ObjectiveWith(_omega, posteriorSamples, priorSamples);

        for (int step = 0; step < steps; step++)
        {
            double c = Math.Max(1e-4, learningRate);

            var delta = new double[_omega.Length];
            var plus = new Vector<T>(_omega.Length);
            var minus = new Vector<T>(_omega.Length);
            for (int i = 0; i < _omega.Length; i++)
            {
                delta[i] = random.NextDouble() < 0.5 ? -1.0 : 1.0;   // Rademacher, as SPSA requires
                double w = NumOps.ToDouble(_omega[i]);
                plus[i] = NumOps.FromDouble(w + (c * delta[i]));
                minus[i] = NumOps.FromDouble(w - (c * delta[i]));
            }

            double fPlus = ObjectiveWith(plus, posteriorSamples, priorSamples);
            double fMinus = ObjectiveWith(minus, posteriorSamples, priorSamples);
            if (double.IsNaN(fPlus) || double.IsNaN(fMinus)) continue;

            var candidate = new Vector<T>(_omega.Length);
            double scale = (fPlus - fMinus) / (2.0 * c);
            for (int i = 0; i < _omega.Length; i++)
            {
                // ASCENT: plus, because the compression lemma is maximized, not minimized.
                double g = scale / delta[i];
                candidate[i] = NumOps.FromDouble(NumOps.ToDouble(_omega[i]) + (learningRate * g));
            }

            double value = ObjectiveWith(candidate, posteriorSamples, priorSamples);
            if (double.IsNaN(value) || value <= best) continue;

            _omega = candidate;
            best = value;
        }

        return best;
    }

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
}
