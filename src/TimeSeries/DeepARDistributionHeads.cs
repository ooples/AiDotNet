using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Tensors;

namespace AiDotNet.TimeSeries;

/// <summary>
/// The predictive distribution DeepAR emits for a single future step, in NORMALIZED space (the model
/// denormalizes before returning to callers). Every head exposes the same three things so the model's
/// prediction/quantile code is head-agnostic: a point forecast (<see cref="MeanNorm"/>), a representative
/// spread for symmetric intervals (<see cref="ScaleNorm"/>), and a sampler for Monte-Carlo predictive paths.
/// Heads that own a closed-form quantile function (e.g. the spline head) also set <see cref="QuantileNorm"/>,
/// which yields genuinely ASYMMETRIC bands — the property the downstream skew-aware sizing relies on.
/// </summary>
internal sealed class DeepARPredictiveDist<T>
{
    /// <summary>Point forecast (the distribution's location/median) in normalized space.</summary>
    public required T MeanNorm { get; init; }

    /// <summary>Representative standard deviation in normalized space (drives symmetric predictive intervals).</summary>
    public required T ScaleNorm { get; init; }

    /// <summary>Draws one sample from the predictive distribution in normalized space.</summary>
    public required Func<Random, T> SampleNorm { get; init; }

    /// <summary>Closed-form quantile function q(p), p in (0,1), in normalized space — null when the head has
    /// no analytic quantile (callers then fall back to Monte-Carlo sampling of <see cref="SampleNorm"/>).</summary>
    public Func<double, T>? QuantileNorm { get; init; }
}

/// <summary>
/// Base class for DeepAR's pluggable predictive-distribution head. It projects the top LSTM hidden state to
/// the parameters of a chosen distribution and owns the training loss + predictive sampling for that
/// distribution. Concrete heads (Gaussian / Student-t / spline-quantile) let DeepAR trade off calibration
/// against tail-weight and skew without touching the recurrence. All projection tensors register as
/// trainable parameters, and this base handles their flatten/serialize uniformly so heads only declare
/// projections and implement the distribution math. Activations are column-major <c>[*, batch]</c>, and every
/// op is an <c>Engine.Tensor*</c> call so a <see cref="GradientTape{T}"/> differentiates the loss w.r.t. the
/// projection weights (BPTT flows back through the shared LSTM stack).
/// </summary>
internal abstract class DeepARDistributionHead<T> : NeuralNetworks.Layers.LayerBase<T>
{
    private readonly List<Tensor<T>> _params = new();
    protected readonly int Hidden;

    protected DeepARDistributionHead(int hiddenSize, int outputDim)
        : base(new[] { hiddenSize }, new[] { outputDim })
    {
        Hidden = hiddenSize;
    }

    /// <summary>Human-readable name recorded in model metadata (e.g. "Gaussian", "StudentT", "Spline").</summary>
    public abstract string LikelihoodName { get; }

    /// <summary>
    /// Batched training loss (scalar tensor), built from tape-tracked engine ops. <paramref name="hiddenSteps"/>
    /// holds the top-layer hidden state <c>[H, B]</c> for each of the L timesteps; <paramref name="obsSteps"/>
    /// holds the corresponding observation input <c>[1, B]</c> (the value the residual-mean skip is added to);
    /// <paramref name="target"/> is the one-step-ahead target window <c>[B, L]</c>.
    /// </summary>
    public abstract Tensor<T> ComputeBatchLoss(
        IReadOnlyList<Tensor<T>> hiddenSteps, IReadOnlyList<Tensor<T>> obsSteps, Tensor<T> target);

    /// <summary>
    /// Eager single-step predictive distribution (batch = 1) in normalized space, from the final hidden state
    /// <paramref name="hLast"/> <c>[H, 1]</c> and the last normalized observation <paramref name="lastObsNorm"/>
    /// (for the residual-mean skip). Same ops as training, run outside a tape.
    /// </summary>
    public abstract DeepARPredictiveDist<T> PredictNorm(Tensor<T> hLast, T lastObsNorm);

    // --- shared projection helpers ------------------------------------------------------------------

    /// <summary>Creates + registers a linear projection: weight <c>[outDim, H]</c> and bias <c>[outDim]</c>.</summary>
    protected (Tensor<T> w, Tensor<T> b) AddProjection(int outDim, Random random)
    {
        double stddev = Math.Sqrt(2.0 / Hidden);
        var w = new Tensor<T>(new[] { outDim, Hidden });
        for (int i = 0; i < w.Length; i++)
            w[i] = NumOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        var b = new Tensor<T>(new[] { outDim });

        RegisterTrainableParameter(w, PersistentTensorRole.Weights);
        RegisterTrainableParameter(b, PersistentTensorRole.Biases);
        _params.Add(w);
        _params.Add(b);
        return (w, b);
    }

    /// <summary>Applies a registered projection to hidden state <paramref name="h"/> <c>[H, B]</c> → <c>[outDim, B]</c>.</summary>
    protected Tensor<T> Linear(Tensor<T> w, Tensor<T> b, Tensor<T> h)
    {
        var o = Engine.TensorMatMul(w, h);
        var bCol = Engine.Reshape(b, new[] { b.Length, 1 });
        return Engine.TensorBroadcastAdd(o, bCol);
    }

    /// <summary>Numerically-stable scalar softplus (matches the model's inference-time softplus).</summary>
    protected T Softplus(T v)
    {
        if (NumOps.GreaterThan(v, NumOps.FromDouble(20.0)))
            return v;
        if (NumOps.LessThan(v, NumOps.FromDouble(-20.0)))
            return NumOps.Exp(v);
        return NumOps.Log(NumOps.Add(NumOps.One, NumOps.Exp(v)));
    }

    /// <summary>Concatenates per-step <c>[1, B]</c> slices into <c>[L, B]</c> then permutes to <c>[B, L]</c>.</summary>
    protected Tensor<T> StackStepsToBL(Tensor<T>[] steps)
    {
        var lb = Engine.TensorConcatenate(steps, axis: 0);
        return Engine.TensorPermute(lb, new[] { 1, 0 });
    }

    // --- LayerBase plumbing (uniform over the registered projection tensors) ------------------------

    public override bool SupportsTraining => true;
    public override void ResetState() { }
    public override void UpdateParameters(T learningRate) { /* tape-based optimizer updates registered params */ }

    public override long ParameterCount
    {
        get
        {
            long count = 0;
            foreach (var p in _params)
                count += p.Length;
            return count;
        }
    }

    public override Vector<T> GetParameters()
    {
        long total = ParameterCount;
        var arr = new T[total];
        int idx = 0;
        foreach (var p in _params)
            for (int i = 0; i < p.Length; i++)
                arr[idx++] = p[i];
        return new Vector<T>(arr);
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int idx = 0;
        foreach (var p in _params)
            for (int i = 0; i < p.Length; i++)
                p[i] = parameters[idx++];
    }

    public override void Serialize(BinaryWriter writer)
    {
        writer.Write(Hidden);
        writer.Write(_params.Count);
        foreach (var p in _params)
            WriteTensor(writer, p);
    }

    public override void Deserialize(BinaryReader reader)
    {
        reader.ReadInt32(); // hidden
        int count = reader.ReadInt32();
        for (int i = 0; i < count && i < _params.Count; i++)
            ReadTensorInto(reader, _params[i]);
    }

    protected static void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        for (int d = 0; d < tensor.Shape.Length; d++)
            writer.Write(tensor.Shape[d]);
        for (int i = 0; i < tensor.Length; i++)
            writer.Write(Convert.ToDouble(tensor[i]));
    }

    protected void ReadTensorInto(BinaryReader reader, Tensor<T> tensor)
    {
        int rank = reader.ReadInt32();
        int total = 1;
        for (int d = 0; d < rank; d++)
            total *= reader.ReadInt32();
        for (int i = 0; i < total; i++)
        {
            double v = reader.ReadDouble();
            if (i < tensor.Length)
                tensor[i] = NumOps.FromDouble(v);
        }
    }
}

/// <summary>
/// Gaussian likelihood head: emits a residual mean (μ = x + Δ(h)) and a softplus scale. The training loss
/// splits into an undivided MSE on μ (so the mean always tracks the target) plus a Gaussian NLL on the
/// detached mean for σ (so the scale learns the residual spread without an "inflate σ to kill the μ gradient"
/// escape hatch). This is the historical DeepAR head, preserved bit-for-bit as the default.
/// </summary>
internal sealed class DeepARGaussianHead<T> : DeepARDistributionHead<T>
{
    private readonly Tensor<T> _meanW, _meanB, _scaleW, _scaleB;

    public override string LikelihoodName => "Gaussian";

    public DeepARGaussianHead(int hiddenSize, int seed = 12345)
        : base(hiddenSize, outputDim: 1)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        (_meanW, _meanB) = AddProjection(1, random);
        (_scaleW, _scaleB) = AddProjection(1, random);
    }

    public override Tensor<T> Forward(Tensor<T> input) => Linear(_meanW, _meanB, input);

    public override Tensor<T> ComputeBatchLoss(
        IReadOnlyList<Tensor<T>> hiddenSteps, IReadOnlyList<Tensor<T>> obsSteps, Tensor<T> target)
    {
        int L = hiddenSteps.Count;
        var meanSteps = new Tensor<T>[L];
        var scaleRawSteps = new Tensor<T>[L];
        for (int t = 0; t < L; t++)
        {
            var delta = Linear(_meanW, _meanB, hiddenSteps[t]);        // [1, B]
            meanSteps[t] = Engine.TensorAdd(obsSteps[t], delta);       // residual skip
            scaleRawSteps[t] = Linear(_scaleW, _scaleB, hiddenSteps[t]);
        }

        var mean = StackStepsToBL(meanSteps);                          // [B, L]
        var scale = Engine.Softplus(StackStepsToBL(scaleRawSteps));    // [B, L]
        var allAxes = Enumerable.Range(0, mean.Shape.Length).ToArray();

        // Mean branch: plain MSE — gradient (μ − y) is independent of σ, so μ always tracks.
        var diffMean = Engine.TensorSubtract(mean, target);
        var meanMse = Engine.ReduceMean(Engine.TensorMultiply(diffMean, diffMean), allAxes, keepDims: false);

        // Scale branch: Gaussian NLL over the detached mean.
        var eps = NumOps.FromDouble(1e-6);
        var muDetached = Engine.StopGradient(mean);
        var diff = Engine.TensorSubtract(target, muDetached);
        var diff2 = Engine.TensorMultiply(diff, diff);
        var variance = Engine.TensorMultiply(scale, scale);
        var twoVar = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(variance, NumOps.FromDouble(2.0)), eps);
        var quad = Engine.TensorDivide(diff2, twoVar);
        var logSigma = Engine.TensorLog(Engine.TensorAddScalar(scale, eps));
        var scaleTerm = Engine.ReduceMean(Engine.TensorAdd(quad, logSigma), allAxes, keepDims: false);

        return Engine.TensorAdd(meanMse, scaleTerm);
    }

    public override DeepARPredictiveDist<T> PredictNorm(Tensor<T> hLast, T lastObsNorm)
    {
        var delta = Linear(_meanW, _meanB, hLast);
        var scaleRaw = Linear(_scaleW, _scaleB, hLast);
        T meanNorm = NumOps.Add(lastObsNorm, delta[0, 0]);
        T scaleNorm = Softplus(scaleRaw[0, 0]);
        if (NumOps.LessThan(scaleNorm, NumOps.FromDouble(1e-6)))
            scaleNorm = NumOps.FromDouble(1e-6);

        return new DeepARPredictiveDist<T>
        {
            MeanNorm = meanNorm,
            ScaleNorm = scaleNorm,
            SampleNorm = rng => NumOps.Add(meanNorm, NumOps.Multiply(scaleNorm, NumOps.FromDouble(rng.NextGaussian()))),
            QuantileNorm = p => NumOps.Add(meanNorm, NumOps.Multiply(scaleNorm, NumOps.FromDouble(DeepARDistMath.Probit(p)))),
        };
    }
}

/// <summary>
/// Student-t likelihood head: same residual mean + softplus scale as the Gaussian, but a Student-t
/// negative log-likelihood with FIXED degrees-of-freedom ν (finance default ν≈4). Heavy tails mean the head
/// stops under-estimating the probability of large moves — the exact failure mode a Gaussian head has on
/// returns — so predictive intervals and the uncertainty they feed to sizing are far better calibrated on
/// fat-tailed data. ν is fixed (not learned) so the log-Γ normalisation constants are true constants and the
/// loss stays fully differentiable in (μ, σ) with standard tape ops — no differentiable log-Γ required.
/// </summary>
internal sealed class DeepARStudentTHead<T> : DeepARDistributionHead<T>
{
    private readonly Tensor<T> _meanW, _meanB, _scaleW, _scaleB;
    private readonly double _nu;
    private readonly T _nuT, _halfNuPlus1, _logNormConst, _stdScale;

    public override string LikelihoodName => "StudentT";

    public DeepARStudentTHead(int hiddenSize, double degreesOfFreedom, int seed = 12345)
        : base(hiddenSize, outputDim: 1)
    {
        // ν must exceed 2 for a finite variance (so predictive std is defined). Clamp defensively.
        _nu = degreesOfFreedom > 2.0001 ? degreesOfFreedom : 2.0001;
        var random = RandomHelper.CreateSeededRandom(seed);
        (_meanW, _meanB) = AddProjection(1, random);
        (_scaleW, _scaleB) = AddProjection(1, random);

        _nuT = NumOps.FromDouble(_nu);
        _halfNuPlus1 = NumOps.FromDouble((_nu + 1.0) / 2.0);
        // Constant term of the t NLL: -logΓ((ν+1)/2) + logΓ(ν/2) + 0.5·log(νπ). Affects only the loss value,
        // not the (μ, σ) gradients, but included so the reported loss is a genuine NLL.
        double logConst = -DeepARDistMath.LogGamma((_nu + 1.0) / 2.0)
                          + DeepARDistMath.LogGamma(_nu / 2.0)
                          + 0.5 * Math.Log(_nu * Math.PI);
        _logNormConst = NumOps.FromDouble(logConst);
        _stdScale = NumOps.FromDouble(Math.Sqrt(_nu / (_nu - 2.0))); // std = σ·sqrt(ν/(ν−2))
    }

    public override Tensor<T> Forward(Tensor<T> input) => Linear(_meanW, _meanB, input);

    public override Tensor<T> ComputeBatchLoss(
        IReadOnlyList<Tensor<T>> hiddenSteps, IReadOnlyList<Tensor<T>> obsSteps, Tensor<T> target)
    {
        int L = hiddenSteps.Count;
        var meanSteps = new Tensor<T>[L];
        var scaleRawSteps = new Tensor<T>[L];
        for (int t = 0; t < L; t++)
        {
            var delta = Linear(_meanW, _meanB, hiddenSteps[t]);
            meanSteps[t] = Engine.TensorAdd(obsSteps[t], delta);
            scaleRawSteps[t] = Linear(_scaleW, _scaleB, hiddenSteps[t]);
        }

        var mean = StackStepsToBL(meanSteps);
        var scale = Engine.Softplus(StackStepsToBL(scaleRawSteps));
        var allAxes = Enumerable.Range(0, mean.Shape.Length).ToArray();

        // Mean branch: undivided MSE (same stability rationale as the Gaussian head).
        var diffMean = Engine.TensorSubtract(mean, target);
        var meanMse = Engine.ReduceMean(Engine.TensorMultiply(diffMean, diffMean), allAxes, keepDims: false);

        // Tail branch: Student-t NLL over the detached mean.
        //   NLL_t = const + log σ + ((ν+1)/2)·log(1 + z²/ν),  z = (y − μ)/σ.
        var eps = NumOps.FromDouble(1e-6);
        var muDetached = Engine.StopGradient(mean);
        var z = Engine.TensorDivide(Engine.TensorSubtract(target, muDetached), Engine.TensorAddScalar(scale, eps));
        var z2OverNu = Engine.TensorDivideScalar(Engine.TensorMultiply(z, z), _nuT);
        var logTail = Engine.TensorMultiplyScalar(
            Engine.TensorLog(Engine.TensorAddScalar(z2OverNu, NumOps.One)), _halfNuPlus1);
        var logSigma = Engine.TensorLog(Engine.TensorAddScalar(scale, eps));
        var perStep = Engine.TensorAddScalar(Engine.TensorAdd(logSigma, logTail), _logNormConst);
        var tailTerm = Engine.ReduceMean(perStep, allAxes, keepDims: false);

        return Engine.TensorAdd(meanMse, tailTerm);
    }

    public override DeepARPredictiveDist<T> PredictNorm(Tensor<T> hLast, T lastObsNorm)
    {
        var delta = Linear(_meanW, _meanB, hLast);
        var scaleRaw = Linear(_scaleW, _scaleB, hLast);
        T meanNorm = NumOps.Add(lastObsNorm, delta[0, 0]);
        T sigma = Softplus(scaleRaw[0, 0]);
        if (NumOps.LessThan(sigma, NumOps.FromDouble(1e-6)))
            sigma = NumOps.FromDouble(1e-6);

        double nu = _nu;
        return new DeepARPredictiveDist<T>
        {
            MeanNorm = meanNorm,
            ScaleNorm = NumOps.Multiply(sigma, _stdScale),
            SampleNorm = rng => NumOps.Add(meanNorm, NumOps.Multiply(sigma, NumOps.FromDouble(DeepARDistMath.SampleStudentT(nu, rng)))),
            QuantileNorm = p => NumOps.Add(meanNorm, NumOps.Multiply(sigma, NumOps.FromDouble(DeepARDistMath.StudentTQuantile(p, nu)))),
        };
    }
}

/// <summary>
/// Spline (monotone piecewise-linear quantile-function) head: instead of assuming a parametric shape, it
/// emits the forecast's quantiles directly on a fixed probability grid and is trained with the pinball
/// (quantile) loss — a proper scoring rule that approximates the CRPS. Monotonicity is guaranteed by
/// construction (q at the first grid point, then cumulative softplus increments), so the predictive
/// distribution can be ASYMMETRIC and multi-modal-ish — the head that actually models skew, which the
/// downstream skew-aware sizing consumes via the closed-form quantile function.
/// </summary>
internal sealed class DeepARSplineHead<T> : DeepARDistributionHead<T>
{
    // Fixed probability grid (must be strictly increasing, symmetric around 0.5 for a sane median).
    private static readonly double[] Grid = { 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95 };
    private const int MedianIndex = 3; // Grid[3] == 0.5

    private readonly Tensor<T> _knotW, _knotB; // projects hidden → [K] raw knot params
    private readonly int _k = Grid.Length;

    public override string LikelihoodName => "Spline";

    public DeepARSplineHead(int hiddenSize, int seed = 12345)
        : base(hiddenSize, outputDim: Grid.Length)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        (_knotW, _knotB) = AddProjection(Grid.Length, random);
    }

    public override Tensor<T> Forward(Tensor<T> input) => Linear(_knotW, _knotB, input);

    // Builds the K monotone knot quantiles for every step as [B, L] tensors (index k → q at Grid[k]).
    // q[0] = obs + raw[0];  q[k] = q[k-1] + softplus(raw[k]) for k>0  → strictly non-decreasing in k.
    private Tensor<T>[] BuildKnotsBL(IReadOnlyList<Tensor<T>> hiddenSteps, IReadOnlyList<Tensor<T>> obsSteps)
    {
        int L = hiddenSteps.Count;
        var rawSteps = new Tensor<T>[L];      // each [K, B]
        for (int t = 0; t < L; t++)
            rawSteps[t] = Linear(_knotW, _knotB, hiddenSteps[t]);

        var knots = new Tensor<T>[_k];        // knots[k] → [B, L]
        for (int k = 0; k < _k; k++)
        {
            var stepK = new Tensor<T>[L];     // [1, B] per step for knot k
            for (int t = 0; t < L; t++)
                stepK[t] = Engine.TensorNarrow(rawSteps[t], 0, k, 1); // [1, B]
            var rawKnotBL = StackStepsToBL(stepK); // [B, L]

            if (k == 0)
            {
                var obsBL = StackStepsToBL(obsSteps.ToArray()); // [B, L]
                knots[0] = Engine.TensorAdd(obsBL, rawKnotBL);  // base quantile = obs + raw[0]
            }
            else
            {
                knots[k] = Engine.TensorAdd(knots[k - 1], Engine.Softplus(rawKnotBL)); // + positive increment
            }
        }

        return knots;
    }

    public override Tensor<T> ComputeBatchLoss(
        IReadOnlyList<Tensor<T>> hiddenSteps, IReadOnlyList<Tensor<T>> obsSteps, Tensor<T> target)
    {
        var knots = BuildKnotsBL(hiddenSteps, obsSteps);
        var allAxes = Enumerable.Range(0, target.Shape.Length).ToArray();
        var half = NumOps.FromDouble(0.5);

        Tensor<T>? loss = null;
        for (int k = 0; k < _k; k++)
        {
            // pinball(τ, q, y) = τ·(y−q) + relu(q−y);  relu(x) = (|x| + x)/2.
            var e = Engine.TensorSubtract(target, knots[k]);                 // y − q
            var negE = Engine.TensorSubtract(knots[k], target);              // q − y
            var reluNegE = Engine.TensorMultiplyScalar(
                Engine.TensorAdd(Engine.TensorAbs(negE), negE), half);
            var pinball = Engine.TensorAdd(Engine.TensorMultiplyScalar(e, NumOps.FromDouble(Grid[k])), reluNegE);
            var term = Engine.ReduceMean(pinball, allAxes, keepDims: false);
            loss = loss is null ? term : Engine.TensorAdd(loss, term);
        }

        // Mean of per-knot pinball terms (keeps the loss scale independent of grid size).
        return Engine.TensorMultiplyScalar(loss!, NumOps.FromDouble(1.0 / _k));
    }

    public override DeepARPredictiveDist<T> PredictNorm(Tensor<T> hLast, T lastObsNorm)
    {
        var raw = Linear(_knotW, _knotB, hLast); // [K, 1]
        var q = new double[_k];
        q[0] = Convert.ToDouble(lastObsNorm) + Convert.ToDouble(raw[0, 0]);
        for (int k = 1; k < _k; k++)
            q[k] = q[k - 1] + Convert.ToDouble(Softplus(raw[k, 0]));

        double median = q[MedianIndex];
        // Robust spread proxy: (q95 − q05) / (2·1.645) ≈ σ for a Gaussian, but honest about the fitted tails.
        double scale = Math.Max((q[_k - 1] - q[0]) / (2.0 * 1.645), 1e-6);

        double InterpQuantile(double p)
        {
            if (p <= Grid[0]) return q[0];
            if (p >= Grid[_k - 1]) return q[_k - 1];
            for (int k = 1; k < _k; k++)
            {
                if (p <= Grid[k])
                {
                    double w = (p - Grid[k - 1]) / (Grid[k] - Grid[k - 1]);
                    return q[k - 1] + w * (q[k] - q[k - 1]);
                }
            }

            return q[_k - 1];
        }

        return new DeepARPredictiveDist<T>
        {
            MeanNorm = NumOps.FromDouble(median),
            ScaleNorm = NumOps.FromDouble(scale),
            // Inverse-CDF sampling through the piecewise-linear quantile function → respects the fitted skew.
            SampleNorm = rng => NumOps.FromDouble(InterpQuantile(rng.NextDouble())),
            QuantileNorm = p => NumOps.FromDouble(InterpQuantile(p)),
        };
    }
}

/// <summary>
/// Small distribution-math helpers (log-Γ, inverse normal CDF, Student-t sampling/quantiles) used by the
/// DeepAR heads. All operate on <see cref="double"/> — they compute per-scalar constants or draw samples
/// outside the gradient tape, so no autodiff is required.
/// </summary>
internal static class DeepARDistMath
{
    // Lanczos approximation to log Γ(x) for x > 0.
    private static readonly double[] LanczosG = {
        676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
        12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    };

    public static double LogGamma(double x)
    {
        if (x < 0.5)
        {
            // Reflection formula: Γ(x)Γ(1−x) = π / sin(πx).
            return Math.Log(Math.PI / Math.Sin(Math.PI * x)) - LogGamma(1.0 - x);
        }

        x -= 1.0;
        double a = 0.99999999999980993;
        double tt = x + 7.5;
        for (int i = 0; i < LanczosG.Length; i++)
            a += LanczosG[i] / (x + i + 1);
        return 0.5 * Math.Log(2 * Math.PI) + (x + 0.5) * Math.Log(tt) - tt + Math.Log(a);
    }

    // Acklam's rational approximation to the inverse standard-normal CDF (probit).
    public static double Probit(double p)
    {
        if (p <= 0.0) return -8.0;
        if (p >= 1.0) return 8.0;

        double[] a = { -39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924 };
        double[] b = { -54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857 };
        double[] c = { -0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878 };
        double[] d = { 0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742 };
        double pLow = 0.02425, pHigh = 1 - pLow;

        if (p < pLow)
        {
            double qq = Math.Sqrt(-2 * Math.Log(p));
            return (((((c[0] * qq + c[1]) * qq + c[2]) * qq + c[3]) * qq + c[4]) * qq + c[5]) /
                   ((((d[0] * qq + d[1]) * qq + d[2]) * qq + d[3]) * qq + 1);
        }

        if (p <= pHigh)
        {
            double qq = p - 0.5;
            double r = qq * qq;
            return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * qq /
                   (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
        }

        double q2 = Math.Sqrt(-2 * Math.Log(1 - p));
        return -(((((c[0] * q2 + c[1]) * q2 + c[2]) * q2 + c[3]) * q2 + c[4]) * q2 + c[5]) /
                ((((d[0] * q2 + d[1]) * q2 + d[2]) * q2 + d[3]) * q2 + 1);
    }

    // Draw a standard Student-t(ν) sample as z / sqrt(g/ν), z~N(0,1), g~ChiSq(ν)=Gamma(ν/2, 2).
    public static double SampleStudentT(double nu, Random rng)
    {
        double z = rng.NextGaussian();
        double g = SampleGamma(nu / 2.0, 2.0, rng);
        double denom = Math.Sqrt(g / nu);
        return denom > 1e-12 ? z / denom : z;
    }

    // Marsaglia–Tsang gamma sampler (shape k > 0, scale θ).
    private static double SampleGamma(double k, double theta, Random rng)
    {
        if (k < 1.0)
        {
            // Boost: Gamma(k) = Gamma(k+1) · U^(1/k).
            double u = rng.NextDouble();
            return SampleGamma(k + 1.0, theta, rng) * Math.Pow(u <= 0 ? 1e-12 : u, 1.0 / k);
        }

        double d = k - 1.0 / 3.0;
        double c = 1.0 / Math.Sqrt(9.0 * d);
        while (true)
        {
            double x = rng.NextGaussian();
            double v = 1.0 + c * x;
            if (v <= 0) continue;
            v = v * v * v;
            double u = rng.NextDouble();
            if (u < 1.0 - 0.0331 * x * x * x * x)
                return d * v * theta;
            if (Math.Log(u <= 0 ? 1e-12 : u) < 0.5 * x * x + d * (1.0 - v + Math.Log(v)))
                return d * v * theta;
        }
    }

    // Student-t quantile via a Cornish-Fisher expansion off the normal quantile (accurate for ν ≳ 3, the
    // regime we use). Symmetric about 0.
    public static double StudentTQuantile(double p, double nu)
    {
        double z = Probit(p);
        double z3 = z * z * z;
        double z5 = z3 * z * z;
        double g1 = (z3 + z) / 4.0;
        double g2 = (5 * z5 + 16 * z3 + 3 * z) / 96.0;
        double g3 = (3 * z5 * z * z + 19 * z5 + 17 * z3 - 15 * z) / 384.0;
        return z + g1 / nu + g2 / (nu * nu) + g3 / (nu * nu * nu);
    }
}
