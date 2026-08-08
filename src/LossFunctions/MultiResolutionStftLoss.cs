using System.Collections.Concurrent;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Multi-resolution STFT loss: spectral convergence plus log-magnitude distance, summed over
/// several STFT resolutions.
/// </summary>
/// <remarks>
/// <para>
/// For each resolution r the loss adds two terms over the magnitude spectrograms of the prediction
/// and the target:
/// </para>
/// <para>
/// spectral convergence <c>L_SC = ||  |STFT(h)| - |STFT(h^)|  ||_F / ||  |STFT(h)|  ||_F</c>, and
/// log-magnitude <c>L_SM = (1/N) ||  log|STFT(h)| - log|STFT(h^)|  ||_1</c>.
/// </para>
/// <para>
/// This is the objective FiNS trains with (Steinmetz, Ithapu and Calamia, WASPAA 2021,
/// arXiv:2107.07503), where the paper reports the multi-resolution STFT loss ALONE produced the
/// best results, at R = 4 resolutions with frame sizes 64, 512, 2048 and 8192. It is also the
/// standard Parallel WaveGAN objective (Yamamoto et al. 2020), so it is reusable by any waveform
/// model, not only FiNS.
/// </para>
/// <para>
/// <b>Why the transform is hand-rolled.</b> <c>IEngine.STFT</c> returns magnitude and phase through
/// <c>out</c> parameters, which places it outside the autodiff graph — the same reason
/// <see cref="APNet2GeneratorLoss{T}"/> cannot use it for its STFT-consistency term. So each
/// resolution keeps constant cosine/sine DFT bases and frames the signal with differentiable tensor
/// ops, giving a magnitude that gradients actually flow through.
/// </para>
/// <para>
/// <b>For Beginners:</b> Comparing two sounds sample by sample is a poor measure of whether they
/// sound alike — shifting a waveform slightly changes every sample while sounding identical.
/// Comparing spectrograms instead asks "does it have the same energy at the same frequencies at
/// roughly the same times", which is much closer to hearing. Doing it at several zoom levels at
/// once matters because a short window resolves sharp clicks and a long window resolves slow
/// decays, and a room impulse response has both.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
public sealed class MultiResolutionStftLoss<T> : LossFunctionBase<T>
{
    private readonly int[] _frameSizes;
    private readonly double _epsilon;

    // Per-resolution constants, built lazily and reused. They carry no gradient, and rebuilding a
    // [frame, bins] basis pair every training step would cost more than the loss itself.
    // CONCURRENT, because a loss instance is shared. Plain Dictionary with an unsynchronized
    // read-then-insert corrupts its internal buckets when two threads miss and insert at once,
    // which surfaces as an intermittent crash or an infinite loop rather than as a wrong number.
    private readonly ConcurrentDictionary<int, (Tensor<T> Window, Tensor<T> Cos, Tensor<T> Sin)> _bases = new();

    // Every constant this loss feeds into the graph is cached and reused, never rebuilt per call.
    // The fused compiled training path TRACES the graph on the first step and REPLAYS it on every
    // step after, capturing the tensor references it saw. A constant allocated fresh inside each
    // ComputeTapeLoss is therefore captured once and then replayed against a tensor the caller has
    // moved on from — and with pooled allocation that buffer can be recycled underneath the plan,
    // which is how the fused step ended up reading NaN while eager evaluation of the same loss
    // stayed finite in both precisions.
    private readonly ConcurrentDictionary<int, Tensor<T>[]> _cosColumns = new();
    private readonly ConcurrentDictionary<int, Tensor<T>[]> _sinColumns = new();
    private readonly ConcurrentDictionary<string, Tensor<T>> _constants = new();

    /// <summary>
    /// Creates the loss over the given STFT frame sizes.
    /// </summary>
    /// <param name="frameSizes">
    /// Frame sizes, one per resolution. Defaults to the paper's {64, 512, 2048, 8192}.
    /// </param>
    /// <param name="epsilon">
    /// Floor added under the square root and inside the logarithm. Magnitudes reach exactly zero in
    /// silent bins, where both sqrt' and log are undefined, so this keeps the gradient finite.
    /// </param>
    public MultiResolutionStftLoss(int[]? frameSizes = null, double epsilon = 1e-7)
    {
        _frameSizes = frameSizes is { Length: > 0 } ? (int[])frameSizes.Clone() : [64, 512, 2048, 8192];
        foreach (int frameSize in _frameSizes)
        {
            if (frameSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(frameSizes), frameSize, "STFT frame sizes must be positive.");
        }
        if (epsilon <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(epsilon), epsilon, "Epsilon must be positive.");
        _epsilon = epsilon;
    }

    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        int length = Math.Min(predicted.Length, target.Length);
        if (length <= 1) return Engine.ReduceMean(Engine.TensorMultiply(
            Engine.TensorSubtract(predicted, target), Engine.TensorSubtract(predicted, target)), [0], keepDims: false);

        // Only reshape when the tensor is not already flat. A no-op Reshape still routes through the
        // engine, and if that op is not tape-recorded it DETACHES the prediction from the graph —
        // which shows up as a loss that is byte-identical at step 1 and step 100.
        var flatPredicted = predicted.Rank == 1 ? predicted : Engine.Reshape(predicted, [predicted.Length]);
        var flatTarget = target.Rank == 1 ? target : Engine.Reshape(target, [target.Length]);
        if (predicted.Length != length) flatPredicted = Engine.TensorNarrow(flatPredicted, 0, 0, length);
        if (target.Length != length) flatTarget = Engine.TensorNarrow(flatTarget, 0, 0, length);

        Tensor<T>? total = null;
        int used = 0;
        foreach (int requested in _frameSizes)
        {
            // A frame longer than the signal yields no frames at all. Rather than skip the
            // resolution outright — which would silently drop a term the caller asked for — fall
            // back to the largest power of two that fits, so short signals (test-scale responses,
            // for instance) still contribute a genuine spectral term at that scale.
            int frameSize = requested <= length ? requested : LargestPowerOfTwoAtMost(length);
            if (frameSize < 2) continue;

            var resolutionLoss = ResolutionLoss(flatPredicted, flatTarget, length, frameSize);
            if (resolutionLoss is null) continue;

            total = total is null ? resolutionLoss : Engine.TensorAdd(total, resolutionLoss);
            used++;
        }

        if (total is null || used == 0)
        {
            var diff = Engine.TensorSubtract(flatPredicted, flatTarget);
            return Engine.ReduceMean(Engine.TensorMultiply(diff, diff), [0], keepDims: false);
        }

        // Reduce to a SCALAR, matching what every other LossFunctionBase returns
        // (MeanSquaredErrorLoss ends with ReduceMean over all axes, keepDims: false). Returning a
        // rank-1 [1] tensor instead leaves the tape without a scalar to seed the backward from, so
        // no gradient reaches the parameters — which presented as a memorization loss that was
        // byte-identical at step 1 and step 100 even though the loss itself differentiates cleanly
        // in isolation (measured max|grad| 8.547E+001 against a leaf input).
        return Engine.ReduceMean(total, AllAxes(total), keepDims: false);
    }

    /// <summary>Spectral convergence + log-magnitude at one resolution, or null if no frame fits.</summary>
    /// <remarks>
    /// Deliberately free of Reshape and Concatenate on tape-connected tensors. Reshape's
    /// tape-backward path does not reliably propagate gradients to its source in the current Tensors
    /// engine — NeuralNetworkBase.TrainWithTape documents the same hazard and reshapes the TARGET
    /// rather than the network output for exactly this reason. An earlier version framed the signal
    /// by reshaping each windowed frame to [1, frameSize] and concatenating, which silently severed
    /// the chain: the loss differentiated perfectly against a leaf input (max|grad| 8.5e+01) while
    /// training left the memorization loss byte-identical at step 1 and step 100.
    ///
    /// Instead each frame is reduced to SCALAR accumulators, so the paper's joint norms over the
    /// whole spectrogram are preserved without ever changing a tape-connected tensor's shape.
    /// </remarks>
    private Tensor<T>? ResolutionLoss(Tensor<T> predicted, Tensor<T> target, int length, int frameSize)
    {
        int hop = Math.Max(1, frameSize / 4);
        int numFrames = 1 + (length - frameSize) / hop;
        if (numFrames <= 0) return null;

        var (window, cos, sin) = GetBasis(frameSize);
        int bins = frameSize / 2 + 1;

        Tensor<T>? differenceSquared = null;   // sum over all frames/bins of (magT - magP)^2
        Tensor<T>? targetSquared = null;       // sum over all frames/bins of magT^2
        Tensor<T>? logAbsolute = null;         // sum over all frames/bins of |log magT - log magP|

        for (int f = 0; f < numFrames; f++)
        {
            var predictedMagnitude = FrameMagnitude(predicted, f * hop, frameSize, bins, window, cos, sin);
            var targetMagnitude = FrameMagnitude(target, f * hop, frameSize, bins, window, cos, sin);

            var difference = Engine.TensorSubtract(targetMagnitude, predictedMagnitude);
            var frameDifferenceSquared = Engine.ReduceSum(Engine.TensorMultiply(difference, difference), [0], keepDims: true);
            var frameTargetSquared = Engine.ReduceSum(Engine.TensorMultiply(targetMagnitude, targetMagnitude), [0], keepDims: true);

            var logDifference = Engine.TensorAbs(Engine.TensorSubtract(
                Engine.TensorLog(AddEpsilon(targetMagnitude)),
                Engine.TensorLog(AddEpsilon(predictedMagnitude))));
            var frameLogAbsolute = Engine.ReduceSum(logDifference, [0], keepDims: true);

            differenceSquared = differenceSquared is null ? frameDifferenceSquared : Engine.TensorAdd(differenceSquared, frameDifferenceSquared);
            targetSquared = targetSquared is null ? frameTargetSquared : Engine.TensorAdd(targetSquared, frameTargetSquared);
            logAbsolute = logAbsolute is null ? frameLogAbsolute : Engine.TensorAdd(logAbsolute, frameLogAbsolute);
        }

        if (differenceSquared is null || targetSquared is null || logAbsolute is null) return null;

        // L_SC = ||targ - pred||_F / ||targ||_F over the whole spectrogram.
        var spectralConvergence = Engine.TensorDivide(
            Engine.TensorSqrt(AddEpsilon(differenceSquared)),
            Engine.TensorSqrt(AddEpsilon(targetSquared)));

        // L_SM = mean |log magT - log magP| over the whole spectrogram.
        var count = Constant(logAbsolute, Math.Max(1, numFrames * bins));
        var logMagnitude = Engine.TensorDivide(logAbsolute, count);

        return Engine.TensorAdd(spectralConvergence, logMagnitude);
    }

    /// <summary>
    /// Magnitude spectrum of ONE frame: window, project onto the DFT bases, then
    /// <c>sqrt(re^2 + im^2)</c>. Shape <c>[bins]</c>, produced without reshaping the input.
    /// </summary>
    private Tensor<T> FrameMagnitude(
        Tensor<T> signal, int offset, int frameSize, int bins,
        Tensor<T> window, Tensor<T> cos, Tensor<T> sin)
    {
        var windowed = Engine.TensorMultiply(Engine.TensorNarrow(signal, 0, offset, frameSize), window);

        // Per-bin projection by elementwise multiply + reduce. A matmul would need the frame as a
        // rank-2 row, and reshaping a tape-connected tensor is the very thing that broke gradients.
        Tensor<T>? magnitude = null;
        for (int k = 0; k < bins; k++)
        {
            var real = Engine.ReduceSum(Engine.TensorMultiply(windowed, FlatColumn(cos, k, frameSize)), [0], keepDims: true);
            var imaginary = Engine.ReduceSum(Engine.TensorMultiply(windowed, FlatColumn(sin, k, frameSize)), [0], keepDims: true);
            var power = Engine.TensorAdd(Engine.TensorMultiply(real, real), Engine.TensorMultiply(imaginary, imaginary));
            var binMagnitude = Engine.TensorSqrt(AddEpsilon(power));
            magnitude = magnitude is null ? binMagnitude : Engine.TensorConcatenate([magnitude, binMagnitude], 0);
        }

        return magnitude!;
    }

    /// <summary>
    /// Column <paramref name="k"/> of a constant [frameSize, bins] basis as a flat [frameSize].
    /// </summary>
    /// <remarks>
    /// Materialized directly rather than narrowed-and-reshaped. These bases are CONSTANTS, so no
    /// gradient needs to flow through them, and building the vector here keeps Reshape away from
    /// the tape entirely.
    /// </remarks>
    private Tensor<T> FlatColumn(Tensor<T> basis, int k, int frameSize)
    {
        bool isCos = ReferenceEquals(basis, _bases[frameSize].Cos);
        var cache = isCos ? _cosColumns : _sinColumns;

        // GetOrAdd rather than read-then-insert: exactly one built array wins, and every caller then
        // sees the SAME tensor instances. That matters beyond thread safety here -- the fused
        // compiled path captures the tensor references it traced, so two racing inserts handing out
        // two different instances of the same constant is the hazard this cache exists to avoid.
        var columns = cache.GetOrAdd(frameSize, _ =>
        {
            int bins = frameSize / 2 + 1;
            var built = new Tensor<T>[bins];
            for (int b = 0; b < bins; b++)
            {
                var column = new Tensor<T>([frameSize]);
                for (int n = 0; n < frameSize; n++) column[n] = basis[n, b];
                built[b] = column;
            }
            return built;
        });

        return columns[k];
    }

    /// <summary>A cached constant shaped like <paramref name="like"/>, filled with <paramref name="value"/>.</summary>
    private Tensor<T> Constant(Tensor<T> like, double value)
    {
        string key = string.Join("x", like.Shape.ToArray()) + "@" + value.ToString("R");
        return _constants.GetOrAdd(key, _ =>
        {
            var ops = MathHelper.GetNumericOperations<T>();
            var t = new Tensor<T>(like.Shape.ToArray());
            T v = ops.FromDouble(value);
            for (int i = 0; i < t.Length; i++) t[i] = v;
            return t;
        });
    }

    /// <summary>
    /// Adds the epsilon floor elementwise, with a constant shaped like its operand.
    /// </summary>
    /// <remarks>
    /// The engine's TensorAdd requires matching shapes rather than broadcasting a scalar, so a
    /// [1]-shaped epsilon against a [frames, bins] magnitude throws
    /// "Tensor shapes must match. Got [1, 17] and [1]".
    /// </remarks>
    private Tensor<T> AddEpsilon(Tensor<T> x) => Engine.TensorAdd(x, Constant(x, _epsilon));


    private static int[] AllAxes(Tensor<T> x) => Enumerable.Range(0, x.Shape.Length).ToArray();

    private static int LargestPowerOfTwoAtMost(int n)
    {
        int p = 1;
        while (p * 2 <= n) p *= 2;
        return p;
    }

    /// <summary>Hann window and the real/imaginary DFT bases for one frame size, built once.</summary>
    private (Tensor<T> Window, Tensor<T> Cos, Tensor<T> Sin) GetBasis(int frameSize)
    {
        // GetOrAdd, for the same reason as the column caches: one built basis wins and every caller
        // sees the same tensor instances, which the fused compiled path depends on after it traces.
        return _bases.GetOrAdd(frameSize, BuildBasis);
    }

    private static (Tensor<T> Window, Tensor<T> Cos, Tensor<T> Sin) BuildBasis(int frameSize)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int bins = frameSize / 2 + 1;

        var window = new Tensor<T>([frameSize]);
        for (int n = 0; n < frameSize; n++)
        {
            double hann = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * n / Math.Max(1, frameSize - 1));
            window[n] = ops.FromDouble(hann);
        }

        // Real DFT bases. Sine is negated so the pair matches e^{-i2*pi*kn/N}; the magnitude is
        // insensitive to that sign, but keeping the convention right makes these reusable.
        var cos = new Tensor<T>([frameSize, bins]);
        var sin = new Tensor<T>([frameSize, bins]);
        for (int n = 0; n < frameSize; n++)
        {
            for (int k = 0; k < bins; k++)
            {
                double angle = 2.0 * Math.PI * k * n / frameSize;
                cos[n, k] = ops.FromDouble(Math.Cos(angle));
                sin[n, k] = ops.FromDouble(-Math.Sin(angle));
            }
        }

        return (window, cos, sin);
    }

    /// <inheritdoc />
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        var loss = ComputeTapeLoss(
            Tensor<T>.FromVector(predicted, [predicted.Length]),
            Tensor<T>.FromVector(actual, [actual.Length]));
        return loss.Length > 0 ? loss[loss.Length - 1] : MathHelper.GetNumericOperations<T>().Zero;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Central finite differences. The analytic gradient of a multi-resolution STFT magnitude is
    /// long-winded, and this path exists only for callers outside the tape; taped training uses
    /// <see cref="ComputeTapeLoss"/>, where the engine differentiates the graph exactly.
    /// </remarks>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var gradient = new Vector<T>(predicted.Length);
        const double step = 1e-4;

        // PERTURB A COPY, NOT THE CALLER'S VECTOR. The probe used to write into `predicted` and
        // restore it afterwards, which left two holes: if CalculateLoss threw between the write and
        // the restore, the caller was handed back a vector still holding a perturbed element; and a
        // vector shared with another thread was observably wrong for the duration of every probe.
        // Copying once costs one allocation per call and removes both, and the method no longer has
        // a side effect on its own input.
        var probe = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            probe[i] = predicted[i];
        }

        for (int i = 0; i < predicted.Length; i++)
        {
            T original = predicted[i];
            probe[i] = ops.Add(original, ops.FromDouble(step));
            double plus = Convert.ToDouble(CalculateLoss(probe, actual));
            probe[i] = ops.Subtract(original, ops.FromDouble(step));
            double minus = Convert.ToDouble(CalculateLoss(probe, actual));
            probe[i] = original;
            gradient[i] = ops.FromDouble((plus - minus) / (2.0 * step));
        }

        return gradient;
    }
}
