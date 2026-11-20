using EnginePadMode = AiDotNet.Tensors.Engines.PadMode;

namespace AiDotNet.Audio.Features;

/// <summary>
/// The centred STFT magnitude the librosa / torch spectrogram front ends compute, as whole-tensor engine ops in
/// <typeparamref name="T"/>: pad by fftSize / 2 on each side, unfold into hop-spaced frames, window, FFT, and the
/// magnitude of bins 0 to fftSize / 2.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal static class EngineStft<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    // The real DFT bases [fftSize, fftSize / 2 + 1] of the sizes the engine's FFT does not cover, built once per size.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<int, (Tensor<T> Cos, Tensor<T> Sin)> Bases = new();

    /// <summary>A fixed coefficient table (a mel filterbank) as a <c>[rows, columns]</c> tensor, converted once.</summary>
    public static Tensor<T> Table(double[,] table)
    {
        if (table is null) throw new ArgumentNullException(nameof(table));
        int rows = table.GetLength(0), columns = table.GetLength(1);
        var flat = new double[rows * columns];
        Buffer.BlockCopy(table, 0, flat, 0, flat.Length * sizeof(double));
        return new Tensor<double>(new[] { rows, columns }, new Vector<double>(flat)).Cast<T>();
    }

    /// <summary>The periodic Hann window of <paramref name="windowLength"/> samples zero-padded to
    /// <paramref name="fftSize"/> about its centre (librosa <c>pad_center</c>): <c>[fftSize]</c>.</summary>
    public static Tensor<T> CenteredHann(int fftSize, int windowLength)
    {
        if (windowLength <= 0 || windowLength > fftSize) throw new ArgumentOutOfRangeException(nameof(windowLength));
        var engine = AiDotNetEngine.Current;
        var window = engine.TensorHannWindow<T>(windowLength, true);
        int left = (fftSize - windowLength) / 2;
        return windowLength == fftSize ? window
            : engine.PadNd(window, new[] { left, fftSize - windowLength - left }, EnginePadMode.Constant, Ops.Zero);
    }

    /// <summary>Frames of the centred STFT of a signal of <paramref name="samples"/> samples
    /// (<c>1 + (samples + 2 (fftSize / 2) - fftSize) / hop</c>, torch.stft with <c>center=True</c>).</summary>
    public static int FrameCount(int samples, int fftSize, int hop) => 1 + (samples + 2 * (fftSize / 2) - fftSize) / hop;

    /// <summary>
    /// The centred STFT magnitude <c>[frames, fftSize / 2 + 1]</c> of a signal, padded by fftSize / 2 with
    /// <paramref name="padMode"/> on each side. The magnitude of each bin is taken by <see cref="Hypot"/>, so a 16-bit
    /// T neither overflows on a loud bin nor underflows on a quiet one.
    /// </summary>
    public static Tensor<T> Magnitude(Tensor<T> audio, int fftSize, int hop, Tensor<T> window, EnginePadMode padMode, double epsilon = 0.0)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        if (AiDotNet.Tensors.Engines.Compilation.GraphMode.IsActive)
        {
            int outFrames = FrameCount(audio.Length, fftSize, hop);
            int outBins = fftSize / 2 + 1;
            int stateTag = unchecked((((fftSize * 397) ^ hop) * 397 ^ (int)padMode) * 397 ^ epsilon.GetHashCode());
            return AiDotNet.Training.CompiledStepHostValue<T>.ComputeStateful(
                "EngineStftMagnitude",
                Tuple.Create(fftSize, hop, padMode, epsilon),
                stateTag,
                new[] { audio, window },
                new[] { outFrames, outBins },
                static (cfg, _, inputs, dst) =>
                {
                    var computed = MagnitudeCore(inputs[0], cfg.Item1, cfg.Item2, inputs[1], cfg.Item3, cfg.Item4);
                    computed.AsSpan().CopyTo(dst.AsWritableSpan());
                });
        }

        return MagnitudeCore(audio, fftSize, hop, window, padMode, epsilon);
    }

    private static Tensor<T> MagnitudeCore(Tensor<T> audio, int fftSize, int hop, Tensor<T> window, EnginePadMode padMode, double epsilon)
    {
        var engine = AiDotNetEngine.Current;
        // A feature of the waveform, not a differentiable layer: off the gradient tape, as the array code it replaces
        // was (the backward of the magnitude and of a log above it is NaN at a silent frame).
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        int pad = fftSize / 2, bins = fftSize / 2 + 1;
        var x = engine.Reshape(audio, new[] { audio.Length });
        var padded = engine.PadNd(x, new[] { pad, pad }, padMode, Ops.Zero);
        int frames = FrameCount(audio.Length, fftSize, hop);
        var framed = engine.TensorMultiply(engine.TensorUnfold(padded, 0, fftSize, hop),
            engine.TensorBroadcastTo(engine.Reshape(window, new[] { 1, fftSize }), new[] { frames, fftSize })).Contiguous();
        Tensor<T> re, im;
        if ((fftSize & (fftSize - 1)) == 0)
        {
            engine.FFT(framed, new Tensor<T>(new[] { frames, fftSize }), out var outRe, out var outIm);
            re = engine.TensorSlice(outRe, new[] { 0, 0 }, new[] { frames, bins });
            im = engine.TensorSlice(outIm, new[] { 0, 0 }, new[] { frames, bins });
        }
        else
        {
            // The engine's FFT is the DFT only for a power-of-two length; any other size goes through the real DFT
            // bases (cos and -sin of 2 pi k n / fftSize), as whisper's 400-point STFT does.
            var (cos, sin) = Bases.GetOrAdd(fftSize, size => AiDotNet.LossFunctions.DftBasis<T>.Build(size, size / 2 + 1, negateSine: true));
            re = engine.TensorMatMul(framed, cos);
            im = engine.TensorMatMul(framed, sin);
        }
        if (epsilon == 0.0) return Hypot(engine, re, im);
        // Retain the caller's epsilon before narrowing: Half cannot represent 1e-10, but can hold sqrt(1e-10).
        var wideRe = AiDotNet.Helpers.TensorCopyHelper.CastLogical<T, double>(re);
        var wideIm = AiDotNet.Helpers.TensorCopyHelper.CastLogical<T, double>(im);
        var squares = engine.TensorAdd(engine.TensorSquare(wideRe), engine.TensorSquare(wideIm));
        return engine.TensorSqrt(engine.TensorAddScalar(squares, epsilon)).Cast<T>();
    }

    /// <summary>
    /// sqrt(re² + im²) per element as m · sqrt(1 + (n / m)²), m = max(|re|, |im|) and n = min(|re|, |im|), 0 where
    /// m is 0. The ratio is at most 1, so neither a component's square can overflow a 16-bit T (past 255) nor a quiet
    /// bin's square underflow to 0: a scale shared by the whole spectrogram (its peak) squares every component more
    /// than ~1.7e-4 below the peak to 0 in Half, which reads a quiet frame as silence.
    /// </summary>
    internal static Tensor<T> Hypot(IEngine engine, Tensor<T> re, Tensor<T> im)
    {
        var a = engine.TensorAbs(re);
        var b = engine.TensorAbs(im);
        var larger = engine.TensorMax(a, b);
        var smaller = engine.TensorMin(a, b);
        var zero = engine.TensorEquals(larger, Ops.Zero);
        // 0 / 0 at a zero bin is replaced before it is used; the ratio of every other bin lies in [0, 1].
        var ratio = engine.TensorWhere(zero, new Tensor<T>(larger._shape), engine.TensorDivide(smaller, larger));
        var factor = engine.TensorSqrt(engine.TensorAddScalar(engine.TensorSquare(ratio), Ops.One));
        var magnitude = engine.TensorMultiply(larger, factor);
        // A non-finite component comes out as sqrt(re² + im²) gives it: NaN if either is NaN, else +∞ if either is
        // infinite. The max and min above drop a NaN beside a finite component, and ∞ / ∞ is NaN where the sum of
        // squares is ∞. s = |re| / 2 + |im| / 2 has exactly that value at such a bin (halved, so two finite
        // components cannot overflow it), and s − s is 0 only where s is finite.
        T half = Ops.FromDouble(0.5);
        var s = engine.TensorAdd(engine.TensorMultiplyScalar(a, half), engine.TensorMultiplyScalar(b, half));
        var finite = engine.TensorEquals(engine.TensorSubtract(s, s), Ops.Zero);
        return engine.TensorWhere(finite, magnitude, s);
    }
}
