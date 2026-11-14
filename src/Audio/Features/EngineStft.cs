using AiDotNet.Helpers;
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
    /// <paramref name="padMode"/> on each side. The components are divided by the largest one before they are
    /// squared, and the magnitude multiplied back, so a 16-bit T cannot overflow; <paramref name="epsilon"/>
    /// is added under the square root (torch's <c>sqrt(re² + im² + eps)</c>), 0 for the plain magnitude.
    /// </summary>
    public static Tensor<T> Magnitude(Tensor<T> audio, int fftSize, int hop, Tensor<T> window, EnginePadMode padMode, double epsilon = 0.0)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        var engine = AiDotNetEngine.Current;
        int pad = fftSize / 2, bins = fftSize / 2 + 1;
        var x = engine.Reshape(audio, new[] { audio.Length });
        var padded = engine.PadNd(x, new[] { pad, pad }, padMode, Ops.Zero);
        int frames = FrameCount(audio.Length, fftSize, hop);
        var framed = engine.TensorMultiply(engine.TensorUnfold(padded, 0, fftSize, hop),
            engine.TensorBroadcastTo(engine.Reshape(window, new[] { 1, fftSize }), new[] { frames, fftSize }));
        engine.FFT(framed.Contiguous(), new Tensor<T>(new[] { frames, fftSize }), out var outRe, out var outIm);
        var re = engine.TensorSlice(outRe, new[] { 0, 0 }, new[] { frames, bins });
        var im = engine.TensorSlice(outIm, new[] { 0, 0 }, new[] { frames, bins });
        // |X| = sqrt(re² + im² + ε) in double, narrowed to T once: a 16-bit T overflows the squares and cannot hold
        // ε = 1e-10 at all, although it holds the resulting magnitude (1e-5 for silence).
        var wideRe = TensorCopyHelper.CastLogical<T, double>(re);
        var wideIm = TensorCopyHelper.CastLogical<T, double>(im);
        var squares = engine.TensorAdd(engine.TensorSquare(wideRe), engine.TensorSquare(wideIm));
        if (epsilon != 0.0)
            squares = engine.TensorAddScalar(squares, epsilon);
        return engine.TensorSqrt(squares).Cast<T>();
    }
}
