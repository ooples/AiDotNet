namespace AiDotNet.Audio.Effects;

/// <summary>
/// Integrated loudness by ITU-R BS.1770-4 as pyloudnorm 0.1 computes it (<c>pyloudnorm.Meter</c> with K-weighting):
/// the K-weighting filter (a high shelf of +4 dB at 1500 Hz, Q 1/√2, then a high pass at 38 Hz, Q 0.5, both the RBJ
/// biquads pyloudnorm derives for the sample rate), the mean square of 400 ms blocks overlapping by 75 %, an absolute gate
/// at −70 LUFS and a relative gate 10 LU below the gated mean; and pyloudnorm's loudness and peak normalization.
/// </summary>
/// <remarks>
/// <para>Each biquad runs as whole-tensor engine ops: its zeros as a three-tap FIR of shifted copies, its poles through
/// the engine's diagonal-pole scan (<see cref="IEngine.ComplexDiagonalSsmScanForward{T}"/>). A conjugate pole pair is one
/// two-state scan (the partial fractions r/(1 − p z⁻¹) + r̄/(1 − p̄ z⁻¹)); real poles, including the high pass's
/// repeated pole at Q 0.5, are two one-state sections y[n] = x[n] + p·y[n−1] in a row. This is the transfer function
/// scipy's <c>lfilter</c> realizes in transposed direct form II, with zero initial state.</para>
/// <para><b>For Beginners:</b> Loudness in LUFS measures how loud audio sounds, weighting frequencies roughly as
/// the ear does; normalizing to a target loudness makes recordings sound equally loud.</para>
/// </remarks>
internal static class LoudnessMeter
{
    /// <summary><c>Meter(rate, block_size).integrated_loudness(data)</c> of mono audio; −∞ for silence.</summary>
    public static double IntegratedLoudness(double[] data, int rate, double blockSize = 0.4, double overlap = 0.75)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        return IntegratedLoudness(Samples(data), rate, blockSize, overlap);
    }

    /// <summary><c>Meter(rate, block_size).integrated_loudness(data)</c> of a mono waveform tensor; −∞ for silence.</summary>
    public static T IntegratedLoudness<T>(Tensor<T> data, int rate, double blockSize = 0.4, double overlap = 0.75)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (data.Length < blockSize * rate)
            throw new ArgumentException("Audio must have length greater than the block size.", nameof(data));
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var filtered = KWeighted(engine, engine.Reshape(data, new[] { data.Length }), rate, clamp: false);

        // pyloudnorm's block bounds, int(block · (j · step) · rate) to int(block · (j · step + 1) · rate), kept exactly:
        // the products are not always whole, so the blocks are gathered by index rather than unfolded.
        double step = 1.0 - overlap;
        double duration = (double)data.Length / rate;
        int blocks = (int)(Math.Round((duration - blockSize) / (blockSize * step), MidpointRounding.ToEven) + 1);
        var lower = new int[blocks];
        var lengths = new int[blocks];
        int widest = 0;
        for (int j = 0; j < blocks; j++)
        {
            lower[j] = (int)(blockSize * (j * step) * rate);
            int upper = Math.Min((int)(blockSize * (j * step + 1) * rate), filtered.Length);
            lengths[j] = Math.Max(0, upper - lower[j]);
            widest = Math.Max(widest, lengths[j]);
        }
        var index = new int[blocks * widest];
        var inside = new int[blocks * widest];
        for (int j = 0; j < blocks; j++)
            for (int k = 0; k < widest; k++)
            {
                bool used = k < lengths[j];
                index[j * widest + k] = used ? lower[j] + k : 0;
                inside[j * widest + k] = used ? 1 : 0;
            }
        var squares = engine.TensorSquare(filtered);
        var gathered = engine.Reshape(engine.TensorIndexSelect(squares, new Tensor<int>(index, new[] { index.Length }), 0), new[] { blocks, widest });
        var masked = engine.TensorWhere(new Tensor<int>(inside, new[] { blocks, widest }).Cast<T>(), gathered,
            new Tensor<T>(new[] { blocks, widest }));
        var z = engine.TensorMultiplyScalar(engine.ReduceSum(masked, new[] { 1 }, keepDims: false), ops.FromDouble(1.0 / (blockSize * rate)));
        var loudness = Level(engine, ops, z);

        // loudness >= −70 (a NaN block fails it, as in numpy).
        var absolute = engine.TensorMax(engine.TensorGreaterThan(loudness, ops.FromDouble(-70.0)), engine.TensorEquals(loudness, ops.FromDouble(-70.0)));
        T relative = ops.Subtract(Level(ops, GatedMean(engine, ops, z, absolute)), ops.FromDouble(10.0));
        var above = engine.TensorMultiply(engine.TensorGreaterThan(loudness, relative), engine.TensorGreaterThan(loudness, ops.FromDouble(-70.0)));
        T gated = GatedMean(engine, ops, z, above);
        if (ops.IsNaN(gated)) gated = ops.Zero;   // np.nan_to_num
        return Level(ops, gated);
    }

    /// <summary><c>pyloudnorm.normalize.loudness</c>: the gain that moves the measured loudness to the target.</summary>
    public static double[] NormalizeLoudness(double[] data, double inputLoudness, double targetLoudness)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        return NormalizeLoudness(Samples(data), inputLoudness, targetLoudness).ToArray();
    }

    /// <summary><c>pyloudnorm.normalize.loudness</c> of a waveform tensor.</summary>
    public static Tensor<T> NormalizeLoudness<T>(Tensor<T> data, double inputLoudness, double targetLoudness)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        double gain = Math.Pow(10.0, (targetLoudness - inputLoudness) / 20.0);
        return AiDotNetEngine.Current.TensorMultiplyScalar(data, MathHelper.GetNumericOperations<T>().FromDouble(gain));
    }

    /// <summary>Torchaudio's mono loudness: clamped biquads, complete 400 ms blocks,
    /// strict absolute/relative gates, and NaN when no block survives.</summary>
    internal static double IntegratedTorchaudioLoudness(double[] data, int rate)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        return IntegratedTorchaudioLoudness(Samples(data), rate);
    }

    /// <summary>Torchaudio's mono loudness of a waveform tensor.</summary>
    internal static T IntegratedTorchaudioLoudness<T>(Tensor<T> data, int rate)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        int block = (int)Math.Round(.4 * rate, MidpointRounding.ToEven);
        int step = (int)Math.Round(block * .25, MidpointRounding.ToEven);
        if (rate <= 0 || block <= 0 || step <= 0 || data.Length < block)
            throw new ArgumentException("Torchaudio loudness requires one complete 400 ms block.", nameof(data));
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var weighted = KWeighted(engine, engine.Reshape(data, new[] { data.Length }), rate, clamp: true);
        var energies = engine.ReduceMean(engine.TensorSquare(engine.TensorUnfold(weighted, 0, block, step)), new[] { 1 }, keepDims: false);
        var levels = Level(engine, ops, energies);
        var absolute = engine.TensorGreaterThan(levels, ops.FromDouble(-70.0));
        if (Count(engine, absolute) == 0) return ops.FromDouble(double.NaN);
        T relative = ops.Subtract(Level(ops, GatedMean(engine, ops, energies, absolute)), ops.FromDouble(10.0));
        var selected = engine.TensorMultiply(absolute, engine.TensorGreaterThan(levels, relative));
        return Count(engine, selected) == 0 ? ops.FromDouble(double.NaN) : Level(ops, GatedMean(engine, ops, energies, selected));
    }

    /// <summary><c>pyloudnorm.normalize.peak</c>: the gain that moves the peak to the target (dBFS).</summary>
    public static double[] NormalizePeak(double[] data, double target)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        return NormalizePeak(Samples(data), target).ToArray();
    }

    /// <summary><c>pyloudnorm.normalize.peak</c> of a waveform tensor.</summary>
    public static Tensor<T> NormalizePeak<T>(Tensor<T> data, double target)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T peak = engine.TensorMaxValue(engine.TensorAbs(data));
        return engine.TensorMultiplyScalar(data, ops.Divide(ops.FromDouble(Math.Pow(10.0, target / 20.0)), peak));
    }

    /// <summary>The K-weighted signal: the high shelf, then the high pass (each clamped to [−1, 1] after it when
    /// <paramref name="clamp"/>, as torchaudio's <c>lfilter(clamp=True)</c>).</summary>
    internal static Tensor<T> KWeighted<T>(IEngine engine, Tensor<T> x, int rate, bool clamp)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var shelf = Filter(engine, Biquad(4.0, 1 / Math.Sqrt(2), 1500.0, rate, highShelf: true), x);
        if (clamp) shelf = engine.TensorClamp(shelf, ops.FromDouble(-1.0), ops.FromDouble(1.0));
        var pass = Filter(engine, Biquad(0.0, 0.5, 38.0, rate, highShelf: false), shelf);
        return clamp ? engine.TensorClamp(pass, ops.FromDouble(-1.0), ops.FromDouble(1.0)) : pass;
    }

    private static Tensor<T> Samples<T>(double[] data) =>
        new Tensor<double>(new[] { data.Length }, new Vector<double>((double[])data.Clone())).Cast<T>();

    private static Tensor<double> Samples(double[] data) => Samples<double>(data);

    // −0.691 + 10 log10(z) per block.
    private static Tensor<T> Level<T>(IEngine engine, INumericOperations<T> ops, Tensor<T> z) =>
        engine.TensorAddScalar(engine.TensorMultiplyScalar(engine.TensorLog10(z), ops.FromDouble(10.0)), ops.FromDouble(-0.691));

    private static T Level<T>(INumericOperations<T> ops, T z) =>
        ops.Add(ops.FromDouble(-0.691), ops.Multiply(ops.FromDouble(10.0 / Math.Log(10.0)), ops.Log(z)));

    private static int Count<T>(IEngine engine, Tensor<T> mask) => mask.Cast<int>().ToArray().Sum();

    // The mean of z over the blocks the 0/1 mask selects; NaN when it selects none (numpy's mean of an empty slice).
    private static T GatedMean<T>(IEngine engine, INumericOperations<T> ops, Tensor<T> z, Tensor<T> mask)
    {
        int count = Count(engine, mask);
        if (count == 0) return ops.FromDouble(double.NaN);
        var selected = engine.TensorWhere(mask, z, new Tensor<T>(z._shape));
        return ops.Divide(engine.TensorSum(selected), ops.FromDouble(count));
    }

    // IIRfilter.generate_coefficients: b and a, divided by a0.
    private static (double[] B, double[] A) Biquad(double gainDb, double q, double frequency, int rate, bool highShelf)
    {
        double a = Math.Pow(10, gainDb / 40.0);
        double w0 = 2.0 * Math.PI * (frequency / rate);
        double alpha = Math.Sin(w0) / (2.0 * q);
        double b0, b1, b2, a0, a1, a2;
        if (highShelf)
        {
            b0 = a * (a + 1 + (a - 1) * Math.Cos(w0) + 2 * Math.Sqrt(a) * alpha);
            b1 = -2 * a * (a - 1 + (a + 1) * Math.Cos(w0));
            b2 = a * (a + 1 + (a - 1) * Math.Cos(w0) - 2 * Math.Sqrt(a) * alpha);
            a0 = a + 1 - (a - 1) * Math.Cos(w0) + 2 * Math.Sqrt(a) * alpha;
            a1 = 2 * (a - 1 - (a + 1) * Math.Cos(w0));
            a2 = a + 1 - (a - 1) * Math.Cos(w0) - 2 * Math.Sqrt(a) * alpha;
        }
        else
        {
            b0 = (1 + Math.Cos(w0)) / 2;
            b1 = -(1 + Math.Cos(w0));
            b2 = (1 + Math.Cos(w0)) / 2;
            a0 = 1 + alpha;
            a1 = -2 * Math.Cos(w0);
            a2 = 1 - alpha;
        }
        return (new[] { b0 / a0, b1 / a0, b2 / a0 }, new[] { a0 / a0, a1 / a0, a2 / a0 });
    }

    /// <summary>
    /// scipy.signal.lfilter(b, a, x) of a biquad with zero initial state: the zeros as b0·x[n] + b1·x[n−1] + b2·x[n−2],
    /// then the poles of 1 + a1 z⁻¹ + a2 z⁻² through the diagonal-pole scan.
    /// </summary>
    internal static Tensor<T> Filter<T>(IEngine engine, (double[] B, double[] A) coefficients, Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var (b, a) = coefficients;
        int n = x.Length;
        var flat = engine.Reshape(x, new[] { n });
        Tensor<T> Delayed(int k) => k == 0 ? flat
            : engine.TensorSlice(engine.PadNd(flat, new[] { k, 0 }, AiDotNet.Tensors.Engines.PadMode.Constant, ops.Zero), new[] { 0 }, new[] { n });
        var zeros = engine.TensorAdd(engine.TensorAdd(
            engine.TensorMultiplyScalar(flat, ops.FromDouble(b[0])),
            engine.TensorMultiplyScalar(Delayed(1), ops.FromDouble(b[1]))),
            engine.TensorMultiplyScalar(Delayed(2), ops.FromDouble(b[2])));

        // Poles: the roots of z² + a1 z + a2. A discriminant that is zero up to rounding (the Q 0.5 high pass) is a
        // repeated real pole; only a clearly negative one is a conjugate pair.
        double a1 = a[1], a2 = a[2];
        double discriminant = a1 * a1 - 4 * a2;
        double tolerance = 1e-9 * Math.Max(1.0, a1 * a1);
        if (discriminant < -tolerance)
        {
            double re = -a1 / 2, im = Math.Sqrt(-discriminant) / 2;
            // 1 / ((1 − p z⁻¹)(1 − p̄ z⁻¹)) = r / (1 − p z⁻¹) + r̄ / (1 − p̄ z⁻¹), r = p / (p − p̄) = ½ − i·re / (2 im).
            return Scan(engine, ops, zeros, new[] { re, re }, new[] { im, -im }, new[] { 0.5, 0.5 }, new[] { -re / (2 * im), re / (2 * im) });
        }
        double root = Math.Sqrt(Math.Max(discriminant, 0.0)) / 2;
        var first = Scan(engine, ops, zeros, new[] { -a1 / 2 + root }, new[] { 0.0 }, new[] { 1.0 }, new[] { 0.0 });
        return Scan(engine, ops, first, new[] { -a1 / 2 - root }, new[] { 0.0 }, new[] { 1.0 }, new[] { 0.0 });
    }

    // y[t] = Σ_s Re(C_s h_s[t]), h_s[t] = A_s h_s[t−1] + u[t], over one channel.
    private static Tensor<T> Scan<T>(IEngine engine, INumericOperations<T> ops, Tensor<T> u,
        double[] transitionRe, double[] transitionIm, double[] outputRe, double[] outputIm)
    {
        int n = u.Length, states = transitionRe.Length;
        Tensor<T> Constant(double[] values, int[] shape) => new Tensor<double>(shape, new Vector<double>(values)).Cast<T>();
        var ones = new double[states];
        for (int s = 0; s < states; s++) ones[s] = 1.0;
        var output = engine.ComplexDiagonalSsmScanForward(
            engine.Reshape(u, new[] { 1, n, 1, 1 }),
            Constant(transitionRe, new[] { 1, states }),
            Constant(transitionIm, new[] { 1, states }),
            Constant(ones, new[] { 1, states, 1 }),
            Constant(new double[states], new[] { 1, states, 1 }),
            Constant(outputRe, new[] { 1, 1, states }),
            Constant(outputIm, new[] { 1, 1, states }),
            new Tensor<T>(new[] { 1, 1 }));
        return engine.Reshape(output, new[] { n });
    }
}
