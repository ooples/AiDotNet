using AiDotNet.Tensors;

namespace AiDotNet.Augmentation.Audio;

/// <summary>
/// Adds background noise to audio data.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This augmentation adds random noise to audio,
/// simulating real-world recording conditions like background hum, ambient sounds,
/// or electronic interference. This helps models become robust to noisy inputs.</para>
/// <para><b>SNR (Signal-to-Noise Ratio):</b>
/// <list type="bullet">
/// <item>Higher SNR = less noise (cleaner audio)</item>
/// <item>Lower SNR = more noise (noisier audio)</item>
/// <item>20 dB = barely audible noise</item>
/// <item>10 dB = noticeable noise</item>
/// <item>0 dB = signal and noise are equal</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AudioNoise<T> : AudioAugmenterBase<T>
{
    /// <summary>
    /// Gets the minimum signal-to-noise ratio in dB.
    /// </summary>
    public double MinSnrDb { get; }

    /// <summary>
    /// Gets the maximum signal-to-noise ratio in dB.
    /// </summary>
    public double MaxSnrDb { get; }

    /// <summary>
    /// Gets or sets the type of noise to add.
    /// </summary>
    public NoiseType NoiseType { get; set; } = NoiseType.White;

    /// <summary>
    /// Creates a new audio noise augmentation.
    /// </summary>
    /// <param name="minSnrDb">Minimum SNR in dB (default: 10.0).</param>
    /// <param name="maxSnrDb">Maximum SNR in dB (default: 30.0).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.3).</param>
    /// <param name="sampleRate">Sample rate of the audio in Hz (default: 16000).</param>
    public AudioNoise(
        double minSnrDb = 10.0,
        double maxSnrDb = 30.0,
        double probability = 0.3,
        int sampleRate = 16000) : base(probability, sampleRate)
    {
        if (minSnrDb > maxSnrDb)
        {
            throw new ArgumentException("Minimum SNR must be less than or equal to maximum SNR.");
        }

        MinSnrDb = minSnrDb;
        MaxSnrDb = maxSnrDb;
    }

    /// <inheritdoc />
    protected override Tensor<T> ApplyAugmentation(Tensor<T> data, AugmentationContext<T> context)
    {
        double snrDb = context.GetRandomDouble(MinSnrDb, MaxSnrDb);
        return AddNoise(data, snrDb, context);
    }

    private Tensor<T> AddNoise(Tensor<T> waveform, double snrDb, AugmentationContext<T> context)
    {
        int samples = GetSampleCount(waveform);
        var result = waveform.Clone();

        if (samples == 0)
        {
            return result;
        }

        // Calculate signal power over the first `samples` elements
        var engine = Engine;
        var signal = engine.TensorSlice(engine.Reshape(waveform, [waveform.Length]), [0], [samples]);

        T signalPower = NumOps.Divide(engine.TensorSum(engine.TensorSquare(signal)), NumOps.FromDouble(samples));

        // Convert SNR from dB to linear ratio
        // SNR_db = 10 * log10(signal_power / noise_power)
        // noise_power = signal_power / 10^(SNR_db/10)
        double snrLinear = Math.Pow(10, snrDb / 10.0);
        T noiseStd = NumOps.Sqrt(NumOps.Divide(signalPower, NumOps.FromDouble(snrLinear)));

        // Noise state starts fresh for each clip, so every generator is a function of this clip's draws
        var noise = NoiseType switch
        {
            NoiseType.Pink => PinkNoise(engine, context, samples, noiseStd),
            NoiseType.Brown => BrownNoise(engine, context, samples, noiseStd),
            _ => engine.TensorMultiplyScalar(AugmentationRandom<T>.StandardNormal(context, [samples]), noiseStd)
        };

        // Add noise and clip to valid range [-1, 1]
        var noisy = engine.TensorClamp(engine.TensorAdd(signal, noise), NumOps.FromDouble(-1.0), NumOps.One);
        noisy.Contiguous().CopyTo(result.AsWritableSpan().Slice(0, samples));

        return result;
    }

    // Voss-McCartney pink-noise generator (McCartney 1999, refined version).
    // Maintains a set of independent random "octave" sources updated at
    // geometrically decreasing rates: octave k flips every 2^k samples.
    // Summing the octaves yields a power spectrum that approximates 1/f over
    // about log2(NumOctaves) decades — far closer to true pink noise than a
    // single first-order IIR (which only approximates 1/f near one corner
    // frequency). Using 16 octaves covers the full audible range at 44.1 kHz.
    private const int PinkOctaves = 16;

    /// <summary>
    /// Pink noise for a whole clip. Draw order matches a per-sample generator: every octave is
    /// seeded first (the default all-zero state would leave clips shorter than 2^15 samples without
    /// the slowest octaves), then each sample draws a new value for one octave and a white term.
    /// </summary>
    private static Tensor<T> PinkNoise(IEngine engine, AugmentationContext<T> context, int samples, T stdDev)
    {
        var draws = engine.TensorMultiplyScalar(
            AugmentationRandom<T>.StandardNormal(context, [PinkOctaves + 2 * samples]), stdDev);
        var perSample = engine.Reshape(engine.TensorSlice(draws, [PinkOctaves], [2 * samples]), [samples, 2]);

        // Held octave values: the seeds followed by every sample's octave update
        var held = engine.TensorConcatenate(
            [engine.TensorSlice(draws, [0], [PinkOctaves]), engine.TensorSliceAxis(perSample, 1, 0).Contiguous()], 0);

        // Sample i (counter i + 1) updates octave = trailing zeros of the counter, capped at the top
        // octave (McCartney's "improved" variant: octave 0 every sample, octave k every 2^k)
        var counter = engine.TensorArange<int>(1, samples + 1, 1);
        var octave = new Tensor<int>([samples]);
        for (int k = 1; k < PinkOctaves; k++)
        {
            int period = 1 << k;
            var remainder = engine.TensorSubtract(counter, engine.TensorMultiplyScalar(engine.TensorDivideScalar(counter, period), period));
            octave = engine.TensorAdd(octave, engine.TensorEquals(remainder, 0));
        }

        // McCartney's "improved" Voss generator refreshes only one held octave per sample, so the
        // highest-frequency band needs an always-resampled white term; sum it with every octave's
        // latest value (seed index k, or PinkOctaves + j for the latest update j at or before i)
        var sum = engine.TensorSliceAxis(perSample, 1, 1).Contiguous();
        var updateIndex = engine.TensorAddScalar(engine.TensorArange<int>(0, samples, 1), PinkOctaves);
        var none = engine.TensorAddScalar(new Tensor<int>([samples]), -1);
        for (int k = 0; k < PinkOctaves; k++)
        {
            var latest = engine.TensorCumMax(engine.TensorWhere(engine.TensorEquals(octave, k), updateIndex, none), 0);
            var source = engine.TensorMax(latest, k);
            sum = engine.TensorAdd(sum, engine.TensorTake(held, source));
        }

        // Normalise by sqrt(NumOctaves + 1) so the output variance matches the input stdDev's
        // intent: summing (PinkOctaves + 1) independent N(0, sigma^2) values scales the variance
        // linearly with the count, so divide by the square root to restore sigma.
        return engine.TensorDivideScalar(sum, NumOps.FromDouble(Math.Sqrt(PinkOctaves + 1)));
    }

    /// <summary>
    /// Brown noise (leaky random walk) for a whole clip: state = (state + 0.02 * white) * 0.999 per
    /// sample, starting from 0. Evaluated in blocks: within a block the state is A * white + d * carry
    /// with A[t, j] = 0.02 * 0.999^(t - j + 1) for j &lt;= t and d[t] = 0.999^(t + 1).
    /// </summary>
    private static Tensor<T> BrownNoise(IEngine engine, AugmentationContext<T> context, int samples, T stdDev)
    {
        const int Block = 256;
        var white = engine.TensorMultiplyScalar(AugmentationRandom<T>.StandardNormal(context, [samples]), stdDev);
        int blocks = (samples + Block - 1) / Block;
        int padded = blocks * Block;
        if (padded > samples)
        {
            white = engine.TensorConcatenate([white, new Tensor<T>([padded - samples])], 0);
        }

        T logDecay = NumOps.FromDouble(Math.Log(0.999));
        var t = engine.Reshape(engine.TensorArange<int>(0, Block, 1), [Block, 1]);
        var j = engine.Reshape(engine.TensorArange<int>(0, Block, 1), [1, Block]);
        var lag = engine.TensorAddScalar(
            engine.TensorSubtract(engine.TensorBroadcastTo(t, [Block, Block]), engine.TensorBroadcastTo(j, [Block, Block])), 1);
        var causal = engine.TensorGreaterThan(lag, 0).Cast<T>();
        var transfer = engine.TensorMultiply(
            engine.TensorMultiplyScalar(engine.TensorExp(engine.TensorMultiplyScalar(lag.Cast<T>(), logDecay)), NumOps.FromDouble(0.02)),
            causal);
        var carryDecay = engine.TensorExp(engine.TensorMultiplyScalar(engine.TensorAddScalar(t, 1).Cast<T>(), logDecay));

        var columns = engine.TensorTranspose(engine.Reshape(white, [blocks, Block]));
        var states = new Tensor<T>[blocks];
        var carry = new Tensor<T>([1, 1]);
        for (int b = 0; b < blocks; b++)
        {
            var input = engine.TensorSlice(columns, [0, b], [Block, 1]);
            var state = engine.TensorAdd(
                engine.TensorMatMul(transfer, input),
                engine.TensorMultiply(carryDecay, engine.TensorBroadcastTo(carry, [Block, 1])));
            states[b] = state;
            carry = engine.TensorSlice(state, [Block - 1, 0], [1, 1]);
        }

        var walk = engine.Reshape(engine.TensorConcatenate(states, 0), [padded]);
        return engine.TensorSlice(walk, [0], [samples]);
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["minSnrDb"] = MinSnrDb;
        parameters["maxSnrDb"] = MaxSnrDb;
        parameters["noiseType"] = NoiseType.ToString();
        return parameters;
    }
}

/// <summary>
/// Types of audio noise.
/// </summary>
public enum NoiseType
{
    /// <summary>
    /// White noise (flat spectrum, all frequencies equal).
    /// </summary>
    White,

    /// <summary>
    /// Pink noise (1/f spectrum, more natural sounding).
    /// </summary>
    Pink,

    /// <summary>
    /// Brown/Brownian noise (1/f² spectrum, deep rumble).
    /// </summary>
    Brown
}
