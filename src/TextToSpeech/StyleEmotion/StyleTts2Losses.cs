using AiDotNet.Audio.Features;
using AiDotNet.Audio.Foundations;
using AiDotNet.Helpers;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>
/// torchaudio's <c>Resample</c> (windowed-sinc interpolation with a Hann window, 6 zero crossings, rolloff 0.99) as a
/// fixed strided convolution, differentiable on the tape — StyleTTS 2 resamples its 24 kHz waveforms to WavLM's 16 kHz
/// with it.
/// </summary>
internal sealed class TorchaudioResample<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _orig;
    private readonly int _new;
    private readonly int _width;
    private readonly Tensor<T> _kernel;     // [new, 1, 1, 2·width + orig]

    public TorchaudioResample(IEngine engine, int originalRate, int newRate, int lowpassFilterWidth = 6, double rolloff = 0.99)
    {
        _engine = engine;
        int gcd = Gcd(originalRate, newRate);
        _orig = originalRate / gcd;
        _new = newRate / gcd;
        double baseFrequency = Math.Min(_orig, _new) * rolloff;
        _width = (int)Math.Ceiling(lowpassFilterWidth * _orig / baseFrequency);
        int taps = 2 * _width + _orig;
        _kernel = new Tensor<T>(new[] { _new, 1, 1, taps });
        double scale = baseFrequency / _orig;
        for (int phase = 0; phase < _new; phase++)
            for (int i = 0; i < taps; i++)
            {
                double t = (-(double)phase / _new + (i - _width) / (double)_orig) * baseFrequency;
                t = Math.Max(-lowpassFilterWidth, Math.Min(lowpassFilterWidth, t));
                double window = Math.Cos(t * Math.PI / lowpassFilterWidth / 2);
                window *= window;
                t *= Math.PI;
                double sinc = t == 0 ? 1.0 : Math.Sin(t) / t;
                _kernel[phase, 0, 0, i] = NumOps.FromDouble(sinc * window * scale);
            }
    }

    private static int Gcd(int a, int b) => b == 0 ? a : Gcd(b, a % b);

    /// <summary>The resampled waveform <c>[⌈new · samples / orig⌉]</c> of <paramref name="audio"/> <c>[samples]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> audio)
    {
        int length = audio.Length;
        var padded = VocoderOps.ZeroPad(_engine, _engine.Reshape(audio, new[] { 1, 1, length }), _width, _width + _orig);
        var y = _engine.Conv2D(_engine.Reshape(padded, new[] { 1, 1, 1, padded.Shape[2] }), _kernel, new[] { 1, _orig }, new[] { 0, 0 }, new[] { 1, 1 });
        int frames = y.Shape[3];
        var interleaved = _engine.Reshape(_engine.TensorTranspose(_engine.Reshape(y, new[] { _new, frames })), new[] { frames * _new });
        int target = (int)Math.Ceiling((double)_new * length / _orig);
        return _engine.TensorSlice(interleaved, new[] { 0 }, new[] { target });
    }
}

/// <summary>
/// StyleTTS 2's training objectives (yl4579/StyleTTS2 <c>losses.py</c>): the multi-resolution mel spectral-convergence
/// loss, HiFi-GAN's least-squares adversarial and feature-matching losses with the truncated pointwise relativistic loss
/// (TPR, τ = 0.04), and the WavLM losses — feature matching on its hidden states and the SLM discriminator's adversarial
/// losses.
/// </summary>
internal sealed class StyleTts2Losses<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<(CenteredComplexStft<T> Stft, Tensor<T> Mel)> _resolutions = new();
    private readonly TorchaudioResample<T> _resample;
    private const double MelMean = -4.0, MelStd = 4.0, TprTau = 0.04;

    public StyleTts2Losses(IEngine engine, int sampleRate, int slmSampleRate)
    {
        _engine = engine;
        // STFTLoss: torchaudio MelSpectrogram(sample_rate 24000, n_fft, win_length, hop_length, Hann) with its defaults —
        // 128 HTK mel bins from 0 to the Nyquist frequency, power 2, centred with reflect padding.
        foreach (var (fft, hop, window) in new[] { (1024, 120, 600), (2048, 240, 1200), (512, 50, 240) })
        {
            var basis = MelFilterbanks.Htk(sampleRate, fft, 128, 0.0, Math.Floor(sampleRate / 2.0));
            var mel = new Tensor<T>(new[] { 128, fft / 2 + 1 });
            for (int m = 0; m < 128; m++)
                for (int k = 0; k < fft / 2 + 1; k++) mel[m, k] = NumOps.FromDouble(basis[m, k]);
            _resolutions.Add((new CenteredComplexStft<T>(engine, fft, hop, window), mel));
        }
        _resample = new TorchaudioResample<T>(engine, sampleRate, slmSampleRate);
    }

    private Tensor<T> Mean(Tensor<T> x) =>
        _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.Reshape(x, new[] { x.Length }), new[] { 0 }, keepDims: true), NumOps.FromDouble(1.0 / x.Length));

    private Tensor<T> Sum(Tensor<T> x) => _engine.ReduceSum(_engine.Reshape(x, new[] { x.Length }), new[] { 0 }, keepDims: true);

    private Tensor<T> AddAll(IEnumerable<Tensor<T>> terms) => terms.Aggregate((a, b) => _engine.TensorAdd(a, b));

    /// <summary>The 16 kHz waveform the SLM reads.</summary>
    public Tensor<T> Resample(Tensor<T> audio) => _resample.Forward(audio);

    private Tensor<T> NormalizedLogMel(CenteredComplexStft<T> stft, Tensor<T> melBasis, Tensor<T> audio)
    {
        var (re, im) = stft.Forward(_engine.Reshape(audio, new[] { audio.Length }));
        int bins = re.Shape[1], frames = re.Shape[2];
        var power = _engine.TensorAdd(_engine.TensorMultiply(re, re), _engine.TensorMultiply(im, im));
        var mel = _engine.TensorMatMul(melBasis, _engine.Reshape(power, new[] { bins, frames }));
        var log = _engine.TensorLog(_engine.TensorAddScalar(mel, NumOps.FromDouble(1e-5)));
        return _engine.TensorMultiplyScalar(_engine.TensorAddScalar(log, NumOps.FromDouble(-MelMean)), NumOps.FromDouble(1.0 / MelStd));
    }

    /// <summary>The mean over three resolutions of ‖y − x‖₁ / ‖y‖₁ on normalized log-mel spectrograms (the reference's
    /// <c>MultiResolutionSTFTLoss</c>).</summary>
    public Tensor<T> MultiResolutionMel(Tensor<T> generated, Tensor<T> real)
    {
        var terms = new List<Tensor<T>>();
        foreach (var (stft, mel) in _resolutions)
        {
            var x = NormalizedLogMel(stft, mel, generated);
            Tensor<T> y;
            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>()) y = NormalizedLogMel(stft, mel, real);
            terms.Add(_engine.TensorDivide(Sum(_engine.TensorAbs(_engine.TensorSubtract(y, x))), Sum(_engine.TensorAbs(y))));
        }
        return _engine.TensorMultiplyScalar(AddAll(terms), NumOps.FromDouble(1.0 / terms.Count));
    }

    private Tensor<T> OneMinusSquared(Tensor<T> score)
    {
        var d = _engine.TensorAddScalar(_engine.TensorNegate(score), NumOps.One);
        return Mean(_engine.TensorMultiply(d, d));
    }

    // The truncated pointwise relativistic loss of Li et al. (2022) (discriminator_TPRLS_loss): m = median(dr − dg);
    // L = mean(((dr − dg) − m)² over dr < dg + m); τ − relu(τ − L). torch.median is the lower middle value.
    private Tensor<T> Tpr(List<Tensor<T>> real, List<Tensor<T>> generated)
    {
        var terms = new List<Tensor<T>>();
        for (int d = 0; d < real.Count; d++)
        {
            int n = real[d].Length;
            var diff = _engine.Reshape(_engine.TensorSubtract(real[d], generated[d]), new[] { n });
            var order = Enumerable.Range(0, n).OrderBy(i => NumOps.ToDouble(diff[i])).ThenBy(i => i).ToArray();
            var median = _engine.TensorSlice(diff, new[] { order[(n - 1) / 2] }, new[] { 1 });
            double m = NumOps.ToDouble(median[0]);
            var mask = new Tensor<T>(new[] { n });
            int count = 0;
            for (int i = 0; i < n; i++)
                if (NumOps.ToDouble(real[d][i]) < NumOps.ToDouble(generated[d][i]) + m)
                {
                    mask[i] = NumOps.One;
                    count++;
                }
            var centered = _engine.TensorSubtract(diff, _engine.TensorTile(median, new[] { n }));
            // The mean over an empty selection is NaN in the reference; it never selects nothing when m is a median of
            // distinct values, and the term is then τ − relu(τ − NaN).
            var relative = _engine.TensorMultiplyScalar(Sum(_engine.TensorMultiply(_engine.TensorMultiply(centered, centered), mask)),
                NumOps.FromDouble(count == 0 ? double.NaN : 1.0 / count));
            var tau = new Tensor<T>(new[] { 1 });
            tau[0] = NumOps.FromDouble(TprTau);
            terms.Add(_engine.TensorSubtract(tau, _engine.ReLU(_engine.TensorSubtract(tau, relative))));
        }
        return AddAll(terms);
    }

    /// <summary>The generator's adversarial objective against the wave discriminators (the reference's
    /// <c>GeneratorLoss</c>): least-squares scores, feature matching ×2 and the relativistic loss.</summary>
    public Tensor<T> Generator(List<(Tensor<T> Score, List<Tensor<T>> Features)> onReal, List<(Tensor<T> Score, List<Tensor<T>> Features)> onGenerated)
    {
        var adversarial = AddAll(onGenerated.Select(o => OneMinusSquared(o.Score)));
        var features = new List<Tensor<T>>();
        for (int d = 0; d < onReal.Count; d++)
            for (int l = 0; l < onReal[d].Features.Count; l++)
                features.Add(Mean(_engine.TensorAbs(_engine.TensorSubtract(onReal[d].Features[l], onGenerated[d].Features[l]))));
        var matching = _engine.TensorMultiplyScalar(AddAll(features), NumOps.FromDouble(2.0));
        // generator_TPRLS_loss swaps the roles: its "real" is the generated scores.
        var tpr = Tpr(onGenerated.Select(o => o.Score).ToList(), onReal.Select(o => o.Score).ToList());
        return _engine.TensorAdd(_engine.TensorAdd(adversarial, matching), tpr);
    }

    /// <summary>The wave discriminators' objective (the reference's <c>DiscriminatorLoss</c>).</summary>
    public Tensor<T> Discriminator(List<(Tensor<T> Score, List<Tensor<T>> Features)> onReal, List<(Tensor<T> Score, List<Tensor<T>> Features)> onGenerated)
    {
        var terms = new List<Tensor<T>>();
        for (int d = 0; d < onReal.Count; d++)
            terms.Add(_engine.TensorAdd(OneMinusSquared(onReal[d].Score), Mean(_engine.TensorMultiply(onGenerated[d].Score, onGenerated[d].Score))));
        return _engine.TensorAdd(AddAll(terms), Tpr(onReal.Select(o => o.Score).ToList(), onGenerated.Select(o => o.Score).ToList()));
    }

    /// <summary>WavLM's hidden states of a 24 kHz waveform (resampled to 16 kHz), each <c>[frames, hidden]</c>.</summary>
    public IReadOnlyList<Tensor<T>> SlmStates(WavLmNetwork<T> slm, Tensor<T> audio)
    {
        return slm.HiddenStates(Resample(audio));
    }

    /// <summary>The reference's <c>WavLMLoss.forward</c>: Σ over WavLM's hidden states of the mean absolute difference.</summary>
    public Tensor<T> SlmFeatureMatching(WavLmNetwork<T> slm, Tensor<T> real, Tensor<T> generated)
    {
        IReadOnlyList<Tensor<T>> realStates;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>()) realStates = SlmStates(slm, real);
        var generatedStates = SlmStates(slm, generated);
        return AddAll(realStates.Zip(generatedStates, (r, g) => Mean(_engine.TensorAbs(_engine.TensorSubtract(r, g)))));
    }

    /// <summary>The stacked hidden states <c>[1, layers · hidden, frames]</c> the SLM discriminator reads (layer-major
    /// channels, as <c>torch.stack(states, 1).transpose(-1, -2).flatten(1, 2)</c>).</summary>
    public Tensor<T> Stacked(IReadOnlyList<Tensor<T>> states)
    {
        var channelsFirst = states.Select(s => _engine.TensorTranspose(s)).ToArray();                       // [hidden, frames]
        var stacked = _engine.TensorConcatenate(channelsFirst, 0);
        return _engine.Reshape(stacked, new[] { 1, stacked.Shape[0], stacked.Shape[1] });
    }

    /// <summary>The SLM generator loss (<c>WavLMLoss.generator</c>): mean (1 − D(states))².</summary>
    public Tensor<T> SlmGenerator(WavLmNetwork<T> slm, StyleTts2SlmDiscriminator<T> discriminator, Tensor<T> generated) =>
        OneMinusSquared(discriminator.Forward(Stacked(SlmStates(slm, generated))));

    /// <summary>The SLM discriminator's scores of a waveform with WavLM outside the tape
    /// (<c>WavLMLoss.discriminator_forward</c>).</summary>
    public Tensor<T> SlmScores(WavLmNetwork<T> slm, StyleTts2SlmDiscriminator<T> discriminator, Tensor<T> audio)
    {
        Tensor<T> stacked;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>()) stacked = Stacked(SlmStates(slm, audio));
        return discriminator.Forward(stacked);
    }

    /// <summary>The SLM discriminator loss (<c>WavLMLoss.discriminator</c>): mean (1 − D(real))² + mean D(generated)².</summary>
    public Tensor<T> SlmDiscriminator(WavLmNetwork<T> slm, StyleTts2SlmDiscriminator<T> discriminator, Tensor<T> real, Tensor<T> generated)
    {
        var fake = SlmScores(slm, discriminator, generated);
        return _engine.TensorAdd(OneMinusSquared(SlmScores(slm, discriminator, real)), Mean(_engine.TensorMultiply(fake, fake)));
    }

    /// <summary>The mean absolute difference of two tensors of the same length.</summary>
    public Tensor<T> L1(Tensor<T> a, Tensor<T> b) =>
        Mean(_engine.TensorAbs(_engine.TensorSubtract(_engine.Reshape(a, new[] { a.Length }), _engine.Reshape(b, new[] { b.Length }))));

    /// <summary>
    /// The differentiable upsampling of §3.2.2 (<c>slmadv.py</c>): with the duration bins' sigmoids q, d = Σq per phoneme,
    /// l = round(Σd) and centres c = cumsum(d) − d/2, each phoneme's logits correlated with a Gaussian
    /// exp(−(t − (l − c))² / 2σ²) over t &lt; l, then a softmax over the phonemes: the alignment <c>[L, l]</c>.
    /// </summary>
    public Tensor<T> DifferentiableAlignment(Tensor<T> logits, int frames, double sigma)
    {
        int length = logits.Shape[0], bins = logits.Shape[1];
        var durations = _engine.ReduceSum(_engine.Sigmoid(logits), new[] { 1 }, keepDims: true);                          // [L, 1]
        var lower = new Tensor<T>(new[] { length, length });
        for (int i = 0; i < length; i++)
            for (int j = 0; j <= i; j++) lower[i, j] = NumOps.One;
        var centres = _engine.TensorSubtract(_engine.TensorMatMul(lower, durations), _engine.TensorMultiplyScalar(durations, NumOps.FromDouble(0.5)));            // [L, 1]
        var offsets = new Tensor<T>(new[] { length, frames });
        for (int i = 0; i < length; i++)
            for (int t = 0; t < frames; t++) offsets[i, t] = NumOps.FromDouble(t - frames);
        var distance = _engine.TensorAdd(offsets, _engine.TensorTile(centres, new[] { 1, frames }));                    // t − (l − c)
        var kernel = _engine.TensorExp(_engine.TensorMultiplyScalar(_engine.TensorMultiply(distance, distance), NumOps.FromDouble(-0.5 / (sigma * sigma))));      // h [L, l]
        // conv1d(logits, h, padding l − 1, groups L)[..., :l]: out[i, j] = Σ_{n ≤ j} logits[i, n] · h[i, l − 1 − j + n].
        var index = new int[frames * bins];
        var mask = new Tensor<T>(new[] { 1, frames, bins });
        for (int j = 0; j < frames; j++)
            for (int n = 0; n < bins; n++)
            {
                bool inside = n <= j;
                index[j * bins + n] = inside ? frames - 1 - j + n : 0;
                if (inside) mask[0, j, n] = NumOps.One;
            }
        var gathered = _engine.Reshape(_engine.TensorGather(kernel, new Tensor<int>(new[] { index.Length }, new Vector<int>(index)), axis: 1),
            new[] { length, frames, bins });
        var weights = _engine.TensorMultiply(gathered, _engine.TensorTile(mask, new[] { length, 1, 1 }));
        var correlated = _engine.Reshape(_engine.BatchMatMul(weights, _engine.Reshape(logits, new[] { length, bins, 1 })), new[] { length, frames });
        return _engine.Softmax(correlated, axis: 0);
    }
}
