using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// The discriminators CosyVoice trains HiFT against (the reference's <c>MultipleDiscriminator</c>): HiFi-GAN's
/// multi-period discriminator (periods 2, 3, 5, 7, 11) and a multi-resolution spectrogram discriminator — for each STFT
/// (1024/120/600, 2048/240/1200, 512/50/240), five weight-normalized 2-D convolutions with leaky ReLU (0.1) and an output
/// convolution over the magnitude spectrogram.
/// </summary>
internal sealed class CosyVoiceHiFTDiscriminators<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const double Slope = 0.1;
    private readonly IEngine _engine;
    private readonly HiFiGanDiscriminators<T> _periods;
    private readonly List<SpectrogramDiscriminator> _spectrograms = new();
    private readonly List<LayerBase<T>> _layers = new();

    public CosyVoiceHiFTDiscriminators(IEngine engine, int[]? periods = null, (int Fft, int Hop, int Window)[]? resolutions = null,
        int[]? periodChannels = null)
    {
        _engine = engine;
        _periods = new HiFiGanDiscriminators<T>(engine, periods ?? new[] { 2, 3, 5, 7, 11 }, scales: 0, useScaleDiscriminator: false,
            periodChannels: periodChannels);
        _layers.AddRange(_periods.Layers);
        foreach (var (fft, hop, window) in resolutions ?? new[] { (1024, 120, 600), (2048, 240, 1200), (512, 50, 240) })
            _spectrograms.Add(new SpectrogramDiscriminator(this, fft, hop, window));
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;
    internal HiFiGanDiscriminators<T> Periods => _periods;

    /// <summary>Every discriminator's score and feature maps for <paramref name="audio"/> <c>[samples]</c>.</summary>
    public List<(Tensor<T> Score, List<Tensor<T>> Features)> Forward(Tensor<T> audio)
    {
        var results = _periods.Forward(audio);
        foreach (var d in _spectrograms) results.Add(d.Forward(audio));
        return results;
    }

    /// <summary>Loads the reference's discriminator state dict (<c>mpd.discriminators.*</c>, <c>mrd.discriminators.*</c>).</summary>
    public void LoadTorch(IReadOnlyDictionary<string, double[]> state, string prefix = "")
    {
        (double[] V, double[] G, double[] B) Normed(string p) =>
            state.ContainsKey(p + ".parametrizations.weight.original1")
                ? (state[p + ".parametrizations.weight.original1"], state[p + ".parametrizations.weight.original0"], state[p + ".bias"])
                : (state[p + ".weight_v"], state[p + ".weight_g"], state[p + ".bias"]);
        var periodLayers = _periods.Layers.Cast<NormedConv1DLayer<T>>().ToList();
        int perDiscriminator = 6;
        for (int d = 0; d < periodLayers.Count / perDiscriminator; d++)
            for (int j = 0; j < perDiscriminator; j++)
            {
                string p = j < 5 ? $"{prefix}mpd.discriminators.{d}.convs.{j}" : $"{prefix}mpd.discriminators.{d}.conv_post";
                var (v, g, b) = Normed(p);
                periodLayers[d * perDiscriminator + j].LoadTorchWeights(v, g, b);
            }
        for (int d = 0; d < _spectrograms.Count; d++)
        {
            var s = _spectrograms[d];
            for (int j = 0; j < s.Convs.Count; j++)
            {
                var (v, g, b) = Normed($"{prefix}mrd.discriminators.{d}.discriminators.{j}");
                s.Convs[j].LoadTorchWeights(v, b, gain: g);
            }
            var (ov, og, ob) = Normed($"{prefix}mrd.discriminators.{d}.out");
            s.Output.LoadTorchWeights(ov, ob, gain: og);
        }
    }

    private Tensor<T> LeakyRelu(Tensor<T> x) =>
        _engine.TensorAdd(_engine.TensorMultiplyScalar(x, NumOps.FromDouble(Slope)),
            _engine.TensorMultiplyScalar(_engine.ReLU(x), NumOps.FromDouble(1 - Slope)));

    /// <summary>One spectrogram discriminator: |STFT| (center, reflect, Hann window of the window length) → convs.</summary>
    private sealed class SpectrogramDiscriminator
    {
        private readonly CosyVoiceHiFTDiscriminators<T> _owner;
        private readonly int _fft, _hop;
        private readonly Tensor<T> _basis;     // [2·bins, 1, 1, fft]

        public SpectrogramDiscriminator(CosyVoiceHiFTDiscriminators<T> owner, int fft, int hop, int window)
        {
            _owner = owner;
            _fft = fft;
            _hop = hop;
            NormedConv2DLayer<T> Conv(int input, int output, int kh, int kw, int sh, int sw, int ph, int pw)
            {
                var layer = new NormedConv2DLayer<T>(input, output, kh, kw, sh, sw, ph, pw, 1, ConvolutionNormalization.Weight);
                owner._layers.Add(layer);
                return layer;
            }
            Convs.Add(Conv(1, 32, 3, 9, 1, 1, 1, 4));
            for (int i = 0; i < 3; i++) Convs.Add(Conv(32, 32, 3, 9, 1, 2, 1, 4));
            Convs.Add(Conv(32, 32, 3, 3, 1, 1, 1, 1));
            Output = Conv(32, 1, 3, 3, 1, 1, 1, 1);
            // torch.hann_window(win) (periodic), zero-padded to the FFT size about the centre, as torch.stft pads it.
            var w = new double[fft];
            int offset = (fft - window) / 2;
            for (int n = 0; n < window; n++) w[offset + n] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / window);
            int bins = fft / 2 + 1;
            _basis = new Tensor<T>(new[] { 2 * bins, 1, 1, fft });
            for (int k = 0; k < bins; k++)
                for (int n = 0; n < fft; n++)
                {
                    double a = 2 * Math.PI * k * n / fft;
                    _basis[k, 0, 0, n] = NumOps.FromDouble(w[n] * Math.Cos(a));
                    bool realOnly = k == 0 || 2 * k == fft;
                    _basis[bins + k, 0, 0, n] = NumOps.FromDouble(realOnly ? 0.0 : -w[n] * Math.Sin(a));
                }
        }

        public List<NormedConv2DLayer<T>> Convs { get; } = new();
        public NormedConv2DLayer<T> Output { get; }

        public (Tensor<T> Score, List<Tensor<T>> Features) Forward(Tensor<T> audio)
        {
            var engine = _owner._engine;
            int pad = _fft / 2, bins = _fft / 2 + 1;
            var padded = VocoderOps.ReflectPad(engine, engine.Reshape(audio, new[] { 1, 1, audio.Length }), pad, pad);
            var spectrum = engine.Conv2D(engine.Reshape(padded, new[] { 1, 1, 1, padded.Shape[2] }), _basis, new[] { 1, _hop }, new[] { 0, 0 }, new[] { 1, 1 });
            int frames = spectrum.Shape[3];
            var re = engine.TensorSlice(spectrum, new[] { 0, 0, 0, 0 }, new[] { 1, bins, 1, frames });
            var im = engine.TensorSlice(spectrum, new[] { 0, bins, 0, 0 }, new[] { 1, bins, 1, frames });
            var magnitude = engine.TensorSqrt(engine.TensorAdd(engine.TensorMultiply(re, re), engine.TensorMultiply(im, im)));
            // |stft|.transpose(2, 1) then unsqueeze(1): [1, 1, frames, bins].
            var h = engine.Reshape(engine.TensorTranspose(engine.Reshape(magnitude, new[] { bins, frames })), new[] { 1, 1, frames, bins });
            var features = new List<Tensor<T>>();
            foreach (var conv in Convs)
            {
                h = _owner.LeakyRelu(conv.Forward(h));
                features.Add(h);
            }
            h = Output.Forward(h);
            features.Add(h);
            return (h, features);
        }
    }
}

/// <summary>
/// CosyVoice's HiFT GAN objectives (the reference's <c>HiFiGan</c> with Matcha's LSGAN losses): the generator's
/// adversarial loss, feature matching (×2, weighted 2), the 80-bin log-mel L1 (weighted 45), the truncated pointwise
/// relativistic loss (TPR, τ = 0.04) and the F0 L1; the discriminator's LSGAN loss plus TPR.
/// </summary>
internal sealed class CosyVoiceHiFTLosses<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly DifferentiableMel<T> _mel;

    public CosyVoiceHiFTLosses(IEngine engine, int sampleRate, double melWeight = 45, double featureWeight = 2.0,
        double tprWeight = 1.0, double tprTau = 0.04, int melFft = 1024, int melHop = 256, int melWindow = 1024, int melBins = 80)
    {
        _engine = engine;
        // matcha's mel_spectrogram(n_fft 1024, 80 mels, hop 256, window 1024, fmin 0, fmax None = sr / 2, center False).
        _mel = new DifferentiableMel<T>(engine, sampleRate, melFft, melHop, melWindow, melBins, 0, sampleRate / 2.0);
        MelWeight = melWeight;
        FeatureWeight = featureWeight;
        TprWeight = tprWeight;
        TprTau = tprTau;
    }

    public double MelWeight { get; }
    public double FeatureWeight { get; }
    public double TprWeight { get; }
    public double TprTau { get; }

    private Tensor<T> Mean(Tensor<T> x) =>
        _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.Reshape(x, new[] { x.Length }), new[] { 0 }, keepDims: false), NumOps.FromDouble(1.0 / x.Length));

    private Tensor<T> Scalar(double v)
    {
        var t = new Tensor<T>(new[] { 1 });
        t[0] = NumOps.FromDouble(v);
        return t;
    }

    private Tensor<T> Add(Tensor<T>? a, Tensor<T> b) => a is null ? _engine.Reshape(b, new[] { 1 }) : _engine.TensorAdd(a, _engine.Reshape(b, new[] { 1 }));

    // tpr_loss: m = median(dr − dg); L = mean(((dr − dg) − m)² over dr < dg + m); τ − relu(τ − L).
    private Tensor<T> Tpr(List<Tensor<T>> real, List<Tensor<T>> generated)
    {
        Tensor<T>? loss = null;
        for (int d = 0; d < real.Count; d++)
        {
            var diff = _engine.Reshape(_engine.TensorSubtract(real[d], generated[d]), new[] { real[d].Length });
            int n = diff.Length;
            // torch.median: the lower of the two middle values; its gradient goes to that element.
            var order = Enumerable.Range(0, n).OrderBy(i => NumOps.ToDouble(diff[i])).ThenBy(i => i).ToArray();
            int medianIndex = order[(n - 1) / 2];
            var median = _engine.TensorSlice(diff, new[] { medianIndex }, new[] { 1 });
            double m = NumOps.ToDouble(median[0]);
            var mask = new Tensor<T>(new[] { n });
            int count = 0;
            for (int i = 0; i < n; i++)
            {
                if (NumOps.ToDouble(real[d][i]) < NumOps.ToDouble(generated[d][i]) + m)
                {
                    mask[i] = NumOps.One;
                    count++;
                }
            }
            var centered = _engine.TensorSubtract(diff, _engine.TensorTile(median, new[] { n }));
            var squares = _engine.TensorMultiply(_engine.TensorMultiply(centered, centered), mask);
            // The mean over an empty selection is NaN, as torch.mean of an empty tensor.
            var relative = _engine.TensorMultiplyScalar(_engine.ReduceSum(squares, new[] { 0 }, keepDims: true), NumOps.FromDouble(1.0 / count));
            var tau = Scalar(TprTau);
            loss = Add(loss, _engine.TensorSubtract(tau, _engine.ReLU(_engine.TensorSubtract(tau, relative))));
        }
        return loss!;
    }

    /// <summary>The generator's loss and its parts, for real and generated audio <c>[samples]</c>, their discriminator
    /// outputs, and the predicted and target F0 <c>[frames]</c>.</summary>
    public (Tensor<T> Total, Tensor<T> Adversarial, Tensor<T> FeatureMatching, Tensor<T> Mel, Tensor<T> Tpr, Tensor<T> F0) Generator(
        Tensor<T> real, Tensor<T> generated, List<(Tensor<T> Score, List<Tensor<T>> Features)> onReal,
        List<(Tensor<T> Score, List<Tensor<T>> Features)> onGenerated, Tensor<T> f0, Tensor<T> pitch)
    {
        Tensor<T>? adversarial = null, features = null;
        foreach (var (score, _) in onGenerated)
        {
            var oneMinus = _engine.TensorSubtract(_engine.TensorAddScalar(new Tensor<T>(score._shape), NumOps.One), score);
            adversarial = Add(adversarial, Mean(_engine.TensorMultiply(oneMinus, oneMinus)));
        }
        for (int d = 0; d < onReal.Count; d++)
            for (int l = 0; l < onReal[d].Features.Count; l++)
                features = Add(features, Mean(_engine.TensorAbs(_engine.TensorSubtract(onReal[d].Features[l], onGenerated[d].Features[l]))));
        features = _engine.TensorMultiplyScalar(features!, NumOps.FromDouble(2.0));
        var mel = Mean(_engine.TensorAbs(_engine.TensorSubtract(_mel.Forward(generated), _mel.Forward(real))));
        var tpr = TprWeight != 0
            ? Tpr(onGenerated.Select(o => o.Score).ToList(), onReal.Select(o => o.Score).ToList())
            : Scalar(0);
        var f0Loss = Mean(_engine.TensorAbs(_engine.TensorSubtract(f0, pitch)));
        var total = _engine.TensorAdd(adversarial!, _engine.TensorMultiplyScalar(features, NumOps.FromDouble(FeatureWeight)));
        total = _engine.TensorAdd(total, _engine.Reshape(_engine.TensorMultiplyScalar(mel, NumOps.FromDouble(MelWeight)), new[] { 1 }));
        total = _engine.TensorAdd(total, _engine.TensorMultiplyScalar(tpr, NumOps.FromDouble(TprWeight)));
        total = _engine.TensorAdd(total, _engine.Reshape(f0Loss, new[] { 1 }));
        return (total, adversarial!, features, mel, tpr, f0Loss);
    }

    /// <summary>The discriminator's loss: LSGAN on real (target 1) and generated (target 0) scores, plus TPR.</summary>
    public Tensor<T> Discriminator(List<(Tensor<T> Score, List<Tensor<T>> Features)> onReal, List<(Tensor<T> Score, List<Tensor<T>> Features)> onGenerated)
    {
        Tensor<T>? loss = null;
        for (int d = 0; d < onReal.Count; d++)
        {
            var real = onReal[d].Score;
            var oneMinus = _engine.TensorSubtract(_engine.TensorAddScalar(new Tensor<T>(real._shape), NumOps.One), real);
            loss = Add(loss, _engine.TensorAdd(Mean(_engine.TensorMultiply(oneMinus, oneMinus)), Mean(_engine.TensorMultiply(onGenerated[d].Score, onGenerated[d].Score))));
        }
        if (TprWeight != 0)
            loss = _engine.TensorAdd(loss!, _engine.TensorMultiplyScalar(Tpr(onReal.Select(o => o.Score).ToList(), onGenerated.Select(o => o.Score).ToList()), NumOps.FromDouble(TprWeight)));
        return loss!;
    }
}
