using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.TextToSpeech.StyleEmotion;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// GPT-SoVITS's flow-matching synthesizer for v3, v4 and V5 (<c>module/models.py</c> <c>SynthesizerTrnV3</c>): the
/// SoVITS text encoder over the quantized semantic tokens (its hidden state, not the prior), a 1×1 bridge to 512 channels
/// with LeakyReLU, nearest upsampling to the mel rate (×1.875 for v3's 24 kHz/256 mels, ×2 after), the WaveNet encoder
/// <c>wns1</c> conditioned on the style vector, and the conditional flow matching DiT that turns those features and a
/// prompt mel into a normalized mel spectrogram.
/// </summary>
internal sealed class GptSoVitsCfmSynthesizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    /// <summary>The normalized mel's range: <c>norm_spec</c> maps log-mels in [−12, 2] to [−1, 1].</summary>
    public const double SpecMin = -12, SpecMax = 2;

    /// <summary>The release's bridge and wns1 width (<c>inter_channels2</c>).</summary>
    public const int ReleaseBridgeChannels = 512;

    public GptSoVitsCfmSynthesizer(IEngine engine, List<LayerBase<T>> layers, GptSoVitsSynthesizerConfiguration configuration,
        GptSoVitsDiTConfiguration dit, Random random, int bridgeChannels = ReleaseBridgeChannels)
    {
        var c = configuration;
        if (!c.IsCfm) throw new ArgumentException($"{c.Version} synthesizes with the GAN (GptSoVitsSynthesizer).", nameof(configuration));
        if (c.Gin != c.MrteChannels) throw new ArgumentException("MRTE adds the style vector, so its width must equal gin.", nameof(configuration));
        if (dit.TextDim != bridgeChannels) throw new ArgumentException($"The DiT reads the bridge's {bridgeChannels}-channel features.", nameof(dit));
        _engine = engine;
        Configuration = c;
        SslProjection = new NormedConv1DLayer<T>(c.SslDim, c.SslDim, c.SemanticRate25Hz ? 2 : 1, c.SemanticRate25Hz ? 2 : 1, 1, 1, 0, false,
            ConvolutionNormalization.None);
        layers.Add(SslProjection);
        Quantizer = new GptSoVitsQuantizer<T>(engine, c.SslDim, c.Bins, random);
        TextEncoder = new GptSoVitsTextEncoder<T>(engine, layers, c, random);
        ReferenceEncoder = new GptSoVitsMelStyleEncoder<T>(engine, layers, random, c.ReferenceChannels, c.Gin, c.StyleHidden, c.Dropout > 0 ? 0.1 : 0.0);
        // The release declares the bridge over inter_channels but feeds it the hidden state; the two are equal (192) in every config.
        Bridge = MeloOps.Conv(layers, c.Hidden, bridgeChannels, 1);
        MelEncoder = new GptSoVitsMaskedEncoder<T>(engine, layers, bridgeChannels, bridgeChannels, bridgeChannels, 5, 1, 8, c.Gin);
        Cfm = new GptSoVitsCfm<T>(engine, new GptSoVitsDiT<T>(engine, layers, dit with { UseStepEmbedding = !c.IsV5 }, random), c.IsV5);
    }

    public GptSoVitsSynthesizerConfiguration Configuration { get; }
    public NormedConv1DLayer<T> SslProjection { get; }
    public GptSoVitsQuantizer<T> Quantizer { get; }
    public GptSoVitsTextEncoder<T> TextEncoder { get; }
    public GptSoVitsMelStyleEncoder<T> ReferenceEncoder { get; }
    public NormedConv1DLayer<T> Bridge { get; }

    /// <summary><c>wns1</c>.</summary>
    public GptSoVitsMaskedEncoder<T> MelEncoder { get; }

    public GptSoVitsCfm<T> Cfm { get; }

    /// <summary>The mel frames per text-encoder frame: 1.875 for v3 (24 kHz, hop 256), 2 after (32 kHz, hop 320).</summary>
    public double UpsampleFactor => Configuration.Version == "v3" ? 1.875 : 2.0;

    /// <summary>The release's synthesis chunking: the reference frames kept (T_ref) and the frames per CFM call (T_chunk).</summary>
    public (int Reference, int Chunk) ChunkFrames => Configuration.Version == "v3" ? (468, 934) : (500, 1000);

    /// <summary>The release's default sampling steps: 32 for v3 and v5dev, 8 for v4, 4 for v5turbo.</summary>
    public int DefaultSteps => Configuration.Version switch { "v4" => 8, "v5turbo" => 4, _ => 32 };

    /// <summary>The release's default classifier-free guidance: 1.3 for v5dev, else none.</summary>
    public double DefaultCfgRate => Configuration.Version == "v5dev" ? 1.3 : 0.0;

    /// <summary>The style vector <c>[1, gin, 1]</c> of a reference linear spectrogram <c>[1, 1025, T]</c> (its first 704
    /// rows).</summary>
    public Tensor<T> Style(Tensor<T> reference, bool training = false, Random? random = null)
    {
        var input = _engine.TensorSlice(reference, new[] { 0, 0, 0 }, new[] { 1, Configuration.ReferenceChannels, reference.Shape[2] });
        return ReferenceEncoder.Forward(input, training, random ?? new Random(0));
    }

    /// <summary><c>F.interpolate(mode="nearest")</c> along time by a scale factor: ⌊T·s⌋ frames, frame i reading
    /// ⌊i · float(1/s)⌋ (ATen's single-precision <c>nearest_idx</c>).</summary>
    internal static Tensor<T> NearestUpsample(IEngine engine, Tensor<T> x, double scale)
    {
        int t = x.Shape[2], size = (int)Math.Floor(t * scale);
        var index = new Tensor<int>(new[] { size });
        float inverse = (float)(1.0 / scale);
        for (int i = 0; i < size; i++)
        {
            int source = size == t ? i : size == 2 * t ? i >> 1 : (int)Math.Floor((float)(i * inverse));
            index[i] = Math.Min(source, t - 1);
        }
        return GptSoVitsTextEncoder<T>.TimeSelect(engine, x, index);
    }

    private Tensor<T> Features(Tensor<T> quantized, IReadOnlyList<int> text, Tensor<T> ge, double speed, int melLength, bool training, Random? random)
    {
        var (hidden, _, _) = TextEncoder.Encode(quantized, text, ge, speed, training, random);
        var fea = StyleTtsOps.LeakyRelu(_engine, Bridge.Forward(hidden), 0.01);
        fea = NearestUpsample(_engine, fea, UpsampleFactor);
        return MelEncoder.Forward(fea, melLength, ge);
    }

    /// <summary>
    /// <c>decode_encp(codes, text, refer, ge, speed)</c>: the CFM's conditioning features <c>[1, 512, T]</c> of semantic
    /// codes and phone ids in the style <paramref name="ge"/>, wns1's mask at <c>⌊codes·(3.875 for v3, else 4)⌋</c>
    /// frames at speed 1 (else that over the speed, plus 1).
    /// </summary>
    public Tensor<T> DecodeFeatures(IReadOnlyList<int> codes, IReadOnlyList<int> text, Tensor<T> ge, double speed = 1.0)
    {
        double rate = Configuration.Version == "v3" ? 3.875 : 4;
        int length = speed == 1.0 ? (int)(codes.Count * rate) : (int)(codes.Count * rate / speed) + 1;
        var quantized = Quantizer.Decode(codes);
        if (Configuration.SemanticRate25Hz) quantized = NearestUpsample(_engine, quantized, 2.0);
        return Features(quantized, text, ge, speed, length, false, null);
    }

    /// <summary><c>norm_spec</c>: log-mel to [−1, 1].</summary>
    public Tensor<T> Normalize(Tensor<T> mel) =>
        _engine.TensorAddScalar(_engine.TensorMultiplyScalar(_engine.TensorAddScalar(mel, NumOps.FromDouble(-SpecMin)), NumOps.FromDouble(2 / (SpecMax - SpecMin))),
            NumOps.FromDouble(-1));

    /// <summary><c>denorm_spec</c>: [−1, 1] to log-mel.</summary>
    public Tensor<T> Denormalize(Tensor<T> mel) =>
        _engine.TensorAddScalar(_engine.TensorMultiplyScalar(_engine.TensorAddScalar(mel, NumOps.One), NumOps.FromDouble((SpecMax - SpecMin) / 2)),
            NumOps.FromDouble(SpecMin));

    // [1, C, T] ↔ [T, C].
    private Tensor<T> Rows(Tensor<T> x) => _engine.TensorTranspose(_engine.Reshape(x, new[] { x.Shape[1], x.Shape[2] }));

    private Tensor<T> ChannelsFirst(Tensor<T> rows) => _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, rows.Shape[1], rows.Shape[0] });

    private static Tensor<T> Frames(IEngine engine, Tensor<T> x, int start, int count) => engine.TensorNarrow(x, 2, start, count);

    /// <summary>
    /// The normalized mel <c>[1, 100, T]</c> of the target features as <c>inference_webui.get_tts_wav</c> builds it from
    /// the reference's features and normalized mel (both <c>[1, ·, T_ref]</c>, trimmed to the shorter): v3/v4 keep the
    /// last T_ref reference frames and synthesize T_chunk − T_ref target frames per call, each call's last reference-length
    /// frames prompting the next; V5 (<c>synthesize_v5_mel</c>) keeps the original reference as the prompt and replaces
    /// only its last 32 frames with the previous chunk's. <paramref name="noise"/> draws each call's standard normal start
    /// <c>[T, 100]</c>.
    /// </summary>
    public Tensor<T> SynthesizeMel(Tensor<T> referenceFeatures, Tensor<T> referenceMel, Tensor<T> targetFeatures, int steps, double cfgRate,
        Func<int[], Tensor<T>> noise)
    {
        int frames = Math.Min(referenceMel.Shape[2], referenceFeatures.Shape[2]);
        var (refLimit, chunk) = ChunkFrames;
        int keep = Math.Min(frames, Configuration.IsV5 ? 500 : refLimit);
        var mel = Frames(_engine, referenceMel, frames - keep, keep);
        var fea = Frames(_engine, referenceFeatures, frames - keep, keep);
        int total = targetFeatures.Shape[2];
        int chunkLength = Configuration.IsV5 ? Math.Min(1000 - keep, 640) : chunk - keep;
        var results = new List<Tensor<T>>();
        Tensor<T> rollingMel = mel, rollingFea = fea;
        for (int start = 0, index = 0; start < total; start += chunkLength, index++)
        {
            var target = Frames(_engine, targetFeatures, start, Math.Min(chunkLength, total - start));
            Tensor<T> prompt, features;
            if (Configuration.IsV5)
            {
                if (index == 0) { prompt = mel; features = fea; }
                else
                {
                    int tail = Math.Min(Math.Min(32, keep), Math.Min(rollingMel.Shape[2], rollingFea.Shape[2]));
                    int prefix = keep - tail;
                    var melTail = Frames(_engine, rollingMel, rollingMel.Shape[2] - tail, tail);
                    var feaTail = Frames(_engine, rollingFea, rollingFea.Shape[2] - tail, tail);
                    // A reference no longer than the tail is replaced whole.
                    prompt = prefix == 0 ? melTail : _engine.TensorConcatenate(new[] { Frames(_engine, mel, 0, prefix), melTail }, 2);
                    features = prefix == 0 ? feaTail : _engine.TensorConcatenate(new[] { Frames(_engine, fea, 0, prefix), feaTail }, 2);
                }
            }
            else { prompt = rollingMel; features = rollingFea; }
            var mu = Rows(_engine.TensorConcatenate(new[] { features, target }, 2));
            var generated = Cfm.Inference(mu, Rows(prompt), steps, 1.0, Configuration.IsV5 ? cfgRate : 0.0, noise(new[] { mu.Shape[0], Cfm.Estimator.Configuration.MelDim }));
            int promptLength = prompt.Shape[2];
            var output = ChannelsFirst(_engine.TensorNarrow(generated, 0, promptLength, generated.Shape[0] - promptLength));
            results.Add(output);
            // mel2 = cfm_res[:, :, -T_min:] and fea_ref = fea_todo_chunk[:, :, -T_min:] (V5's rolling tail alike).
            int outFrames = output.Shape[2], m = Math.Min(keep, outFrames), f = Math.Min(keep, target.Shape[2]);
            rollingMel = Frames(_engine, output, outFrames - m, m);
            rollingFea = Frames(_engine, target, target.Shape[2] - f, f);
        }
        return results.Count == 1 ? results[0] : _engine.TensorConcatenate(results.ToArray(), 2);
    }

    /// <summary>The training loss's draws (<c>torch.rand</c> for the prompt fraction and the time, <c>torch.randn_like</c>
    /// for the noise <c>[T, mel]</c>, the shortcut step's exponent when the bootstrap fires).</summary>
    internal sealed record LossDraws(double PromptFraction, double Time, Tensor<T> Noise, int? ShortcutExponent);

    /// <summary>
    /// <c>forward(ssl, y, mel, ..., text, ..., mel_lengths)</c> for one example: the style of the spectrogram, the frozen
    /// (when <paramref name="freezeQuantizer"/>, as s2.json sets) SSL projection, quantizer and text encoder, the bridge,
    /// upsampling and wns1 over the mel's frames, a prompt of ⌊U·⅔·frames⌋ frames, and the CFM loss. The draws come from
    /// <paramref name="random"/> unless <paramref name="draws"/> fixes them.
    /// </summary>
    internal Tensor<T> Loss(Tensor<T> ssl, Tensor<T> spectrogram, Tensor<T> mel, IReadOnlyList<int> text, bool freezeQuantizer, Random random,
        bool training = true, LossDraws? draws = null)
    {
        var ge = Style(spectrogram, training, random);
        var projected = SslProjection.Forward(ssl);
        var (quantized, _, _) = Quantizer.Forward(freezeQuantizer ? MeloOps.Detach(projected) : projected, training: training && !freezeQuantizer, random);
        quantized = NearestUpsample(_engine, quantized, 2.0);
        if (quantized.Shape[2] != spectrogram.Shape[2])
            throw new ArgumentException($"The SSL features give {quantized.Shape[2]} frames for a {spectrogram.Shape[2]}-frame spectrogram.", nameof(ssl));
        int melFrames = mel.Shape[2];
        Tensor<T> hidden;
        if (freezeQuantizer)
        {
            // The frozen encoder runs in evaluation under no_grad, though the style vector it reads is not detached.
            var (h, _, _) = TextEncoder.Encode(quantized, text, ge, 1.0, training: false, random);
            hidden = MeloOps.Detach(h);
        }
        else hidden = TextEncoder.Encode(quantized, text, ge, 1.0, training, random).Hidden;
        var fea = NearestUpsample(_engine, StyleTtsOps.LeakyRelu(_engine, Bridge.Forward(hidden), 0.01), UpsampleFactor);
        fea = MelEncoder.Forward(fea, melFrames, ge);
        double fraction = draws?.PromptFraction ?? random.NextDouble();
        int promptLength = (int)Math.Floor(fraction * (melFrames * 2.0 / 3.0));
        int minimum = Math.Min(melFrames, fea.Shape[2]);
        var target = Rows(Frames(_engine, mel, 0, minimum));
        var mu = Rows(Frames(_engine, fea, 0, minimum));
        return draws is null
            ? Cfm.Loss(target, mu, promptLength, random, training)
            : Cfm.Loss(target, mu, promptLength, draws.Time, draws.Noise, draws.ShortcutExponent, training, random);
    }

    /// <summary>Loads the s2 v3/v4/V5 checkpoint's <c>weight</c>.</summary>
    public void Load(Func<string, double[]?> read)
    {
        double[] Require(string name) => read(name) ?? throw new KeyNotFoundException($"The SoVITS checkpoint has no '{name}'.");
        MeloOps.LoadConv(Require, SslProjection, "ssl_proj");
        Quantizer.Load(read, "quantizer.vq.layers.0._codebook.");
        TextEncoder.Load(Require, "enc_p.");
        ReferenceEncoder.Load(_engine, Require, "ref_enc.");
        MeloOps.LoadConv(Require, Bridge, "bridge.0");
        MelEncoder.Load(Require, "wns1.");
        Cfm.Estimator.Load(Require, "cfm.estimator.");
    }
}
