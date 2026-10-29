using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>How GLM-4-Voice's decoder turns a sequence of speech tokens into audio.</summary>
public enum GlmVoiceDecoding
{
    /// <summary>
    /// The paper's streaming decoder (§3.2), as the release's <c>AudioDecoder.stream_inference</c> runs it: blocks of
    /// b = 0.8 s (10 tokens, the encoder's attention block). Block n's mel is generated from the first n·b seconds of
    /// tokens with the mel returned so far as the prompt, then vocoded through <c>token2wav</c> as
    /// <see cref="ReleaseStreaming"/> describes.
    /// </summary>
    PaperBlocks,

    /// <summary>
    /// The release's web demo (<c>web_demo.py</c> with <c>AudioDecoder.token2wav</c>): blocks of 25, 50, 100, 150 and then
    /// 200 tokens. Each block's mel is cross-faded over 34 frames (a Hamming window) with the previous block's held-back
    /// tail. HiFT runs per block, carrying one mel frame and 256 excitation samples across blocks. The mel each call
    /// returns joins the prompt of the next.
    /// </summary>
    ReleaseStreaming,

    /// <summary>The release's <c>offline_inference</c>: one flow pass over all the tokens, then HiFT.</summary>
    Offline,
}

/// <summary>The sizes of GLM-4-Voice's decoder: the CosyVoice flow (block-causal Conformer, nearest regulator, cosine CFM)
/// and HiFT, with the release's streaming constants.</summary>
internal sealed record GlmVoiceDecoderConfiguration(CosyVoiceFlowConfiguration Flow, HiftConfiguration Hift, int BlockTokens = 10,
    int OverlapTokens = 5, int CacheMelFrames = 1, int NoiseSeed = 42)
{
    /// <summary>The release's <c>glm-4-voice-decoder/config.yaml</c>.</summary>
    public static GlmVoiceDecoderConfiguration Release(CosyVoiceFlowLoss loss = CosyVoiceFlowLoss.AbsoluteError) => new(
        new CosyVoiceFlowConfiguration(VocabSize: 16384, InputSize: 512, OutputSize: 80, SpeakerDim: 192, InputFrameRate: 12.5,
            SampleRate: 22050, HopSize: 256,
            Encoder: new WenetEncoderConfiguration(512, 512, 8, 2048, 6, WenetInputLayer.Linear, Swish: true, Conformer: true,
                StaticChunkSize: 10, Dropout: 0.1, PositionalDropout: 0.1, AttentionDropout: 0.1, BlockNormEpsilon: 1e-5),
            RegulatorStages: 4, DecoderChannels: new[] { 256, 256 }, DecoderHeads: 8, DecoderHeadDim: 64, DecoderBlocksPerLevel: 4,
            DecoderMidBlocks: 12, SigmaMin: 1e-6, TrainingCfgRate: 0.2, InferenceCfgRate: 0.7, Timesteps: 10, Loss: loss,
            Regulator: CosyVoiceRegulator.Nearest, PromptRatio: 0.8, CosineTrainingTime: true),
        new HiftConfiguration(InChannels: 80, BaseChannels: 512, Harmonics: 8, SampleRate: 22050, NsfAlpha: 0.1, NsfSigma: 0.003,
            VoicedThreshold: 10, UpsampleRates: new[] { 8, 8 }, UpsampleKernels: new[] { 16, 16 }, FftSize: 16, HopSize: 4,
            ResblockKernels: new[] { 3, 7, 11 }, ResblockDilations: new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } },
            SourceKernels: new[] { 7, 11 }, SourceDilations: new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 } }, LeakySlope: 0.1,
            AudioLimit: 0.99, F0Channels: 512));
}

/// <summary>
/// GLM-4-Voice's speech decoder (THUDM/GLM-4-Voice <c>flow_inference.py</c> over its <c>cosyvoice</c> fork): speech tokens
/// at 12.5 Hz to a 22.05 kHz waveform through CosyVoice's flow matching (<see cref="CosyVoiceFlow{T}"/>) and HiFT
/// (<see cref="HiftGenerator{T}"/>).
/// </summary>
/// <remarks>
/// <para>The fork's flow differs from CosyVoice's. Its Conformer attends in blocks of 10 tokens: each token sees its own
/// block and every earlier block (<c>BlockRelPositionMultiHeadedAttention</c>'s grid mask). Tokens are stretched to frames
/// by nearest-neighbour interpolation, and at inference over the whole sequence, prompt included. Training's prompt prefix
/// reaches 80 % of the mel, and its t follows the cosine schedule.</para>
/// <para>Every flow call starts from the same noise, as the release re-seeds torch with 42 in <c>ConditionalCFM.forward</c>;
/// here a generator seeded with <see cref="GlmVoiceDecoderConfiguration.NoiseSeed"/> is created per call.</para>
/// </remarks>
internal sealed class GlmVoiceDecoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public GlmVoiceDecoder(IEngine engine, List<LayerBase<T>> flowLayers, List<LayerBase<T>> hiftLayers, GlmVoiceDecoderConfiguration configuration,
        Random initialization)
    {
        _engine = engine;
        Configuration = configuration;
        if (configuration.Hift.UpsampleScale != configuration.Flow.HopSize)
            throw new ArgumentException($"HiFT gives {configuration.Hift.UpsampleScale} samples per mel frame; the flow's hop is {configuration.Flow.HopSize}.",
                nameof(configuration));
        Flow = new CosyVoiceFlow<T>(engine, flowLayers, configuration.Flow, initialization);
        Hift = new HiftGenerator<T>(engine, hiftLayers, configuration.Hift, initialization);
    }

    public GlmVoiceDecoderConfiguration Configuration { get; }
    public CosyVoiceFlow<T> Flow { get; }
    public HiftGenerator<T> Hift { get; }

    /// <summary>Mel frames held back to cross-fade with the next block: <c>int(5 / 12.5 · 22050 / 256)</c> = 34.</summary>
    public int OverlapFrames => (int)(Configuration.OverlapTokens / Configuration.Flow.InputFrameRate * Configuration.Flow.SampleRate / Configuration.Flow.HopSize);

    /// <summary>Excitation samples carried between blocks: the cached mel frames times the hop.</summary>
    public int CacheSamples => Configuration.CacheMelFrames * Configuration.Flow.HopSize;

    /// <summary>The state <c>token2wav</c> keeps per stream between calls.</summary>
    internal sealed class StreamState
    {
        public Tensor<T>? Overlap { get; set; }
        public Tensor<T>? CacheMel { get; set; }
        public Tensor<T>? CacheSource { get; set; }
    }

    /// <summary>The release's per-call noise <c>[1, mel channels, frames]</c>: standard normals from a fresh generator.</summary>
    internal Tensor<T> CallNoise(int frames)
    {
        var random = new Random(Configuration.NoiseSeed);
        var noise = new Tensor<T>(new[] { 1, Configuration.Flow.OutputSize, frames });
        for (int i = 0; i < noise.Length; i++)
            noise[i] = NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble()));
        return noise;
    }

    /// <summary>The flow's mel <c>[1, C, frames]</c> of <paramref name="tokens"/> after a prompt; <paramref name="noise"/>
    /// (default: <see cref="CallNoise"/>) covers the prompt's frames and these.</summary>
    public Tensor<T> Mel(IReadOnlyList<int> tokens, IReadOnlyList<int> promptTokens, Tensor<T>? promptMel, Tensor<T> xvector,
        Func<int, Tensor<T>>? noise = null)
    {
        int frames = Flow.MelFrames(promptTokens.Count + tokens.Count);
        using var _ = new NoGradScope<T>();
        return Flow.Inference(tokens, promptTokens, promptMel, xvector, (noise ?? CallNoise)(frames));
    }

    /// <summary>
    /// HiFT on a mel <c>[1, C, frames]</c> (<c>HiFTGenerator.forward</c> with <c>cache_source</c>): the waveform and its
    /// excitation <c>[1, 1, samples]</c>, whose first samples are replaced by <paramref name="cacheSource"/>.
    /// <paramref name="source"/> stands in for the random excitation (F0 per frame to <c>[1, 1, samples]</c>).
    /// </summary>
    public (Tensor<T> Speech, Tensor<T> Source) Vocode(Tensor<T> mel, Tensor<T>? cacheSource, Random random, Func<Tensor<T>, Tensor<T>>? source = null)
    {
        using var _ = new NoGradScope<T>();
        var f0 = Hift.PredictF0(mel);
        var s = source is null ? Hift.Source(f0, random) : source(f0);
        if (cacheSource is not null && cacheSource.Length > 0)
        {
            int cached = cacheSource.Length, samples = s.Shape[2];
            s = _engine.TensorConcatenate(new[]
            {
                _engine.Reshape(cacheSource, new[] { 1, 1, cached }),
                _engine.TensorSlice(s, new[] { 0, 0, cached }, new[] { 1, 1, samples - cached }),
            }, 2);
        }
        return (Hift.Decode(mel, s), s);
    }

    private Tensor<T> Frames(Tensor<T> mel, int start, int count) =>
        _engine.TensorSlice(mel, new[] { 0, 0, start }, new[] { 1, mel.Shape[1], count });

    /// <summary>
    /// <c>AudioDecoder.token2wav</c>: the flow's mel of <paramref name="tokens"/> after the prompt, cross-faded with the
    /// stream's held-back frames, preceded by its cached mel frame. Unless <paramref name="finalize"/>, the last
    /// <see cref="OverlapFrames"/> are held back, HiFT runs on the rest and the last <see cref="CacheSamples"/> samples
    /// are withheld (to be regenerated with the next block). Returns the speech <c>[samples]</c> and the mel
    /// <c>[1, C, frames]</c> as the release returns it.
    /// </summary>
    public (Tensor<T> Speech, Tensor<T> Mel) TokenToWave(IReadOnlyList<int> tokens, IReadOnlyList<int> promptTokens, Tensor<T>? promptMel,
        Tensor<T> xvector, StreamState state, bool finalize, Random random, Func<int, Tensor<T>>? noise = null,
        Func<Tensor<T>, Tensor<T>>? source = null)
    {
        var mel = Mel(tokens, promptTokens, promptMel, xvector, noise);
        int overlap = OverlapFrames;
        if (state.Overlap is not null)
        {
            // fade_in_out: the first `overlap` frames become mel·w[:L] + previous tail·w[L:], w = np.hamming(2L).
            int frames = mel.Shape[2], channels = mel.Shape[1], previous = state.Overlap.Shape[2];
            if (frames < overlap || previous < overlap)
                throw new ArgumentException($"A streamed block needs at least {overlap} mel frames to cross-fade; it has {frames}.", nameof(tokens));
            var faded = mel.Clone();
            for (int f = 0; f < overlap; f++)
            {
                double fadeIn = 0.54 - 0.46 * Math.Cos(2 * Math.PI * f / (2 * overlap - 1));
                double fadeOut = 0.54 - 0.46 * Math.Cos(2 * Math.PI * (overlap + f) / (2 * overlap - 1));
                for (int c = 0; c < channels; c++)
                    faded[0, c, f] = NumOps.FromDouble(NumOps.ToDouble(mel[0, c, f]) * fadeIn
                        + NumOps.ToDouble(state.Overlap[0, c, previous - overlap + f]) * fadeOut);
            }
            mel = faded;
        }
        Tensor<T>? cacheSource = null;
        if (state.CacheMel is not null)
        {
            mel = _engine.TensorConcatenate(new[] { state.CacheMel, mel }, 2);
            cacheSource = state.CacheSource;
        }
        if (finalize)
        {
            var (speech, _) = Vocode(mel, cacheSource, random, source);
            state.Overlap = state.CacheMel = state.CacheSource = null;
            return (speech, mel);
        }
        int total = mel.Shape[2];
        if (total <= overlap)
            throw new ArgumentException($"A streamed block needs more than {overlap} mel frames; it has {total}.", nameof(tokens));
        state.Overlap = Frames(mel, total - overlap, overlap);
        mel = Frames(mel, 0, total - overlap);
        var (blockSpeech, blockSource) = Vocode(mel, cacheSource, random, source);
        int kept = mel.Shape[2], samples = blockSpeech.Length, cache = CacheSamples, cacheFrames = Configuration.CacheMelFrames;
        state.CacheMel = Frames(mel, kept - cacheFrames, cacheFrames);
        state.CacheSource = _engine.TensorSlice(blockSource, new[] { 0, 0, blockSource.Shape[2] - cache }, new[] { 1, 1, cache });
        return (_engine.TensorSlice(blockSpeech, new[] { 0 }, new[] { samples - cache }), mel);
    }

    /// <summary>The web demo's block sizes: 25, 50, 100, 150, then 200 tokens.</summary>
    internal static readonly int[] ReleaseBlocks = { 25, 50, 100, 150, 200 };

    /// <summary>
    /// The waveform <c>[samples]</c> of speech tokens, decoded as <paramref name="decoding"/> says, with the speaker
    /// x-vector <paramref name="xvector"/> (the release passes zeros).
    /// </summary>
    public Tensor<T> Decode(IReadOnlyList<int> tokens, Tensor<T> xvector, GlmVoiceDecoding decoding, Random random,
        Func<int, Tensor<T>>? noise = null, Func<Tensor<T>, Tensor<T>>? source = null)
    {
        if (tokens.Count == 0) throw new ArgumentException("There are no speech tokens to decode.", nameof(tokens));
        var none = Array.Empty<int>();
        switch (decoding)
        {
            case GlmVoiceDecoding.Offline:
                return TokenToWave(tokens, none, null, xvector, new StreamState(), finalize: true, random, noise, source).Speech;
            default:
            {
                bool paper = decoding == GlmVoiceDecoding.PaperBlocks;
                var state = new StreamState();
                var speeches = new List<Tensor<T>>();
                var mels = new List<Tensor<T>>();
                int index = 0, start = 0;
                while (start < tokens.Count)
                {
                    int size = paper ? Configuration.BlockTokens : ReleaseBlocks[Math.Min(index++, ReleaseBlocks.Length - 1)];
                    int count = Math.Min(size, tokens.Count - start);
                    bool finalize = start + count >= tokens.Count;
                    var promptMel = mels.Count == 0 ? null : RowsOf(_engine.TensorConcatenate(mels.ToArray(), 2));
                    var (speech, mel) = TokenToWave(tokens.Skip(start).Take(count).ToList(), tokens.Take(start).ToList(), promptMel, xvector,
                        state, finalize, random, noise, source);
                    speeches.Add(speech);
                    mels.Add(mel);
                    start += count;
                }
                return _engine.TensorConcatenate(speeches.ToArray(), 0);
            }
        }
    }

    // A mel [1, C, frames] as the flow's prompt rows [frames, C].
    private Tensor<T> RowsOf(Tensor<T> mel) => _engine.TensorTranspose(_engine.Reshape(mel, new[] { mel.Shape[1], mel.Shape[2] }));

    /// <summary>
    /// The paper's streaming fine-tuning sample (§3.2): the first n·b seconds of an utterance, n drawn uniformly from
    /// 1 … ⌈tokens / block⌉, so that in expectation every truncation of the utterance is trained on. The last n is the
    /// whole utterance.
    /// </summary>
    public (int[] Tokens, Tensor<T> Mel) Truncate(IReadOnlyList<int> tokens, Tensor<T> mel, Random random)
    {
        int block = Configuration.BlockTokens, blocks = (tokens.Count + block - 1) / block;
        int n = random.Next(1, blocks + 1);
        if (n == blocks) return (tokens.ToArray(), mel);
        int count = n * block, frames = Math.Min(mel.Shape[2], Flow.MelFrames(count));
        return (tokens.Take(count).ToArray(), Frames(mel, 0, frames));
    }

    /// <summary>Loads the release's <c>flow.pt</c> and <c>hift.pt</c> states. <c>hift.pt</c> keeps weight norm's
    /// <c>weight_g</c>/<c>weight_v</c>, read here as the parametrization's gain and direction.</summary>
    public void Load(Func<string, int[], double[]> readFlow, Func<string, int[], double[]> readHift, Func<string, bool> hiftHas)
    {
        Flow.LoadTorchWeights(readFlow, "");
        Hift.LoadTorchWeights((name, shape) =>
        {
            if (!hiftHas(name))
            {
                if (name.EndsWith(".parametrizations.weight.original0", StringComparison.Ordinal))
                    name = name.Substring(0, name.Length - ".parametrizations.weight.original0".Length) + ".weight_g";
                else if (name.EndsWith(".parametrizations.weight.original1", StringComparison.Ordinal))
                    name = name.Substring(0, name.Length - ".parametrizations.weight.original1".Length) + ".weight_v";
            }
            return readHift(name, shape);
        });
    }
}
