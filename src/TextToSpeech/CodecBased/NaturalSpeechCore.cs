using AiDotNet.Helpers;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes and choices of NaturalSpeech's networks (the paper's Table 8 by default).</summary>
internal sealed record NaturalSpeechConfiguration
{
    public int Vocab { get; init; } = 178;
    public int Hidden { get; init; } = 192;
    public int Filter { get; init; } = 768;
    public int Heads { get; init; } = 2;
    public int EncoderLayers { get; init; } = 6;
    public int EncoderKernel { get; init; } = 3;
    public double Dropout { get; init; } = 0.1;
    public NaturalSpeechPhonemeEncoder Encoder { get; init; } = NaturalSpeechPhonemeEncoder.FeedForwardTransformer;
    public int MaxPositions { get; init; } = 1000;
    public int DurationLayers { get; init; } = 3;
    public int DurationFilter { get; init; } = 192;
    public int DurationKernel { get; init; } = 3;
    public double DurationDropout { get; init; } = 0.5;
    public NaturalSpeechUpsampling Upsampling { get; init; } = NaturalSpeechUpsampling.Paper;
    public int UpsamplingKernel { get; init; } = 3;
    public int UpsamplingConvOutput { get; init; } = 8;
    public int UpsamplingQ { get; init; } = 4;
    public int UpsamplingP { get; init; } = 2;
    public int SpecChannels { get; init; } = 513;
    public int Couplings { get; init; } = 4;
    public int CouplingLayers { get; init; } = 4;
    public int[] UpsampleRates { get; init; } = { 8, 8, 2, 2 };
    public int[] UpsampleKernels { get; init; } = { 16, 16, 4, 4 };
    public int UpsampleInitialChannels { get; init; } = 256;
    public int[] ResblockKernels { get; init; } = { 3, 7, 11 };
    public int[][] ResblockDilations { get; init; } = { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };
    public int MemorySize { get; init; } = 1000;
    public int MemoryHeads { get; init; } = 2;
    public int SupPhonemes { get; init; }
}

/// <summary>
/// NaturalSpeech's networks (Tan et al. 2022, §3): the phoneme encoder, the differentiable durator (duration predictor and
/// learnable upsampling to the frame-level prior p(z′|y)), the phoneme-level prior two linear layers give for monotonic
/// alignment search in warm-up, the posterior encoder q(z|x) over the linear spectrogram, the bidirectional
/// prior/posterior flow f (mean-only affine couplings), the VAE memory and the HiFi-GAN waveform decoder.
/// </summary>
internal sealed class NaturalSpeechCore<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<(NormedConv1DLayer<T> Pre, MeloWaveNet<T> Net, NormedConv1DLayer<T> Post)> _couplings = new();

    public NaturalSpeechCore(IEngine engine, NaturalSpeechConfiguration configuration, Random random, List<LayerBase<T>> phonemeLayers,
        List<LayerBase<T>> duratorLayers, List<LayerBase<T>> flowLayers, List<LayerBase<T>> posteriorLayers, List<LayerBase<T>> decoderLayers,
        List<LayerBase<T>> memoryLayers, List<LayerBase<T>> pretrainingLayers)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        int h = c.Hidden;
        // Phoneme encoder θpho.
        // One extra row each for pre-training's [MASK] token.
        Embedding = IndexTtsOps.Embedding(phonemeLayers, random, c.Vocab + 1, h, Math.Pow(h, -0.5));
        if (c.SupPhonemes > 0) SupEmbedding = IndexTtsOps.Embedding(phonemeLayers, random, c.SupPhonemes + 1, h, Math.Pow(h, -0.5));
        if (c.Encoder == NaturalSpeechPhonemeEncoder.FeedForwardTransformer)
        {
            for (int i = 0; i < c.EncoderLayers; i++)
            {
                var block = new FeedForwardTransformerBlock<T>(h, c.Heads, c.Filter, c.EncoderKernel, c.EncoderKernel, c.Dropout);
                phonemeLayers.Add(block);
                FftBlocks.Add(block);
            }
        }
        else VitsEncoder = new MeloAttentionEncoder<T>(engine, phonemeLayers, h, c.Filter, c.Heads, c.EncoderLayers, c.EncoderKernel, c.Dropout, 0);
        // Mixed-phoneme pre-training heads (masked phoneme and sup-phoneme prediction).
        PhonemeHead = IndexTtsOps.Linear(engine, pretrainingLayers, random, h, c.Vocab);
        if (c.SupPhonemes > 0) SupHead = IndexTtsOps.Linear(engine, pretrainingLayers, random, h, c.SupPhonemes);
        // Durator θdur.
        DurationPredictor = new NaturalSpeechDurationPredictor<T>(engine, duratorLayers, h, c.DurationFilter, c.DurationKernel, c.DurationLayers, c.DurationDropout);
        Upsampler = new NaturalSpeechUpsampler<T>(engine, duratorLayers, random, c.Upsampling, h, c.UpsamplingKernel, c.UpsamplingConvOutput,
            c.UpsamplingQ, c.UpsamplingP, h);
        PhonemeMean = IndexTtsOps.Linear(engine, duratorLayers, random, h, h);
        PhonemeLogScale = IndexTtsOps.Linear(engine, duratorLayers, random, h, h);
        // Bidirectional prior/posterior θbpp: mean-only couplings (the scaling discarded, §4.1).
        int half = h / 2;
        for (int i = 0; i < c.Couplings; i++)
        {
            var pre = MeloOps.Conv(flowLayers, half, h, 1);
            var net = new MeloWaveNet<T>(engine, flowLayers, h, 5, 1, c.CouplingLayers, 0);
            var post = MeloOps.Conv(flowLayers, h, half, 1);
            post.LoadTorchWeights(new double[half * h], null, new double[half]);
            _couplings.Add((pre, net, post));
        }
        Posterior = new MeloPosteriorEncoder<T>(engine, posteriorLayers, c.SpecChannels, h, h, 0);
        Decoder = new HiFiGanGenerator<T>(engine, h, c.UpsampleInitialChannels, c.UpsampleRates, c.UpsampleKernels, c.ResblockKernels, c.ResblockDilations,
            resblockType1: true, plainEnds: true, initialization: random);
        decoderLayers.AddRange(Decoder.Layers);
        Memory = new NaturalSpeechMemoryBank<T>(engine, memoryLayers, random, h, c.MemorySize, c.MemoryHeads);
        if (c.Encoder == NaturalSpeechPhonemeEncoder.FeedForwardTransformer)
        {
            // FastSpeech's sinusoidal positions.
            _positions = new double[c.MaxPositions * h];
            for (int p = 0; p < c.MaxPositions; p++)
                for (int i = 0; i < h; i++)
                {
                    double angle = p / Math.Pow(10000, 2.0 * Math.Floor(i / 2.0) / h);
                    _positions[p * h + i] = i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle);
                }
        }
    }

    private readonly double[]? _positions;

    public NaturalSpeechConfiguration Configuration { get; }
    public TiedEmbeddingLayer<T> Embedding { get; }
    public TiedEmbeddingLayer<T>? SupEmbedding { get; }
    public List<FeedForwardTransformerBlock<T>> FftBlocks { get; } = new();
    public MeloAttentionEncoder<T>? VitsEncoder { get; }
    public DenseLayer<T> PhonemeHead { get; }
    public DenseLayer<T>? SupHead { get; }
    public NaturalSpeechDurationPredictor<T> DurationPredictor { get; }
    public NaturalSpeechUpsampler<T> Upsampler { get; }
    public DenseLayer<T> PhonemeMean { get; }
    public DenseLayer<T> PhonemeLogScale { get; }
    public MeloPosteriorEncoder<T> Posterior { get; }
    public HiFiGanGenerator<T> Decoder { get; }
    public NaturalSpeechMemoryBank<T> Memory { get; }

    /// <summary>Whether the decoder reads the memory bank (after warm-up).</summary>
    public bool UseMemory { get; set; }

    /// <summary>
    /// The phoneme hidden states <c>[1, h, n]</c> of phoneme ids (with their sup-phoneme ids when the model has
    /// sup-phonemes): the embeddings scaled by √h, plus positions for the FFT encoder, through the encoder.
    /// </summary>
    public Tensor<T> Encode(IReadOnlyList<int> phonemes, IReadOnlyList<int>? supPhonemes, bool training, Random random)
    {
        var c = Configuration;
        int n = phonemes.Count, h = c.Hidden;
        if (phonemes.Any(p => p < 0 || p > c.Vocab)) throw new ArgumentOutOfRangeException(nameof(phonemes), $"Phoneme ids lie in [0, {c.Vocab}] (the last is [MASK]).");
        var x = _engine.TensorMultiplyScalar(Embedding.Forward(IndexTtsOps.Ids<T>(phonemes)), NumOps.FromDouble(Math.Sqrt(h)));   // [n, h]
        if (SupEmbedding is not null)
        {
            if (supPhonemes is null || supPhonemes.Count != n) throw new ArgumentException("Each phoneme needs its sup-phoneme id.", nameof(supPhonemes));
            x = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(SupEmbedding.Forward(IndexTtsOps.Ids<T>(supPhonemes)), NumOps.FromDouble(Math.Sqrt(h))));
        }
        if (VitsEncoder is not null)
            return VitsEncoder.Forward(_engine.Reshape(_engine.TensorTranspose(x), new[] { 1, h, n }), null, training, random);
        if (n > c.MaxPositions) throw new ArgumentException($"The FFT encoder has positions for {c.MaxPositions} phonemes.", nameof(phonemes));
        var pos = new Tensor<T>(new[] { n, h });
        for (int i = 0; i < n * h; i++) pos.SetFlat(i, NumOps.FromDouble(_positions![i]));
        var rows = _engine.Reshape(_engine.TensorAdd(x, pos), new[] { 1, n, h });
        foreach (var block in FftBlocks)
        {
            block.SetTrainingMode(training);
            rows = block.Forward(rows);
        }
        return _engine.Reshape(_engine.TensorTranspose(_engine.Reshape(rows, new[] { n, h })), new[] { 1, h, n });
    }

    /// <summary>The phoneme-level prior <c>(mean, log-scale)</c>, each <c>[1, h, n]</c> (two linear layers on the hidden
    /// states, App. F), which monotonic alignment search matches to the simplified posterior in warm-up.</summary>
    public (Tensor<T> Mean, Tensor<T> LogScale) PhonemePrior(Tensor<T> hidden)
    {
        int h = Configuration.Hidden, n = hidden.Shape[2];
        var rows = _engine.TensorTranspose(_engine.Reshape(hidden, new[] { h, n }));
        Tensor<T> Channels(Tensor<T> r) => _engine.Reshape(_engine.TensorTranspose(r), new[] { 1, h, n });
        return (Channels(PhonemeMean.Forward(rows)), Channels(PhonemeLogScale.Forward(rows)));
    }

    /// <summary>The flow f (forward: z′ → z) or f⁻¹ (backward: z → z′).</summary>
    public Tensor<T> Flow(Tensor<T> x, bool backward)
    {
        if (backward)
        {
            // f⁻¹: VITS's flow direction (coupling, then flip).
            foreach (var coupling in _couplings) x = MeloOps.Flip(_engine, Couple(x, coupling, false));
            return x;
        }
        for (int i = _couplings.Count - 1; i >= 0; i--) x = Couple(MeloOps.Flip(_engine, x), _couplings[i], true);
        return x;
    }

    private Tensor<T> Couple(Tensor<T> x, (NormedConv1DLayer<T> Pre, MeloWaveNet<T> Net, NormedConv1DLayer<T> Post) c, bool reverse)
    {
        int half = Configuration.Hidden / 2;
        var x0 = MeloOps.Channels(_engine, x, 0, half);
        var x1 = MeloOps.Channels(_engine, x, half, half);
        var m = c.Post.Forward(c.Net.Forward(c.Pre.Forward(x0), null));
        return _engine.TensorConcatenate(new[] { x0, reverse ? _engine.TensorSubtract(x1, m) : _engine.TensorAdd(x1, m) }, 1);
    }

    /// <summary>The waveform <c>[samples]</c> of latents <c>[1, h, T]</c> through the memory (when used) and the decoder.</summary>
    public Tensor<T> Decode(Tensor<T> z)
    {
        var input = UseMemory ? Memory.Forward(z) : z;
        var audio = Decoder.Forward(input);
        return _engine.Reshape(audio, new[] { audio.Length });
    }

    /// <summary>The durator's log durations <c>[n]</c> (detached input).</summary>
    public Tensor<T> LogDurations(Tensor<T> hidden, bool training, Random random)
    {
        var logw = DurationPredictor.Forward(hidden, training, random);
        return _engine.Reshape(logw, new[] { hidden.Shape[2] });
    }

    /// <summary>Loads heatz123/naturalspeech's generator checkpoint names (for its variants of the modules).</summary>
    public void LoadHeatz123(Func<string, double[]?> read) => LoadHeatz123(TorchTensorSource.From(read));

    /// <summary>Loads heatz123/naturalspeech's generator checkpoint names from <paramref name="source"/>.</summary>
    internal void LoadHeatz123(TorchTensorSource source)
    {
        // The VITS modules shared with MeloTTS still take their values as arrays.
        double[] Require(string n) => source.Read(n, Array.Empty<int>());
        if (VitsEncoder is null) throw new InvalidOperationException("heatz123's checkpoint has the VITS phoneme encoder; build with NaturalSpeechPhonemeEncoder.Vits.");
        Embedding.LoadLeadingRows(source, "enc_p.emb.weight");
        VitsEncoder.LoadTorch(_engine, Require, "enc_p.encoder.");
        DurationPredictor.Load(source, "dp.");
        Upsampler.LoadHeatz123(source, "learnable_upsampling.");
        for (int i = 0; i < _couplings.Count; i++)
        {
            var (pre, net, post) = _couplings[i];
            NaturalSpeechOps.LoadConv(source, pre, $"flow.flows.{2 * i}.pre");
            net.LoadTorch(Require, $"flow.flows.{2 * i}.enc.");
            NaturalSpeechOps.LoadConv(source, post, $"flow.flows.{2 * i}.post");
        }
        Posterior.LoadTorch(Require, "enc_q.");
        Decoder.LoadTorch(n => source.TryRead(n), "dec.");
        if (source.Contains("memory_bank.memory_bank")) Memory.Load(source, "memory_bank.");
    }
}
