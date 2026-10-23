using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>The sizes of MeloTTS's synthesizer (myshell-ai/MeloTTS <c>configs/config.json</c> by default).</summary>
internal sealed record MeloConfiguration(
    int Vocabulary, int Tones, int Languages, int SpectrogramChannels, int SegmentFrames, int InterChannels = 192,
    int HiddenChannels = 192, int FilterChannels = 768, int Heads = 2, int Layers = 6, int KernelSize = 3, double Dropout = 0.1,
    int[]? ResblockKernels = null, int[][]? ResblockDilations = null, int[]? UpsampleRates = null, int UpsampleInitialChannels = 512,
    int[]? UpsampleKernels = null, int Speakers = 256, int SpeakerChannels = 256, int FlowCouplings = 4, int FlowLayers = 3,
    double DurationDropout = 0.5, double MasNoiseInitial = 0.01, double MasNoiseDecay = 2e-6)
{
    public int[] Kernels => ResblockKernels ?? new[] { 3, 7, 11 };
    public int[][] Dilations => ResblockDilations ?? new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };
    public int[] Rates => UpsampleRates ?? new[] { 8, 8, 2, 2, 2 };
    public int[] UpKernels => UpsampleKernels ?? new[] { 16, 16, 8, 2, 2 };
    public int HopSize => Rates.Aggregate(1, (a, b) => a * b);
}

/// <summary>The quantities one MeloTTS training forward produces (<c>SynthesizerTrn.forward</c>'s outputs).</summary>
internal sealed record MeloTrainingOutputs<T>(
    Tensor<T> Waveform, Tensor<T> DurationLoss, int[] Durations, Tensor<T> Hidden, Tensor<T> PriorMean, Tensor<T> PriorLogScale,
    Tensor<T> PosteriorMean, Tensor<T> PosteriorLogScale, Tensor<T> Z, Tensor<T> ZPrior, Tensor<T> PredictedLogDurations,
    Tensor<T> AlignedLogDurations, int SliceStart, Tensor<T> Speaker);

/// <summary>
/// MeloTTS's synthesizer (myshell-ai/MeloTTS <c>models.SynthesizerTrn</c>, from Bert-VITS2): VITS with VITS2's
/// speaker-conditioned text encoder, transformer coupling flows and noise-scaled alignment search, per-symbol tone,
/// language and BERT features, and durations from both a stochastic and a deterministic predictor.
/// </summary>
internal sealed class MeloSynthesizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public MeloSynthesizer(IEngine engine, List<LayerBase<T>> layers, MeloConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        TextEncoder = new MeloTextEncoder<T>(engine, layers, c.Vocabulary, c.Tones, c.Languages, c.InterChannels, c.HiddenChannels,
            c.FilterChannels, c.Heads, c.Layers, c.KernelSize, c.Dropout, c.SpeakerChannels);
        Generator = new HiFiGanGenerator<T>(engine, c.InterChannels, c.UpsampleInitialChannels, c.Rates, c.UpKernels, c.Kernels, c.Dilations,
            resblockType1: true, conditionChannels: c.SpeakerChannels, plainEnds: true);
        layers.AddRange(Generator.Layers);
        PosteriorEncoder = new MeloPosteriorEncoder<T>(engine, layers, c.SpectrogramChannels, c.InterChannels, c.HiddenChannels, c.SpeakerChannels);
        Flow = new MeloTransformerFlow<T>(engine, layers, c.InterChannels, c.HiddenChannels, c.FilterChannels, c.Heads, c.FlowLayers, 5,
            c.Dropout, c.FlowCouplings, c.SpeakerChannels);
        StochasticDuration = new MeloStochasticDurationPredictor<T>(engine, layers, c.HiddenChannels, 3, c.DurationDropout, 4, c.SpeakerChannels);
        Duration = new MeloDurationPredictor<T>(engine, layers, c.HiddenChannels, 256, 3, c.DurationDropout, c.SpeakerChannels);
        layers.Add(SpeakerEmbedding = new TiedEmbeddingLayer<T>(c.Speakers, c.SpeakerChannels));
    }

    public MeloConfiguration Configuration { get; }
    public MeloTextEncoder<T> TextEncoder { get; }
    public HiFiGanGenerator<T> Generator { get; }
    public MeloPosteriorEncoder<T> PosteriorEncoder { get; }
    public MeloTransformerFlow<T> Flow { get; }
    public MeloStochasticDurationPredictor<T> StochasticDuration { get; }
    public MeloDurationPredictor<T> Duration { get; }
    public TiedEmbeddingLayer<T> SpeakerEmbedding { get; }

    /// <summary>The speaker embedding <c>[1, gin, 1]</c> (<c>emb_g(sid)</c>).</summary>
    public Tensor<T> Speaker(int id) =>
        _engine.Reshape(_engine.TensorSlice(SpeakerEmbedding.Table, new[] { id, 0 }, new[] { 1, Configuration.SpeakerChannels }),
            new[] { 1, Configuration.SpeakerChannels, 1 });

    private static double Gaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
        return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }

    private static Tensor<T> Gaussian(int[] shape, Random random, double scale)
    {
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = NumOps.FromDouble(scale * Gaussian(random));
        return t;
    }

    /// <summary>
    /// <c>SynthesizerTrn.forward</c> for one utterance: symbols, tones, languages, BERT features <c>[1, 1024, L]</c> and
    /// <c>[1, 768, L]</c>, the linear spectrogram <c>[1, spec, T]</c> and the speaker. The random draws — the posterior's
    /// noise <c>[1, inter, T]</c>, the alignment search's noise <c>[T, L]</c> (scaled by std(neg_cent) and
    /// <paramref name="masNoiseScale"/>), the stochastic duration predictor's posterior noise <c>[1, 2, L]</c> and the
    /// decoder segment's start — come from <paramref name="random"/> unless given.
    /// </summary>
    public MeloTrainingOutputs<T> Forward(IReadOnlyList<int> symbols, IReadOnlyList<int> tones, IReadOnlyList<int> languages, Tensor<T> bert,
        Tensor<T> jaBert, Tensor<T> spectrogram, int speaker, double masNoiseScale, bool training, Random random,
        Tensor<T>? posteriorNoise = null, double[,]? masNoise = null, Tensor<T>? durationNoise = null, int? sliceStart = null,
        bool durationTraining = true)
    {
        var c = Configuration;
        int tokens = symbols.Count, frames = spectrogram.Shape[2];
        var g = Speaker(speaker);
        var (hidden, mP, logsP) = TextEncoder.Forward(symbols, tones, languages, bert, jaBert, g, training, random);
        posteriorNoise ??= Gaussian(new[] { 1, c.InterChannels, frames }, random, 1.0);
        var (z, mQ, logsQ) = PosteriorEncoder.Forward(spectrogram, g, posteriorNoise);
        var zP = Flow.Forward(z, g, reverse: false, training, random);

        // Monotonic alignment search on the prior's log-likelihood of z_p [T, L], with noise σ = std · scale.
        int[] durations;
        int d = c.InterChannels;
        var negCent = new double[frames, tokens];
        double sum = 0, sumSquares = 0;
        for (int f = 0; f < frames; f++)
            for (int l = 0; l < tokens; l++)
            {
                double v = 0;
                for (int k = 0; k < d; k++)
                {
                    double lp = NumOps.ToDouble(logsP[0, k, l]), m = NumOps.ToDouble(mP[0, k, l]), zz = NumOps.ToDouble(zP[0, k, f]);
                    double s = Math.Exp(-2 * lp);
                    v += -0.5 * Math.Log(2 * Math.PI) - lp - 0.5 * zz * zz * s + zz * m * s - 0.5 * m * m * s;
                }
                negCent[f, l] = v;
                sum += v;
                sumSquares += v * v;
            }
        if (masNoiseScale > 0)
        {
            int n = frames * tokens;
            double mean = sum / n, std = Math.Sqrt(Math.Max(0, (sumSquares - n * mean * mean) / (n - 1)));
            for (int f = 0; f < frames; f++)
                for (int l = 0; l < tokens; l++)
                    negCent[f, l] += std * (masNoise is null ? Gaussian(random) : masNoise[f, l]) * masNoiseScale;
        }
        var scores = new double[tokens, frames];
        for (int f = 0; f < frames; f++)
            for (int l = 0; l < tokens; l++) scores[l, f] = negCent[f, l];
        durations = MonotonicAlignment.MaximumPath(scores);
        var attn = Path(durations, frames);                                                                  // [L, T]

        var w = new Tensor<T>(new[] { 1, 1, tokens });
        var logwAligned = new Tensor<T>(new[] { 1, 1, tokens });
        for (int l = 0; l < tokens; l++)
        {
            w[0, 0, l] = NumOps.FromDouble(durations[l]);
            logwAligned[0, 0, l] = NumOps.FromDouble(Math.Log(durations[l] + 1e-6));
        }
        var sdpLoss = _engine.TensorMultiplyScalar(StochasticDuration.NegativeLogLikelihood(hidden, w, g, durationTraining && training, random, durationNoise),
            NumOps.FromDouble(1.0 / tokens));
        var logw = Duration.Forward(hidden, g, durationTraining && training, random);
        var difference = _engine.TensorSubtract(logw, logwAligned);
        var dpLoss = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(difference, difference), new[] { 0, 1, 2 }, keepDims: false),
            NumOps.FromDouble(1.0 / tokens));
        var durationLoss = _engine.TensorAdd(_engine.Reshape(dpLoss, new[] { 1 }), _engine.Reshape(sdpLoss, new[] { 1 }));

        var expandedMean = Expand(mP, attn);
        var expandedLogScale = Expand(logsP, attn);
        int start = sliceStart ?? (int)(random.NextDouble() * (frames - c.SegmentFrames + 1));
        var segment = _engine.TensorSlice(z, new[] { 0, 0, start }, new[] { 1, c.InterChannels, c.SegmentFrames });
        var waveform = Generator.Forward(segment, g);
        return new MeloTrainingOutputs<T>(waveform, durationLoss, durations, hidden, expandedMean, expandedLogScale, mQ, logsQ, z, zP,
            logw, logwAligned, start, g);
    }

    // generate_path: token l covers the frames [Σ_{<l} d, Σ_{≤l} d); [L, T].
    private static Tensor<T> Path(IReadOnlyList<int> durations, int frames)
    {
        var path = new Tensor<T>(new[] { durations.Count, frames });
        int start = 0;
        for (int l = 0; l < durations.Count; l++)
        {
            for (int f = start; f < Math.Min(frames, start + durations[l]); f++) path[l, f] = NumOps.One;
            start += durations[l];
        }
        return path;
    }

    // [1, d, L] stretched by an alignment [L, T] to [1, d, T].
    private Tensor<T> Expand(Tensor<T> x, Tensor<T> attn)
    {
        int d = x.Shape[1], length = x.Shape[2];
        return _engine.Reshape(_engine.TensorMatMul(_engine.Reshape(x, new[] { d, length }), attn), new[] { 1, d, attn.Shape[1] });
    }

    /// <summary>
    /// <c>SynthesizerTrn.infer</c>: log-durations blended from the stochastic predictor (noise σ
    /// <paramref name="durationNoiseScale"/>, weight <paramref name="sdpRatio"/>) and the deterministic one, ceiled and scaled
    /// by <paramref name="lengthScale"/>; the expanded prior sampled with σ <paramref name="noiseScale"/>, inverted
    /// through the flow and decoded. The duration noise (already scaled) and the prior's unit noise may be supplied.
    /// </summary>
    public (Tensor<T> Waveform, int[] Durations) Infer(IReadOnlyList<int> symbols, IReadOnlyList<int> tones, IReadOnlyList<int> languages,
        Tensor<T> bert, Tensor<T> jaBert, int speaker, Random random, double noiseScale = 0.667, double lengthScale = 1.0,
        double durationNoiseScale = 0.8, double sdpRatio = 0.0, Tensor<T>? durationNoise = null, Func<int, Tensor<T>>? priorNoise = null)
    {
        using var _ = new NoGradScope<T>();
        int tokens = symbols.Count;
        var g = Speaker(speaker);
        var (hidden, mP, logsP) = TextEncoder.Forward(symbols, tones, languages, bert, jaBert, g, false, random);
        var stochastic = StochasticDuration.Sample(hidden, g, random, durationNoiseScale, durationNoise);
        var deterministic = Duration.Forward(hidden, g, false, random);
        var durations = new int[tokens];
        for (int l = 0; l < tokens; l++)
        {
            double logw = NumOps.ToDouble(stochastic[0, 0, l]) * sdpRatio + NumOps.ToDouble(deterministic[0, 0, l]) * (1 - sdpRatio);
            durations[l] = (int)Math.Ceiling(Math.Exp(logw) * lengthScale);
        }
        int frames = Math.Max(1, durations.Sum());
        var attn = Path(durations, frames);
        var mean = Expand(mP, attn);
        var logScale = Expand(logsP, attn);
        var eps = priorNoise?.Invoke(frames) ?? Gaussian(new[] { 1, Configuration.InterChannels, frames }, random, 1.0);
        var zP = _engine.TensorAdd(mean, _engine.TensorMultiplyScalar(_engine.TensorMultiply(eps, _engine.TensorExp(logScale)), NumOps.FromDouble(noiseScale)));
        var z = Flow.Forward(zP, g, reverse: true, false, random);
        return (Generator.Forward(z, g), durations);
    }

    /// <summary>
    /// train.py's kl_loss: Σ (logs_p − logs_q − ½ + ½ (z_p − m_p)² e^{−2 logs_p}) over the frames, divided by their count.
    /// </summary>
    public Tensor<T> KlLoss(MeloTrainingOutputs<T> o)
    {
        var e = _engine;
        var diff = e.TensorSubtract(o.ZPrior, o.PriorMean);
        var kl = e.TensorAdd(e.TensorAddScalar(e.TensorSubtract(o.PriorLogScale, o.PosteriorLogScale), NumOps.FromDouble(-0.5)),
            e.TensorMultiplyScalar(e.TensorMultiply(e.TensorMultiply(diff, diff), e.TensorExp(e.TensorMultiplyScalar(o.PriorLogScale, NumOps.FromDouble(-2)))),
                NumOps.FromDouble(0.5)));
        return e.TensorMultiplyScalar(e.ReduceSum(kl, new[] { 0, 1, 2 }, keepDims: false), NumOps.FromDouble(1.0 / o.Z.Shape[2]));
    }

    /// <summary>Loads a MeloTTS generator checkpoint's state dict (<c>model['net_g']</c>, PyTorch names).</summary>
    public void LoadTorch(Func<string, double[]?> read)
    {
        double[] Require(string name) => read(name) ?? throw new KeyNotFoundException(name);
        TextEncoder.LoadTorch(Require, "enc_p.");
        Generator.LoadTorch(read, "dec.");
        PosteriorEncoder.LoadTorch(Require, "enc_q.");
        Flow.LoadTorch(Require, "flow.");
        StochasticDuration.LoadTorch(Require, "sdp.");
        Duration.LoadTorch(Require, "dp.");
        SpeakerEmbedding.LoadTable(Require("emb_g.weight"));
    }
}
