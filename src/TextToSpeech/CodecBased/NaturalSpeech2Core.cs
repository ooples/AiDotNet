using AiDotNet.Helpers;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>NaturalSpeech 2's network sizes (<see cref="NaturalSpeech2Options"/> resolved against its codec).</summary>
internal sealed record Ns2Configuration
{
    public int Vocab { get; init; } = 100;
    public int Latent { get; init; } = 256;
    public int Hidden { get; init; } = 512;
    public int Heads { get; init; } = 8;
    public int EncoderLayers { get; init; } = 6;
    public int EncoderFilter { get; init; } = 2048;
    public int EncoderKernel { get; init; } = 9;
    public double EncoderDropout { get; init; } = 0.2;
    public bool StyleAdaptiveEncoder { get; init; }
    public int PromptLayers { get; init; } = 6;
    public NaturalSpeech2Positions Positions { get; init; } = NaturalSpeech2Positions.Sinusoidal;
    public int DurationLayers { get; init; } = 30;
    public int DurationKernel { get; init; } = 3;
    public int PitchLayers { get; init; } = 30;
    public int PitchKernel { get; init; } = 5;
    public int PredictorAttentionEvery { get; init; } = 3;
    public double PredictorDropout { get; init; } = 0.5;
    public double PitchMin { get; init; } = 50;
    public double PitchMax { get; init; } = 1100;
    public int PitchBins { get; init; } = 512;
    public int QueryTokens { get; init; } = 32;
    public int DenoiserLayers { get; init; } = 40;
    public int DenoiserDilation { get; init; } = 2;
    public int? DenoiserDilationCycle { get; init; }
    public int DenoiserAttentionEvery { get; init; } = 3;
    public int DenoiserAttentionOffset { get; init; } = 2;
    public double DenoiserAttentionDropout { get; init; } = 0.1;
}

/// <summary>
/// NaturalSpeech 2's networks (Shen et al. 2023, §3.2-3.3): the phoneme encoder, the duration and pitch predictors with
/// prompt attention, the speech prompt encoder with its query-token attention, and the WaveNet denoiser of the latent
/// diffusion model; and the variance-preserving SDE that model is trained and sampled on.
/// </summary>
internal sealed class NaturalSpeech2Core<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly Ns2Configuration _c;
    private double[] _pitchBins;

    public NaturalSpeech2Core(IEngine engine, Ns2Configuration c, Random random, List<LayerBase<T>> priorLayers, List<LayerBase<T>> promptLayers,
        List<LayerBase<T>> denoiserLayers)
    {
        _engine = engine;
        _c = c;
        int h = c.Hidden;
        PhonemeEncoder = new Ns2TransformerEncoder<T>(engine, priorLayers, random, c.Vocab, h, c.Heads, c.EncoderFilter, c.EncoderKernel, c.EncoderLayers,
            c.EncoderDropout, c.StyleAdaptiveEncoder, c.Positions);
        DurationPredictor = new Ns2VariancePredictor<T>(engine, priorLayers, random, h, h, c.DurationKernel, c.DurationLayers, c.PredictorAttentionEvery,
            c.Heads, c.PredictorDropout);
        PitchPredictor = new Ns2VariancePredictor<T>(engine, priorLayers, random, h, h, c.PitchKernel, c.PitchLayers, c.PredictorAttentionEvery,
            c.Heads, c.PredictorDropout);
        PitchEmbedding = IndexTtsOps.Embedding(priorLayers, random, c.PitchBins, h, 1.0);
        // exp(linspace(log min, log max, bins − 1)): the bucket boundaries of torch.bucketize.
        _pitchBins = new double[c.PitchBins - 1];
        for (int i = 0; i < _pitchBins.Length; i++)
            _pitchBins[i] = Math.Exp(Math.Log(c.PitchMin) + (Math.Log(c.PitchMax) - Math.Log(c.PitchMin)) * i / Math.Max(1, _pitchBins.Length - 1));

        if (c.Latent != h) PromptProjection = Ns2Ops.Linear(engine, promptLayers, random, c.Latent, h, Ns2Ops.Normal(random, h * c.Latent, 0.02));
        PromptEncoder = new Ns2TransformerEncoder<T>(engine, promptLayers, random, 0, h, c.Heads, c.EncoderFilter, c.EncoderKernel, c.PromptLayers,
            c.EncoderDropout, false, c.Positions);
        QueryEmbedding = IndexTtsOps.Embedding(promptLayers, random, c.QueryTokens, h, 1.0);
        QueryAttention = new Ns2Attention<T>(engine, promptLayers, random, h, c.Heads);

        Denoiser = new Ns2WaveNet<T>(engine, denoiserLayers, random, c.Latent, h, c.Latent, c.DenoiserLayers,
            i => c.DenoiserDilationCycle is int cycle ? 1 << (i % cycle) : c.DenoiserDilation,
            i => (i + c.DenoiserAttentionEvery - c.DenoiserAttentionOffset) % c.DenoiserAttentionEvery == 0, c.Heads, c.DenoiserAttentionDropout);
    }

    public Ns2TransformerEncoder<T> PhonemeEncoder { get; }
    public Ns2VariancePredictor<T> DurationPredictor { get; }
    public Ns2VariancePredictor<T> PitchPredictor { get; }
    public TiedEmbeddingLayer<T> PitchEmbedding { get; }
    public DenseLayer<T>? PromptProjection { get; }
    public Ns2TransformerEncoder<T> PromptEncoder { get; }
    public TiedEmbeddingLayer<T> QueryEmbedding { get; }
    public Ns2Attention<T> QueryAttention { get; }
    public Ns2WaveNet<T> Denoiser { get; }

    /// <summary>The pitch bucket boundaries (Hz).</summary>
    public IReadOnlyList<double> PitchBins => _pitchBins;

    /// <summary><c>torch.bucketize(pitch, bins)</c>: the number of boundaries strictly below the value.</summary>
    public int PitchToken(double pitch)
    {
        int lo = 0, hi = _pitchBins.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (_pitchBins[mid] < pitch) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    /// <summary>
    /// The speech prompt (§3.3): the prompt's latent <c>[1, latent, T′]</c> projected to the hidden width and encoded to rows
    /// <c>[T′, d]</c>, and the m query tokens' attention over them <c>[m, d]</c>.
    /// </summary>
    public (Tensor<T> Prompt, Tensor<T> Query) Prompt(Tensor<T> promptLatent, bool training, Random? random)
    {
        var rows = Ns2Ops.Rows(_engine, promptLatent);
        if (PromptProjection is not null) rows = PromptProjection.Forward(rows);
        var prompt = PromptEncoder.Forward(rows, null, training, random);
        var ids = Enumerable.Range(0, _c.QueryTokens).ToArray();
        var query = QueryAttention.Forward(QueryEmbedding.Forward(IndexTtsOps.Ids<T>(ids)), prompt);
        return (prompt, query);
    }

    /// <summary>The phoneme encoder's rows <c>[n, d]</c> (style-adaptive on the prompt when so configured) and the
    /// predicted log-durations <c>[n]</c> (of 1 + d, Amphion's target).</summary>
    public (Tensor<T> Hidden, Tensor<T> LogDurations) Phonemes(IReadOnlyList<int> phonemes, Tensor<T> prompt, bool training, Random? random)
    {
        var hidden = PhonemeEncoder.Forward(phonemes, _c.StyleAdaptiveEncoder ? prompt : null, training, random);
        return (hidden, DurationPredictor.Forward(hidden, prompt, training, random));
    }

    /// <summary>dur_pred_round: clamp(round(exp(x) − 1), min 0), rounding half to even as torch.round does.</summary>
    public static int[] RoundDurations(Tensor<T> logDurations)
    {
        var durations = new int[logDurations.Length];
        for (int i = 0; i < durations.Length; i++)
            durations[i] = Math.Max(0, (int)Math.Round(Math.Exp(NumOps.ToDouble(logDurations.GetFlat(i))) - 1, MidpointRounding.ToEven));
        return durations;
    }

    /// <summary>
    /// The prior's condition (§3.2): the hidden rows expanded by <paramref name="durations"/> (the length regulator), the
    /// predicted log-pitch <c>[T]</c>, and c = expanded + the embedding of the ground-truth pitch's bucket, or of the
    /// predicted pitch's when <paramref name="pitch"/> is null.
    /// </summary>
    public (Tensor<T> Condition, Tensor<T> LogPitch) Condition(Tensor<T> hidden, IReadOnlyList<int> durations, IReadOnlyList<double>? pitch,
        Tensor<T> prompt, bool training, Random? random)
    {
        var expanded = Expand(hidden, durations);
        var logPitch = PitchPredictor.Forward(expanded, prompt, training, random);
        var tokens = new int[expanded.Shape[0]];
        for (int f = 0; f < tokens.Length; f++)
            tokens[f] = PitchToken(pitch is not null ? pitch[f] : Math.Exp(NumOps.ToDouble(logPitch[f])));
        return (_engine.TensorAdd(expanded, PitchEmbedding.Forward(IndexTtsOps.Ids<T>(tokens))), logPitch);
    }

    /// <summary>Rows <c>[n, d]</c> repeated by integer durations to <c>[Σd, d]</c> (the length regulator).</summary>
    public Tensor<T> Expand(Tensor<T> rows, IReadOnlyList<int> durations)
    {
        var index = new List<int>();
        for (int j = 0; j < durations.Count; j++)
            for (int r = 0; r < durations[j]; r++) index.Add(j);
        if (index.Count == 0) throw new ArgumentException("The durations expand to no frames.", nameof(durations));
        return _engine.TensorIndexSelect(rows, new Tensor<int>(new[] { index.Count }, new Vector<int>(index.ToArray())), 0);
    }

    internal void SetPitchBins(double[] bins)
    {
        if (bins.Length != _pitchBins.Length) throw new ArgumentException($"Expected {_pitchBins.Length} pitch bins, got {bins.Length}.");
        _pitchBins = (double[])bins.Clone();
    }

    /// <summary>Loads Amphion's <c>NaturalSpeech2</c> state dict (all but the codec's quantizer).</summary>
    internal void Load(TorchTensorSource source)
    {
        PhonemeEncoder.Load(source, "prior_encoder.encoder.");
        DurationPredictor.Load(source, "prior_encoder.duration_predictor.");
        PitchPredictor.Load(source, "prior_encoder.pitch_predictor.");
        PitchEmbedding.LoadLeadingRows(source, "prior_encoder.pitch_embedding.weight");
        if (source.TryRead("prior_encoder.pitch_bins") is { } bins) SetPitchBins(bins);
        if (PromptProjection is not null)
            TorchParameters.Linear(_engine, PromptProjection, _c.Latent, _c.Hidden, source, "prompt_lin.weight", "prompt_lin.bias");
        PromptEncoder.Load(source, "prompt_encoder.");
        QueryEmbedding.LoadLeadingRows(source, "query_emb.weight");
        QueryAttention.Load(source, "query_attn.");
        Denoiser.Load(source, "diffusion.diff_estimator.");
    }

    // ---------------------------------------------------------------- the diffusion (Eqs. 2-5)

    /// <summary>∫₀ᵗ β_s ds for the linear schedule β_t = β_min + (β_max − β_min) t.</summary>
    public static double CumulativeBeta(double t, double betaMin, double betaMax) => betaMin * t + 0.5 * (betaMax - betaMin) * t * t;

    public static double Beta(double t, double betaMin, double betaMax) => betaMin + (betaMax - betaMin) * t;
}
