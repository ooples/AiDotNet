using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Fun-CosyVoice3's <c>CausalMaskedDiffWithDiT</c>: S3-v3 tokens are embedded, passed through the causal
/// pre-lookahead layer and repeated to two mel frames; a 22-block causal DiT predicts the conditional flow field.
/// </summary>
internal sealed class CosyVoice3Flow<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public CosyVoice3Flow(IEngine engine, List<LayerBase<T>> layers, CosyVoice3Options options, Random random)
    {
        _engine = engine;
        Options = options;
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        double Uniform(int fanIn) => (2 * random.NextDouble() - 1) / Math.Sqrt(fanIn);

        InputEmbedding = Add(new TiedEmbeddingLayer<T>(options.CodebookSize, options.DiTInputDim));
        InputEmbedding.Reinitialize(() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) *
            Math.Cos(2.0 * Math.PI * random.NextDouble()));
        SpeakerProjection = Add(new DenseLayer<T>(options.DiTInputDim, (IActivationFunction<T>)new IdentityActivation<T>()));
        TorchParameters.Linear(engine, SpeakerProjection, options.SpeakerEmbeddingDim, options.DiTInputDim,
            Enumerable.Range(0, options.SpeakerEmbeddingDim * options.DiTInputDim).Select(_ => Uniform(options.SpeakerEmbeddingDim)).ToArray(),
            Enumerable.Range(0, options.DiTInputDim).Select(_ => Uniform(options.SpeakerEmbeddingDim)).ToArray());

        LookaheadConv1 = MeloOps.Conv(layers, options.DiTInputDim, options.PreLookaheadChannels,
            options.Lookahead + 1, 1, 0, 1);
        LookaheadConv2 = MeloOps.Conv(layers, options.PreLookaheadChannels, options.DiTInputDim, 3, 1, 0, 1);
        Estimator = new GptSoVitsDiT<T>(engine, layers, new GptSoVitsDiTConfiguration(
            Dim: options.DiTDim, Depth: options.DiTDepth, Heads: options.DiTHeads, HeadDim: options.DiTHeadDim,
            FfMult: options.DiTFeedForwardMultiplier, MelDim: options.MelChannels, TextDim: options.DiTInputDim,
            ConvLayers: 0, Dropout: options.DropoutRate, UseStepEmbedding: false, SpeakerDim: options.DiTInputDim,
            CausalPosition: true, StaticChunkSize: options.StaticChunkSize * options.TokenMelRatio), random);
    }

    public CosyVoice3Options Options { get; }
    public TiedEmbeddingLayer<T> InputEmbedding { get; }
    public DenseLayer<T> SpeakerProjection { get; }
    public NormedConv1DLayer<T> LookaheadConv1 { get; }
    public NormedConv1DLayer<T> LookaheadConv2 { get; }
    public GptSoVitsDiT<T> Estimator { get; }

    public int MelFrames(int tokens) => tokens * Options.TokenMelRatio;

    private Tensor<T> Pad(Tensor<T> x, int left, int right)
    {
        var parts = new List<Tensor<T>>(3);
        if (left > 0) parts.Add(new Tensor<T>(new[] { x.Shape[0], x.Shape[1], left }));
        parts.Add(x);
        if (right > 0) parts.Add(new Tensor<T>(new[] { x.Shape[0], x.Shape[1], right }));
        return parts.Count == 1 ? x : _engine.TensorConcatenate(parts.ToArray(), 2);
    }

    private Tensor<T> Channels(Tensor<T> rows) =>
        _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, rows.Shape[1], rows.Shape[0] });

    private Tensor<T> Rows(Tensor<T> channels) =>
        _engine.TensorTranspose(_engine.Reshape(channels, new[] { channels.Shape[1], channels.Shape[2] }));

    /// <summary>The pre-lookahead residual followed by nearest repetition to two mel frames per token.</summary>
    internal Tensor<T> Encode(IReadOnlyList<int> tokens)
    {
        var ids = new Tensor<T>(new[] { tokens.Count });
        for (int i = 0; i < tokens.Count; i++) ids[i] = NumOps.FromDouble(tokens[i]);
        var embedded = InputEmbedding.Forward(ids);
        var h = LookaheadConv1.Forward(Pad(Channels(embedded), 0, Options.Lookahead));
        h = _engine.LeakyReLU(h, NumOps.FromDouble(0.01));
        h = LookaheadConv2.Forward(Pad(h, 2, 0));
        var looked = _engine.TensorAdd(Rows(h), embedded);
        var index = new Tensor<int>(new[] { MelFrames(tokens.Count) });
        for (int i = 0; i < index.Length; i++) index[i] = i / Options.TokenMelRatio;
        return _engine.TensorIndexSelect(looked, index, 0);
    }

    internal Tensor<T> ProjectSpeaker(Tensor<T> xvector)
    {
        if (xvector.Length != Options.SpeakerEmbeddingDim)
            throw new ArgumentException($"The speaker embedding needs {Options.SpeakerEmbeddingDim} values.", nameof(xvector));
        double norm = 0;
        for (int i = 0; i < xvector.Length; i++) norm += Math.Pow(NumOps.ToDouble(xvector[i]), 2);
        norm = Math.Max(Math.Sqrt(norm), 1e-12);
        var normalized = _engine.TensorMultiplyScalar(_engine.Reshape(xvector, new[] { 1, xvector.Length }), NumOps.FromDouble(1.0 / norm));
        return _engine.Reshape(SpeakerProjection.Forward(normalized), new[] { Options.DiTInputDim });
    }

    private Tensor<T> Field(Tensor<T> x, Tensor<T> mu, Tensor<T> speaker, Tensor<T> condition, double time,
        bool training, Random? random, bool streaming) =>
        Estimator.Forward(x, condition, Estimator.TextEmbedding(mu, drop: false), Estimator.TimeEmbedding(time, null),
            dropAudioCond: false, training, random, speaker, streaming);

    /// <summary>Ten-step cosine Euler sampling with classifier-free guidance.</summary>
    internal Tensor<T> Sample(Tensor<T> mu, Tensor<T> speaker, Tensor<T> condition, Tensor<T> noise, bool streaming = false)
    {
        int steps = Options.DiTTimesteps;
        var span = Enumerable.Range(0, steps + 1).Select(i => 1 - Math.Cos(i / (double)steps * 0.5 * Math.PI)).ToArray();
        var x = noise;
        var zeroMu = _engine.TensorMultiplyScalar(mu, NumOps.Zero);
        var zeroSpeaker = _engine.TensorMultiplyScalar(speaker, NumOps.Zero);
        var zeroCondition = _engine.TensorMultiplyScalar(condition, NumOps.Zero);
        double time = span[0];
        for (int step = 0; step < steps; step++)
        {
            double dt = span[step + 1] - time;
            var conditional = Field(x, mu, speaker, condition, time, false, null, streaming);
            var unconditional = Field(x, zeroMu, zeroSpeaker, zeroCondition, time, false, null, streaming);
            var guided = _engine.TensorSubtract(
                _engine.TensorMultiplyScalar(conditional, NumOps.FromDouble(1 + Options.DiTInferenceCfgRate)),
                _engine.TensorMultiplyScalar(unconditional, NumOps.FromDouble(Options.DiTInferenceCfgRate)));
            x = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(guided, NumOps.FromDouble(dt)));
            time = span[step + 1];
        }
        return x;
    }

    public Tensor<T> Inference(IReadOnlyList<int> tokens, IReadOnlyList<int> promptTokens, Tensor<T>? promptMel,
        Tensor<T> xvector, Tensor<T> noise, bool streaming = false)
    {
        if (tokens.Count == 0) throw new ArgumentException("There are no speech tokens to decode.", nameof(tokens));
        int promptFrames = promptTokens.Count == 0 ? 0 : promptMel?.Shape[0]
            ?? throw new ArgumentException("Prompt tokens need the prompt mel.", nameof(promptMel));
        int textFrames = MelFrames(tokens.Count), total = promptFrames + textFrames;
        if (promptFrames != MelFrames(promptTokens.Count))
            throw new ArgumentException($"The prompt has {promptFrames} mel frames but {promptTokens.Count} tokens require {MelFrames(promptTokens.Count)}.", nameof(promptMel));
        if (noise.Length != total * Options.MelChannels)
            throw new ArgumentException($"The noise needs {total * Options.MelChannels} values.", nameof(noise));
        var mu = Encode(promptTokens.Concat(tokens).ToList());
        var condition = new Tensor<T>(new[] { total, Options.MelChannels });
        if (promptMel is not null)
            for (int f = 0; f < promptFrames; f++)
                for (int m = 0; m < Options.MelChannels; m++) condition[f, m] = promptMel[f, m];
        var noiseRows = _engine.TensorTranspose(_engine.Reshape(noise, new[] { Options.MelChannels, total }));
        var sampled = Sample(mu, ProjectSpeaker(xvector), condition, noiseRows, streaming);
        var generated = _engine.TensorSlice(sampled, new[] { promptFrames, 0 }, new[] { textFrames, Options.MelChannels });
        return _engine.Reshape(_engine.TensorTranspose(generated), new[] { 1, Options.MelChannels, textFrames });
    }

    public Tensor<T> TrainingLoss(IReadOnlyList<int> tokens, Tensor<T> mel, Tensor<T> xvector, bool training, Random random)
    {
        int frames = mel.Shape[2];
        if (frames != MelFrames(tokens.Count))
            throw new ArgumentException($"{tokens.Count} speech tokens require {MelFrames(tokens.Count)} mel frames, not {frames}.", nameof(mel));
        bool streaming = random.NextDouble() < 0.5;
        var target = _engine.TensorTranspose(_engine.Reshape(mel, new[] { Options.MelChannels, frames }));
        var mu = Encode(tokens);
        var speaker = ProjectSpeaker(xvector);
        int prompt = random.NextDouble() < 0.5 ? 0 : random.Next((int)(0.3 * frames) + 1);
        var condition = new Tensor<T>(new[] { frames, Options.MelChannels });
        for (int f = 0; f < prompt; f++)
            for (int m = 0; m < Options.MelChannels; m++) condition[f, m] = target[f, m];
        var noise = new Tensor<T>(new[] { frames, Options.MelChannels });
        for (int i = 0; i < noise.Length; i++)
            noise.SetFlat(i, NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble())));
        double time = random.NextDouble(), sigma = Options.DiTSigmaMin;
        var y = _engine.TensorAdd(_engine.TensorMultiplyScalar(noise, NumOps.FromDouble(1 - (1 - sigma) * time)),
            _engine.TensorMultiplyScalar(target, NumOps.FromDouble(time)));
        var velocity = _engine.TensorSubtract(target, _engine.TensorMultiplyScalar(noise, NumOps.FromDouble(1 - sigma)));
        if (random.NextDouble() < Options.DiTTrainingCfgRate)
        {
            mu = _engine.TensorMultiplyScalar(mu, NumOps.Zero);
            speaker = _engine.TensorMultiplyScalar(speaker, NumOps.Zero);
            condition = _engine.TensorMultiplyScalar(condition, NumOps.Zero);
        }
        var difference = _engine.TensorSubtract(Field(y, mu, speaker, condition, time, training, random, streaming), velocity);
        return _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(difference, difference), new[] { 0, 1 }, false),
            NumOps.FromDouble(1.0 / difference.Length));
    }

    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        int input = Options.DiTInputDim, look = Options.PreLookaheadChannels;
        InputEmbedding.LoadTable(read("input_embedding.weight", new[] { Options.CodebookSize, input }));
        TorchParameters.Linear(_engine, SpeakerProjection, Options.SpeakerEmbeddingDim, input,
            read("spk_embed_affine_layer.weight", new[] { input, Options.SpeakerEmbeddingDim }),
            read("spk_embed_affine_layer.bias", new[] { input }));
        LookaheadConv1.LoadTorchWeights(read("pre_lookahead_layer.conv1.weight", new[] { look, input, Options.Lookahead + 1 }), null,
            read("pre_lookahead_layer.conv1.bias", new[] { look }));
        LookaheadConv2.LoadTorchWeights(read("pre_lookahead_layer.conv2.weight", new[] { input, look, 3 }), null,
            read("pre_lookahead_layer.conv2.bias", new[] { input }));
        Estimator.Load(name => read(name, Array.Empty<int>()), "decoder.estimator.");
    }
}
