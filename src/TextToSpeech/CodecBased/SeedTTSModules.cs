using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Codecs;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Trainable convolutional speech tokenizer using the report's cited DAC family.</summary>
internal sealed class SeedTTSSpeechTokenizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly SeedTTSOptions _options;
    private readonly DacEncoder<T> _analysis;
    internal FactorizedVectorQuantizerLayer<T> Quantizer { get; }
    internal IReadOnlyList<LayerBase<T>> Layers { get; }

    public SeedTTSSpeechTokenizer(IEngine engine, List<LayerBase<T>> layers, SeedTTSOptions options)
    {
        _engine = engine;
        _options = options;
        int start = layers.Count;
        _analysis = new DacEncoder<T>(engine, options.TokenizerChannels, options.CodecRates, options.AcousticDimension, layers);
        Quantizer = Own(layers, new FactorizedVectorQuantizerLayer<T>(options.AcousticDimension, 1,
            options.CodebookSize, options.TokenizerCodeDimension, lossesOnRawVectors: true));
        Layers = layers.Skip(start).ToList();
    }

    private Tensor<T> AnalyzeChannels(Tensor<T> audio)
    {
        int hop = _options.SamplesPerCodecFrame;
        int length = (audio.Length + hop - 1) / hop * hop;
        var input = _engine.Reshape(audio, new[] { audio.Length });
        if (length > audio.Length) input = _engine.TensorConcatenate(new[] { input, new Tensor<T>(new[] { length - audio.Length }) }, 0);
        return _analysis.Forward(_engine.Reshape(input, new[] { 1, 1, length }));
    }

    private Tensor<T> Rows(Tensor<T> x) => _engine.TensorTranspose(_engine.Reshape(x, new[] { x.Shape[1], x.Shape[2] }));
    public Tensor<T> Analyze(Tensor<T> audio) => Rows(AnalyzeChannels(audio));
    internal Tensor<T> QuantizeForTraining(Tensor<T> audio) => Rows(Quantizer.Forward(AnalyzeChannels(audio)));

    public Tensor<T> Encode(Tensor<T> audio)
    {
        var features = AnalyzeChannels(audio);
        var codes = Quantizer.Encode(features);
        int frames = codes.GetLength(1);
        var tokens = new Tensor<T>(new[] { frames });
        for (int frame = 0; frame < frames; frame++)
        {
            tokens[frame] = NumOps.FromDouble(codes[0, frame]);
        }
        return tokens;
    }

    public Tensor<T> Embed(Tensor<T> tokens)
    {
        var codes = new int[1, tokens.Length];
        for (int i = 0; i < tokens.Length; i++) codes[0, i] = (int)Math.Round(NumOps.ToDouble(tokens[i]));
        return Rows(Quantizer.Decode(codes));
    }

    public Tensor<T> Frame(Tensor<T> audio)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        if (audio.Length == 0) throw new ArgumentException("Audio must contain at least one sample.", nameof(audio));
        int samples = _options.SamplesPerCodecFrame;
        int frames = Math.Max(1, (audio.Length + samples - 1) / samples);
        var framed = new Tensor<T>(new[] { frames, samples });
        for (int frame = 0; frame < frames; frame++)
            for (int sample = 0; sample < samples; sample++)
            {
                int index = frame * samples + sample;
                framed[frame, sample] = index < audio.Length ? audio[index] : NumOps.Zero;
            }
        return framed;
    }

    private static TLayer Own<TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }
}

/// <summary>The causal speech-token LM. Text and prompt positions are context only; loss is applied to speech tokens.</summary>
internal sealed class SeedTTSTokenLanguageModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly SeedTTSOptions _options;
    private readonly LlamaNetwork<T> _transformer;
    private readonly int _speechBos;
    private readonly int _speechEos;

    public SeedTTSTokenLanguageModel(IEngine engine, List<LayerBase<T>> layers, SeedTTSOptions options)
    {
        _engine = engine;
        _options = options;
        _speechBos = options.CodebookSize;
        _speechEos = options.CodebookSize + 1;
        TextEmbedding = Own(layers, new TiedEmbeddingLayer<T>(options.VocabSize, options.LLMDim));
        SpeechEmbedding = Own(layers, new TiedEmbeddingLayer<T>(options.CodebookSize + 2, options.LLMDim));
        SpeechHead = Own(layers, new BiasFreeLinearLayer<T>(options.LLMDim, options.CodebookSize + 2));
        _transformer = new LlamaNetwork<T>(engine, layers, new LlamaConfiguration
        {
            VocabSize = 0,
            Hidden = options.LLMDim,
            Layers = options.NumLLMLayers,
            Heads = options.NumHeads,
            KeyValueHeads = options.NumHeads,
            HeadDim = options.LLMDim / options.NumHeads,
            Intermediate = options.LLMDim * 4,
            IncludeHead = false,
        });
        _transformer.InitializeLikeReference(new Random(options.GenerationSeed));
    }

    public TiedEmbeddingLayer<T> TextEmbedding { get; }
    public TiedEmbeddingLayer<T> SpeechEmbedding { get; }
    public BiasFreeLinearLayer<T> SpeechHead { get; }

    public Tensor<T> Loss(IReadOnlyList<int> text, IReadOnlyList<int> speech, IReadOnlyList<int>? promptSpeech = null)
    {
        if (speech.Count == 0) throw new ArgumentException("At least one target speech token is required.", nameof(speech));
        var inputs = new int[speech.Count + 1];
        inputs[0] = _speechBos;
        for (int i = 0; i < speech.Count; i++) inputs[i + 1] = CheckedSpeech(speech[i]);
        var targets = new int[speech.Count + 1];
        for (int i = 0; i < speech.Count; i++) targets[i] = CheckedSpeech(speech[i]);
        targets[^1] = _speechEos;

        var prepared = Prepare(text, promptSpeech, inputs);
        var hidden = _transformer.Forward(prepared.Embeddings).Hidden;
        var speechHidden = _engine.TensorSlice(hidden, new[] { 0, prepared.SpeechStart, 0 },
            new[] { 1, inputs.Length, _options.LLMDim });
        return CrossEntropy(SpeechHead.Forward(speechHidden), targets);
    }

    public int[] Generate(IReadOnlyList<int> text, IReadOnlyList<int>? promptSpeech, Random random)
    {
        var prepared = Prepare(text, promptSpeech, new[] { _speechBos });
        var cache = new LlamaCache<T>();
        var output = _transformer.Forward(prepared.Embeddings, null, null, cache, lastLogitsOnly: true).Hidden;
        var generated = new List<int>();
        int budget = Math.Min(_options.MaxCodecFrames,
            Math.Max(_options.MinimumGeneratedCodecFrames, Math.Max(1, text.Count) * _options.FramesPerTextToken));

        for (int step = 0; step < budget; step++)
        {
            var last = _engine.TensorSlice(output, new[] { 0, output.Shape[1] - 1, 0 },
                new[] { 1, 1, _options.LLMDim });
            int next = Sample(SpeechHead.Forward(last), random);
            if (next == _speechEos && step >= _options.MinimumGeneratedCodecFrames)
                break;
            if (next >= _options.CodebookSize)
                next = ArgMaxCodec(SpeechHead.Forward(last));
            generated.Add(next);
            var nextEmbedding = SpeechEmbedding.Forward(IdTensor(new[] { next }));
            output = _transformer.Forward(nextEmbedding, null, null, cache, lastLogitsOnly: true).Hidden;
        }
        return generated.Count == 0 ? new[] { 0 } : generated.ToArray();
    }

    public Tensor<T> TextConditionLogits(IReadOnlyList<int> text)
    {
        var prepared = Prepare(text, null, new[] { _speechBos });
        var hidden = _transformer.Forward(prepared.Embeddings).Hidden;
        return SpeechHead.Forward(hidden);
    }

    private (Tensor<T> Embeddings, int SpeechStart) Prepare(
        IReadOnlyList<int> text,
        IReadOnlyList<int>? promptSpeech,
        IReadOnlyList<int> speechInput)
    {
        int[] textIds = text.Count == 0 ? new[] { 0 } : text.Take(_options.MaxTextLength)
            .Select(id => MathPolyfill.Clamp(id, 0, _options.VocabSize - 1)).ToArray();
        int[] prompt = promptSpeech is null ? Array.Empty<int>() : promptSpeech
            .Skip(Math.Max(0, promptSpeech.Count - _options.MaxPromptCodecFrames)).Select(CheckedSpeech).ToArray();
        var parts = new List<Tensor<T>>();
        if (prompt.Length > 0) parts.Add(SpeechEmbedding.Forward(IdTensor(prompt)));
        parts.Add(TextEmbedding.Forward(IdTensor(textIds)));
        int speechStart = prompt.Length + textIds.Length;
        parts.Add(SpeechEmbedding.Forward(IdTensor(speechInput.Select(CheckedSpecialSpeech).ToArray())));
        return (_engine.TensorConcatenate(parts.ToArray(), axis: 1), speechStart);
    }

    private Tensor<T> CrossEntropy(Tensor<T> logits, IReadOnlyList<int> labels)
    {
        int length = logits.Shape[1], vocabulary = logits.Shape[2];
        var picks = new int[length];
        for (int i = 0; i < length; i++) picks[i] = i * vocabulary + labels[i];
        var logProbabilities = _engine.Reshape(_engine.TensorLogSoftmax(logits, axis: 2),
            new[] { length * vocabulary, 1 });
        var index = new Tensor<int>(new[] { picks.Length });
        for (int i = 0; i < picks.Length; i++) index[i] = picks[i];
        var selected = _engine.TensorIndexSelect(logProbabilities, index, 0);
        return _engine.TensorMultiplyScalar(_engine.ReduceSum(selected, null), NumOps.FromDouble(-1.0 / length));
    }

    private int Sample(Tensor<T> logits, Random random)
    {
        if (_options.Temperature <= 0) throw new InvalidOperationException("Temperature must be positive.");
        int vocabulary = logits.Shape[2];
        int take = _options.TopK <= 0 ? vocabulary : Math.Min(_options.TopK, vocabulary);
        var ranked = Enumerable.Range(0, vocabulary)
            .Select(token => (Token: token, Value: NumOps.ToDouble(logits[0, 0, token]) / _options.Temperature))
            .OrderByDescending(pair => pair.Value).Take(take).ToArray();
        double max = ranked[0].Value;
        double total = ranked.Sum(pair => Math.Exp(pair.Value - max));
        double draw = random.NextDouble() * total;
        foreach (var pair in ranked)
        {
            draw -= Math.Exp(pair.Value - max);
            if (draw <= 0) return pair.Token;
        }
        return ranked[^1].Token;
    }

    private int ArgMaxCodec(Tensor<T> logits)
    {
        int best = 0;
        double value = double.NegativeInfinity;
        for (int token = 0; token < _options.CodebookSize; token++)
        {
            double candidate = NumOps.ToDouble(logits[0, 0, token]);
            if (candidate > value) { value = candidate; best = token; }
        }
        return best;
    }

    private Tensor<T> IdTensor(IReadOnlyList<int> ids)
    {
        var tensor = new Tensor<T>(new[] { 1, ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[0, i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    private int CheckedSpeech(int id) => MathPolyfill.Clamp(id, 0, _options.CodebookSize - 1);
    private int CheckedSpecialSpeech(int id) => MathPolyfill.Clamp(id, 0, _options.CodebookSize + 1);

    private static TLayer Own<TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }
}

/// <summary>The report's token-conditioned diffusion transformer, including explicit timbre-reference conditioning.</summary>
internal sealed class SeedTTSTokenDiffusion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly SeedTTSOptions _options;
    private readonly TiedEmbeddingLayer<T> _tokens;
    private readonly FullyConnectedLayer<T> _noisyInput;
    private readonly FullyConnectedLayer<T> _timbre;
    private readonly List<SeedTTSDiffusionBlock<T>> _blocks = new();
    private readonly FullyConnectedLayer<T> _output;
    private readonly PositionalEncodingLayer<T> _positions;
    private readonly double[] _alphaBar;
    private readonly Random _trainingRandom;

    public SeedTTSTokenDiffusion(IEngine engine, List<LayerBase<T>> layers, SeedTTSOptions options)
    {
        _engine = engine;
        _options = options;
        if (options.DiffusionTrainingSteps < 2 || options.DiffusionSteps < 1 ||
            options.DiffusionSteps > options.DiffusionTrainingSteps ||
            !(options.DiffusionBetaStart > 0 && options.DiffusionBetaStart <= options.DiffusionBetaEnd && options.DiffusionBetaEnd < 1))
            throw new ArgumentException("Native Seed diffusion requires 0 < betaStart <= betaEnd < 1 and 1 <= sampling steps <= training steps.", nameof(options));
        if (options.AcousticDimension <= 0 || options.NumHeads <= 0 || options.AcousticDimension % options.NumHeads != 0)
            throw new ArgumentException("Seed diffusion width must be positive and divisible by its head count.", nameof(options));
        _alphaBar = new double[options.DiffusionTrainingSteps + 1];
        _alphaBar[0] = 1;
        for (int i = 1; i < _alphaBar.Length; i++)
        {
            double beta = options.DiffusionBetaStart + (options.DiffusionBetaEnd - options.DiffusionBetaStart) *
                (i - 1) / (options.DiffusionTrainingSteps - 1);
            _alphaBar[i] = _alphaBar[i - 1] * (1 - beta);
        }
        _trainingRandom = new Random(options.GenerationSeed + 17);
        _tokens = Own(layers, new TiedEmbeddingLayer<T>(options.CodebookSize, options.AcousticDimension));
        _noisyInput = Own(layers, new FullyConnectedLayer<T>(options.AcousticDimension,
            options.AcousticDimension, new IdentityActivation<T>()));
        _timbre = Own(layers, new FullyConnectedLayer<T>(options.SpeakerEmbeddingDim,
            options.AcousticDimension, new IdentityActivation<T>()));
        _positions = Own(layers, new PositionalEncodingLayer<T>(Math.Max(1, options.MaxCodecFrames), options.AcousticDimension));
        for (int i = 0; i < options.DiffusionLayers; i++)
            _blocks.Add(new SeedTTSDiffusionBlock<T>(engine, layers, options.AcousticDimension, options.NumHeads));
        _output = Own(layers, new FullyConnectedLayer<T>(options.AcousticDimension,
            options.AcousticDimension, new IdentityActivation<T>()));
    }

    internal double TrainingAlphaBar(int timestep)
    {
        if (timestep < 0 || timestep >= _alphaBar.Length) throw new ArgumentOutOfRangeException(nameof(timestep));
        return _alphaBar[timestep];
    }

    // Standard VP/DDPM corruption with a clean-feature prediction objective. The Seed report
    // does not specify its schedule: these exposed native choices are not ByteDance settings.
    internal Tensor<T> Corrupt(Tensor<T> clean, Tensor<T> noise, int timestep)
    {
        if (!clean.Shape.ToArray().SequenceEqual(noise.Shape.ToArray()))
            throw new ArgumentException("Diffusion noise must match the clean feature shape.", nameof(noise));
        double alpha = TrainingAlphaBar(timestep);
        return _engine.TensorAdd(_engine.TensorMultiplyScalar(clean, NumOps.FromDouble(Math.Sqrt(alpha))),
            _engine.TensorMultiplyScalar(noise, NumOps.FromDouble(Math.Sqrt(1 - alpha))));
    }

    internal Tensor<T> DdimStep(Tensor<T> state, Tensor<T> clean, int timestep, int nextTimestep)
    {
        if (timestep < 1 || nextTimestep < 0 || nextTimestep >= timestep)
            throw new ArgumentOutOfRangeException(nameof(nextTimestep));
        double alpha = TrainingAlphaBar(timestep), nextAlpha = TrainingAlphaBar(nextTimestep);
        var epsilon = _engine.TensorMultiplyScalar(_engine.TensorSubtract(state,
            _engine.TensorMultiplyScalar(clean, NumOps.FromDouble(Math.Sqrt(alpha)))),
            NumOps.FromDouble(1 / Math.Sqrt(1 - alpha)));
        return _engine.TensorAdd(_engine.TensorMultiplyScalar(clean, NumOps.FromDouble(Math.Sqrt(nextAlpha))),
            _engine.TensorMultiplyScalar(epsilon, NumOps.FromDouble(Math.Sqrt(1 - nextAlpha))));
    }

    public Tensor<T> Refine(Tensor<T> tokens, Tensor<T>? timbre)
    {
        var random = new Random(_options.GenerationSeed + TokenHash(tokens));
        var noise = new Tensor<T>(new[] { tokens.Length, _options.AcousticDimension });
        for (int i = 0; i < noise.Length; i++) noise[i] = NumOps.FromDouble(StandardNormal(random));
        return Refine(tokens, timbre, noise);
    }

    internal Tensor<T> Refine(Tensor<T> tokens, Tensor<T>? timbre, Tensor<T> initialNoise)
    {
        if (tokens.Length == 0 || initialNoise.Rank != 2 || initialNoise.Shape[0] != tokens.Length ||
            initialNoise.Shape[1] != _options.AcousticDimension)
            throw new ArgumentException("Initial diffusion noise must have shape [tokens, acoustic width].", nameof(initialNoise));
        var state = initialNoise;
        int total = _alphaBar.Length - 1;
        for (int step = _options.DiffusionSteps; step >= 1; step--)
        {
            int timestep = (int)Math.Round(step * (double)total / (double)_options.DiffusionSteps);
            int nextTimestep = (int)Math.Round((step - 1.0) * total / (double)_options.DiffusionSteps);
            var clean = Predict(state, tokens, timbre, timestep / (double)total);
            state = DdimStep(state, clean, timestep, nextTimestep);
        }
        return state;
    }

    public Tensor<T> DenoisingLoss(Tensor<T> tokens, Tensor<T> target, Tensor<T>? timbre, Random? random = null)
    {
        random ??= _trainingRandom;
        int timestep = random.Next(1, _alphaBar.Length);
        var noise = new Tensor<T>(target.Shape.ToArray());
        for (int i = 0; i < noise.Length; i++) noise[i] = NumOps.FromDouble(StandardNormal(random));
        return DenoisingLossAtTimestep(tokens, target, timbre, timestep, noise);
    }

    internal Tensor<T> DenoisingLossAtTimestep(Tensor<T> tokens, Tensor<T> target, Tensor<T>? timbre,
        int timestep, Tensor<T> noise)
    {
        if (timestep < 1) throw new ArgumentOutOfRangeException(nameof(timestep));
        var prediction = Predict(Corrupt(target, noise, timestep), tokens, timbre, timestep / (double)(_alphaBar.Length - 1));
        var difference = _engine.TensorSubtract(prediction, target);
        return _engine.ReduceMean(_engine.TensorMultiply(difference, difference), new[] { 0, 1 }, keepDims: false);
    }

    private Tensor<T> Predict(Tensor<T> noisy, Tensor<T> tokens, Tensor<T>? timbre, double time)
    {
        int frames = noisy.Shape[0], width = _options.AcousticDimension;
        if (noisy.Rank != 2 || noisy.Shape[1] != width || tokens.Length != frames)
            throw new ArgumentException("Seed diffusion requires one token per continuous feature frame.", nameof(tokens));
        var ids = _engine.Reshape(tokens, new[] { 1, tokens.Length });
        var condition = _tokens.Forward(ids);
        var hidden = _noisyInput.Forward(_engine.Reshape(noisy, new[] { 1, frames, width }));
        hidden = _engine.TensorAdd(hidden, condition);
        hidden = _engine.TensorAdd(hidden, TimeEmbedding(frames, time));
        if (timbre is not null)
        {
            var projected = _timbre.Forward(_engine.Reshape(timbre, new[] { 1, _options.SpeakerEmbeddingDim }));
            hidden = _engine.TensorAdd(hidden, _engine.TensorTile(
                _engine.Reshape(projected, new[] { 1, 1, width }), new[] { 1, frames, 1 }));
        }
        hidden = _positions.Forward(hidden);
        foreach (var block in _blocks) hidden = block.Forward(hidden);
        return _engine.Reshape(_output.Forward(hidden), new[] { frames, width });
    }

    private Tensor<T> TimeEmbedding(int frames, double time)
    {
        int width = _options.AcousticDimension;
        var embedding = new Tensor<T>(new[] { 1, frames, width });
        for (int frame = 0; frame < frames; frame++)
            for (int d = 0; d < width; d += 2)
            {
                double angle = time / Math.Pow(10000.0, d / (double)Math.Max(1, width));
                embedding[0, frame, d] = NumOps.FromDouble(Math.Sin(angle));
                if (d + 1 < width) embedding[0, frame, d + 1] = NumOps.FromDouble(Math.Cos(angle));
            }
        return embedding;
    }

    private int TokenHash(Tensor<T> tokens)
    {
        int hash = 17;
        for (int i = 0; i < tokens.Length; i++) hash = unchecked(hash * 31 + (int)NumOps.ToDouble(tokens[i]));
        return hash;
    }

    private static double StandardNormal(Random random)
    {
        double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static TLayer Own<TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }
}


/// <summary>Separately trained convolutional DAC-family acoustic vocoder, as cited in Seed-TTS section 2.</summary>
internal sealed class SeedTTSAcousticVocoder<T>
{
    private readonly IEngine _engine;
    private readonly SeedTTSOptions _options;
    private readonly DacDecoder<T> _waveform;
    internal IReadOnlyList<LayerBase<T>> Layers { get; }

    public SeedTTSAcousticVocoder(IEngine engine, List<LayerBase<T>> layers, SeedTTSOptions options)
    {
        _engine = engine;
        _options = options;
        int start = layers.Count;
        _waveform = new DacDecoder<T>(engine, options.AcousticDimension, options.VocoderHiddenDimension,
            options.CodecRates.AsEnumerable().Reverse().ToArray(), layers, exactStrideLengths: true);
        Layers = layers.Skip(start).ToList();
    }

    public Tensor<T> Frames(Tensor<T> acoustic)
    {
        var channels = _engine.Reshape(_engine.TensorTranspose(acoustic), new[] { 1, _options.AcousticDimension, acoustic.Shape[0] });
        var audio = _waveform.Forward(channels);
        return _engine.Reshape(audio, new[] { acoustic.Shape[0], _options.SamplesPerCodecFrame });
    }

    public Tensor<T> Decode(Tensor<T> acoustic)
    {
        var frames = Frames(acoustic);
        return _engine.Reshape(frames, new[] { frames.Length });
    }

    public Tensor<T> ReconstructionLoss(Tensor<T> acoustic, Tensor<T> targetFrames)
    {
        var difference = _engine.TensorSubtract(Frames(acoustic), targetFrames);
        return _engine.ReduceMean(_engine.TensorMultiply(difference, difference), new[] { 0, 1 }, keepDims: false);
    }

    private static TLayer Own<TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }
}
