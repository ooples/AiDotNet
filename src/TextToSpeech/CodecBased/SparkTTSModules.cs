using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Audio.Codecs;
using AiDotNet.Audio.Foundations;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The released Qwen2.5 causal LM, including Spark's exact added-token ranges.</summary>
internal sealed class SparkTokenLanguageModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly SparkTTSOptions _options;

    public SparkTokenLanguageModel(IEngine engine, List<LayerBase<T>> layers, SparkTTSOptions options)
    {
        _engine = engine;
        _options = options;
        Transformer = new LlamaNetwork<T>(engine, layers, new LlamaConfiguration
        {
            VocabSize = options.VocabSize,
            Hidden = options.LLMDim,
            Layers = options.NumLLMLayers,
            Heads = options.NumHeads,
            KeyValueHeads = options.KeyValueHeads,
            HeadDim = options.LLMDim / options.NumHeads,
            Intermediate = options.LmIntermediateDim,
            RmsEpsilon = options.RmsNormEpsilon,
            RopeTheta = options.RopeTheta,
            AttentionBias = true,
            TieWordEmbeddings = options.TieWordEmbeddings,
            IncludeHead = true,
        });
        Transformer.InitializeLikeReference(new Random(options.GenerationSeed));
    }

    public LlamaNetwork<T> Transformer { get; }

    public int[] GenerateSemantic(IReadOnlyList<int> prompt, Random random)
    {
        if (_options.ConstrainedSpeechDecoding)
            return GenerateRange(prompt, _options.SemanticTokenOffset, _options.CodebookSize,
                _options.MaxCodecFrames, _options.MinimumGeneratedCodecFrames, random, _options.EndSemanticToken);
        var tokens = GenerateAll(prompt, random).Where(id => id >= _options.SemanticTokenOffset
            && id < _options.SemanticTokenOffset + _options.CodebookSize).Select(id => id - _options.SemanticTokenOffset).ToArray();
        if (tokens.Length == 0) throw new InvalidOperationException("Spark generated no semantic speech tokens.");
        return tokens;
    }

    public int[] GenerateGlobal(IReadOnlyList<int> prompt, Random random)
        => GenerateRange(prompt, _options.GlobalTokenOffset, _options.GlobalCodebookSize,
            _options.GlobalTokenCount, _options.GlobalTokenCount, random);

    /// <summary>Released full-vocabulary Qwen sampling, including grammar and fine style attributes.</summary>
    public int[] GenerateAll(IReadOnlyList<int> prompt, Random random)
        => GenerateRange(prompt, 0, _options.VocabSize, _options.MaxCodecFrames, 0, random, _options.EosToken);

    public Tensor<T> Logits(IReadOnlyList<int> ids)
        => Transformer.ForwardWithLogits(Transformer.Embed(ToMatrix(ids))).Logits;

    /// <summary>Output-token likelihood conditioned on the text and speaker prefix (paper Eq. 2/3).</summary>
    public Tensor<T> Loss(IReadOnlyList<int> sequence, int firstTargetIndex = 1)
    {
        if (firstTargetIndex < 1 || firstTargetIndex >= sequence.Count) throw new ArgumentOutOfRangeException(nameof(firstTargetIndex));
        int[] input = sequence.Take(sequence.Count - 1).Select(Checked).ToArray();
        int[] target = sequence.Skip(firstTargetIndex).Select(Checked).ToArray();
        var logits = Transformer.ForwardWithLogits(Transformer.Embed(ToMatrix(input))).Logits;
        int length = target.Length, vocabulary = _options.VocabSize;
        logits = _engine.TensorNarrow(logits, 1, firstTargetIndex - 1, length);
        var logProbabilities = _engine.Reshape(_engine.TensorLogSoftmax(logits, 2), new[] { length * vocabulary, 1 });
        var indices = new Tensor<int>(new[] { length });
        for (int i = 0; i < length; i++) indices[i] = i * vocabulary + target[i];
        var selected = _engine.TensorIndexSelect(logProbabilities, indices, 0);
        return _engine.TensorMultiplyScalar(_engine.ReduceSum(selected, null), NumOps.FromDouble(-1.0 / length));
    }

    private int[] GenerateRange(IReadOnlyList<int> prompt, int offset, int count, int maximum, int minimum, Random random, int? stopToken = null)
    {
        if (offset < 0 || offset + count > _options.VocabSize)
            throw new InvalidOperationException("The configured Spark added-token range is outside VocabSize.");
        if (stopToken.HasValue) Checked(stopToken.Value);
        if (prompt.Count == 0) throw new ArgumentException("Spark requires a task prompt.", nameof(prompt));
        int[] initial = prompt.Select(Checked).ToArray();
        var cache = new LlamaCache<T>();
        var output = Transformer.Forward(Transformer.Embed(ToMatrix(initial)), null, null, cache, lastLogitsOnly: true).Logits!;
        var generated = new List<int>(maximum);
        for (int step = 0; step < maximum; step++)
        {
            var candidates = Enumerable.Range(offset, count);
            if (stopToken.HasValue)
            {
                candidates = candidates.Where(id => id != stopToken.Value);
                if (step >= minimum) candidates = candidates.Append(stopToken.Value);
            }
            int token = Sample(output, candidates, random);
            if (token == stopToken) break;
            generated.Add(token - offset);
            if (step + 1 < maximum)
                output = Transformer.Forward(Transformer.Embed(ToMatrix(new[] { token })), null, null, cache,
                    lastLogitsOnly: true).Logits!;
        }
        return generated.ToArray();
    }

    private int Sample(Tensor<T> logits, IEnumerable<int> candidates, Random random)
    {
        if (_options.Temperature <= 0) throw new InvalidOperationException("Temperature must be positive.");
        var ranked = candidates
            .Select(token => (Token: token, Value: NumOps.ToDouble(logits[0, 0, token]) / _options.Temperature))
            .OrderByDescending(pair => pair.Value).ThenBy(pair => pair.Token).ToList();
        if (_options.TopK > 0 && ranked.Count > _options.TopK)
        {
            // HF retains every logit tied at the kth threshold.
            double threshold = ranked[_options.TopK - 1].Value;
            ranked.RemoveAll(pair => pair.Value < threshold);
        }
        double max = ranked[0].Value;
        var weighted = ranked.Select(pair => (pair.Token, Weight: Math.Exp(pair.Value - max))).ToList();
        HuggingFaceNucleusSampling.Retain(weighted, _options.TopP);
        double total = weighted.Sum(pair => pair.Weight);
        double draw = random.NextDouble() * total;
        foreach (var pair in weighted)
        {
            draw -= pair.Weight;
            if (draw <= 0) return pair.Token;
        }
        return weighted[^1].Token;
    }

    private int Checked(int id) => id >= 0 && id < _options.VocabSize ? id
        : throw new ArgumentOutOfRangeException(nameof(id), id, "Spark token is outside the configured vocabulary.");

    private static int[,] ToMatrix(IReadOnlyList<int> ids)
    {
        var matrix = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) matrix[0, i] = ids[i];
        return matrix;
    }
}

/// <summary>Spark's Vocos ConvNeXt backbone, including continuous d-vector AdaLayerNorm.</summary>
internal sealed class SparkVocosBackbone<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _dim;
    private readonly int _condition;
    private readonly NormedConv1DLayer<T> _embed;
    private readonly LayerNormalizationLayer<T> _inputNorm;
    private readonly FullyConnectedLayer<T>? _inputScale, _inputShift;
    private readonly List<Block> _blocks = new();
    private readonly LayerNormalizationLayer<T> _finalNorm;

    private sealed record Block(NormedConv1DLayer<T> Depthwise, LayerNormalizationLayer<T> Norm,
        FullyConnectedLayer<T>? Scale, FullyConnectedLayer<T>? Shift, NormedConv1DLayer<T> Expand,
        NormedConv1DLayer<T> Project, ChannelScaleLayer<T> Gamma);

    public SparkVocosBackbone(IEngine engine, List<LayerBase<T>> owner, Random random, int input, int dim,
        int intermediate, int blocks, int condition = 0)
    {
        _engine = engine;
        _dim = dim;
        _condition = condition;
        double Normal()
        {
            double u1 = 1 - random.NextDouble(), u2 = random.NextDouble();
            return 0.02 * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        }
        NormedConv1DLayer<T> Conv(int cin, int cout, int kernel, int groups, int padding)
        {
            var layer = new NormedConv1DLayer<T>(cin, cout, kernel, 1, 1, groups, padding, false,
                ConvolutionNormalization.None);
            layer.Reinitialize(Normal);
            owner.Add(layer);
            return layer;
        }
        FullyConnectedLayer<T> Condition(int output)
        {
            var layer = new FullyConnectedLayer<T>(condition, output, new IdentityActivation<T>());
            owner.Add(layer);
            return layer;
        }

        _embed = Conv(input, dim, 7, 1, 3);
        _inputNorm = new LayerNormalizationLayer<T>(dim, 1e-6);
        if (condition > 0) { _inputScale = Condition(dim); _inputShift = Condition(dim); }
        else owner.Add(_inputNorm);
        for (int i = 0; i < blocks; i++)
        {
            var depthwise = Conv(dim, dim, 7, dim, 3);
            var norm = new LayerNormalizationLayer<T>(dim, 1e-6);
            FullyConnectedLayer<T>? scale = null, shift = null;
            if (condition > 0) { scale = Condition(dim); shift = Condition(dim); }
            else owner.Add(norm);
            var expand = Conv(dim, intermediate, 1, 1, 0);
            var project = Conv(intermediate, dim, 1, 1, 0);
            var gamma = new ChannelScaleLayer<T>(dim, 1.0 / blocks);
            owner.Add(gamma);
            _blocks.Add(new Block(depthwise, norm, scale, shift, expand, project, gamma));
        }
        _finalNorm = new LayerNormalizationLayer<T>(dim, 1e-6);
        owner.Add(_finalNorm);
    }

    /// <summary>Maps channel-first features to channel-first features at the same frame rate.</summary>
    public Tensor<T> Forward(Tensor<T> features, Tensor<T>? condition = null)
    {
        var x = Normalize(_inputNorm, _inputScale, _inputShift, _embed.Forward(features), condition);
        foreach (var block in _blocks)
        {
            var h = Normalize(block.Norm, block.Scale, block.Shift, block.Depthwise.Forward(x), condition);
            h = block.Project.Forward(ActivationHelper.ExactGelu(block.Expand.Forward(h), _engine));
            x = _engine.TensorAdd(x, block.Gamma.Forward(h));
        }
        return ChannelNorm(_finalNorm, x);
    }

    private Tensor<T> Normalize(LayerNormalizationLayer<T> norm, FullyConnectedLayer<T>? scale,
        FullyConnectedLayer<T>? shift, Tensor<T> x, Tensor<T>? condition)
    {
        var normalized = ChannelNorm(norm, x);
        if (scale is null || shift is null) return normalized;
        if (condition is null) throw new ArgumentNullException(nameof(condition), "AdaLayerNorm needs a speaker d-vector.");
        int time = x.Shape[2];
        var row = condition.Rank == 1 ? _engine.Reshape(condition, new[] { 1, condition.Length }) : condition;
        var s = _engine.TensorTile(_engine.Reshape(scale.Forward(row), new[] { 1, _dim, 1 }), new[] { 1, 1, time });
        var b = _engine.TensorTile(_engine.Reshape(shift.Forward(row), new[] { 1, _dim, 1 }), new[] { 1, 1, time });
        return _engine.TensorAdd(_engine.TensorMultiply(normalized, s), b);
    }

    internal void LoadTorchWeights(Func<string, int[], double[]> read, string prefix = "")
    {
        void Conv(NormedConv1DLayer<T> layer, string name, bool linear = false)
        {
            var shape = layer.TorchWeightShape;
            layer.LoadTorchWeights(read(prefix + name + ".weight", linear ? shape.Take(2).ToArray() : shape),
                null, read(prefix + name + ".bias", new[] { layer.OutputChannels }));
        }
        void Norm(LayerNormalizationLayer<T> layer, FullyConnectedLayer<T>? scale, FullyConnectedLayer<T>? shift, string name)
        {
            if (scale is null || shift is null)
                TorchParameters.LayerNorm(_engine, layer, read(prefix + name + ".weight", new[] { _dim }), read(prefix + name + ".bias", new[] { _dim }));
            else
            {
                void Linear(FullyConnectedLayer<T> target, string suffix) => target.SetParameters(new Vector<T>(
                    read(prefix + name + suffix + ".weight", new[] { _dim, _condition })
                        .Concat(read(prefix + name + suffix + ".bias", new[] { _dim })).Select(NumOps.FromDouble).ToArray()));
                Linear(scale, ".scale"); Linear(shift, ".shift");
            }
        }
        Conv(_embed, "embed"); Norm(_inputNorm, _inputScale, _inputShift, "norm");
        for (int i = 0; i < _blocks.Count; i++)
        {
            var b = _blocks[i]; string p = $"convnext.{i}.";
            Conv(b.Depthwise, p + "dwconv"); Norm(b.Norm, b.Scale, b.Shift, p + "norm");
            Conv(b.Expand, p + "pwconv1", true); Conv(b.Project, p + "pwconv2", true);
            b.Gamma.SetParameters(new Vector<T>(read(prefix + p + "gamma", new[] { _dim }).Select(NumOps.FromDouble).ToArray()));
        }
        Norm(_finalNorm, null, null, "final_layer_norm");
    }

    private Tensor<T> ChannelNorm(LayerNormalizationLayer<T> norm, Tensor<T> x)
    {
        int batch = x.Shape[0], channels = x.Shape[1], time = x.Shape[2];
        var rows = _engine.TensorPermute(x, new[] { 0, 2, 1 });
        rows = norm.Forward(_engine.Reshape(rows, new[] { batch * time, channels }));
        return _engine.TensorPermute(_engine.Reshape(rows, new[] { batch, time, channels }), new[] { 0, 2, 1 });
    }
}

/// <summary>The ECAPA -> Perceiver -> residual-FSQ global speaker tokenizer in BiCodec.</summary>
internal sealed class SparkSpeakerTokenizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly SparkTTSOptions _options;
    private readonly CenteredLogMel<T> _mel;
    private readonly EcapaTdnnBackbone<T> _ecapa;
    private readonly IndexTtsPerceiver<T> _perceiver;
    private readonly FullyConnectedLayer<T> _projectIn, _projectOut, _dVector;

    public SparkSpeakerTokenizer(IEngine engine, List<LayerBase<T>> owner, SparkTTSOptions options, Random random)
    {
        _engine = engine;
        _options = options;
        _mel = new CenteredLogMel<T>(engine, options.SampleRate, options.MelFftSize, options.HopSize,
            options.MelWindowSize, options.MelChannels, options.MelMinimumFrequency, options.SampleRate / 2.0,
            1e-5, logarithm: false);
        int c = options.SpeakerEcapaChannels;
        _ecapa = new EcapaTdnnBackbone<T>(new[] { c, c, c, c, c * 3 }, new[] { 5, 3, 3, 3, 1 },
            new[] { 1, 2, 3, 4, 1 }, 8, 128, 128, options.SpeakerEmbeddingDim, sparkLayout: true);
        foreach (var layer in _ecapa.Layers) owner.Add((LayerBase<T>)layer);
        _perceiver = new IndexTtsPerceiver<T>(engine, owner, random, options.SpeakerLatentDim,
            options.SpeakerContextDim, options.GlobalTokenCount, options.PerceiverHeads, 4, options.PerceiverDepth);
        _projectIn = Own(owner, new FullyConnectedLayer<T>(options.SpeakerLatentDim, options.FsqLevels.Length,
            new IdentityActivation<T>()));
        _projectOut = Own(owner, new FullyConnectedLayer<T>(options.FsqLevels.Length, options.SpeakerLatentDim,
            new IdentityActivation<T>()));
        _dVector = Own(owner, new FullyConnectedLayer<T>(options.GlobalTokenCount * options.SpeakerLatentDim,
            options.SpeakerEmbeddingDim, new IdentityActivation<T>()));
    }

    public (Tensor<T> GlobalTokens, Tensor<T> DVector) Encode(Tensor<T> audio)
    {
        var coordinates = Coordinates(audio);
        var tokens = Quantize(coordinates.Coordinates);
        return (tokens, Decode(tokens));
    }

    internal (Tensor<T> GlobalTokens, Tensor<T> DVector) EncodeContext(Tensor<T> context)
    {
        var tokens = Quantize(_projectIn.Forward(_perceiver.Forward(context)));
        return (tokens, Decode(tokens));
    }

    internal void LoadTorchWeights(Func<string, int[], double[]> read, string prefix = "speaker_encoder.", bool includeEcapa = true)
    {
        if (includeEcapa)
        {
            _ecapa.LoadSparkFrameWeights((name, shape) => read(prefix + "speaker_encoder." + name, shape), _options.MelChannels);
            _ecapa.LoadSparkPoolingWeights((name, shape) => read(prefix + "speaker_encoder." + name, shape));
        }
        _perceiver.Load(_engine, read, prefix + "perceiver_sampler.");
        void Linear(FullyConnectedLayer<T> layer, string name, int input, int output)
            => layer.SetParameters(new Vector<T>(read(prefix + name + ".weight", new[] { output, input })
                .Concat(read(prefix + name + ".bias", new[] { output })).Select(NumOps.FromDouble).ToArray()));
        Linear(_projectIn, "quantizer.project_in", _options.SpeakerLatentDim, _options.FsqLevels.Length);
        Linear(_projectOut, "quantizer.project_out", _options.FsqLevels.Length, _options.SpeakerLatentDim);
        Linear(_dVector, "project", _options.GlobalTokenCount * _options.SpeakerLatentDim, _options.SpeakerEmbeddingDim);
    }

    private (Tensor<T> Latents, Tensor<T> Coordinates) Coordinates(Tensor<T> audio)
    {
        // The released ECAPA checkpoint is an inference feature extractor. Keep its running statistics fixed even when
        // Qwen or the BiCodec decoder is optimized with a batch of one reference utterance.
        foreach (var norm in _ecapa.Layers.OfType<BatchNormalizationLayer<T>>()) norm.SetTrainingMode(false);
        var channelFirst = _mel.Forward(ReferenceClip(audio)); // [1, 128, frames], raw Slaney magnitude mel
        var context = _ecapa.ForwardFrameFeatures(channelFirst); // [1, 1536, frames]
        context = _engine.Reshape(_engine.TensorPermute(context, new[] { 0, 2, 1 }),
            new[] { context.Shape[2], context.Shape[1] });
        var latents = _perceiver.Forward(context); // [32, 128]
        var coordinates = _projectIn.Forward(latents);
        return (latents, coordinates);
    }

    internal (Tensor<T> DVector, Tensor<T> WarmupLoss) EncodeTraining(Tensor<T> audio, bool warmup)
    {
        var (latents, projected) = Coordinates(audio);
        var columns = new List<Tensor<T>>();
        for (int d = 0; d < _options.FsqLevels.Length; d++)
        {
            int level = _options.FsqLevels[d];
            double half = (level - 1) * 1.001 / 2, offset = level % 2 == 0 ? 0.5 : 0;
            double shift = 0.5 * Math.Log((1 + offset / half) / (1 - offset / half));
            var bounded = _engine.TensorAddScalar(_engine.TensorMultiplyScalar(_engine.Tanh(
                _engine.TensorAddScalar(_engine.TensorNarrow(projected, 1, d, 1), NumOps.FromDouble(shift))), NumOps.FromDouble(half)), NumOps.FromDouble(-offset));
            var rounded = new Tensor<T>(bounded.Shape.ToArray());
            for (int i = 0; i < rounded.Length; i++) rounded[i] = NumOps.FromDouble(Math.Round(NumOps.ToDouble(bounded[i]), MidpointRounding.ToEven));
            var detached = new Tensor<T>(bounded.Shape.ToArray(), new Vector<T>(bounded.ToArray()));
            columns.Add(_engine.TensorMultiplyScalar(_engine.TensorAdd(bounded, _engine.TensorSubtract(rounded, detached)), NumOps.FromDouble(1.0 / Math.Floor(level / 2.0))));
        }
        var reconstructed = _projectOut.Forward(_engine.TensorConcatenate(columns.ToArray(), 1));
        Tensor<T> VectorOf(Tensor<T> rows) => _engine.Reshape(_dVector.Forward(_engine.Reshape(
            _engine.TensorTranspose(rows), new[] { 1, _options.GlobalTokenCount * _options.SpeakerLatentDim })), new[] { _options.SpeakerEmbeddingDim });
        var teacher = VectorOf(latents);
        var student = VectorOf(reconstructed);
        var label = new Tensor<T>(teacher.Shape.ToArray(), new Vector<T>(teacher.ToArray()));
        var warmupLoss = _engine.ReduceMean(_engine.TensorAbs(_engine.TensorSubtract(student, label)), new[] { 0 }, false);
        return (warmup ? teacher : student, warmupLoss);
    }

    public Tensor<T> Decode(Tensor<T> tokens)
    {
        if (tokens.Length != _options.GlobalTokenCount)
            throw new ArgumentException($"Spark BiCodec needs exactly {_options.GlobalTokenCount} global tokens.", nameof(tokens));
        int dimensions = _options.FsqLevels.Length;
        var coordinates = new Tensor<T>(new[] { _options.GlobalTokenCount, dimensions });
        for (int t = 0; t < _options.GlobalTokenCount; t++)
        {
            int index = MathPolyfill.Clamp((int)Math.Round(NumOps.ToDouble(tokens[t])), 0, _options.GlobalCodebookSize - 1);
            for (int d = 0; d < dimensions; d++)
            {
                int level = _options.FsqLevels[d];
                int digit = index % level;
                index /= level;
                coordinates[t, d] = NumOps.FromDouble((digit - Math.Floor(level / 2.0)) / Math.Floor(level / 2.0));
            }
        }
        var latents = _projectOut.Forward(coordinates);
        return _engine.Reshape(_dVector.Forward(_engine.Reshape(_engine.TensorTranspose(latents),
            new[] { 1, _options.GlobalTokenCount * _options.SpeakerLatentDim })), new[] { _options.SpeakerEmbeddingDim });
    }

    private Tensor<T> Quantize(Tensor<T> coordinates)
    {
        var tokens = new Tensor<T>(new[] { _options.GlobalTokenCount });
        for (int t = 0; t < tokens.Length; t++)
        {
            int index = 0, place = 1;
            for (int d = 0; d < _options.FsqLevels.Length; d++)
            {
                int level = _options.FsqLevels[d];
                double half = (level - 1) * 1.001 / 2, offset = level % 2 == 0 ? 0.5 : 0;
                double shift = 0.5 * Math.Log((1 + offset / half) / (1 - offset / half));
                double bounded = Math.Tanh(NumOps.ToDouble(coordinates[t, d]) + shift) * half - offset;
                int digit = MathPolyfill.Clamp((int)Math.Round(bounded, MidpointRounding.ToEven) + level / 2, 0, level - 1);
                index += digit * place;
                place *= level;
            }
            tokens[t] = NumOps.FromDouble(index);
        }
        return tokens;
    }

    private Tensor<T> ReferenceClip(Tensor<T> audio)
    {
        int length = Math.Max(_options.MelFftSize, (int)Math.Round(_options.SampleRate * _options.ReferenceSegmentSeconds));
        length = length / _options.HopSize * _options.HopSize;
        var clip = new Tensor<T>(new[] { length });
        for (int i = 0; i < length; i++) clip[i] = audio[i % audio.Length];
        return clip;
    }

    private static TLayer Own<TLayer>(List<LayerBase<T>> owner, TLayer layer) where TLayer : LayerBase<T>
    { owner.Add(layer); return layer; }
}

/// <summary>Spark's non-aliased Snake1d: x + sin²(alpha*x)/(alpha+1e-9).</summary>
[LayerCategory(LayerCategory.Activation)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 3, 5", TestConstructorArgs = "3")]
[ElementWiseShape(Note = "Spark's per-channel Snake activation preserves the waveform feature shape.")]
[AutoParameters]
internal sealed partial class SparkSnakeLayer<T> : LayerBase<T>
{
    private readonly int _channels;
    private Tensor<T> _alpha;

    public SparkSnakeLayer([LayerState] int channels) : base(new[] { channels }, new[] { channels })
    {
        _channels = channels;
        _alpha = new Tensor<T>(new[] { channels });
        for (int i = 0; i < channels; i++) _alpha[i] = NumOps.One;
        RegisterTrainableParameter(_alpha, PersistentTensorRole.Weights);
    }

    public override bool SupportsTraining => true;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        int batch = input.Shape[0], time = input.Shape[2];
        var alpha = Engine.TensorTile(Engine.Reshape(_alpha, new[] { 1, _channels, 1 }), new[] { batch, 1, time });
        var sine = Engine.TensorSin(Engine.TensorMultiply(alpha, input));
        return Engine.TensorAdd(input, Engine.TensorDivide(Engine.TensorMultiply(sine, sine),
            Engine.TensorAddScalar(alpha, NumOps.FromDouble(1e-9))));
    }

    public override void ResetState() { }
}

/// <summary>The exact DAC-style Spark WaveGenerator: four transposed-convolution blocks and Snake residual units.</summary>
internal sealed class SparkWaveGenerator<T>
{
    private readonly IEngine _engine;
    private readonly int _samplesPerFrame;
    private readonly NormedConv1DLayer<T> _first, _last;
    private readonly List<(SparkSnakeLayer<T> Activation, NormedConv1DLayer<T> Up, SparkWaveResidual<T>[] Units)> _blocks = new();
    private readonly SparkSnakeLayer<T> _finalActivation;

    public SparkWaveGenerator(IEngine engine, List<LayerBase<T>> owner, SparkTTSOptions options)
    {
        _engine = engine;
        _samplesPerFrame = options.WaveDecoderRates.Aggregate(1, (product, rate) => product * rate);
        _first = Conv(owner, options.BiCodecDim, options.WaveDecoderChannels, 7, 1, 3, false);
        int channels = options.WaveDecoderChannels;
        for (int i = 0; i < options.WaveDecoderRates.Length; i++)
        {
            int rate = options.WaveDecoderRates[i], kernel = options.WaveDecoderKernels[i], output = channels / 2;
            var activation = Own(owner, new SparkSnakeLayer<T>(channels));
            var up = Conv(owner, channels, output, kernel, rate, (kernel - rate) / 2, true);
            var units = new[] { 1, 3, 9 }.Select(d => new SparkWaveResidual<T>(engine, owner, output, d)).ToArray();
            _blocks.Add((activation, up, units));
            channels = output;
        }
        _finalActivation = Own(owner, new SparkSnakeLayer<T>(channels));
        _last = Conv(owner, channels, 1, 7, 1, 3, false);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        var x = _first.Forward(input);
        foreach (var (activation, up, units) in _blocks)
        {
            x = up.Forward(activation.Forward(x));
            foreach (var unit in units) x = unit.Forward(x);
        }
        var waveform = _engine.Reshape(_engine.Tanh(_last.Forward(_finalActivation.Forward(x))),
            new[] { x.Shape[0] * x.Shape[2] });
        int expected = input.Shape[0] * input.Shape[2] * _samplesPerFrame;
        if (waveform.Length < expected)
            throw new InvalidOperationException($"WaveGenerator produced {waveform.Length} samples; {expected} were required.");
        return waveform.Length == expected ? waveform : _engine.TensorNarrow(waveform, 0, 0, expected);
    }

    internal void LoadTorchWeights(Func<string, int[], double[]> read, string prefix = "decoder.")
    {
        SparkTorchWeights<T>.Conv(_first, read, prefix + "model.0");
        for (int i = 0; i < _blocks.Count; i++)
        {
            var b = _blocks[i]; string p = prefix + $"model.{i + 1}.block.";
            b.Activation.SetParameters(new Vector<T>(read(p + "0.alpha", new[] { 1, b.Up.InputChannels, 1 })
                .Select(MathHelper.GetNumericOperations<T>().FromDouble).ToArray()));
            SparkTorchWeights<T>.Conv(b.Up, read, p + "1");
            for (int j = 0; j < b.Units.Length; j++) b.Units[j].LoadTorchWeights(read, p + $"{j + 2}.block.");
        }
        string final = prefix + $"model.{_blocks.Count + 1}";
        _finalActivation.SetParameters(new Vector<T>(read(final + ".alpha", new[] { 1, _last.InputChannels, 1 })
            .Select(MathHelper.GetNumericOperations<T>().FromDouble).ToArray()));
        SparkTorchWeights<T>.Conv(_last, read, prefix + $"model.{_blocks.Count + 2}");
    }

    private static NormedConv1DLayer<T> Conv(List<LayerBase<T>> owner, int input, int output, int kernel,
        int stride, int padding, bool transpose)
    {
        var layer = new NormedConv1DLayer<T>(input, output, kernel, stride, 1, 1, padding, transpose,
            ConvolutionNormalization.Weight);
        owner.Add(layer);
        return layer;
    }

    private static TLayer Own<TLayer>(List<LayerBase<T>> owner, TLayer layer) where TLayer : LayerBase<T>
    { owner.Add(layer); return layer; }
}

internal sealed class SparkWaveResidual<T>
{
    private readonly IEngine _engine;
    private readonly SparkSnakeLayer<T> _activation1, _activation2;
    private readonly NormedConv1DLayer<T> _conv1, _conv2;

    public SparkWaveResidual(IEngine engine, List<LayerBase<T>> owner, int channels, int dilation)
    {
        _engine = engine;
        _activation1 = Own(owner, new SparkSnakeLayer<T>(channels));
        _conv1 = Conv(owner, channels, 7, dilation, 3 * dilation);
        _activation2 = Own(owner, new SparkSnakeLayer<T>(channels));
        _conv2 = Conv(owner, channels, 1, 1, 0);
    }

    public Tensor<T> Forward(Tensor<T> input)
        => _engine.TensorAdd(input, _conv2.Forward(_activation2.Forward(_conv1.Forward(_activation1.Forward(input)))));

    internal void LoadTorchWeights(Func<string, int[], double[]> read, string prefix)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        _activation1.SetParameters(new Vector<T>(read(prefix + "0.alpha", new[] { 1, _conv1.InputChannels, 1 }).Select(ops.FromDouble).ToArray()));
        SparkTorchWeights<T>.Conv(_conv1, read, prefix + "1");
        _activation2.SetParameters(new Vector<T>(read(prefix + "2.alpha", new[] { 1, _conv2.InputChannels, 1 }).Select(ops.FromDouble).ToArray()));
        SparkTorchWeights<T>.Conv(_conv2, read, prefix + "3");
    }

    private static NormedConv1DLayer<T> Conv(List<LayerBase<T>> owner, int channels, int kernel, int dilation, int padding)
    {
        var layer = new NormedConv1DLayer<T>(channels, channels, kernel, 1, dilation, 1, padding, false,
            ConvolutionNormalization.Weight);
        owner.Add(layer);
        return layer;
    }

    private static TLayer Own<TLayer>(List<LayerBase<T>> owner, TLayer layer) where TLayer : LayerBase<T>
    { owner.Add(layer); return layer; }
}

/// <summary>BiCodec: XLSR feature mixture, Vocos encoder, factorized VQ, speaker FSQ, conditional Vocos and WaveGenerator.</summary>
internal sealed class SparkBiCodec<T>
{
    private readonly IEngine _engine;
    private readonly SparkTTSOptions _options;
    private readonly Wav2Vec2Network<T> _wav2Vec;
    private readonly SparkVocosBackbone<T> _semanticBackbone, _semanticStage1, _semanticStage2;
    private readonly FullyConnectedLayer<T> _semanticProjection;
    private readonly FactorizedVectorQuantizerLayer<T> _semanticQuantizer;
    private readonly SparkSpeakerTokenizer<T> _speaker;
    private readonly FullyConnectedLayer<T> _preProjection, _outputProjection, _postnetPreProjection, _postnetProjection;
    private readonly SparkVocosBackbone<T> _preStage1, _preStage2, _conditionalPrenet;
    private readonly SparkVocosBackbone<T> _postnet, _postnetStage1, _postnetStage2;
    private readonly SparkWaveGenerator<T> _waveGenerator;

    public SparkBiCodec(IEngine engine, List<LayerBase<T>> owner, SparkTTSOptions options, Random random)
    {
        _engine = engine;
        _options = options;
        _wav2Vec = new Wav2Vec2Network<T>(engine, owner, new Wav2Vec2Configuration
        {
            ConvDims = options.Wav2VecConvDims,
            ConvKernels = options.Wav2VecConvKernels,
            ConvStrides = options.Wav2VecConvStrides,
            ConvBias = true,
            LayerNormFeatureEncoder = true,
            HiddenSize = options.Wav2VecHiddenDim,
            Layers = options.Wav2VecLayers,
            Heads = options.Wav2VecHeads,
            IntermediateSize = options.Wav2VecIntermediateDim,
            PositionalConvGroups = ValidGroupCount(options.Wav2VecHiddenDim, 16),
            StableLayerNorm = true,
            LayerDrop = 0,
            MaskTimeProbability = 0,
        });
        _wav2Vec.InitializeLikeReference(random);
        int trainableStart = owner.Count;
        _semanticBackbone = new SparkVocosBackbone<T>(engine, owner, random, options.Wav2VecHiddenDim,
            options.VocosDim, options.VocosIntermediateDim, options.SemanticEncoderLayers);
        _semanticStage1 = new SparkVocosBackbone<T>(engine, owner, random, options.VocosDim,
            options.VocosDim, options.VocosIntermediateDim, 2);
        _semanticStage2 = new SparkVocosBackbone<T>(engine, owner, random, options.VocosDim,
            options.VocosDim, options.VocosIntermediateDim, 2);
        _semanticProjection = Own(owner, new FullyConnectedLayer<T>(options.VocosDim, options.BiCodecDim,
            new IdentityActivation<T>()));
        _semanticQuantizer = Own(owner, new FactorizedVectorQuantizerLayer<T>(options.BiCodecDim, 1,
            options.CodebookSize, options.SemanticCodeDimension, lossesOnRawVectors: true, useL1Loss: true));
        _speaker = new SparkSpeakerTokenizer<T>(engine, owner, options, random);
        _preProjection = Own(owner, new FullyConnectedLayer<T>(options.BiCodecDim, options.VocosDim,
            new IdentityActivation<T>()));
        _preStage1 = new SparkVocosBackbone<T>(engine, owner, random, options.VocosDim,
            options.VocosDim, options.VocosIntermediateDim, 2);
        _preStage2 = new SparkVocosBackbone<T>(engine, owner, random, options.VocosDim,
            options.VocosDim, options.VocosIntermediateDim, 2);
        _conditionalPrenet = new SparkVocosBackbone<T>(engine, owner, random, options.VocosDim,
            options.VocosDim, options.VocosIntermediateDim, options.PrenetLayers, options.SpeakerEmbeddingDim);
        _outputProjection = Own(owner, new FullyConnectedLayer<T>(options.VocosDim, options.BiCodecDim,
            new IdentityActivation<T>()));
        _postnetPreProjection = Own(owner, new FullyConnectedLayer<T>(options.BiCodecDim, options.VocosDim,
            new IdentityActivation<T>()));
        _postnetStage1 = new SparkVocosBackbone<T>(engine, owner, random, options.VocosDim,
            options.VocosDim, options.VocosIntermediateDim, 2);
        _postnetStage2 = new SparkVocosBackbone<T>(engine, owner, random, options.VocosDim,
            options.VocosDim, options.VocosIntermediateDim, 2);
        _postnet = new SparkVocosBackbone<T>(engine, owner, random, options.VocosDim,
            options.VocosDim, options.VocosIntermediateDim, options.PostnetLayers);
        _postnetProjection = Own(owner, new FullyConnectedLayer<T>(options.VocosDim, options.Wav2VecHiddenDim,
            new IdentityActivation<T>()));
        _waveGenerator = new SparkWaveGenerator<T>(engine, owner, options);
        TrainableLayers = owner.Skip(trainableStart).ToList();
    }

    internal IReadOnlyList<LayerBase<T>> TrainableLayers { get; }

    public FactorizedVectorQuantizerLayer<T> SemanticQuantizer => _semanticQuantizer;
    public Wav2Vec2Network<T> Wav2Vec => _wav2Vec;
    public SparkSpeakerTokenizer<T> Speaker => _speaker;

    internal void LoadTorchWeights(Func<string, int[], double[]> read, bool includeSpeaker = true)
    {
        LoadFeatureWeights(read);
        SparkTorchWeights<T>.Conv(_semanticQuantizer.InProjection(0), read, "quantizer.in_project");
        SparkTorchWeights<T>.Conv(_semanticQuantizer.OutProjection(0), read, "quantizer.out_project");
        var table = _semanticQuantizer.Codebook(0);
        var values = read("quantizer.codebook.weight", new[] { _options.CodebookSize, _options.SemanticCodeDimension });
        for (int i = 0; i < table.Length; i++) table[i] = MathHelper.GetNumericOperations<T>().FromDouble(values[i]);
        _engine.InvalidatePersistentTensor(table);
        _waveGenerator.LoadTorchWeights(read);
        if (includeSpeaker) _speaker.LoadTorchWeights(read);
    }

    internal void LoadFeatureWeights(Func<string, int[], double[]> read)
    {
        _semanticBackbone.LoadTorchWeights(read, "encoder.encoder.");
        _semanticStage1.LoadTorchWeights(read, "encoder.downsample.0.1.");
        _semanticStage2.LoadTorchWeights(read, "encoder.downsample.1.1.");
        SparkTorchWeights<T>.Linear(_semanticProjection, read, "encoder.project", _options.VocosDim, _options.BiCodecDim);
        void Decoder(string p, FullyConnectedLayer<T> input, SparkVocosBackbone<T> first, SparkVocosBackbone<T> second,
            SparkVocosBackbone<T> backbone, FullyConnectedLayer<T> output, int outputChannels)
        {
            SparkTorchWeights<T>.Linear(input, read, p + "linear_pre", _options.BiCodecDim, _options.VocosDim);
            first.LoadTorchWeights(read, p + "downsample.0.1."); second.LoadTorchWeights(read, p + "downsample.1.1.");
            backbone.LoadTorchWeights(read, p + "vocos_backbone.");
            SparkTorchWeights<T>.Linear(output, read, p + "linear", _options.VocosDim, outputChannels);
        }
        Decoder("prenet.", _preProjection, _preStage1, _preStage2, _conditionalPrenet, _outputProjection, _options.BiCodecDim);
        Decoder("postnet.", _postnetPreProjection, _postnetStage1, _postnetStage2, _postnet, _postnetProjection, _options.Wav2VecHiddenDim);
    }

    public (Tensor<T> Semantic, Tensor<T> Global) Tokenize(Tensor<T> audio)
    {
        using var _ = new NoGradScope<T>();
        var semantic = EncodeSemanticFeatures(audio);
        int[,] ids = _semanticQuantizer.Encode(semantic);
        var tokens = new Tensor<T>(new[] { ids.GetLength(1) });
        for (int i = 0; i < tokens.Length; i++) tokens[i] = MathHelper.GetNumericOperations<T>().FromDouble(ids[0, i]);
        return (tokens, _speaker.Encode(audio).GlobalTokens);
    }

    public Tensor<T> EncodeSemantic(Tensor<T> audio) => Tokenize(audio).Semantic;
    public Tensor<T> EncodeGlobal(Tensor<T> audio) => _speaker.Encode(audio).GlobalTokens;

    public Tensor<T> Decode(Tensor<T> semanticTokens, Tensor<T> globalTokens)
    {
        int[,] codes = new int[1, semanticTokens.Length];
        for (int i = 0; i < semanticTokens.Length; i++)
            codes[0, i] = MathPolyfill.Clamp((int)Math.Round(MathHelper.GetNumericOperations<T>().ToDouble(semanticTokens[i])), 0,
                _options.CodebookSize - 1);
        var quantized = _semanticQuantizer.Decode(codes);
        var dVector = _speaker.Decode(globalTokens);
        var x = PrenetFeatures(quantized, dVector);
        var condition = _engine.TensorTile(_engine.Reshape(dVector, new[] { 1, _options.SpeakerEmbeddingDim, 1 }),
            new[] { 1, 1, x.Shape[2] });
        return _waveGenerator.Forward(_engine.TensorAdd(x, condition));
    }

    public (Tensor<T> Waveform, Tensor<T> AuxiliaryLoss) TrainingForward(Tensor<T> audio, bool warmup)
    {
        var latent = EncodeSemanticFeatures(audio);
        var quantized = _semanticQuantizer.Forward(latent);
        var speaker = _speaker.EncodeTraining(audio, warmup);
        var dVector = speaker.DVector;
        var x = PrenetFeatures(quantized, dVector);
        var condition = _engine.TensorTile(_engine.Reshape(dVector, new[] { 1, _options.SpeakerEmbeddingDim, 1 }), new[] { 1, 1, x.Shape[2] });
        var waveform = _waveGenerator.Forward(_engine.TensorAdd(x, condition));
        int samples = Math.Min(audio.Length, waveform.Length);
        waveform = _engine.TensorNarrow(waveform, 0, 0, samples);
        var predictedFeatures = PostnetFeatures(x);
        var target = MixedWav2VecFeatures(audio);
        int frames = Math.Min(predictedFeatures.Shape[2], target.Shape[2]);
        var predicted = _engine.TensorSlice(predictedFeatures, new[] { 0, 0, 0 },
            new[] { 1, predictedFeatures.Shape[1], frames });
        var wanted = _engine.TensorSlice(target, new[] { 0, 0, 0 }, new[] { 1, target.Shape[1], frames });
        var delta = _engine.TensorSubtract(predicted, wanted);
        var featureLoss = _engine.ReduceMean(_engine.TensorMultiply(delta, delta),
            Enumerable.Range(0, delta.Rank).ToArray(), keepDims: false);
        var commitment = _semanticQuantizer.CommitmentLoss ?? Scalar(0);
        var codebook = _semanticQuantizer.CodebookLoss ?? Scalar(0);
        var auxiliary = _engine.TensorAdd(featureLoss, _engine.TensorAdd(
            _engine.TensorMultiplyScalar(commitment, MathHelper.GetNumericOperations<T>().FromDouble(_options.SemanticCommitmentWeight)),
            _engine.TensorMultiplyScalar(codebook, MathHelper.GetNumericOperations<T>().FromDouble(_options.SemanticCodebookWeight))));
        if (warmup) auxiliary = _engine.TensorAdd(auxiliary, speaker.WarmupLoss);
        return (waveform, auxiliary);
    }

    private Tensor<T> EncodeSemanticFeatures(Tensor<T> audio)
        => EncodeFeatures(MixedWav2VecFeatures(audio));

    internal Tensor<T> EncodeFeatures(Tensor<T> features)
    {
        var x = _semanticBackbone.Forward(features);
        x = _semanticStage2.Forward(SamplingIdentity(_semanticStage1.Forward(SamplingIdentity(x))));
        return Rows(_semanticProjection, x);
    }

    internal Tensor<T> PrenetFeatures(Tensor<T> quantized, Tensor<T> dVector)
    {
        var x = Rows(_preProjection, quantized);
        x = _preStage2.Forward(SamplingIdentity(_preStage1.Forward(SamplingIdentity(x))));
        return Rows(_outputProjection, _conditionalPrenet.Forward(x, dVector));
    }

    internal Tensor<T> PostnetFeatures(Tensor<T> features)
    {
        var x = Rows(_postnetPreProjection, features);
        x = _postnetStage2.Forward(SamplingIdentity(_postnetStage1.Forward(SamplingIdentity(x))));
        return Rows(_postnetProjection, _postnet.Forward(x));
    }

    private Tensor<T> MixedWav2VecFeatures(Tensor<T> audio)
        => _engine.TensorPermute(ExtractXlsrFeatures(audio), new[] { 0, 2, 1 });

    internal Tensor<T> ExtractXlsrFeatures(Tensor<T> audio)
    {
        using var frozenTeacher = new NoGradScope<T>();
        // The released Wav2Vec2FeatureExtractor normalizes each utterance before the XLSR backbone.
        var output = _wav2Vec.Forward(NormalizeXlsrWaveform(audio), false, new Random(_options.GenerationSeed));
        var valid = _options.Wav2VecMixedLayers.Select(index => MathPolyfill.Clamp(index, 0, output.LayerStates.Count - 1)).ToArray();
        var mixed = output.LayerStates[valid[0]];
        for (int i = 1; i < valid.Length; i++) mixed = _engine.TensorAdd(mixed, output.LayerStates[valid[i]]);
        mixed = _engine.TensorMultiplyScalar(mixed, MathHelper.GetNumericOperations<T>().FromDouble(1.0 / valid.Length));
        return mixed;
    }

    internal Tensor<T> NormalizeXlsrWaveform(Tensor<T> audio)
    {
        var x = _engine.Reshape(audio, new[] { audio.Length });
        var centered = _engine.TensorSubtract(x, _engine.ReduceMean(x, new[] { 0 }, keepDims: true));
        var variance = _engine.ReduceMean(_engine.TensorMultiply(centered, centered), new[] { 0 }, keepDims: true);
        return _engine.TensorDivide(centered, _engine.TensorSqrt(_engine.TensorAddScalar(variance,
            MathHelper.GetNumericOperations<T>().FromDouble(1e-7))));
    }

    private Tensor<T> Rows(LayerBase<T> layer, Tensor<T> channelFirst)
    {
        int batch = channelFirst.Shape[0], input = channelFirst.Shape[1], frames = channelFirst.Shape[2];
        var rows = _engine.Reshape(_engine.TensorPermute(channelFirst, new[] { 0, 2, 1 }), new[] { batch * frames, input });
        var output = layer.Forward(rows);
        return _engine.TensorPermute(_engine.Reshape(output, new[] { batch, frames, output.Shape[1] }), new[] { 0, 2, 1 });
    }

    private Tensor<T> Scalar(double value)
    {
        var tensor = new Tensor<T>(new[] { 1 });
        tensor[0] = MathHelper.GetNumericOperations<T>().FromDouble(value);
        return tensor;
    }

    // Released SamplingBlock.forward sums conv_res + skip1_res + skip2_res even when both ratios are one.
    private Tensor<T> SamplingIdentity(Tensor<T> x) => _engine.TensorMultiplyScalar(x, MathHelper.GetNumericOperations<T>().FromDouble(3));

    private static int ValidGroupCount(int channels, int preferred)
    {
        int groups = Math.Min(channels, preferred);
        while (channels % groups != 0) groups--;
        return groups;
    }

    private static TLayer Own<TLayer>(List<LayerBase<T>> owner, TLayer layer) where TLayer : LayerBase<T>
    { owner.Add(layer); return layer; }
}

internal static class SparkTorchWeights<T>
{
    internal static void Conv(NormedConv1DLayer<T> layer, Func<string, int[], double[]> read, string name)
    {
        var shape = layer.TorchWeightShape;
        layer.LoadTorchWeights(read(name + ".weight_v", shape), read(name + ".weight_g", new[] { shape[0], 1, 1 }),
            read(name + ".bias", new[] { layer.OutputChannels }));
    }

    internal static void Linear(FullyConnectedLayer<T> layer, Func<string, int[], double[]> read, string name, int input, int output)
        => layer.SetParameters(new Vector<T>(read(name + ".weight", new[] { output, input })
            .Concat(read(name + ".bias", new[] { output })).Select(MathHelper.GetNumericOperations<T>().FromDouble).ToArray()));
}
