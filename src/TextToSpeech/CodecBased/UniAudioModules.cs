using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Codecs;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

internal static class UniAudioOps<T>
{
    internal static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    internal static TLayer Own<TLayer>(List<LayerBase<T>> owner, TLayer layer) where TLayer : LayerBase<T>
    {
        owner.Add(layer);
        return layer;
    }

    internal static Tensor<T> Ids(IReadOnlyList<int> ids)
    {
        var result = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) result[i] = NumOps.FromDouble(ids[i]);
        return result;
    }

    internal static int[] IdsOf(Tensor<T> ids)
    {
        var result = new int[ids.Length];
        for (int i = 0; i < result.Length; i++) result[i] = (int)Math.Round(NumOps.ToDouble(ids[i]));
        return result;
    }

    internal static Tensor<T> CausalMask(int length)
    {
        var mask = new Tensor<T>(new[] { length, length });
        for (int row = 0; row < length; row++)
            for (int column = row + 1; column < length; column++)
                mask[row, column] = NumOps.FromDouble(-1e9);
        return mask;
    }

    internal static Tensor<T> TeacherForcedLoss(IEngine engine, Tensor<T> logits, IReadOnlyList<int> sequence,
        int prefixTokenCount, bool prefixLanguageModel, int paddingToken)
    {
        int start = prefixLanguageModel ? prefixTokenCount : 0;
        if (start < 0 || start >= sequence.Count) throw new ArgumentOutOfRangeException(nameof(prefixTokenCount));
        var scored = engine.TensorSlice(logits, new[] { start, 0 }, new[] { sequence.Count - start, logits.Shape[1] });
        return CrossEntropy(engine, scored, sequence.Skip(start).ToArray(), paddingToken);
    }

    internal static Tensor<T> CrossEntropy(IEngine engine, Tensor<T> logits, IReadOnlyList<int> labels, int? ignoreLabel = null)
    {
        int rows = logits.Shape[0], classes = logits.Shape[1], count = Math.Min(rows, labels.Count);
        if (count == 0) return new Tensor<T>(new[] { 1 });
        var log = engine.Reshape(engine.TensorLogSoftmax(logits, 1), new[] { rows * classes, 1 });
        var selected = Enumerable.Range(0, count).Where(i => !ignoreLabel.HasValue || labels[i] != ignoreLabel.Value).ToArray();
        if (selected.Length == 0) return new Tensor<T>(new[] { 1 });
        var indices = new Tensor<int>(new[] { selected.Length });
        for (int i = 0; i < selected.Length; i++)
            indices[i] = selected[i] * classes + MathPolyfill.Clamp(labels[selected[i]], 0, classes - 1);
        return engine.TensorNegate(engine.ReduceMean(engine.TensorIndexSelect(log, indices, 0), new[] { 0, 1 }, false));
    }
}

/// <summary>The release's pre-LayerNorm Whisper block: Q/V/out have biases, K does not, followed by a 4x GELU MLP.</summary>
internal sealed class UniAudioTransformerBlock<T>
{
    private readonly IEngine _engine;
    private readonly int _dim, _heads;

    internal UniAudioTransformerBlock(IEngine engine, List<LayerBase<T>> owner, int dim, int heads, int feedForward)
    {
        _engine = engine;
        _dim = dim;
        _heads = heads;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        Norm1 = UniAudioOps<T>.Own(owner, new LayerNormalizationLayer<T>(dim));
        Query = UniAudioOps<T>.Own(owner, new DenseLayer<T>(dim, identity, biasMode: BiasMode.Always));
        Key = UniAudioOps<T>.Own(owner, new BiasFreeLinearLayer<T>(dim, dim));
        Value = UniAudioOps<T>.Own(owner, new DenseLayer<T>(dim, identity, biasMode: BiasMode.Always));
        AttentionOut = UniAudioOps<T>.Own(owner, new DenseLayer<T>(dim, identity, biasMode: BiasMode.Always));
        Norm2 = UniAudioOps<T>.Own(owner, new LayerNormalizationLayer<T>(dim));
        FeedForwardIn = UniAudioOps<T>.Own(owner, new DenseLayer<T>(feedForward, identity, biasMode: BiasMode.Always));
        FeedForwardOut = UniAudioOps<T>.Own(owner, new DenseLayer<T>(dim, identity, biasMode: BiasMode.Always));
    }

    internal LayerNormalizationLayer<T> Norm1 { get; }
    internal DenseLayer<T> Query { get; }
    internal BiasFreeLinearLayer<T> Key { get; }
    internal DenseLayer<T> Value { get; }
    internal DenseLayer<T> AttentionOut { get; }
    internal LayerNormalizationLayer<T> Norm2 { get; }
    internal DenseLayer<T> FeedForwardIn { get; }
    internal DenseLayer<T> FeedForwardOut { get; }

    internal Tensor<T> Forward(Tensor<T> x, Tensor<T> mask)
    {
        var normalized = Norm1.Forward(x);
        int length = normalized.Shape[0], headDim = _dim / _heads;
        Tensor<T> Heads(Tensor<T> projected) => _engine.TensorPermute(
            _engine.Reshape(projected, new[] { length, _heads, headDim }), new[] { 1, 0, 2 });
        var q = _engine.TensorMultiplyScalar(Heads(Query.Forward(normalized)),
            UniAudioOps<T>.NumOps.FromDouble(1.0 / Math.Sqrt(headDim)));
        var k = Heads(Key.Forward(normalized));
        var v = Heads(Value.Forward(normalized));
        var scores = _engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 }));
        scores = _engine.TensorAdd(scores, _engine.TensorTile(
            _engine.Reshape(mask, new[] { 1, length, length }), new[] { _heads, 1, 1 }));
        var weights = _engine.Softmax(scores, 2);
        var context = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(weights, v),
            new[] { 1, 0, 2 }), new[] { length, _dim });
        x = _engine.TensorAdd(x, AttentionOut.Forward(context));
        var hidden = ActivationHelper.ExactGelu(FeedForwardIn.Forward(Norm2.Forward(x)), _engine);
        return _engine.TensorAdd(x, FeedForwardOut.Forward(hidden));
    }

    internal void LoadTorchWeights(TorchTensorSource source, string prefix, int feedForward)
    {
        TorchParameters.LayerNorm(_engine, Norm1, source, prefix + "attn_ln.weight", prefix + "attn_ln.bias");
        TorchParameters.LayerNorm(_engine, Norm2, source, prefix + "mlp_ln.weight", prefix + "mlp_ln.bias");
        TorchParameters.Linear(_engine, Query, _dim, _dim, source, prefix + "attn.query.weight", prefix + "attn.query.bias");
        TorchParameters.Linear(Key, _dim, _dim, source, prefix + "attn.key.weight");
        TorchParameters.Linear(_engine, Value, _dim, _dim, source, prefix + "attn.value.weight", prefix + "attn.value.bias");
        TorchParameters.Linear(_engine, AttentionOut, _dim, _dim, source, prefix + "attn.out.weight", prefix + "attn.out.bias");
        TorchParameters.Linear(_engine, FeedForwardIn, _dim, feedForward, source, prefix + "mlp.0.weight", prefix + "mlp.0.bias");
        TorchParameters.Linear(_engine, FeedForwardOut, feedForward, _dim, source, prefix + "mlp.2.weight", prefix + "mlp.2.bias");
    }
}

internal sealed class UniAudioTransformer<T>
{
    private readonly List<UniAudioTransformerBlock<T>> _blocks = new();

    internal UniAudioTransformer(IEngine engine, List<LayerBase<T>> owner, int dim, int heads, int depth, int feedForward)
    {
        for (int i = 0; i < depth; i++) _blocks.Add(new UniAudioTransformerBlock<T>(engine, owner, dim, heads, feedForward));
        FinalNorm = UniAudioOps<T>.Own(owner, new LayerNormalizationLayer<T>(dim));
    }

    internal IReadOnlyList<UniAudioTransformerBlock<T>> Blocks => _blocks;
    internal LayerNormalizationLayer<T> FinalNorm { get; }

    internal Tensor<T> Forward(Tensor<T> x, Tensor<T>? mask = null)
    {
        mask ??= UniAudioOps<T>.CausalMask(x.Shape[0]);
        foreach (var block in _blocks) x = block.Forward(x, mask);
        return FinalNorm.Forward(x);
    }
}

/// <summary>UniAudio's two-scale causal LM: inter-frame global patches and an intra-frame local autoregressive decoder.</summary>
internal sealed class UniAudioMultiScaleTransformer<T>
{
    private readonly IEngine _engine;
    private readonly UniAudioOptions _options;
    private readonly EmbeddingLayer<T> _globalEmbedding, _localEmbedding;
    private readonly LearnedPositionalEmbeddingLayer<T> _globalPosition, _localPosition;
    private readonly UniAudioTransformer<T> _global, _local;
    private readonly DenseLayer<T> _globalToLocal;
    private readonly BiasFreeLinearLayer<T> _head;

    internal UniAudioMultiScaleTransformer(IEngine engine, List<LayerBase<T>> owner, UniAudioOptions options)
    {
        _engine = engine;
        _options = options;
        int patchDim = options.LLMDim / options.NumCodebooks;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        _globalEmbedding = UniAudioOps<T>.Own(owner,
            new EmbeddingLayer<T>(options.JointVocabularySize, patchDim));
        _globalPosition = UniAudioOps<T>.Own(owner,
            new LearnedPositionalEmbeddingLayer<T>(options.MaxContextPatches, options.LLMDim));
        _global = new UniAudioTransformer<T>(engine, owner, options.LLMDim, options.NumHeads,
            options.NumEncoderLayers, options.GlobalFeedForwardDim);
        _localEmbedding = UniAudioOps<T>.Own(owner,
            new EmbeddingLayer<T>(options.JointVocabularySize, options.LLMDim));
        _localPosition = UniAudioOps<T>.Own(owner,
            new LearnedPositionalEmbeddingLayer<T>(options.NumCodebooks, options.LLMDim));
        _local = new UniAudioTransformer<T>(engine, owner, options.LLMDim, options.NumHeads,
            options.NumLLMLayers, options.LocalFeedForwardDim);
        _globalToLocal = UniAudioOps<T>.Own(owner,
            new DenseLayer<T>(options.LLMDim, identity, biasMode: BiasMode.Always));
        _head = UniAudioOps<T>.Own(owner, new BiasFreeLinearLayer<T>(options.LLMDim, options.JointVocabularySize));
    }

    internal int GlobalLayerCount => _global.Blocks.Count;
    internal int LocalLayerCount => _local.Blocks.Count;

    internal void LoadTorchWeights(Func<string, int[], double[]> read) => LoadTorchWeights(TorchTensorSource.From(read));

    internal void LoadTorchWeights(TorchTensorSource source)
    {
        int d = _options.LLMDim, patch = d / _options.NumCodebooks;
        TorchParameters.Parameters(_globalEmbedding, source, "g_emb.weight", new[] { _options.JointVocabularySize, patch });
        TorchParameters.Parameters(_localEmbedding, source, "l_emb.weight", new[] { _options.JointVocabularySize, d });
        TorchParameters.Parameters(_globalPosition, source, "g_pos.weight", new[] { _options.MaxContextPatches, d });
        TorchParameters.Parameters(_localPosition, source, "l_pos.weight", new[] { _options.NumCodebooks, d });
        TorchParameters.LayerNorm(_engine, _global.FinalNorm, source, "g_ln.weight", "g_ln.bias");
        TorchParameters.LayerNorm(_engine, _local.FinalNorm, source, "l_ln.weight", "l_ln.bias");
        for (int i = 0; i < _global.Blocks.Count; i++) _global.Blocks[i].LoadTorchWeights(source, $"g_layers.{i}.", _options.GlobalFeedForwardDim);
        for (int i = 0; i < _local.Blocks.Count; i++) _local.Blocks[i].LoadTorchWeights(source, $"l_layers.{i}.", _options.LocalFeedForwardDim);
        TorchParameters.Linear(_engine, _globalToLocal, patch, d, source, "g2l_linear.weight", "g2l_linear.bias");
        TorchParameters.Linear(_head, d, _options.JointVocabularySize, source, "lm_head.weight");
    }

    /// <summary>Teacher-forced logits <c>[patches*n_q, vocabulary]</c> for every token in <paramref name="sequence"/>.</summary>
    internal Tensor<T> Forward(IReadOnlyList<int> sequence, int prefixTokenCount = 0)
    {
        int nq = _options.NumCodebooks;
        if (sequence.Count == 0 || sequence.Count % nq != 0)
            throw new ArgumentException($"A UniAudio sequence must contain complete {nq}-token patches.", nameof(sequence));
        int patches = sequence.Count / nq;
        var shifted = new int[sequence.Count];
        for (int q = 0; q < nq; q++) shifted[q] = _options.GlobalStartToken;
        for (int i = nq; i < shifted.Length; i++) shifted[i] = sequence[i - nq];
        var global = GlobalHidden(shifted, patches, prefixTokenCount > 0 ? prefixTokenCount / nq + 1 : 0);
        int patchDim = _options.LLMDim / nq;
        var contexts = _globalToLocal.Forward(_engine.Reshape(global, new[] { patches * nq, patchDim }));

        var outputs = new List<Tensor<T>>(patches);
        for (int patch = 0; patch < patches; patch++)
        {
            var localIds = new int[nq];
            localIds[0] = _options.LocalStartToken;
            for (int q = 1; q < nq; q++) localIds[q] = sequence[patch * nq + q - 1];
            var local = _localPosition.Forward(_localEmbedding.Forward(UniAudioOps<T>.Ids(localIds)));
            local = _engine.TensorAdd(local, _engine.TensorSlice(contexts,
                new[] { patch * nq, 0 }, new[] { nq, _options.LLMDim }));
            outputs.Add(_head.Forward(_local.Forward(local)));
        }
        return outputs.Count == 1 ? outputs[0] : _engine.TensorConcatenate(outputs.ToArray(), 0);
    }

    /// <summary>Generates codec frames after a complete condition sequence ending in <c>&lt;audio_start&gt;</c>.</summary>
    internal int[] Generate(IReadOnlyList<int> prefix, int minimumFrames, int maximumFrames, Random random)
    {
        int nq = _options.NumCodebooks;
        if (prefix.Count == 0 || prefix.Count % nq != 0)
            throw new ArgumentException($"A UniAudio prefix must contain complete {nq}-token patches.", nameof(prefix));
        var sequence = new List<int>(prefix);
        var generated = new List<int>();
        for (int frame = 0; frame < maximumFrames; frame++)
        {
            var context = NextGlobalContext(sequence, prefix.Count / nq + 1);
            var patch = new List<int>(nq);
            bool end = false;
            for (int q = 0; q < nq; q++)
            {
                var input = new int[q + 1];
                input[0] = _options.LocalStartToken;
                for (int j = 1; j < input.Length; j++) input[j] = patch[j - 1];
                var local = _localPosition.Forward(_localEmbedding.Forward(UniAudioOps<T>.Ids(input)));
                local = _engine.TensorAdd(local, _engine.TensorSlice(context,
                    new[] { 0, 0 }, new[] { q + 1, _options.LLMDim }));
                var hidden = _local.Forward(local);
                var last = _engine.TensorSlice(hidden,
                    new[] { q, 0 }, new[] { 1, _options.LLMDim });
                int token = Sample(_head.Forward(last), q, frame >= minimumFrames, random);
                if (token == _options.AudioEndToken)
                {
                    end = true;
                    break;
                }
                patch.Add(token);
            }
            if (end && frame >= minimumFrames) break;
            if (patch.Count != nq) continue;
            sequence.AddRange(patch);
            generated.AddRange(patch);
        }
        return generated.ToArray();
    }

    private Tensor<T> GlobalHidden(IReadOnlyList<int> flatPatches, int patches, int prefixPatches = 0)
    {
        int patchDim = _options.LLMDim / _options.NumCodebooks;
        var embedded = _globalEmbedding.Forward(UniAudioOps<T>.Ids(flatPatches));
        var rows = _engine.Reshape(embedded, new[] { patches, _options.NumCodebooks * patchDim });
        var mask = UniAudioOps<T>.CausalMask(patches);
        if (_options.PrefixLanguageModel)
            for (int i = 0; i < Math.Min(patches, prefixPatches); i++)
                for (int j = i + 1; j < Math.Min(patches, prefixPatches); j++) mask[i, j] = UniAudioOps<T>.NumOps.Zero;
        return _global.Forward(_globalPosition.Forward(rows), mask);
    }

    private Tensor<T> NextGlobalContext(IReadOnlyList<int> sequence, int prefixPatches)
    {
        int nq = _options.NumCodebooks, existing = sequence.Count / nq;
        if (existing + 1 > _options.MaxContextPatches)
            throw new InvalidOperationException("UniAudio's global context window is exhausted.");
        var input = new int[(existing + 1) * nq];
        for (int q = 0; q < nq; q++) input[q] = _options.GlobalStartToken;
        for (int i = 0; i < sequence.Count; i++) input[nq + i] = sequence[i];
        var global = GlobalHidden(input, existing + 1, prefixPatches);
        var last = _engine.TensorSlice(global,
            new[] { existing, 0 }, new[] { 1, _options.LLMDim });
        int patchDim = _options.LLMDim / nq;
        return _globalToLocal.Forward(_engine.Reshape(last, new[] { nq, patchDim }));
    }

    private int Sample(Tensor<T> logits, int codebook, bool allowEnd, Random random)
    {
        int first = _options.AudioTokenOffset + codebook * _options.CodebookSize;
        var candidates = new List<(int Id, double Logit)>(_options.CodebookSize + 1);
        for (int id = first; id < first + _options.CodebookSize; id++)
            candidates.Add((id, UniAudioOps<T>.NumOps.ToDouble(logits[0, id])));
        if (allowEnd) candidates.Add((_options.AudioEndToken,
            UniAudioOps<T>.NumOps.ToDouble(logits[0, _options.AudioEndToken])));
        candidates.Sort((a, b) => b.Logit.CompareTo(a.Logit));
        int keep = Math.Min(Math.Max(1, _options.TopK), candidates.Count);
        candidates.RemoveRange(keep, candidates.Count - keep);
        double maximum = candidates[0].Logit, total = 0;
        var weights = new double[candidates.Count];
        for (int i = 0; i < candidates.Count; i++)
        {
            weights[i] = Math.Exp((candidates[i].Logit - maximum) / _options.Temperature);
            total += weights[i];
        }
        double draw = random.NextDouble() * total;
        for (int i = 0; i < candidates.Count; i++)
        {
            draw -= weights[i];
            if (draw <= 0) return candidates[i].Id;
        }
        return candidates[^1].Id;
    }
}

/// <summary>The 16 kHz non-causal SEANet codec, selecting three trained RVQ stages at 1.5 kbps.</summary>
internal sealed class UniAudioCodec<T>
{
    private readonly IEngine _engine;
    private readonly UniAudioOptions _options;
    private readonly SeanetEncoder<T> _encoder;
    private readonly SeanetDecoder<T> _decoder;
    private readonly ResidualVectorQuantizerLayer<T> _quantizer;

    internal UniAudioCodec(IEngine engine, List<LayerBase<T>> owner, UniAudioOptions options)
    {
        _engine = engine;
        _options = options;
        var config = new SeanetConfig
        {
            Channels = 1,
            Dimension = options.CodecDimension,
            Filters = options.CodecFilters,
            ResidualLayers = 1,
            Ratios = (int[])options.CodecRatios.Clone(),
            Normalization = ConvolutionNormalization.Weight,
            KernelSize = 7,
            LastKernelSize = 7,
            ResidualKernelSizes = new[] { 3, 1 },
            DilationBase = 2,
            Causal = false,
            Reflect = true,
            TrueSkip = false,
            Compress = 2,
            LstmLayers = 2,
            TimeGroupNorm = false,
        };
        _encoder = new SeanetEncoder<T>(engine, config, owner);
        _quantizer = UniAudioOps<T>.Own(owner, new ResidualVectorQuantizerLayer<T>(
            options.CodecDimension, options.CodecTotalQuantizers, options.CodebookSize,
            options.CodecCodebookDecay, options.CodecKMeansIterations, options.CodecDeadCodeThreshold,
            commitmentAsMean: true, expireCodesAfterEma: true));
        _decoder = new SeanetDecoder<T>(engine, config, owner);
    }

    internal int HopLength => _options.CodecRatios.Aggregate(1, (product, ratio) => product * ratio);
    internal ResidualVectorQuantizerLayer<T> Quantizer => _quantizer;

    // SEANet's loader still takes each tensor's values.
    internal void LoadTorchWeights(TorchTensorSource source, string prefix = "") =>
        SeanetTorchWeights.Load(source.AsFunction(), _encoder, _decoder, _quantizer, _options.CodecTotalQuantizers,
            _options.CodebookSize, _options.CodecDimension, prefix);

    internal int[,] Encode(Tensor<T> audio)
    {
        var flat = _engine.Reshape(audio, new[] { audio.Length });
        if (flat.Length == 0) throw new ArgumentException("The codec requires a nonempty recording.", nameof(audio));
        var latent = _encoder.Forward(_engine.Reshape(flat, new[] { 1, 1, flat.Length }));
        if (!_quantizer.CodebooksInitialized) _quantizer.InitializeCodebooks(latent);
        return _quantizer.Encode(latent, _options.NumCodebooks);
    }

    internal Tensor<T> Decode(int[,] codes)
    {
        if (codes.GetLength(0) != _options.NumCodebooks)
            throw new ArgumentException($"UniAudio decodes {_options.NumCodebooks} RVQ streams.", nameof(codes));
        var audio = _decoder.Forward(_quantizer.Decode(codes));
        return _engine.Reshape(audio, new[] { audio.Length });
    }

    internal (Tensor<T> Waveform, Tensor<T> Commitment) TrainingForward(Tensor<T> audio, int quantizers)
    {
        if (audio.Length == 0 || audio.Length % HopLength != 0)
            throw new ArgumentException("Codec training uses a nonempty whole number of codec frames.", nameof(audio));
        if (quantizers <= 0 || quantizers > _options.CodecTotalQuantizers)
            throw new ArgumentOutOfRangeException(nameof(quantizers));
        var latent = _encoder.Forward(_engine.Reshape(audio, new[] { 1, 1, audio.Length }));
        int previous = _quantizer.ActiveQuantizers;
        try
        {
            _quantizer.ActiveQuantizers = quantizers;
            var reconstructed = _decoder.Forward(_quantizer.Forward(latent));
            return (_engine.Reshape(reconstructed, new[] { reconstructed.Length }),
                _quantizer.CommitmentLoss ?? new Tensor<T>(new[] { 1 }));
        }
        finally { _quantizer.ActiveQuantizers = previous; }
    }

    internal int[] Flatten(int[,] codes)
    {
        int frames = codes.GetLength(1);
        var flat = new int[frames * _options.NumCodebooks];
        for (int frame = 0; frame < frames; frame++)
            for (int q = 0; q < _options.NumCodebooks; q++)
                flat[frame * _options.NumCodebooks + q] = codes[q, frame] + q * _options.CodebookSize;
        return flat;
    }

    internal int[,] Unflatten(IReadOnlyList<int> tokens)
    {
        if (tokens.Count == 0 || tokens.Count % _options.NumCodebooks != 0)
            throw new ArgumentException($"Codec tokens must contain complete {_options.NumCodebooks}-token frames.", nameof(tokens));
        int frames = tokens.Count / _options.NumCodebooks;
        var codes = new int[_options.NumCodebooks, frames];
        for (int frame = 0; frame < frames; frame++)
            for (int q = 0; q < _options.NumCodebooks; q++)
            {
                int raw = tokens[frame * _options.NumCodebooks + q] - q * _options.CodebookSize;
                codes[q, frame] = MathPolyfill.Clamp(raw, 0, _options.CodebookSize - 1);
            }
        return codes;
    }
}

/// <summary>The optional native CMUDict/ARPAbet frontend. Released checkpoints use UniAudioReleasedTokenizer's alignment inventory.</summary>
internal static class UniAudioPhoneTokenizer
{
    private static readonly string[] Symbols =
    {
        "|", "AH0", "T", "N", "D", "S", "R", "L", "IH1", "DH", "M", "K", "Z", "EH1", "AE1", "IH0",
        "AH1", "W", "HH", ",", "ER0", "P", "IY1", "V", "F", "B", "UW1", "AA1", "AY1", "AO1", "EY1", ".",
        "IY0", "OW1", "NG", "G", "SH", "Y", "AW1", "CH", "ER1", "TH", "UH1", "JH", "'", "OW0", "?", "EH2",
        "!", "IH2", "OY1", "EY2", "AY2", "EH0", "AE2", "UW0", "AA2", "-", "OW2", "AO2", "AH2", "AE0", "ZH",
        "AA0", "UW2", "IY2", "AO0", "AY0", "AW2", "EY0", "UH2", "ER2", "AW0", "...", "UH0", "OY2", "OY0",
        "..", ";", ":",
    };

    private static readonly Dictionary<string, int> Ids = Symbols
        .Select((symbol, id) => (symbol, id)).ToDictionary(pair => pair.symbol, pair => pair.id, StringComparer.Ordinal);

    internal static int[] Tokenize(string text, int maximum)
    {
        var result = new List<int>();
        foreach (string phone in G2pEn.Default.Word(text))
        {
            string symbol = phone == " " ? "|" : phone;
            if (Ids.TryGetValue(symbol, out int id)) result.Add(id);
            if (result.Count == maximum) break;
        }
        if (result.Count == 0) result.Add(0);
        return result.ToArray();
    }
}
