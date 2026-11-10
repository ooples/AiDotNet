using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.TextToSpeech.VoiceCloning;

internal sealed record MetaVoiceGptConfiguration
{
    public int Dimension { get; init; } = 2048;
    public int Layers { get; init; } = 24;
    public int Heads { get; init; } = 16;
    public int Context { get; init; } = 2048;
    public int[] InputVocabularies { get; init; } = new[] { 2562 };
    public int[]? OutputVocabularies { get; init; }
    public bool Causal { get; init; } = true;
    public bool RmsNorm { get; init; } = true;
    public bool SwiGlu { get; init; } = true;
    public int SwiGluMultiple { get; init; } = 256;
    public int SpeakerDimension { get; init; } = 256;
    public double SpeakerDropout { get; init; } = .1;
    public double Dropout { get; init; }
}

/// <summary>The official MetaVoice GPT graph, with learned absolute positions (no RoPE),
/// additive speaker conditioning, tied causal embeddings or separate noncausal codebook heads.</summary>
internal sealed class MetaVoiceGpt<T>
{
    private sealed record Block(MetaVoiceGptNorm<T> AttentionNorm, MetaVoiceGptNorm<T> FeedForwardNorm,
        BiasFreeLinearLayer<T> Qkv, BiasFreeLinearLayer<T> AttentionOutput,
        BiasFreeLinearLayer<T> FeedForwardInput, BiasFreeLinearLayer<T>? Gate, BiasFreeLinearLayer<T> FeedForwardOutput);

    private readonly IEngine _engine;
    private readonly MetaVoiceGptConfiguration _config;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly List<TiedEmbeddingLayer<T>> _embeddings = new();
    private readonly TiedEmbeddingLayer<T> _positions;
    private readonly List<Block> _blocks = new();
    private readonly MetaVoiceGptNorm<T> _finalNorm;
    private readonly BiasFreeLinearLayer<T>? _speaker;
    private readonly List<BiasFreeLinearLayer<T>> _heads = new();
    private readonly Random _random;
    private readonly int _feedForward;
    private readonly IReadOnlyList<LayerBase<T>>? _restoredLayers;
    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    public MetaVoiceGpt(IEngine engine, MetaVoiceGptConfiguration config, int seed = 2294,
        IReadOnlyList<LayerBase<T>>? restoredLayers = null)
    {
        if (config.Dimension <= 0 || config.Heads <= 0 || config.Dimension % config.Heads != 0 || config.Layers <= 0 ||
            config.Context <= 0 || config.InputVocabularies.Length == 0 || config.InputVocabularies.Any(v => v <= 0) ||
            config.Causal != (config.OutputVocabularies is null) || config.SpeakerDropout < 0 || config.SpeakerDropout > 1 ||
            config.Dropout < 0 || config.Dropout >= 1 || double.IsNaN(config.Dropout) ||
            (config.OutputVocabularies is not null && (config.OutputVocabularies.Length == 0 || config.OutputVocabularies.Any(v => v <= 0))))
            throw new ArgumentException("Invalid MetaVoice GPT configuration.", nameof(config));
        if (config.SwiGlu && config.SwiGluMultiple <= 0) throw new ArgumentOutOfRangeException(nameof(config));
        _engine = engine; _config = config; _random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        _restoredLayers = restoredLayers;
        _feedForward = config.SwiGlu
            ? (((int)(2.0 * (4.0 * config.Dimension) / 3) + config.SwiGluMultiple - 1) / config.SwiGluMultiple) * config.SwiGluMultiple
            : 4 * config.Dimension;
        foreach (int size in config.InputVocabularies) _embeddings.Add(Embedding(size));
        _positions = Embedding(config.Context);
        for (int i = 0; i < config.Layers; i++)
            _blocks.Add(new Block(Norm(), Norm(), Linear(config.Dimension, 3 * config.Dimension),
                Linear(config.Dimension, config.Dimension, true), Linear(config.Dimension, _feedForward),
                config.SwiGlu ? Linear(config.Dimension, _feedForward) : null, Linear(_feedForward, config.Dimension, true)));
        _finalNorm = Norm();
        if (config.SpeakerDimension > 0) _speaker = Linear(config.SpeakerDimension, config.Dimension);
        if (config.OutputVocabularies is not null)
            foreach (int size in config.OutputVocabularies) _heads.Add(Linear(config.Dimension, size));
        if (_restoredLayers is not null && _layers.Count != _restoredLayers.Count)
            throw new InvalidDataException("The restored GPT has unconsumed layers.");
    }

    private TLayer Own<TLayer>(Func<TLayer> create, long parameters) where TLayer : LayerBase<T>
    {
        TLayer layer;
        if (_restoredLayers is null) layer = create();
        else if (_layers.Count >= _restoredLayers.Count || _restoredLayers[_layers.Count] is not TLayer restored ||
            restored.GetTrainableParametersWithoutMaterialization().Sum(tensor => (long)tensor.Length) != parameters)
            throw new InvalidDataException("The restored GPT layer differs at index " + _layers.Count + ".");
        else layer = restored;
        _layers.Add(layer);
        return layer;
    }
    private MetaVoiceGptNorm<T> Norm() => Own(() => new MetaVoiceGptNorm<T>(_config.Dimension, _config.RmsNorm), _config.Dimension);
    private TiedEmbeddingLayer<T> Embedding(int size)
    {
        var layer = Own(() => new TiedEmbeddingLayer<T>(size, _config.Dimension), (long)size * _config.Dimension);
        if (_restoredLayers is null) layer.LoadTable(TorchParameters.Gaussian<double>(_random, size * _config.Dimension, .02).ToArray());
        return layer;
    }
    private BiasFreeLinearLayer<T> Linear(int input, int output, bool residual = false)
    {
        var layer = Own(() => new BiasFreeLinearLayer<T>(input, output), (long)input * output);
        if (_restoredLayers is null) layer.SetParameters(TorchParameters.Gaussian<T>(_random, input * output, residual ? .02 / Math.Sqrt(2.0 * _config.Layers) : .02));
        return layer;
    }
    private Tensor<T> Ids(IEnumerable<int> values)
    {
        var array = values.ToArray(); var result = new Tensor<T>(new[] { array.Length });
        var operations = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < array.Length; i++) result[i] = operations.FromDouble(array[i]);
        return result;
    }

    public (Tensor<T> Hidden, Tensor<T>[] Logits) Forward(IReadOnlyList<int[]> tokens, Tensor<T>? speaker = null,
        bool lastOnly = false, bool training = false, double? speakerDropoutDraw = null,
        Func<string, int[], double[]>? dropoutMasks = null)
    {
        if (tokens.Count != _embeddings.Count || tokens.Count == 0 || tokens[0].Length == 0 || tokens[0].Length > _config.Context)
            throw new ArgumentException("MetaVoice token stream count or context length differs.", nameof(tokens));
        int time = tokens[0].Length, dimension = _config.Dimension;
        var hidden = ForwardLayer(_positions, Ids(Enumerable.Range(0, time)));
        for (int i = 0; i < tokens.Count; i++)
        {
            if (tokens[i].Length != time || tokens[i].Any(id => id < 0 || id >= _config.InputVocabularies[i]))
                throw new ArgumentException("Token stream lengths or vocabulary IDs differ.", nameof(tokens));
            hidden = _engine.TensorAdd(hidden, ForwardLayer(_embeddings[i], Ids(tokens[i])));
        }
        if (speaker is not null)
        {
            if (_speaker is null || speaker.Rank != 2 || speaker.Shape[1] != _config.SpeakerDimension ||
                (speaker.Shape[0] != 1 && speaker.Shape[0] != time)) throw new ArgumentException("Expected global or per-token speaker features.", nameof(speaker));
            var projected = ForwardLayer(_speaker, speaker);
            if (training && _config.SpeakerDropout > 0 && (speakerDropoutDraw ?? _random.NextDouble()) < _config.SpeakerDropout)
                projected = _engine.TensorMultiplyScalar(projected, MathHelper.GetNumericOperations<T>().Zero);
            if (projected.Shape[0] == 1) projected = _engine.TensorTile(projected, new[] { time, 1 });
            hidden = _engine.TensorAdd(hidden, projected);
        }
        hidden = Drop(hidden, "embedding", training, dropoutMasks);
        for (int index = 0; index < _blocks.Count; index++)
        {
            var block = _blocks[index]; string prefix = "block." + index;
            hidden = _engine.TensorAdd(hidden, Attention(block, ForwardLayer(block.AttentionNorm, hidden), prefix, training, dropoutMasks));
            var normalized = ForwardLayer(block.FeedForwardNorm, hidden);
            var value = ForwardLayer(block.FeedForwardInput, normalized);
            value = block.Gate is null ? ActivationHelper.ExactGelu(value, _engine)
                : _engine.TensorMultiply(_engine.TensorMultiply(value, _engine.Sigmoid(value)), ForwardLayer(block.Gate, normalized));
            hidden = _engine.TensorAdd(hidden, Drop(ForwardLayer(block.FeedForwardOutput, value), prefix + ".feed_forward", training, dropoutMasks));
        }
        hidden = ForwardLayer(_finalNorm, hidden);
        var output = lastOnly ? _engine.TensorSlice(hidden, new[] { time - 1, 0 }, new[] { 1, dimension }) : hidden;
        var logits = _config.OutputVocabularies is null ? _embeddings.Select(table =>
        {
            using var scope = new WeightRegistry.MaterializeScope<T>(table.GetTrainableParametersWithoutMaterialization());
            return table.Logits(output);
        }).ToArray() : _heads.Select(head => ForwardLayer(head, output)).ToArray();
        return (hidden, logits);
    }

    private static Tensor<T> ForwardLayer(LayerBase<T> layer, Tensor<T> input)
    {
        // This graph branches and reuses tied weights, so it does not execute through
        // NeuralNetworkBase's sequential streaming loop. Pin each layer explicitly.
        using var scope = new WeightRegistry.MaterializeScope<T>(layer.GetTrainableParametersWithoutMaterialization());
        return layer.Forward(input);
    }

    private Tensor<T> Attention(Block block, Tensor<T> input, string prefix, bool training, Func<string, int[], double[]>? dropoutMasks)
    {
        int time = input.Shape[0], dimension = _config.Dimension, heads = _config.Heads, width = dimension / heads;
        var qkv = ForwardLayer(block.Qkv, input);
        Tensor<T> Part(int index) => _engine.TensorPermute(_engine.Reshape(
            _engine.TensorSlice(qkv, new[] { 0, index * dimension }, new[] { time, dimension }), new[] { time, heads, width }), new[] { 1, 0, 2 });
        var score = _engine.TensorMultiplyScalar(_engine.BatchMatMul(Part(0), _engine.TensorPermute(Part(1), new[] { 0, 2, 1 })),
            MathHelper.GetNumericOperations<T>().FromDouble(1 / Math.Sqrt(width)));
        if (_config.Causal)
        {
            var mask = new Tensor<T>(new[] { heads, time, time });
            for (int h = 0; h < heads; h++) for (int i = 0; i < time; i++) for (int j = i + 1; j < time; j++)
                mask[h, i, j] = MathHelper.GetNumericOperations<T>().FromDouble(-1e9);
            score = _engine.TensorAdd(score, mask);
        }
        var attended = _engine.BatchMatMul(Drop(_engine.Softmax(score, 2), prefix + ".attention_probs", training, dropoutMasks), Part(2));
        return Drop(ForwardLayer(block.AttentionOutput, _engine.Reshape(_engine.TensorPermute(attended, new[] { 1, 0, 2 }),
            new[] { time, dimension })), prefix + ".attention_output", training, dropoutMasks);
    }

    private Tensor<T> Drop(Tensor<T> input, string name, bool training, Func<string, int[], double[]>? masks)
    {
        if (!training || _config.Dropout == 0) return input;
        var values = masks?.Invoke(name, input.Shape.ToArray());
        if (values is not null && values.Length != input.Length) throw new ArgumentException("Dropout mask dimensions differ.", nameof(masks));
        var numeric = MathHelper.GetNumericOperations<T>();
        var mask = new Tensor<T>(input.Shape.ToArray()); double keepScale = 1 / (1 - _config.Dropout);
        for (int i = 0; i < mask.Length; i++)
        {
            double value = values is null ? (_random.NextDouble() < _config.Dropout ? 0 : keepScale) : values[i];
            if (value != 0 && Math.Abs(value - keepScale) > 1e-12) throw new ArgumentException("Expected a Bernoulli dropout mask with inverse keep scaling.", nameof(masks));
            mask[i] = numeric.FromDouble(value);
        }
        return _engine.TensorMultiply(input, mask);
    }

    internal IReadOnlyList<LayerBase<T>> FineTuneLayers(int lastBlocks)
    {
        if (lastBlocks < 0 || lastBlocks > _blocks.Count) throw new ArgumentOutOfRangeException(nameof(lastBlocks));
        var selected = new List<LayerBase<T>> { _finalNorm };
        foreach (var block in _blocks.Skip(lastBlocks == 0 ? 0 : _blocks.Count - lastBlocks))
        {
            selected.AddRange(new LayerBase<T>[] { block.AttentionNorm, block.FeedForwardNorm, block.Qkv,
                block.AttentionOutput, block.FeedForwardInput, block.FeedForwardOutput });
            if (block.Gate is not null) selected.Add(block.Gate);
        }
        return selected;
    }

    public void LoadTorch(Func<string, int[], double[]> read) => LoadTorch(TorchTensorSource.From(read));

    /// <summary>Loads the actor's state straight from <paramref name="source"/> into the layers.</summary>
    internal void LoadTorch(TorchTensorSource source)
    {
        int dimension = _config.Dimension;
        for (int i = 0; i < _embeddings.Count; i++) _embeddings[i].LoadTable(source, $"transformer.wtes.{i}.weight");
        _positions.LoadTable(source, "transformer.wpe.weight");
        void LinearLoad(BiasFreeLinearLayer<T> layer, string name, int input, int output)
            => TorchParameters.Linear(layer, input, output, source, name + ".weight");
        for (int i = 0; i < _blocks.Count; i++)
        {
            var block = _blocks[i]; string prefix = "transformer.h." + i;
            block.AttentionNorm.Load(source.Read(prefix + ".ln_1.weight", new[] { dimension }));
            block.FeedForwardNorm.Load(source.Read(prefix + ".ln_2.weight", new[] { dimension }));
            LinearLoad(block.Qkv, prefix + ".attn.c_attn", dimension, 3 * dimension);
            LinearLoad(block.AttentionOutput, prefix + ".attn.c_proj", dimension, dimension);
            LinearLoad(block.FeedForwardInput, prefix + (_config.SwiGlu ? ".mlp.swiglu.w1" : ".mlp.c_fc"), dimension, _feedForward);
            if (block.Gate is not null) LinearLoad(block.Gate, prefix + ".mlp.swiglu.w3", dimension, _feedForward);
            LinearLoad(block.FeedForwardOutput, prefix + ".mlp.c_proj", _feedForward, dimension);
        }
        _finalNorm.Load(source.Read("transformer.ln_f.weight", new[] { dimension }));
        if (_speaker is not null) LinearLoad(_speaker, "speaker_cond_pos", _config.SpeakerDimension, dimension);
        if (_config.OutputVocabularies is null)
        {
            var ops = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < _embeddings.Count; i++)
            {
                // The head must equal the table it is tied to, compared in the table's own precision.
                var table = _embeddings[i].Table;
                var head = new Tensor<T>(table.Shape.ToArray());
                source.CopyTo($"lm_heads.{i}.weight", new[] { _config.InputVocabularies[i], dimension }, head);
                var stored = head.AsSpan();
                var tied = table.AsSpan();
                for (int k = 0; k < stored.Length; k++)
                    if (ops.ToDouble(stored[k]) != ops.ToDouble(tied[k]))
                        throw new InvalidDataException("MetaVoice causal output weights must be tied to their input embeddings.");
            }
        }
        else for (int i = 0; i < _heads.Count; i++) LinearLoad(_heads[i], "lm_heads." + i, dimension, _config.OutputVocabularies[i]);
    }
}
