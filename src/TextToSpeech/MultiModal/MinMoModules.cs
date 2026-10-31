using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// FunASR's Transformer encoder layer (<c>funasr/models/transformer/encoder.py EncoderLayer</c>, ESPnet's): pre-norm
/// (layer norm, epsilon 1e-12) multi-head attention with biased projections, then a pre-norm ReLU feed-forward, each added
/// to its input; optionally causal.
/// </summary>
internal sealed class FunAsrEncoderLayer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _width, _heads;

    public FunAsrEncoderLayer(IEngine engine, List<LayerBase<T>> layers, int width, int heads, int feedForward, Random initialization)
    {
        if (width % heads != 0) throw new ArgumentException("The width must divide by the heads.");
        _engine = engine;
        _width = width;
        _heads = heads;
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        DenseLayer<T> Linear(int inputs, int outputs)
        {
            var layer = Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
            double bound = 1.0 / Math.Sqrt(inputs);
            TorchParameters.Linear(engine, layer, inputs, outputs, Enumerable.Range(0, inputs * outputs).Select(_ => bound * (2 * initialization.NextDouble() - 1)).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => bound * (2 * initialization.NextDouble() - 1)).ToArray());
            return layer;
        }
        Norm1 = Add(new LayerNormalizationLayer<T>(width, 1e-12));
        Query = Linear(width, width);
        Key = Linear(width, width);
        Value = Linear(width, width);
        Output = Linear(width, width);
        Norm2 = Add(new LayerNormalizationLayer<T>(width, 1e-12));
        FeedForward1 = Linear(width, feedForward);
        FeedForward2 = Linear(feedForward, width);
        FeedForwardSize = feedForward;
    }

    public LayerNormalizationLayer<T> Norm1 { get; }
    public DenseLayer<T> Query { get; }
    public DenseLayer<T> Key { get; }
    public DenseLayer<T> Value { get; }
    public DenseLayer<T> Output { get; }
    public LayerNormalizationLayer<T> Norm2 { get; }
    public DenseLayer<T> FeedForward1 { get; }
    public DenseLayer<T> FeedForward2 { get; }
    public int FeedForwardSize { get; }

    /// <summary>The layer on <c>[L, width]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x, bool causal = false)
    {
        int length = x.Shape[0], dk = _width / _heads;
        var h = Norm1.Forward(x);
        Tensor<T> Heads(Tensor<T> t) => _engine.TensorPermute(_engine.Reshape(t, new[] { length, _heads, dk }), new[] { 1, 0, 2 });
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(Heads(Query.Forward(h)), _engine.TensorPermute(Heads(Key.Forward(h)), new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(dk)));
        if (causal && length > 1)
        {
            var mask = new Tensor<T>(new[] { 1, length, length });
            for (int i = 0; i < length; i++)
                for (int j = i + 1; j < length; j++) mask[0, i, j] = NumOps.FromDouble(-1e30);
            scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(mask, new[] { _heads, length, length }));
        }
        var weights = _engine.Softmax(scores, axis: 2);
        var context = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(weights, Heads(Value.Forward(h))), new[] { 1, 0, 2 }), new[] { length, _width });
        x = _engine.TensorAdd(x, Output.Forward(context));
        var f = FeedForward2.Forward(_engine.ReLU(FeedForward1.Forward(Norm2.Forward(x))));
        return _engine.TensorAdd(x, f);
    }

    /// <summary>Loads <c>{prefix}self_attn.linear_q/k/v/out</c>, <c>feed_forward.w_1/w_2</c>, <c>norm1</c>, <c>norm2</c>.</summary>
    public void LoadTorchWeights(Func<string, double[]?> read, string prefix)
    {
        double[] Read(string name) => read(prefix + name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{prefix}{name}'.");
        void Dense(DenseLayer<T> layer, string name, int inputs, int outputs) =>
            TorchParameters.Linear(_engine, layer, inputs, outputs, Read(name + ".weight"), Read(name + ".bias"));
        TorchParameters.LayerNorm(_engine, Norm1, Read("norm1.weight"), Read("norm1.bias"));
        Dense(Query, "self_attn.linear_q", _width, _width);
        Dense(Key, "self_attn.linear_k", _width, _width);
        Dense(Value, "self_attn.linear_v", _width, _width);
        Dense(Output, "self_attn.linear_out", _width, _width);
        TorchParameters.LayerNorm(_engine, Norm2, Read("norm2.weight"), Read("norm2.bias"));
        Dense(FeedForward1, "feed_forward.w_1", _width, FeedForwardSize);
        Dense(FeedForward2, "feed_forward.w_2", FeedForwardSize, _width);
    }
}

/// <summary>
/// MinMo's input projector (Table 1: "2 Transformer layers and 1 CNN layer for dimensional transformation and 2×
/// downsampling"), as the same team's FunASR LLM-ASR builds it (<c>funasr/models/llm_asr/adaptor.py Transformer</c>):
/// k consecutive encoder frames stacked (zero-padded to a multiple of k) — a stride-k convolution of kernel k — through
/// Linear, ReLU and Linear into the LLM's width, then Transformer layers whose feed-forward is a quarter of the width.
/// </summary>
internal sealed class MinMoInputProjector<T>
{
    private readonly IEngine _engine;

    public MinMoInputProjector(IEngine engine, List<LayerBase<T>> layers, int encoderWidth, int downsample, int feedForward, int llmWidth,
        int transformerLayers, int heads, Random initialization)
    {
        _engine = engine;
        EncoderWidth = encoderWidth;
        Downsample = downsample;
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        DenseLayer<T> Linear(int inputs, int outputs)
        {
            var layer = Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
            double bound = 1.0 / Math.Sqrt(inputs);
            TorchParameters.Linear(engine, layer, inputs, outputs, Enumerable.Range(0, inputs * outputs).Select(_ => bound * (2 * initialization.NextDouble() - 1)).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => bound * (2 * initialization.NextDouble() - 1)).ToArray());
            return layer;
        }
        Linear1 = Linear(encoderWidth * downsample, feedForward);
        Linear2 = Linear(feedForward, llmWidth);
        for (int i = 0; i < transformerLayers; i++)
            Blocks.Add(new FunAsrEncoderLayer<T>(engine, layers, llmWidth, heads, llmWidth / 4, initialization));
        FeedForward = feedForward;
        LlmWidth = llmWidth;
    }

    public int EncoderWidth { get; }
    public int Downsample { get; }
    public int FeedForward { get; }
    public int LlmWidth { get; }
    public DenseLayer<T> Linear1 { get; }
    public DenseLayer<T> Linear2 { get; }
    public List<FunAsrEncoderLayer<T>> Blocks { get; } = new();

    /// <summary>The LLM-width embeddings <c>[(T − 1)/k + 1, width]</c> of encoder states <c>[T, encoder width]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> encoded)
    {
        int length = encoded.Shape[0], chunks = (length - 1) / Downsample + 1, pad = chunks * Downsample - length;
        var x = pad > 0 ? _engine.TensorConcatenate(new[] { encoded, new Tensor<T>(new[] { pad, EncoderWidth }) }, 0) : encoded;
        x = _engine.Reshape(x, new[] { chunks, EncoderWidth * Downsample });
        x = Linear2.Forward(_engine.ReLU(Linear1.Forward(x)));
        foreach (var block in Blocks) x = block.Forward(x);
        return x;
    }

    /// <summary>Loads the adaptor's state (<c>{prefix}linear1</c>, <c>linear2</c>, <c>blocks.{i}.*</c>).</summary>
    public void LoadTorchWeights(Func<string, double[]?> read, string prefix = "audio_adaptor.")
    {
        double[] Read(string name) => read(prefix + name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{prefix}{name}'.");
        TorchParameters.Linear(_engine, Linear1, EncoderWidth * Downsample, FeedForward, Read("linear1.weight"), Read("linear1.bias"));
        TorchParameters.Linear(_engine, Linear2, FeedForward, LlmWidth, Read("linear2.weight"), Read("linear2.bias"));
        for (int i = 0; i < Blocks.Count; i++) Blocks[i].LoadTorchWeights(read, $"{prefix}blocks.{i}.");
    }
}

/// <summary>
/// MinMo's full duplex predictor (§3.1, Table 1: one Transformer layer and a linear softmax, 18M parameters, randomly
/// initialized): the LLM's last hidden states projected to the predictor's width, one causal Transformer layer, and a
/// linear map to the <see cref="MinMoDuplexAction"/> logits at every position.
/// </summary>
internal sealed class MinMoDuplexPredictor<T>
{
    private readonly IEngine _engine;

    public MinMoDuplexPredictor(IEngine engine, List<LayerBase<T>> layers, int llmWidth, int width, int heads, int feedForward, Random initialization)
    {
        _engine = engine;
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        DenseLayer<T> Linear(int inputs, int outputs)
        {
            var layer = Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
            double bound = 1.0 / Math.Sqrt(inputs);
            TorchParameters.Linear(engine, layer, inputs, outputs, Enumerable.Range(0, inputs * outputs).Select(_ => bound * (2 * initialization.NextDouble() - 1)).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => bound * (2 * initialization.NextDouble() - 1)).ToArray());
            return layer;
        }
        Input = Linear(llmWidth, width);
        Layer = new FunAsrEncoderLayer<T>(engine, layers, width, heads, feedForward, initialization);
        Head = Linear(width, Enum.GetValues(typeof(MinMoDuplexAction)).Length);
    }

    public DenseLayer<T> Input { get; }
    public FunAsrEncoderLayer<T> Layer { get; }
    public DenseLayer<T> Head { get; }

    /// <summary>The action logits <c>[L, actions]</c> of hidden states <c>[L, LLM width]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> hidden) => Head.Forward(Layer.Forward(Input.Forward(hidden), causal: true));
}
