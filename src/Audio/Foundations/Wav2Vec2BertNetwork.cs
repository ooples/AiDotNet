using AiDotNet.Audio.Features;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Audio.Foundations;

/// <summary>The shape of a Wav2Vec2-BERT encoder; the defaults are w2v-BERT 2.0's (<c>facebook/w2v-bert-2.0</c>).</summary>
internal sealed record Wav2Vec2BertConfiguration
{
    /// <summary>The stacked filterbank width (two 80-bin frames, 160).</summary>
    public int FeatureInputDim { get; init; } = 160;

    public int HiddenSize { get; init; } = 1024;

    public int Layers { get; init; } = 24;

    public int Heads { get; init; } = 16;

    public int IntermediateSize { get; init; } = 4096;

    /// <summary>The convolution module's causal depthwise kernel (31).</summary>
    public int DepthwiseKernel { get; init; } = 31;

    /// <summary>How far back a relative-key distance is resolved (64) before it is clamped.</summary>
    public int LeftMaxPositions { get; init; } = 64;

    /// <summary>How far ahead a relative-key distance is resolved (8) before it is clamped.</summary>
    public int RightMaxPositions { get; init; } = 8;

    public double LayerNormEpsilon { get; init; } = 1e-5;

    /// <summary>The filterbank's mel bins (80).</summary>
    public int MelBins { get; init; } = 80;

    /// <summary>Filterbank frames stacked into one input row (2).</summary>
    public int Stride { get; init; } = 2;
}

/// <summary>
/// The input features w2v-BERT 2.0 reads (transformers' <c>SeamlessM4TFeatureExtractor</c>): Kaldi log mel filterbanks of
/// the 16 kHz waveform scaled to 16-bit range (80 bins, Povey window, 25 ms frames every 10 ms, pre-emphasis 0.97, DC
/// removal, snipped edges, a 20 Hz floor), each bin normalized over time to zero mean and unit (unbiased) variance, an odd
/// last frame dropped and every two frames stacked into one 160-wide row.
/// </summary>
/// <remarks>The reference casts the waveform to float32 before the filterbank; this works in the model's precision.</remarks>
internal sealed class Wav2Vec2BertFeatures<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly KaldiFbank<T> _fbank;
    private readonly int _stride;

    public Wav2Vec2BertFeatures(IEngine engine, Wav2Vec2BertConfiguration configuration)
    {
        _engine = engine;
        _stride = configuration.Stride;
        _fbank = new KaldiFbank<T>(engine, 16000, configuration.MelBins, WindowType.Povey);
    }

    /// <summary>The rows the features of <paramref name="samples"/> samples have.</summary>
    public int Rows(int samples) => _fbank.Frames(samples) / _stride;

    /// <summary>The features <c>[rows, mels · stride]</c> of a 16 kHz waveform.</summary>
    public Tensor<T> Forward(IReadOnlyList<double> audio)
    {
        var scaled = new Tensor<T>(new[] { audio.Count });
        for (int i = 0; i < audio.Count; i++) scaled[i] = NumOps.FromDouble(audio[i] * 32768.0);
        var fbank = _fbank.Forward(scaled);                                                                  // [F, mels]
        int frames = fbank.Shape[0], mels = fbank.Shape[1];
        if (frames < 2) throw new ArgumentException("The waveform needs at least two filterbank frames.", nameof(audio));
        var mean = _engine.ReduceMean(fbank, new[] { 0 }, keepDims: true);
        var centered = _engine.TensorSubtract(fbank, _engine.TensorTile(mean, new[] { frames, 1 }));
        var variance = _engine.TensorDivideScalar(_engine.ReduceSum(_engine.TensorSquare(centered), new[] { 0 }, keepDims: true),
            NumOps.FromDouble(frames - 1));
        var deviation = _engine.TensorSqrt(_engine.TensorAddScalar(variance, NumOps.FromDouble(1e-7)));
        var normalized = _engine.TensorDivide(centered, _engine.TensorTile(deviation, new[] { frames, 1 }));
        int rows = frames / _stride;
        if (rows * _stride != frames) normalized = _engine.TensorSlice(normalized, new[] { 0, 0 }, new[] { rows * _stride, mels });
        return _engine.Reshape(normalized, new[] { rows, mels * _stride });
    }
}

/// <summary>
/// Wav2Vec2-BERT (Barrault et al. 2023, "Seamless: Multilingual Expressive and Streaming Speech Translation", the w2v-BERT
/// 2.0 encoder) as transformers' <c>Wav2Vec2BertModel</c> computes its hidden states in evaluation mode: a layer norm and
/// linear projection of the stacked filterbanks, then Conformer layers. Each layer is a half-step feed-forward (layer
/// norm, linear, swish, linear, ×½, residual), self-attention with relative-key position scores (a learned embedding of
/// the key − query distance clamped to [−64, 8], dotted with the query and scaled like the content scores), the
/// convolution module (layer norm, pointwise convolution to twice the width, GLU, a causal depthwise convolution, layer
/// norm, swish, pointwise convolution, residual), a second half-step feed-forward and a final layer norm.
/// </summary>
/// <remarks>For frozen use (X-codec2's semantic encoder): no masking, dropout or LayerDrop. Only the first
/// <c>layerCount</c> layers are built, since X-codec2 reads <c>hidden_states[16]</c>. Its layers are added to the owner's
/// list.</remarks>
internal sealed class Wav2Vec2BertNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    /// <param name="layerCount">The Conformer layers to build (all of them when null).</param>
    public Wav2Vec2BertNetwork(IEngine engine, List<LayerBase<T>> layers, Wav2Vec2BertConfiguration configuration, int? layerCount = null)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.HiddenSize % c.Heads != 0) throw new ArgumentException($"The width {c.HiddenSize} must divide into {c.Heads} heads.");
        int count = layerCount ?? c.Layers;
        if (count < 0 || count > c.Layers) throw new ArgumentOutOfRangeException(nameof(layerCount));
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        DenseLayer<T> Linear(int outputs) => Own(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        LayerNormalizationLayer<T> Norm(int width) => Own(new LayerNormalizationLayer<T>(width, c.LayerNormEpsilon));
        int h = c.HiddenSize, e = h / c.Heads;
        ProjectionNorm = Norm(c.FeatureInputDim);
        Projection = Linear(h);
        for (int i = 0; i < count; i++)
        {
            EncoderLayers.Add(new Layer(
                Norm(h), Linear(c.IntermediateSize), Linear(h),
                Norm(h), Linear(h), Linear(h), Linear(h), Linear(h),
                Own(new TiedEmbeddingLayer<T>(c.LeftMaxPositions + c.RightMaxPositions + 1, e)),
                Norm(h), Own(new BiasFreeLinearLayer<T>(h, 2 * h)),
                Own(new NormedConv1DLayer<T>(h, h, c.DepthwiseKernel, 1, 1, h, 0, false, ConvolutionNormalization.None, useBias: false)),
                Norm(h), Own(new BiasFreeLinearLayer<T>(h, h)),
                Norm(h), Linear(c.IntermediateSize), Linear(h),
                Norm(h)));
        }
    }

    /// <summary>A Conformer layer's modules.</summary>
    internal sealed record Layer(
        LayerNormalizationLayer<T> FeedForward1Norm, DenseLayer<T> FeedForward1In, DenseLayer<T> FeedForward1Out,
        LayerNormalizationLayer<T> AttentionNorm, DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Output,
        TiedEmbeddingLayer<T> Distance,
        LayerNormalizationLayer<T> ConvolutionNorm, BiasFreeLinearLayer<T> Pointwise1, NormedConv1DLayer<T> Depthwise,
        LayerNormalizationLayer<T> DepthwiseNorm, BiasFreeLinearLayer<T> Pointwise2,
        LayerNormalizationLayer<T> FeedForward2Norm, DenseLayer<T> FeedForward2In, DenseLayer<T> FeedForward2Out,
        LayerNormalizationLayer<T> FinalNorm);

    public Wav2Vec2BertConfiguration Configuration { get; }

    public LayerNormalizationLayer<T> ProjectionNorm { get; }

    public DenseLayer<T> Projection { get; }

    public List<Layer> EncoderLayers { get; } = new();

    /// <summary>The hidden states <c>[rows, hidden]</c> of features <c>[rows, 160]</c>, as transformers'
    /// <c>output_hidden_states</c> lists them: the projected features, then every built layer's output.</summary>
    public IReadOnlyList<Tensor<T>> HiddenStates(Tensor<T> features)
    {
        if (features.Rank != 2 || features.Shape[1] != Configuration.FeatureInputDim)
            throw new ArgumentException($"Expected features [rows, {Configuration.FeatureInputDim}].", nameof(features));
        var hidden = Projection.Forward(ProjectionNorm.Forward(features));
        var distances = Distances(hidden.Shape[0]);
        var states = new List<Tensor<T>> { hidden };
        foreach (var layer in EncoderLayers)
        {
            hidden = Encode(layer, hidden, distances);
            states.Add(hidden);
        }
        return states;
    }

    // Row i holds, for each key j, the index i·P + clamp(j − i, −left, right) + left into the flattened [L·P] query-distance
    // scores (P distances).
    private Tensor<int> Distances(int length)
    {
        int left = Configuration.LeftMaxPositions, right = Configuration.RightMaxPositions, p = left + right + 1;
        var index = new Tensor<int>(new[] { length * length });
        for (int i = 0; i < length; i++)
            for (int j = 0; j < length; j++)
                index[i * length + j] = i * p + Math.Min(Math.Max(j - i, -left), right) + left;
        return index;
    }

    private Tensor<T> Swish(Tensor<T> x) => _engine.TensorMultiply(x, _engine.Sigmoid(x));

    private Tensor<T> FeedForward(LayerNormalizationLayer<T> norm, DenseLayer<T> input, DenseLayer<T> output, Tensor<T> x)
        => _engine.TensorAdd(_engine.TensorMultiplyScalar(output.Forward(Swish(input.Forward(norm.Forward(x)))), NumOps.FromDouble(0.5)), x);

    private Tensor<T> Heads(DenseLayer<T> projection, Tensor<T> x)
    {
        int length = x.Shape[0], heads = Configuration.Heads, e = Configuration.HiddenSize / heads;
        return _engine.TensorPermute(_engine.Reshape(projection.Forward(x), new[] { length, heads, e }), new[] { 1, 0, 2 });
    }

    private Tensor<T> Attention(Layer l, Tensor<T> x, Tensor<int> distances)
    {
        int length = x.Shape[0], heads = Configuration.Heads, e = Configuration.HiddenSize / heads;
        int p = Configuration.LeftMaxPositions + Configuration.RightMaxPositions + 1;
        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(e));
        var q = Heads(l.Query, x);
        var k = Heads(l.Key, x);
        var v = Heads(l.Value, x);
        var content = _engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 }));                 // [H, L, L]
        // einsum("bhld,lrd->bhlr", q, E[clamp(r − l)]): every query against each of the P distance embeddings, then each
        // (query, key) pair picks its distance.
        var perDistance = _engine.Reshape(_engine.TensorMatMul(_engine.Reshape(q, new[] { heads * length, e }), _engine.TensorTranspose(l.Distance.Table)),
            new[] { heads, length * p });
        var position = _engine.Reshape(_engine.TensorIndexSelect(perDistance, distances, 1), new[] { heads, length, length });
        var scores = _engine.TensorMultiplyScalar(_engine.TensorAdd(content, position), scale);
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), v);
        return l.Output.Forward(_engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { length, heads * e }));
    }

    private Tensor<T> Convolution(Layer l, Tensor<T> x)
    {
        int length = x.Shape[0], h = Configuration.HiddenSize, kernel = Configuration.DepthwiseKernel;
        var expanded = l.Pointwise1.Forward(l.ConvolutionNorm.Forward(x));                                // [L, 2H]
        var gated = _engine.TensorMultiply(_engine.TensorSlice(expanded, new[] { 0, 0 }, new[] { length, h }),
            _engine.Sigmoid(_engine.TensorSlice(expanded, new[] { 0, h }, new[] { length, h })));
        // The causal depthwise convolution: k − 1 zeros in front.
        var padded = _engine.TensorConcatenate(new[] { new Tensor<T>(new[] { kernel - 1, h }), gated }, 0);
        var channels = _engine.Reshape(_engine.TensorTranspose(padded), new[] { 1, h, length + kernel - 1 });
        var depthwise = _engine.TensorTranspose(_engine.Reshape(l.Depthwise.Forward(channels), new[] { h, length }));
        return l.Pointwise2.Forward(Swish(l.DepthwiseNorm.Forward(depthwise)));
    }

    private Tensor<T> Encode(Layer l, Tensor<T> x, Tensor<int> distances)
    {
        x = FeedForward(l.FeedForward1Norm, l.FeedForward1In, l.FeedForward1Out, x);
        x = _engine.TensorAdd(x, Attention(l, l.AttentionNorm.Forward(x), distances));
        x = _engine.TensorAdd(x, Convolution(l, x));
        x = FeedForward(l.FeedForward2Norm, l.FeedForward2In, l.FeedForward2Out, x);
        return l.FinalNorm.Forward(x);
    }

    /// <summary>Loads a transformers <c>Wav2Vec2BertModel</c> state dict, each name under <paramref name="prefix"/>
    /// (X-codec2 keeps it under <c>semantic_model.</c>). Layers beyond those built are not read.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix = "")
    {
        var c = Configuration;
        int h = c.HiddenSize, e = h / c.Heads, f = c.IntermediateSize;
        void Lin(DenseLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, read($"{prefix}{name}.weight", new[] { outputs, inputs }),
                read($"{prefix}{name}.bias", new[] { outputs }));
        void Norm(LayerNormalizationLayer<T> layer, string name, int width)
            => TorchParameters.LayerNorm(_engine, layer, read($"{prefix}{name}.weight", new[] { width }), read($"{prefix}{name}.bias", new[] { width }));
        Norm(ProjectionNorm, "feature_projection.layer_norm", c.FeatureInputDim);
        Lin(Projection, "feature_projection.projection", c.FeatureInputDim, h);
        for (int i = 0; i < EncoderLayers.Count; i++)
        {
            var l = EncoderLayers[i];
            string p = $"encoder.layers.{i}";
            Norm(l.FeedForward1Norm, $"{p}.ffn1_layer_norm", h);
            Lin(l.FeedForward1In, $"{p}.ffn1.intermediate_dense", h, f);
            Lin(l.FeedForward1Out, $"{p}.ffn1.output_dense", f, h);
            Norm(l.AttentionNorm, $"{p}.self_attn_layer_norm", h);
            Lin(l.Query, $"{p}.self_attn.linear_q", h, h);
            Lin(l.Key, $"{p}.self_attn.linear_k", h, h);
            Lin(l.Value, $"{p}.self_attn.linear_v", h, h);
            Lin(l.Output, $"{p}.self_attn.linear_out", h, h);
            l.Distance.LoadTable(read($"{prefix}{p}.self_attn.distance_embedding.weight", new[] { c.LeftMaxPositions + c.RightMaxPositions + 1, e }));
            Norm(l.ConvolutionNorm, $"{p}.conv_module.layer_norm", h);
            TorchParameters.Linear(l.Pointwise1, h, 2 * h, read($"{prefix}{p}.conv_module.pointwise_conv1.weight", new[] { 2 * h, h, 1 }));
            l.Depthwise.LoadTorchWeights(read($"{prefix}{p}.conv_module.depthwise_conv.weight", new[] { h, 1, c.DepthwiseKernel }), null, new double[h]);
            Norm(l.DepthwiseNorm, $"{p}.conv_module.depthwise_layer_norm", h);
            TorchParameters.Linear(l.Pointwise2, h, h, read($"{prefix}{p}.conv_module.pointwise_conv2.weight", new[] { h, h, 1 }));
            Norm(l.FeedForward2Norm, $"{p}.ffn2_layer_norm", h);
            Lin(l.FeedForward2In, $"{p}.ffn2.intermediate_dense", h, f);
            Lin(l.FeedForward2Out, $"{p}.ffn2.output_dense", f, h);
            Norm(l.FinalNorm, $"{p}.final_layer_norm", h);
        }
    }
}
