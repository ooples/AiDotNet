using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Audio.Foundations;

/// <summary>The shape of a WavLM encoder; the defaults are WavLM-base-plus's (<c>microsoft/wavlm-base-plus</c>).</summary>
internal sealed record WavLmConfiguration
{
    public int[] ConvDims { get; init; } = { 512, 512, 512, 512, 512, 512, 512 };
    public int[] ConvKernels { get; init; } = { 10, 3, 3, 3, 3, 2, 2 };
    public int[] ConvStrides { get; init; } = { 5, 2, 2, 2, 2, 2, 2 };
    public int HiddenSize { get; init; } = 768;
    public int Layers { get; init; } = 12;
    public int Heads { get; init; } = 12;
    public int IntermediateSize { get; init; } = 3072;
    public int PositionalConvKernel { get; init; } = 128;
    public int PositionalConvGroups { get; init; } = 16;
    public int RelativePositionBuckets { get; init; } = 320;
    public int RelativePositionMaxDistance { get; init; } = 800;
    public double LayerNormEpsilon { get; init; } = 1e-5;
}

/// <summary>
/// WavLM (Chen et al. 2022, "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing") as
/// transformers' <c>WavLMModel</c> computes its hidden states in evaluation mode: wav2vec 2.0's convolutional feature
/// encoder (group norm after the first layer), feature projection and convolutional position embedding, then post-norm
/// Transformer layers whose attention adds a gated relative position bias — the first layer's bucketed bias table, gated
/// per head and query by <c>a (b · c − 1) + 2</c> with <c>a, b</c> sigmoids of a linear map of the head's input.
/// </summary>
/// <remarks>For frozen use (StyleTTS 2's speech language model): no masking, dropout or LayerDrop. Its layers are added
/// to the owner's list.</remarks>
internal sealed class WavLmNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public WavLmNetwork(IEngine engine, List<LayerBase<T>> layers, WavLmConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.ConvDims.Length == 0 || c.ConvDims.Length != c.ConvKernels.Length || c.ConvDims.Length != c.ConvStrides.Length)
            throw new ArgumentException("The feature encoder needs matching conv dims, kernels and strides.");
        if (c.HiddenSize % c.Heads != 0) throw new ArgumentException($"The width {c.HiddenSize} must divide into {c.Heads} heads.");
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        DenseLayer<T> Linear(int outputs) => Own(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        LayerNormalizationLayer<T> Norm(int width) => Own(new LayerNormalizationLayer<T>(width, c.LayerNormEpsilon));
        for (int i = 0; i < c.ConvDims.Length; i++)
        {
            int inputs = i == 0 ? 1 : c.ConvDims[i - 1];
            Convs.Add(Own(new BiasFreeLinearLayer<T>(c.ConvKernels[i] * inputs, c.ConvDims[i])));
        }
        FirstNorm = Own(new GroupNormalizationLayer<T>(c.ConvDims[0], c.ConvDims[0], 1e-5));
        ProjectionNorm = Norm(c.ConvDims[^1]);
        Projection = Linear(c.HiddenSize);
        PositionalConv = Own(new Wav2Vec2PositionalConvLayer<T>(c.HiddenSize, c.PositionalConvKernel, c.PositionalConvGroups));
        EncoderNorm = Norm(c.HiddenSize);
        RelativeEmbedding = Own(new TiedEmbeddingLayer<T>(c.RelativePositionBuckets, c.Heads));
        for (int i = 0; i < c.Layers; i++)
        {
            var constant = Own(new TiedEmbeddingLayer<T>(1, c.Heads));
            constant.Reinitialize(() => 1.0);
            EncoderLayers.Add(new Layer(Linear(c.HiddenSize), Linear(c.HiddenSize), Linear(c.HiddenSize), Linear(c.HiddenSize), Linear(8), constant,
                Norm(c.HiddenSize), Linear(c.IntermediateSize), Linear(c.HiddenSize), Norm(c.HiddenSize)));
        }
    }

    /// <summary>An encoder layer: attention projections, the position-bias gate, its norm, the feed-forward and the final
    /// norm.</summary>
    internal sealed record Layer(DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Output, DenseLayer<T> Gate,
        TiedEmbeddingLayer<T> GateConstant, LayerNormalizationLayer<T> AttentionNorm, DenseLayer<T> Intermediate, DenseLayer<T> FeedForwardOutput,
        LayerNormalizationLayer<T> FinalNorm);

    public WavLmConfiguration Configuration { get; }
    public List<BiasFreeLinearLayer<T>> Convs { get; } = new();
    public GroupNormalizationLayer<T> FirstNorm { get; }
    public LayerNormalizationLayer<T> ProjectionNorm { get; }
    public DenseLayer<T> Projection { get; }
    public Wav2Vec2PositionalConvLayer<T> PositionalConv { get; }
    public LayerNormalizationLayer<T> EncoderNorm { get; }
    public TiedEmbeddingLayer<T> RelativeEmbedding { get; }
    public List<Layer> EncoderLayers { get; } = new();

    /// <summary>The number of frames the feature encoder makes of <paramref name="samples"/> samples.</summary>
    public int Frames(int samples)
    {
        int length = samples;
        for (int i = 0; i < Configuration.ConvKernels.Length; i++)
            length = (length - Configuration.ConvKernels[i]) / Configuration.ConvStrides[i] + 1;
        return length;
    }

    /// <summary>The hidden states <c>[frames, hidden]</c> of a 16 kHz waveform <c>[samples]</c>: the encoder's input
    /// (after the position embedding and layer norm) and every layer's output (transformers'
    /// <c>output_hidden_states</c>).</summary>
    public IReadOnlyList<Tensor<T>> HiddenStates(Tensor<T> audio)
    {
        var c = Configuration;
        int samples = audio.Length;
        if (Frames(samples) <= 0)
            throw new ArgumentException($"{samples} samples are fewer than WavLM's receptive field.", nameof(audio));
        var h = _engine.Reshape(audio, new[] { samples, 1 });
        for (int i = 0; i < Convs.Count; i++) h = ConvBlock(i, h);
        var hidden = Projection.Forward(ProjectionNorm.Forward(h));                                            // [L, H]
        int length = hidden.Shape[0];
        var batched = _engine.Reshape(hidden, new[] { 1, length, c.HiddenSize });
        batched = _engine.TensorAdd(batched, ActivationHelper.ExactGelu(PositionalConv.Forward(batched), _engine));
        hidden = EncoderNorm.Forward(_engine.Reshape(batched, new[] { length, c.HiddenSize }));
        var states = new List<Tensor<T>> { hidden };
        var bias = PositionBias(length);
        foreach (var layer in EncoderLayers)
        {
            hidden = Encode(layer, hidden, bias);
            states.Add(hidden);
        }
        return states;
    }

    // A feature-encoder layer on [L, Cin]: the strided kernel windows through the conv's linear map, the first layer's
    // group norm (one group per channel, over time), GELU.
    private Tensor<T> ConvBlock(int index, Tensor<T> x)
    {
        var c = Configuration;
        int length = x.Shape[0], channels = x.Shape[1], kernel = c.ConvKernels[index], stride = c.ConvStrides[index];
        int frames = (length - kernel) / stride + 1;
        var indices = new int[frames * kernel];
        for (int t = 0; t < frames; t++)
            for (int k = 0; k < kernel; k++) indices[t * kernel + k] = t * stride + k;
        var windows = _engine.Reshape(_engine.TensorGather(x, new Tensor<int>(new[] { indices.Length }, new Vector<int>(indices)), axis: 0),
            new[] { frames, kernel * channels });
        var y = Convs[index].Forward(windows);
        if (index == 0)
        {
            int width = y.Shape[1];
            var normed = FirstNorm.Forward(_engine.Reshape(_engine.TensorTranspose(y), new[] { 1, width, frames, 1 }));
            y = _engine.TensorTranspose(_engine.Reshape(normed, new[] { width, frames }));
        }
        return ActivationHelper.ExactGelu(y, _engine);
    }

    // transformers WavLMAttention._relative_positions_bucket (its logarithm in single precision, as torch computes it on
    // a float32 tensor).
    private int Bucket(int relative)
    {
        int buckets = Configuration.RelativePositionBuckets / 2, maxExact = buckets / 2;
        int bucket = relative > 0 ? buckets : 0;
        int n = Math.Abs(relative);
        if (n < maxExact) return bucket + n;
        float large = (float)Math.Log((float)n / maxExact) / (float)Math.Log((double)Configuration.RelativePositionMaxDistance / maxExact)
            * (buckets - maxExact);
        return bucket + Math.Min(maxExact + (int)large, buckets - 1);
    }

    /// <summary>The relative position bias <c>[heads, L, L]</c> of key − query, from the first layer's table.</summary>
    private Tensor<T> PositionBias(int length)
    {
        var buckets = new Tensor<T>(new[] { length * length });
        for (int i = 0; i < length; i++)
            for (int j = 0; j < length; j++) buckets[i * length + j] = NumOps.FromDouble(Bucket(j - i));
        var values = RelativeEmbedding.Forward(buckets);                                                        // [L·L, heads]
        return _engine.Reshape(_engine.TensorTranspose(values), new[] { Configuration.Heads, length, length });
    }

    private Tensor<T> Heads(DenseLayer<T> projection, Tensor<T> x)
    {
        int length = x.Shape[0], heads = Configuration.Heads, e = Configuration.HiddenSize / heads;
        return _engine.TensorPermute(_engine.Reshape(projection.Forward(x), new[] { length, heads, e }), new[] { 1, 0, 2 });
    }

    private Tensor<T> Attention(Layer l, Tensor<T> x, Tensor<T> bias)
    {
        int length = x.Shape[0], heads = Configuration.Heads, e = Configuration.HiddenSize / heads;
        // The gate: from the input split into heads, a linear map to eight values summed in two groups of four,
        // a = σ(first), b = σ(second), gate = a (b · c − 1) + 2 per head and query.
        var split = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(x, new[] { length, heads, e }), new[] { 1, 0, 2 }), new[] { heads * length, e });
        var sums = _engine.Sigmoid(_engine.ReduceSum(_engine.Reshape(l.Gate.Forward(split), new[] { heads * length, 2, 4 }), new[] { 2 }, keepDims: false));
        var a = _engine.TensorSlice(sums, new[] { 0, 0 }, new[] { heads * length, 1 });
        var b = _engine.TensorSlice(sums, new[] { 0, 1 }, new[] { heads * length, 1 });
        var constant = _engine.Reshape(_engine.TensorTile(_engine.Reshape(l.GateConstant.Forward(new Tensor<T>(new[] { 1 })), new[] { heads, 1 }),
            new[] { 1, length }), new[] { heads * length, 1 });
        var gate = _engine.TensorAddScalar(_engine.TensorMultiply(a, _engine.TensorAddScalar(_engine.TensorMultiply(b, constant), NumOps.FromDouble(-1.0))),
            NumOps.FromDouble(2.0));
        var gated = _engine.TensorMultiply(_engine.TensorTile(_engine.Reshape(gate, new[] { heads, length, 1 }), new[] { 1, 1, length }), bias);
        var q = Heads(l.Query, x);
        var k = Heads(l.Key, x);
        var v = Heads(l.Value, x);
        var scores = _engine.TensorAdd(_engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(e))), gated);
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), v);
        return l.Output.Forward(_engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { length, heads * e }));
    }

    private Tensor<T> Encode(Layer l, Tensor<T> x, Tensor<T> bias)
    {
        x = l.AttentionNorm.Forward(_engine.TensorAdd(x, Attention(l, x, bias)));
        var hidden = l.FeedForwardOutput.Forward(ActivationHelper.ExactGelu(l.Intermediate.Forward(x), _engine));
        return l.FinalNorm.Forward(_engine.TensorAdd(x, hidden));
    }

    /// <summary>Loads a transformers <c>WavLMModel</c> state dict (the position convolution's weight norm under its old
    /// <c>weight_g</c>/<c>weight_v</c> names or the parametrized ones).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, Func<string, bool> has)
    {
        var c = Configuration;
        int h = c.HiddenSize, e = h / c.Heads;
        void Lin(DenseLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, read($"{name}.weight", new[] { outputs, inputs }), read($"{name}.bias", new[] { outputs }));
        void Norm(LayerNormalizationLayer<T> layer, string name, int width)
            => TorchParameters.LayerNorm(_engine, layer, read($"{name}.weight", new[] { width }), read($"{name}.bias", new[] { width }));
        for (int i = 0; i < Convs.Count; i++)
        {
            int inputs = i == 0 ? 1 : c.ConvDims[i - 1], kernel = c.ConvKernels[i], outputs = c.ConvDims[i];
            var w = read($"feature_extractor.conv_layers.{i}.conv.weight", new[] { outputs, inputs, kernel });
            // [out, in, k] → columns k·in + c of the unfolded window.
            var unfolded = new double[outputs * kernel * inputs];
            for (int o = 0; o < outputs; o++)
                for (int ch = 0; ch < inputs; ch++)
                    for (int k = 0; k < kernel; k++) unfolded[o * kernel * inputs + k * inputs + ch] = w[(o * inputs + ch) * kernel + k];
            TorchParameters.Linear(Convs[i], kernel * inputs, outputs, unfolded);
        }
        var gamma = read("feature_extractor.conv_layers.0.layer_norm.weight", new[] { c.ConvDims[0] });
        var beta = read("feature_extractor.conv_layers.0.layer_norm.bias", new[] { c.ConvDims[0] });
        var gt = FirstNorm.GetGammaTensor();
        var bt = FirstNorm.GetBetaTensor();
        for (int j = 0; j < gamma.Length; j++)
        {
            gt[j] = NumOps.FromDouble(gamma[j]);
            bt[j] = NumOps.FromDouble(beta[j]);
        }
        _engine.InvalidatePersistentTensor(gt);
        _engine.InvalidatePersistentTensor(bt);
        Norm(ProjectionNorm, "feature_projection.layer_norm", c.ConvDims[^1]);
        Lin(Projection, "feature_projection.projection", c.ConvDims[^1], h);
        int perGroup = h / c.PositionalConvGroups, kernelLength = c.PositionalConvKernel;
        const string positional = "encoder.pos_conv_embed.conv";
        bool parametrized = has($"{positional}.parametrizations.weight.original0");
        var direction = read(parametrized ? $"{positional}.parametrizations.weight.original1" : $"{positional}.weight_v", new[] { h, perGroup, kernelLength });
        var magnitude = read(parametrized ? $"{positional}.parametrizations.weight.original0" : $"{positional}.weight_g", new[] { 1, 1, kernelLength });
        var (pv, pg, pb) = PositionalConv.WeightNormParameters;
        for (int i = 0; i < direction.Length; i++) pv[i] = NumOps.FromDouble(direction[i]);
        for (int t = 0; t < kernelLength; t++) pg[t] = NumOps.FromDouble(magnitude[t]);
        var positionalBias = read($"{positional}.bias", new[] { h });
        for (int i = 0; i < h; i++) pb[i] = NumOps.FromDouble(positionalBias[i]);
        _engine.InvalidatePersistentTensor(pv);
        _engine.InvalidatePersistentTensor(pg);
        _engine.InvalidatePersistentTensor(pb);
        Norm(EncoderNorm, "encoder.layer_norm", h);
        RelativeEmbedding.LoadTable(read("encoder.layers.0.attention.rel_attn_embed.weight", new[] { c.RelativePositionBuckets, c.Heads }));
        for (int i = 0; i < EncoderLayers.Count; i++)
        {
            var l = EncoderLayers[i];
            string p = $"encoder.layers.{i}";
            Lin(l.Query, $"{p}.attention.q_proj", h, h);
            Lin(l.Key, $"{p}.attention.k_proj", h, h);
            Lin(l.Value, $"{p}.attention.v_proj", h, h);
            Lin(l.Output, $"{p}.attention.out_proj", h, h);
            Lin(l.Gate, $"{p}.attention.gru_rel_pos_linear", e, 8);
            l.GateConstant.LoadTable(read($"{p}.attention.gru_rel_pos_const", new[] { 1, c.Heads, 1, 1 }));
            Norm(l.AttentionNorm, $"{p}.layer_norm", h);
            Lin(l.Intermediate, $"{p}.feed_forward.intermediate_dense", h, c.IntermediateSize);
            Lin(l.FeedForwardOutput, $"{p}.feed_forward.output_dense", c.IntermediateSize, h);
            Norm(l.FinalNorm, $"{p}.final_layer_norm", h);
        }
    }
}
