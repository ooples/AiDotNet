using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Audio.Foundations;

/// <summary>The sizes and switches of a HuBERT network (fairseq's <c>HubertConfig</c>; defaults are HuBERT-Large).</summary>
internal sealed record HubertConfiguration
{
    public int[] ConvDims { get; init; } = { 512, 512, 512, 512, 512, 512, 512 };
    public int[] ConvKernels { get; init; } = { 10, 3, 3, 3, 3, 2, 2 };
    public int[] ConvStrides { get; init; } = { 5, 2, 2, 2, 2, 2, 2 };

    /// <summary>Layer norm on every feature-encoder layer (Large, <c>extractor_mode="layer_norm"</c>) instead of group norm
    /// on the first (Base).</summary>
    public bool LayerNormFeatureEncoder { get; init; } = true;

    public int HiddenSize { get; init; } = 1024;

    /// <summary>The encoder layers built (the layers above a read layer can be left out).</summary>
    public int Layers { get; init; } = 24;

    public int Heads { get; init; } = 16;
    public int IntermediateSize { get; init; } = 4096;
    public int PositionalConvKernel { get; init; } = 128;
    public int PositionalConvGroups { get; init; } = 16;

    /// <summary>Pre-norm encoder layers (Large, <c>layer_norm_first</c>) instead of post-norm (Base).</summary>
    public bool LayerNormFirst { get; init; } = true;

    public double LayerNormEpsilon { get; init; } = 1e-5;
}

/// <summary>
/// HuBERT (Hsu et al., 2021) as a feature extractor (fairseq's and torchaudio's <c>extract_features</c>): the
/// convolutional feature encoder over the raw 16 kHz waveform (bias-free convolutions, layer or group norm, GELU), the
/// feature projection, the weight-normalized grouped convolutional position embedding, and the Transformer encoder;
/// <see cref="Forward"/> returns the hidden state after every layer, as models that read one of HuBERT's middle layers
/// use it (Vevo reads the 18th). Its layers are added to the owner's list.
/// </summary>
/// <remarks>The network is inference-only: the models that use it keep HuBERT frozen, so it has no masking, LayerDrop or
/// dropout.</remarks>
internal sealed class HubertNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record ConvLayer(BiasFreeLinearLayer<T> Conv, LayerBase<T>? Norm);

    internal sealed record EncoderLayer(DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Output,
        LayerNormalizationLayer<T> AttentionNorm, DenseLayer<T> Intermediate, DenseLayer<T> FeedForwardOutput, LayerNormalizationLayer<T> FinalNorm);

    public HubertNetwork(IEngine engine, List<LayerBase<T>> layers, HubertConfiguration configuration)
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
            var conv = Own(new BiasFreeLinearLayer<T>(c.ConvKernels[i] * inputs, c.ConvDims[i]));
            LayerBase<T>? norm = c.LayerNormFeatureEncoder
                ? Own(new LayerNormalizationLayer<T>(c.ConvDims[i], 1e-5))
                : i == 0 ? Own(new GroupNormalizationLayer<T>(c.ConvDims[0], c.ConvDims[0], 1e-5)) : null;
            ConvLayers.Add(new ConvLayer(conv, norm));
        }
        ProjectionNorm = Norm(c.ConvDims[^1]);
        Projection = Linear(c.HiddenSize);
        PositionalConv = Own(new Wav2Vec2PositionalConvLayer<T>(c.HiddenSize, c.PositionalConvKernel, c.PositionalConvGroups));
        EncoderNorm = Norm(c.HiddenSize);
        for (int i = 0; i < c.Layers; i++)
            EncoderLayers.Add(new EncoderLayer(Linear(c.HiddenSize), Linear(c.HiddenSize), Linear(c.HiddenSize), Linear(c.HiddenSize),
                Norm(c.HiddenSize), Linear(c.IntermediateSize), Linear(c.HiddenSize), Norm(c.HiddenSize)));
    }

    public HubertConfiguration Configuration { get; }
    public List<ConvLayer> ConvLayers { get; } = new();
    public LayerNormalizationLayer<T> ProjectionNorm { get; }
    public DenseLayer<T> Projection { get; }
    public Wav2Vec2PositionalConvLayer<T> PositionalConv { get; }
    public LayerNormalizationLayer<T> EncoderNorm { get; }
    public List<EncoderLayer> EncoderLayers { get; } = new();

    /// <summary>The number of frames the feature encoder makes of <paramref name="samples"/> samples.</summary>
    public int Frames(int samples)
    {
        int length = samples;
        for (int i = 0; i < Configuration.ConvKernels.Length; i++)
            length = (length - Configuration.ConvKernels[i]) / Configuration.ConvStrides[i] + 1;
        return length;
    }

    /// <summary>
    /// The hidden states <c>[frames, hidden]</c> of a waveform <c>[samples]</c>: after the position embedding (index 0)
    /// and after each of the first <paramref name="layers"/> encoder layers (torchaudio's
    /// <c>extract_features(num_layers)</c>; with pre-norm layers the final norm is not applied).
    /// </summary>
    public IReadOnlyList<Tensor<T>> Forward(Tensor<T> audio, int layers)
    {
        var c = Configuration;
        if (layers < 0 || layers > EncoderLayers.Count) throw new ArgumentOutOfRangeException(nameof(layers), $"The network holds {EncoderLayers.Count} layers.");
        int samples = audio.Length;
        if (Frames(samples) <= 0) throw new ArgumentException($"{samples} samples are fewer than the feature encoder's receptive field.", nameof(audio));
        var h = _engine.Reshape(audio, new[] { samples, 1 });                                          // [S, 1]
        for (int i = 0; i < ConvLayers.Count; i++) h = ConvBlock(ConvLayers[i], h, c.ConvKernels[i], c.ConvStrides[i]);
        var hidden = Projection.Forward(ProjectionNorm.Forward(h));
        hidden = _engine.TensorAdd(hidden, ActivationHelper.ExactGelu(Positional(hidden), _engine));
        if (!c.LayerNormFirst) hidden = EncoderNorm.Forward(hidden);
        var states = new List<Tensor<T>> { hidden };
        for (int i = 0; i < layers; i++)
        {
            hidden = Encode(EncoderLayers[i], hidden);
            states.Add(hidden);
        }
        return states;
    }

    // The position convolution reads [B, L, H].
    private Tensor<T> Positional(Tensor<T> x) =>
        _engine.Reshape(PositionalConv.Forward(_engine.Reshape(x, new[] { 1, x.Shape[0], x.Shape[1] })), x.Shape.ToArray());

    // A feature-encoder layer on [L, Cin]: the strided kernel windows through the conv's linear map, its norm and GELU.
    private Tensor<T> ConvBlock(ConvLayer layer, Tensor<T> x, int kernel, int stride)
    {
        int length = x.Shape[0], channels = x.Shape[1];
        int frames = (length - kernel) / stride + 1;
        var indices = new int[frames * kernel];
        for (int t = 0; t < frames; t++)
            for (int k = 0; k < kernel; k++) indices[t * kernel + k] = t * stride + k;
        var windows = _engine.Reshape(_engine.TensorGather(x, new Tensor<int>(new[] { indices.Length }, new Vector<int>(indices)), axis: 0),
            new[] { frames, kernel * channels });
        var y = layer.Conv.Forward(windows);
        if (layer.Norm is GroupNormalizationLayer<T> group)
        {
            int c = y.Shape[1];
            var normed = group.Forward(_engine.Reshape(_engine.TensorTranspose(y), new[] { 1, c, frames, 1 }));
            y = _engine.TensorTranspose(_engine.Reshape(normed, new[] { c, frames }));
        }
        else if (layer.Norm is not null)
        {
            y = layer.Norm.Forward(y);
        }
        return ActivationHelper.ExactGelu(y, _engine);
    }

    private Tensor<T> Attention(EncoderLayer l, Tensor<T> x)
    {
        int length = x.Shape[0], heads = Configuration.Heads, e = Configuration.HiddenSize / heads;
        Tensor<T> Heads(DenseLayer<T> projection) =>
            _engine.TensorPermute(_engine.Reshape(projection.Forward(x), new[] { length, heads, e }), new[] { 1, 0, 2 });
        var q = Heads(l.Query);
        var k = Heads(l.Key);
        var v = Heads(l.Value);
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(e)));
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), v);
        return l.Output.Forward(_engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { length, heads * e }));
    }

    private Tensor<T> FeedForward(EncoderLayer l, Tensor<T> x) =>
        l.FeedForwardOutput.Forward(ActivationHelper.ExactGelu(l.Intermediate.Forward(x), _engine));

    private Tensor<T> Encode(EncoderLayer l, Tensor<T> x)
    {
        if (Configuration.LayerNormFirst)
        {
            x = _engine.TensorAdd(x, Attention(l, l.AttentionNorm.Forward(x)));
            return _engine.TensorAdd(x, FeedForward(l, l.FinalNorm.Forward(x)));
        }
        x = l.AttentionNorm.Forward(_engine.TensorAdd(x, Attention(l, x)));
        return l.FinalNorm.Forward(_engine.TensorAdd(x, FeedForward(l, x)));
    }

    /// <summary>fairseq's initialization (as the transformers port reproduces it): Kaiming-normal feature-encoder
    /// convolutions, N(0, 0.02) encoder linears with zero biases, the projection U(±√(1/in)), the position convolution
    /// N(0, 2·√(1/(k·C))) with weight norm's gain at its norm.</summary>
    public void InitializeLikeReference(Random random)
    {
        var c = Configuration;
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        double[] Gaussian(int count, double std)
        {
            var a = new double[count];
            for (int i = 0; i < count; i++) a[i] = std * Normal();
            return a;
        }
        for (int i = 0; i < ConvLayers.Count; i++)
        {
            int inputs = (i == 0 ? 1 : c.ConvDims[i - 1]) * c.ConvKernels[i];
            TorchParameters.Linear(ConvLayers[i].Conv, inputs, c.ConvDims[i], Gaussian(c.ConvDims[i] * inputs, Math.Sqrt(2.0 / inputs)));
        }
        double bound = Math.Sqrt(1.0 / c.ConvDims[^1]);
        var weight = new double[c.HiddenSize * c.ConvDims[^1]];
        var bias = new double[c.HiddenSize];
        for (int i = 0; i < weight.Length; i++) weight[i] = (2 * random.NextDouble() - 1) * bound;
        for (int i = 0; i < bias.Length; i++) bias[i] = (2 * random.NextDouble() - 1) * bound;
        TorchParameters.Linear(_engine, Projection, c.ConvDims[^1], c.HiddenSize, weight, bias);
        int perGroup = c.HiddenSize / c.PositionalConvGroups;
        LoadPositional(Gaussian(c.HiddenSize * perGroup * c.PositionalConvKernel, 2 * Math.Sqrt(1.0 / ((double)c.PositionalConvKernel * c.HiddenSize))),
            null, new double[c.HiddenSize]);
        void Linear(DenseLayer<T> layer, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, Gaussian(outputs * inputs, 0.02), new double[outputs]);
        foreach (var l in EncoderLayers)
        {
            Linear(l.Query, c.HiddenSize, c.HiddenSize);
            Linear(l.Key, c.HiddenSize, c.HiddenSize);
            Linear(l.Value, c.HiddenSize, c.HiddenSize);
            Linear(l.Output, c.HiddenSize, c.HiddenSize);
            Linear(l.Intermediate, c.HiddenSize, c.IntermediateSize);
            Linear(l.FeedForwardOutput, c.IntermediateSize, c.HiddenSize);
        }
    }

    // v [C, C/G, K] and g [K] (null: g = ‖v‖ per tap) and the bias.
    private void LoadPositional(double[] direction, double[]? magnitude, double[] bias)
    {
        var c = Configuration;
        int perGroup = c.HiddenSize / c.PositionalConvGroups, k = c.PositionalConvKernel;
        var (v, g, b) = PositionalConv.WeightNormParameters;
        for (int i = 0; i < direction.Length; i++) v[i] = NumOps.FromDouble(direction[i]);
        for (int t = 0; t < k; t++)
        {
            double value;
            if (magnitude is not null) value = magnitude[t];
            else
            {
                double sum = 0;
                for (int o = 0; o < c.HiddenSize; o++)
                    for (int i = 0; i < perGroup; i++)
                    {
                        double d = direction[(o * perGroup + i) * k + t];
                        sum += d * d;
                    }
                value = Math.Sqrt(sum);
            }
            g[t] = NumOps.FromDouble(value);
        }
        for (int i = 0; i < bias.Length; i++) b[i] = NumOps.FromDouble(bias[i]);
        _engine.InvalidatePersistentTensor(v);
        _engine.InvalidatePersistentTensor(g);
        _engine.InvalidatePersistentTensor(b);
    }

    /// <summary>
    /// Loads transformers' <c>HubertModel</c> names (also under <c>hubert.</c>), the position convolution's weight norm as
    /// <c>weight_g</c>/<c>weight_v</c> or as parametrizations. <paramref name="take"/> returns null for a missing tensor;
    /// the masked-frame embedding and the layers past <see cref="HubertConfiguration.Layers"/> are not read.
    /// </summary>
    public void LoadTorchWeights(Func<string, double[]?> take)
    {
        var c = Configuration;
        int h = c.HiddenSize;
        double[] Read(string name) => take(name) ?? take("hubert." + name) ?? throw new InvalidDataException($"The HuBERT checkpoint has no tensor '{name}'.");
        double[] ReadEither(string name, string alternative) =>
            take(name) ?? take("hubert." + name) ?? take(alternative) ?? take("hubert." + alternative)
            ?? throw new InvalidDataException($"The HuBERT checkpoint has no tensor '{name}' or '{alternative}'.");
        void Lin(DenseLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, Read($"{name}.weight"), Read($"{name}.bias"));
        void Norm(LayerNormalizationLayer<T> layer, string name)
            => TorchParameters.LayerNorm(_engine, layer, Read($"{name}.weight"), Read($"{name}.bias"));
        for (int i = 0; i < ConvLayers.Count; i++)
        {
            int inputs = i == 0 ? 1 : c.ConvDims[i - 1], kernel = c.ConvKernels[i], outputs = c.ConvDims[i];
            var w = Read($"feature_extractor.conv_layers.{i}.conv.weight");
            // [out, in, k] → columns k·in + c of the unfolded window.
            var unfolded = new double[outputs * kernel * inputs];
            for (int o = 0; o < outputs; o++)
                for (int ch = 0; ch < inputs; ch++)
                    for (int k = 0; k < kernel; k++) unfolded[o * kernel * inputs + k * inputs + ch] = w[(o * inputs + ch) * kernel + k];
            TorchParameters.Linear(ConvLayers[i].Conv, kernel * inputs, outputs, unfolded);
            string normName = $"feature_extractor.conv_layers.{i}.layer_norm";
            if (ConvLayers[i].Norm is LayerNormalizationLayer<T> ln) Norm(ln, normName);
            else if (ConvLayers[i].Norm is GroupNormalizationLayer<T> gn)
            {
                var gamma = Read($"{normName}.weight");
                var beta = Read($"{normName}.bias");
                var gt = gn.GetGammaTensor();
                var bt = gn.GetBetaTensor();
                for (int j = 0; j < outputs; j++)
                {
                    gt[j] = NumOps.FromDouble(gamma[j]);
                    bt[j] = NumOps.FromDouble(beta[j]);
                }
                _engine.InvalidatePersistentTensor(gt);
                _engine.InvalidatePersistentTensor(bt);
            }
        }
        Norm(ProjectionNorm, "feature_projection.layer_norm");
        Lin(Projection, "feature_projection.projection", c.ConvDims[^1], h);
        LoadPositional(ReadEither("encoder.pos_conv_embed.conv.parametrizations.weight.original1", "encoder.pos_conv_embed.conv.weight_v"),
            ReadEither("encoder.pos_conv_embed.conv.parametrizations.weight.original0", "encoder.pos_conv_embed.conv.weight_g"),
            Read("encoder.pos_conv_embed.conv.bias"));
        Norm(EncoderNorm, "encoder.layer_norm");
        for (int i = 0; i < EncoderLayers.Count; i++)
        {
            var l = EncoderLayers[i];
            string p = $"encoder.layers.{i}";
            Lin(l.Query, $"{p}.attention.q_proj", h, h);
            Lin(l.Key, $"{p}.attention.k_proj", h, h);
            Lin(l.Value, $"{p}.attention.v_proj", h, h);
            Lin(l.Output, $"{p}.attention.out_proj", h, h);
            Norm(l.AttentionNorm, $"{p}.layer_norm");
            Lin(l.Intermediate, $"{p}.feed_forward.intermediate_dense", h, c.IntermediateSize);
            Lin(l.FeedForwardOutput, $"{p}.feed_forward.output_dense", c.IntermediateSize, h);
            Norm(l.FinalNorm, $"{p}.final_layer_norm");
        }
    }
}
