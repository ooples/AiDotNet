using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// The sizes and switches of a wav2vec 2.0 network (the reference's <c>Wav2Vec2Config</c>; defaults are the BASE model).
/// </summary>
internal sealed record Wav2Vec2Configuration
{
    public int[] ConvDims { get; init; } = { 512, 512, 512, 512, 512, 512, 512 };
    public int[] ConvKernels { get; init; } = { 10, 3, 3, 3, 3, 2, 2 };
    public int[] ConvStrides { get; init; } = { 5, 2, 2, 2, 2, 2, 2 };
    public bool ConvBias { get; init; }

    /// <summary>The feature encoder's normalization: group norm on the first layer (BASE) or layer norm on every layer
    /// (LARGE-LV60, "layer").</summary>
    public bool LayerNormFeatureEncoder { get; init; }

    public int HiddenSize { get; init; } = 768;
    public int Layers { get; init; } = 12;
    public int Heads { get; init; } = 12;
    public int IntermediateSize { get; init; } = 3072;
    public int PositionalConvKernel { get; init; } = 128;
    public int PositionalConvGroups { get; init; } = 16;

    /// <summary>Pre-norm encoder layers with a final norm (LARGE-LV60) instead of post-norm layers (BASE).</summary>
    public bool StableLayerNorm { get; init; }

    public double LayerNormEpsilon { get; init; } = 1e-5;
    public double FeatureProjectionDropout { get; init; }
    public double HiddenDropout { get; init; } = 0.1;
    public double AttentionDropout { get; init; } = 0.1;
    public double ActivationDropout { get; init; } = 0.1;
    public double LayerDrop { get; init; } = 0.1;

    /// <summary>SpecAugment time masking during training: the probability a frame starts a masked span.</summary>
    public double MaskTimeProbability { get; init; } = 0.05;
    public int MaskTimeLength { get; init; } = 10;
    public int MaskTimeMinMasks { get; init; } = 2;
}

/// <summary>
/// The wav2vec 2.0 network (Baevski et al., NeurIPS 2020) as fairseq and its transformers port build it: a 7-layer
/// convolutional feature encoder over the raw waveform, a feature projection (layer norm, linear), a weight-normalized
/// grouped convolutional position embedding, and a Transformer encoder (post-norm for BASE, pre-norm "stable layer norm"
/// for LARGE-LV60). In training it applies the reference's SpecAugment time masking with a learned mask embedding and
/// LayerDrop.
/// </summary>
/// <remarks>
/// Shared by <see cref="Wav2Vec2{T}"/> (representations), <see cref="AiDotNet.Audio.SpeechRecognition.Wav2Vec2Model{T}"/>
/// (CTC speech recognition) and <see cref="AiDotNet.Audio.LanguageIdentification.Wav2Vec2LanguageIdentifier{T}"/>
/// (utterance classification), which add their heads. Its layers are added to the owner's list.
/// </remarks>
internal sealed class Wav2Vec2Network<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public Wav2Vec2Network(IEngine engine, List<LayerBase<T>> layers, Wav2Vec2Configuration configuration)
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
            LayerBase<T> conv = c.ConvBias ? Linear(c.ConvDims[i]) : Own(new BiasFreeLinearLayer<T>(c.ConvKernels[i] * inputs, c.ConvDims[i]));
            LayerBase<T>? norm = c.LayerNormFeatureEncoder
                ? Own(new LayerNormalizationLayer<T>(c.ConvDims[i], 1e-5))
                : i == 0 ? Own(new GroupNormalizationLayer<T>(c.ConvDims[0], c.ConvDims[0], 1e-5)) : null;
            ConvLayers.Add(new ConvLayer(conv, norm));
        }
        ProjectionNorm = Norm(c.ConvDims[^1]);
        Projection = Linear(c.HiddenSize);
        MaskEmbedding = Own(new BiasFreeLinearLayer<T>(1, c.HiddenSize));
        PositionalConv = Own(new Wav2Vec2PositionalConvLayer<T>(c.HiddenSize, c.PositionalConvKernel, c.PositionalConvGroups));
        EncoderNorm = Norm(c.HiddenSize);
        for (int i = 0; i < c.Layers; i++)
            EncoderLayers.Add(new EncoderLayer(Linear(c.HiddenSize), Linear(c.HiddenSize), Linear(c.HiddenSize), Linear(c.HiddenSize),
                Norm(c.HiddenSize), Linear(c.IntermediateSize), Linear(c.HiddenSize), Norm(c.HiddenSize)));
    }

    /// <summary>A feature-encoder layer: the convolution (a linear map of the unfolded kernel window) and its norm.</summary>
    internal sealed record ConvLayer(LayerBase<T> Conv, LayerBase<T>? Norm);

    /// <summary>An encoder layer: attention projections, its norm, the feed-forward and the final norm.</summary>
    internal sealed record EncoderLayer(DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Output,
        LayerNormalizationLayer<T> AttentionNorm, DenseLayer<T> Intermediate, DenseLayer<T> FeedForwardOutput, LayerNormalizationLayer<T> FinalNorm);

    public Wav2Vec2Configuration Configuration { get; }
    public List<ConvLayer> ConvLayers { get; } = new();
    public LayerNormalizationLayer<T> ProjectionNorm { get; }
    public DenseLayer<T> Projection { get; }

    /// <summary>The learned embedding that replaces masked frames (a [1 → H] map of a constant one).</summary>
    public BiasFreeLinearLayer<T> MaskEmbedding { get; }

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

    // ------------------------------------------------------------------ the forward

    /// <summary>The outputs of one forward: the encoder output, the feature-encoder output after its layer norm and
    /// before it (<paramref name="RawFeatures"/>), the hidden states, and the time mask applied (null when none).</summary>
    internal sealed record Output(Tensor<T> Hidden, Tensor<T> Features, IReadOnlyList<Tensor<T>> LayerStates,
        Tensor<T> RawFeatures, bool[,]? Mask);

    /// <summary>
    /// The encoder output <c>[B, L, H]</c> of waveforms <c>[B, S]</c> (or <c>[S]</c>), the feature-encoder output
    /// <c>[B, L, C]</c> after its layer norm, and the hidden state after the position embedding and after every layer.
    /// </summary>
    /// <remarks>
    /// <para>With <paramref name="freezeFeatureEncoder"/> the convolutional feature encoder runs without gradients
    /// (fine-tuning: "the feature encoder is not trained"); <paramref name="featureGradientScale"/> scales its
    /// gradients instead (pre-training: "scale down the gradients for the feature encoder by a factor of 10").</para>
    /// <para>Time masking replaces the frames of <paramref name="mask"/> by the mask embedding; without one, training
    /// draws spans with <paramref name="maskProbability"/> (the configuration's when null).</para>
    /// </remarks>
    public Output Forward(Tensor<T> audio, bool training, Random random, bool freezeFeatureEncoder = false,
        bool[,]? mask = null, double? maskProbability = null, double featureGradientScale = 1.0)
    {
        var c = Configuration;
        // [S], [B, S], or [B, …] with every trailing axis read as samples.
        var x = audio.Rank == 1 ? _engine.Reshape(audio, new[] { 1, audio.Shape[0] })
            : audio.Rank == 2 ? audio
            : _engine.Reshape(audio, new[] { audio.Shape[0], audio.Length / audio.Shape[0] });
        int batch = x.Shape[0];
        if (Frames(x.Shape[1]) <= 0)
            throw new ArgumentException($"{x.Shape[1]} samples are fewer than the feature encoder's receptive field.", nameof(audio));
        var h = _engine.Reshape(x, new[] { batch, x.Shape[1], 1 });                                              // [B, S, 1]
        using (freezeFeatureEncoder ? new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>() : null)
            for (int i = 0; i < ConvLayers.Count; i++) h = ConvBlock(ConvLayers[i], h, c.ConvKernels[i], c.ConvStrides[i]);
        if (featureGradientScale != 1.0 && !freezeFeatureEncoder)
        {
            // GradMultiply: the value unchanged, the gradient scaled.
            var constant = new Tensor<T>(h._shape, new Vector<T>(h.ToArray()));
            h = _engine.TensorAdd(_engine.TensorMultiplyScalar(h, NumOps.FromDouble(featureGradientScale)),
                _engine.TensorMultiplyScalar(constant, NumOps.FromDouble(1.0 - featureGradientScale)));
        }
        var raw = h;
        var features = ProjectionNorm.Forward(h);
        var hidden = Drop(Dense(Projection, features), c.FeatureProjectionDropout, training, random);
        double probability = maskProbability ?? c.MaskTimeProbability;
        bool[,]? applied = mask ?? (training && probability > 0 ? DrawMask(batch, hidden.Shape[1], probability, random) : null);
        if (applied is not null) hidden = ApplyMask(hidden, applied);

        var states = new List<Tensor<T>>();
        hidden = _engine.TensorAdd(hidden, ActivationHelper.ExactGelu(PositionalConv.Forward(hidden), _engine));
        if (!c.StableLayerNorm) hidden = EncoderNorm.Forward(hidden);
        hidden = Drop(hidden, c.HiddenDropout, training, random);
        states.Add(hidden);
        foreach (var layer in EncoderLayers)
        {
            // LayerDrop: in training each layer is skipped with probability LayerDrop.
            if (!(training && random.NextDouble() < c.LayerDrop)) hidden = Encode(layer, hidden, training, random);
            states.Add(hidden);
        }
        if (c.StableLayerNorm) hidden = EncoderNorm.Forward(hidden);
        return new Output(hidden, features, states, raw, applied);
    }

    // ------------------------------------------------------------------ building blocks

    private Tensor<T> Dense(LayerBase<T> layer, Tensor<T> x)
    {
        int batch = x.Shape[0], length = x.Shape[1];
        var y = layer.Forward(_engine.Reshape(x, new[] { batch * length, x.Shape[2] }));
        return _engine.Reshape(y, new[] { batch, length, y.Shape[1] });
    }

    private Tensor<T> Drop(Tensor<T> x, double rate, bool training, Random random)
    {
        if (!training || rate <= 0) return x;
        var mask = new Tensor<T>(x._shape);
        var keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        for (int i = 0; i < mask.Length; i++) mask[i] = random.NextDouble() < rate ? NumOps.Zero : keep;
        return _engine.TensorMultiply(x, mask);
    }

    // A feature-encoder layer on [B, L, Cin]: the strided kernel windows [B, Lout, K·Cin] through the conv's linear map,
    // then its norm and GELU.
    private Tensor<T> ConvBlock(ConvLayer layer, Tensor<T> x, int kernel, int stride)
    {
        int batch = x.Shape[0], length = x.Shape[1], channels = x.Shape[2];
        int frames = (length - kernel) / stride + 1;
        var indices = new int[frames * kernel];
        for (int t = 0; t < frames; t++)
            for (int k = 0; k < kernel; k++) indices[t * kernel + k] = t * stride + k;
        var windows = _engine.Reshape(_engine.TensorGather(x, new Tensor<int>(new[] { indices.Length }, new Vector<int>(indices)), axis: 1),
            new[] { batch, frames, kernel * channels });
        var y = Dense(layer.Conv, windows);
        if (layer.Norm is GroupNormalizationLayer<T> group)
        {
            // Per channel over time; the layer reads rank 4 as [B, C, H, W].
            int c = y.Shape[2];
            var normed = group.Forward(_engine.Reshape(_engine.TensorPermute(y, new[] { 0, 2, 1 }), new[] { batch, c, frames, 1 }));
            y = _engine.TensorPermute(_engine.Reshape(normed, new[] { batch, c, frames }), new[] { 0, 2, 1 });
        }
        else if (layer.Norm is not null)
            y = layer.Norm.Forward(y);
        return ActivationHelper.ExactGelu(y, _engine);
    }

    // [B, L, H] → [B·heads, L, H/heads].
    private Tensor<T> Heads(DenseLayer<T> projection, Tensor<T> x)
    {
        int batch = x.Shape[0], length = x.Shape[1], heads = Configuration.Heads, e = Configuration.HiddenSize / heads;
        var y = _engine.Reshape(Dense(projection, x), new[] { batch, length, heads, e });
        return _engine.Reshape(_engine.TensorPermute(y, new[] { 0, 2, 1, 3 }), new[] { batch * heads, length, e });
    }

    private Tensor<T> Attention(EncoderLayer l, Tensor<T> x, bool training, Random random)
    {
        int batch = x.Shape[0], length = x.Shape[1], heads = Configuration.Heads, e = Configuration.HiddenSize / heads;
        var q = Heads(l.Query, x);
        var k = Heads(l.Key, x);
        var v = Heads(l.Value, x);
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(e)));
        var weights = Drop(_engine.Softmax(scores, axis: 2), Configuration.AttentionDropout, training, random);
        var context = _engine.Reshape(_engine.BatchMatMul(weights, v), new[] { batch, heads, length, e });
        var merged = _engine.Reshape(_engine.TensorPermute(context, new[] { 0, 2, 1, 3 }), new[] { batch, length, heads * e });
        return Dense(l.Output, merged);
    }

    private Tensor<T> FeedForward(EncoderLayer l, Tensor<T> x, bool training, Random random)
    {
        var c = Configuration;
        var hidden = Drop(ActivationHelper.ExactGelu(Dense(l.Intermediate, x), _engine), c.ActivationDropout, training, random);
        return Drop(Dense(l.FeedForwardOutput, hidden), c.HiddenDropout, training, random);
    }

    private Tensor<T> Encode(EncoderLayer l, Tensor<T> x, bool training, Random random)
    {
        var c = Configuration;
        if (c.StableLayerNorm)
        {
            x = _engine.TensorAdd(x, Drop(Attention(l, l.AttentionNorm.Forward(x), training, random), c.HiddenDropout, training, random));
            return _engine.TensorAdd(x, FeedForward(l, l.FinalNorm.Forward(x), training, random));
        }
        x = l.AttentionNorm.Forward(_engine.TensorAdd(x, Drop(Attention(l, x, training, random), c.HiddenDropout, training, random)));
        return l.FinalNorm.Forward(_engine.TensorAdd(x, FeedForward(l, x, training, random)));
    }

    /// <summary>
    /// SpecAugment time masks (transformers <c>_compute_mask_indices</c>): per utterance ⌊p·L/M + U(0, 1)⌋ spans of M
    /// frames (at least <see cref="Wav2Vec2Configuration.MaskTimeMinMasks"/>), starts drawn without replacement.
    /// </summary>
    public bool[,] DrawMask(int batch, int length, double probability, Random random)
    {
        var c = Configuration;
        int span = Math.Min(c.MaskTimeLength, length);
        var mask = new bool[batch, length];
        for (int b = 0; b < batch; b++)
        {
            int spans = (int)(probability * length / span + random.NextDouble());
            spans = Math.Max(spans, c.MaskTimeMinMasks);
            if (spans * span > length) spans = length / span;
            foreach (var start in Enumerable.Range(0, length - span + 1).OrderBy(_ => random.Next()).Take(spans))
                for (int t = start; t < start + span; t++) mask[b, t] = true;
        }
        return mask;
    }

    // The masked frames replaced by the learned mask embedding.
    private Tensor<T> ApplyMask(Tensor<T> hidden, bool[,] mask)
    {
        var c = Configuration;
        int batch = hidden.Shape[0], length = hidden.Shape[1], width = c.HiddenSize;
        if (mask.GetLength(0) != batch || mask.GetLength(1) != length)
            throw new ArgumentException($"The mask is [{mask.GetLength(0)}, {mask.GetLength(1)}]; the frames are [{batch}, {length}].");
        var keep = new Tensor<T>(new[] { batch, length, 1 });
        var masked = new Tensor<T>(new[] { batch, length, 1 });
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < length; t++)
            {
                keep[b, t, 0] = mask[b, t] ? NumOps.Zero : NumOps.One;
                masked[b, t, 0] = mask[b, t] ? NumOps.One : NumOps.Zero;
            }
        var embedding = _engine.Reshape(MaskEmbedding.Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.One }))), new[] { 1, 1, width });
        return _engine.TensorAdd(
            _engine.TensorMultiply(hidden, _engine.TensorTile(keep, new[] { 1, 1, width })),
            _engine.TensorMultiply(_engine.TensorTile(embedding, new[] { batch, length, 1 }), _engine.TensorTile(masked, new[] { 1, 1, width })));
    }

    // ------------------------------------------------------------------ initialization and loading

    /// <summary>
    /// The reference's initialization (transformers <c>Wav2Vec2PreTrainedModel._init_weights</c>): Kaiming-normal
    /// convolutions, N(0, 0.02) encoder linears with zero biases, the feature projection U(±√(1/in)), the position
    /// convolution N(0, 2·√(1/(k·C_in))) with the weight norm's magnitude matching it, unit norms, a U(0, 1) mask
    /// embedding.
    /// </summary>
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
            var w = Gaussian(c.ConvDims[i] * inputs, Math.Sqrt(2.0 / inputs));
            if (ConvLayers[i].Conv is BiasFreeLinearLayer<T> plain) TorchParameters.Linear(plain, inputs, c.ConvDims[i], w);
            else
            {
                double bound = 1.0 / Math.Sqrt(inputs);
                var b = new double[c.ConvDims[i]];
                for (int j = 0; j < b.Length; j++) b[j] = (2 * random.NextDouble() - 1) * bound;
                TorchParameters.Linear(_engine, (DenseLayer<T>)ConvLayers[i].Conv, inputs, c.ConvDims[i], w, b);
            }
        }
        void Linear(DenseLayer<T> layer, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, Gaussian(outputs * inputs, 0.02), new double[outputs]);
        // Wav2Vec2FeatureProjection: U(±√(1/in)) for the weight and the bias.
        double projectionBound = Math.Sqrt(1.0 / c.ConvDims[^1]);
        var projectionWeight = new double[c.HiddenSize * c.ConvDims[^1]];
        var projectionBias = new double[c.HiddenSize];
        for (int i = 0; i < projectionWeight.Length; i++) projectionWeight[i] = (2 * random.NextDouble() - 1) * projectionBound;
        for (int i = 0; i < projectionBias.Length; i++) projectionBias[i] = (2 * random.NextDouble() - 1) * projectionBound;
        TorchParameters.Linear(_engine, Projection, c.ConvDims[^1], c.HiddenSize, projectionWeight, projectionBias);
        var mask = new double[c.HiddenSize];
        for (int i = 0; i < mask.Length; i++) mask[i] = random.NextDouble();
        TorchParameters.Linear(MaskEmbedding, 1, c.HiddenSize, mask);
        int perGroup = c.HiddenSize / c.PositionalConvGroups;
        // Wav2Vec2PositionalConvEmbedding: N(0, 2·√(1/(k·C_in))) on the weight, then weight_norm's g = ‖v‖ per tap.
        var direction = Gaussian(c.HiddenSize * perGroup * c.PositionalConvKernel, 2 * Math.Sqrt(1.0 / (c.PositionalConvKernel * c.HiddenSize)));
        LoadPositional(direction, null, new double[c.HiddenSize]);
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

    // v [C, C/G, K] and g [K] (null: g = ‖v‖ per tap, as weight_norm initializes it) and the bias.
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

    /// <summary>Loads parameters named as transformers' <c>Wav2Vec2Model</c> state dict (prefix included by the caller's
    /// <paramref name="read"/>).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int h = c.HiddenSize;
        void Lin(DenseLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, read($"{name}.weight", new[] { outputs, inputs }), read($"{name}.bias", new[] { outputs }));
        void Norm(LayerNormalizationLayer<T> layer, string name, int width)
            => TorchParameters.LayerNorm(_engine, layer, read($"{name}.weight", new[] { width }), read($"{name}.bias", new[] { width }));
        for (int i = 0; i < ConvLayers.Count; i++)
        {
            int inputs = i == 0 ? 1 : c.ConvDims[i - 1], kernel = c.ConvKernels[i], outputs = c.ConvDims[i];
            var w = read($"feature_extractor.conv_layers.{i}.conv.weight", new[] { outputs, inputs, kernel });
            // [out, in, k] → columns k·in + c of the unfolded window.
            var unfolded = new double[outputs * kernel * inputs];
            for (int o = 0; o < outputs; o++)
                for (int ch = 0; ch < inputs; ch++)
                    for (int k = 0; k < kernel; k++) unfolded[o * kernel * inputs + k * inputs + ch] = w[(o * inputs + ch) * kernel + k];
            if (ConvLayers[i].Conv is BiasFreeLinearLayer<T> plain) TorchParameters.Linear(plain, kernel * inputs, outputs, unfolded);
            else TorchParameters.Linear(_engine, (DenseLayer<T>)ConvLayers[i].Conv, kernel * inputs, outputs, unfolded,
                read($"feature_extractor.conv_layers.{i}.conv.bias", new[] { outputs }));
            string normName = $"feature_extractor.conv_layers.{i}.layer_norm";
            if (ConvLayers[i].Norm is LayerNormalizationLayer<T> ln) Norm(ln, normName, outputs);
            else if (ConvLayers[i].Norm is GroupNormalizationLayer<T> gn)
            {
                var gamma = read($"{normName}.weight", new[] { outputs });
                var beta = read($"{normName}.bias", new[] { outputs });
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
        Norm(ProjectionNorm, "feature_projection.layer_norm", c.ConvDims[^1]);
        Lin(Projection, "feature_projection.projection", c.ConvDims[^1], h);
        TorchParameters.Linear(MaskEmbedding, 1, h, read("masked_spec_embed", new[] { h }));
        int perGroup = h / c.PositionalConvGroups;
        LoadPositional(read("encoder.pos_conv_embed.conv.parametrizations.weight.original1", new[] { h, perGroup, c.PositionalConvKernel }),
            read("encoder.pos_conv_embed.conv.parametrizations.weight.original0", new[] { 1, 1, c.PositionalConvKernel }),
            read("encoder.pos_conv_embed.conv.bias", new[] { h }));
        Norm(EncoderNorm, "encoder.layer_norm", h);
        for (int i = 0; i < EncoderLayers.Count; i++)
        {
            var l = EncoderLayers[i];
            string p = $"encoder.layers.{i}";
            Lin(l.Query, $"{p}.attention.q_proj", h, h);
            Lin(l.Key, $"{p}.attention.k_proj", h, h);
            Lin(l.Value, $"{p}.attention.v_proj", h, h);
            Lin(l.Output, $"{p}.attention.out_proj", h, h);
            Norm(l.AttentionNorm, $"{p}.layer_norm", h);
            Lin(l.Intermediate, $"{p}.feed_forward.intermediate_dense", h, c.IntermediateSize);
            Lin(l.FeedForwardOutput, $"{p}.feed_forward.output_dense", c.IntermediateSize, h);
            Norm(l.FinalNorm, $"{p}.final_layer_norm", h);
        }
    }
}
