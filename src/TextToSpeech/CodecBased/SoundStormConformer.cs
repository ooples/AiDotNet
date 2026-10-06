using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Channel layer normalization over axis 1 of <c>[batch, channels, time]</c> with a learned scale and no shift, the
/// variance clamped below rather than offset: <c>(x − mean) · max(var, ε)^-1/2 · γ</c> (SoundStorm's
/// <c>ChanLayerNorm</c>, ε = 1e-6).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LayerCategory(LayerCategory.Normalization)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 3, 5", TestConstructorArgs = "3")]
[ElementWiseShape(Note = "Normalizes across channels; the shape is carried through.")]
[AutoParameters]
internal sealed partial class ChannelLayerNormLayer<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly double _epsilon;

    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _gamma;

    public override bool SupportsTraining => true;

    public ChannelLayerNormLayer([LayerState] int channels, [LayerState] double epsilon = 1e-6)
        : base(new[] { channels }, new[] { channels })
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        _channels = channels;
        _epsilon = epsilon;
        _gamma = new Tensor<T>(new[] { channels });
        for (int c = 0; c < channels; c++) _gamma[c] = NumOps.One;
        RegisterTrainableParameter(_gamma, PersistentTensorRole.NormalizationParams);
    }

    internal Tensor<T> Gamma => _gamma;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[1] != _channels)
            throw new ArgumentException($"Expected [batch, {_channels}, time].", nameof(input));
        int batch = input.Shape[0], time = input.Shape[2];
        var mean = Engine.ReduceMean(input, new[] { 1 }, keepDims: true);
        var centred = Engine.TensorSubtract(input, Engine.TensorTile(mean, new[] { 1, _channels, 1 }));
        var variance = Engine.ReduceMean(Engine.TensorMultiply(centred, centred), new[] { 1 }, keepDims: true);
        var clamped = Engine.TensorMax(variance, NumOps.FromDouble(_epsilon));
        var inverse = Engine.TensorPow(clamped, NumOps.FromDouble(-0.5));
        var gamma = Engine.TensorTile(Engine.Reshape(_gamma, new[] { 1, _channels, 1 }), new[] { batch, 1, time });
        return Engine.TensorMultiply(Engine.TensorMultiply(centred, Engine.TensorTile(inverse, new[] { 1, _channels, 1 })), gamma);
    }

    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var invariant = System.Globalization.CultureInfo.InvariantCulture;
        metadata["Channels"] = _channels.ToString(invariant);
        metadata["Epsilon"] = _epsilon.ToString("R", invariant);
        return metadata;
    }
}

/// <summary>Loads PyTorch <c>nn.Linear</c> and <c>nn.LayerNorm</c> parameters into the repository's layers.</summary>
internal static class TorchParameters
{
    /// <summary>Loads <c>weight [out, in]</c> (and <c>bias [out]</c>) into a dense layer, materializing it first.</summary>
    public static void Linear<T>(IEngine engine, DenseLayer<T> layer, int inputs, int outputs, double[] weight, double[]? bias)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            layer.Forward(new Tensor<T>(new[] { 1, inputs }));
        var w = layer.GetWeights();                                                                       // [in, out]
        for (int o = 0; o < outputs; o++)
            for (int i = 0; i < inputs; i++) w[i, o] = ops.FromDouble(weight[o * inputs + i]);
        engine.InvalidatePersistentTensor(w);
        var b = layer.GetBiases();
        for (int o = 0; o < outputs; o++) b[o] = ops.FromDouble(bias is null ? 0.0 : bias[o]);
        engine.InvalidatePersistentTensor(b);
    }

    /// <summary>Loads <c>weight [out, in]</c> into a bias-free linear layer.</summary>
    public static void Linear<T>(BiasFreeLinearLayer<T> layer, int inputs, int outputs, double[] weight)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var transposed = new Vector<T>(inputs * outputs);
        for (int o = 0; o < outputs; o++)
            for (int i = 0; i < inputs; i++) transposed[i * outputs + o] = ops.FromDouble(weight[o * inputs + i]);
        layer.SetParameters(transposed);
    }

    /// <summary>Loads an <c>nn.LayerNorm</c>'s weight and bias.</summary>
    public static void LayerNorm<T>(IEngine engine, LayerNormalizationLayer<T> layer, double[] weight, double[] bias)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var gamma = layer.GetGammaTensor();
        var beta = layer.GetBetaTensor();
        for (int i = 0; i < weight.Length; i++)
        {
            gamma[i] = ops.FromDouble(weight[i]);
            beta[i] = ops.FromDouble(bias[i]);
        }
        engine.InvalidatePersistentTensor(gamma);
        engine.InvalidatePersistentTensor(beta);
    }
}

/// <summary>The sizes of a SoundStorm Conformer (lucidrains/soundstorm-pytorch <c>Conformer</c>, as Pheme configures it).</summary>
internal sealed record SoundStormConformerConfiguration(
    int Dim, int Layers, int Heads, int HeadDim = 64, int FeedForwardMultiplier = 4, int ConvExpansion = 2,
    int ConvKernel = 31, double Dropout = 0.0);

/// <summary>
/// A Conformer block as SoundStorm builds it (Borsos et al. 2023 after Gulati et al. 2020; lucidrains
/// <c>ConformerBlock</c>): a half-step feed-forward, self-attention with rotary position embeddings, the convolution
/// module, another half-step feed-forward — each a pre-norm residual — and a final LayerNorm.
/// </summary>
/// <remarks>
/// The attention projects to <c>heads × headDim</c> (independent of the model width) with bias-free query and
/// key/value projections and a biased output projection, and scales scores by <c>headDim^-1/2</c>. The feed-forward is
/// Linear → Swish → dropout → Linear → dropout. The convolution module is LayerNorm → pointwise convolution to twice
/// the expanded width → GLU → depthwise convolution with "same" padding → Swish → channel LayerNorm → pointwise
/// convolution → dropout.
/// </remarks>
internal sealed class SoundStormConformerBlock<T>
{
    private readonly IEngine _engine;
    private readonly SoundStormConformerConfiguration _c;
    private readonly int _inner;
    private readonly int _convInner;

    public SoundStormConformerBlock(IEngine engine, List<LayerBase<T>> layers, SoundStormConformerConfiguration c)
    {
        _engine = engine;
        _c = c;
        _inner = c.Heads * c.HeadDim;
        _convInner = c.Dim * c.ConvExpansion;
        int hidden = c.Dim * c.FeedForwardMultiplier;
        Ff1Norm = Own(layers, new LayerNormalizationLayer<T>(c.Dim));
        Ff1In = Own(layers, Dense(hidden));
        Ff1Out = Own(layers, Dense(c.Dim));
        AttentionNorm = Own(layers, new LayerNormalizationLayer<T>(c.Dim));
        ToQuery = Own(layers, new BiasFreeLinearLayer<T>(c.Dim, _inner));
        ToKeyValue = Own(layers, new BiasFreeLinearLayer<T>(c.Dim, 2 * _inner));
        ToOut = Own(layers, Dense(c.Dim));
        ConvNorm = Own(layers, new LayerNormalizationLayer<T>(c.Dim));
        ConvIn = Own(layers, new NormedConv1DLayer<T>(c.Dim, 2 * _convInner, 1, 1, 1, 1, 0, false, ConvolutionNormalization.None));
        Depthwise = Own(layers, new NormedConv1DLayer<T>(_convInner, _convInner, c.ConvKernel, 1, 1, _convInner, 0, false,
            ConvolutionNormalization.None));
        ConvChannelNorm = Own(layers, new ChannelLayerNormLayer<T>(_convInner));
        ConvOut = Own(layers, new NormedConv1DLayer<T>(_convInner, c.Dim, 1, 1, 1, 1, 0, false, ConvolutionNormalization.None));
        Ff2Norm = Own(layers, new LayerNormalizationLayer<T>(c.Dim));
        Ff2In = Own(layers, Dense(hidden));
        Ff2Out = Own(layers, Dense(c.Dim));
        PostNorm = Own(layers, new LayerNormalizationLayer<T>(c.Dim));
    }

    public LayerNormalizationLayer<T> Ff1Norm { get; }
    public DenseLayer<T> Ff1In { get; }
    public DenseLayer<T> Ff1Out { get; }
    public LayerNormalizationLayer<T> AttentionNorm { get; }
    public BiasFreeLinearLayer<T> ToQuery { get; }
    public BiasFreeLinearLayer<T> ToKeyValue { get; }
    public DenseLayer<T> ToOut { get; }
    public LayerNormalizationLayer<T> ConvNorm { get; }
    public NormedConv1DLayer<T> ConvIn { get; }
    public NormedConv1DLayer<T> Depthwise { get; }
    public ChannelLayerNormLayer<T> ConvChannelNorm { get; }
    public NormedConv1DLayer<T> ConvOut { get; }
    public LayerNormalizationLayer<T> Ff2Norm { get; }
    public DenseLayer<T> Ff2In { get; }
    public DenseLayer<T> Ff2Out { get; }
    public LayerNormalizationLayer<T> PostNorm { get; }

    private static DenseLayer<T> Dense(int outputs) =>
        new(outputs, new IdentityActivation<T>() as IActivationFunction<T>);

    private static TLayer Own<TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }

    /// <summary>One block on <c>[time, dim]</c>; <paramref name="cos"/> and <paramref name="sin"/> are the rotary
    /// tables <c>[time, headDim]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T> cos, Tensor<T> sin, bool training, Random random)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var half = ops.FromDouble(0.5);
        x = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(FeedForward(Ff1Norm.Forward(x), Ff1In, Ff1Out, training, random), half));
        x = _engine.TensorAdd(x, Attention(AttentionNorm.Forward(x), cos, sin, training, random));
        x = _engine.TensorAdd(x, Convolution(x, training, random));
        x = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(FeedForward(Ff2Norm.Forward(x), Ff2In, Ff2Out, training, random), half));
        return PostNorm.Forward(x);
    }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random random) =>
        training && _c.Dropout > 0 ? T5Seq2Seq<T>.Dropout(_engine, x, _c.Dropout, random) : x;

    private Tensor<T> FeedForward(Tensor<T> x, DenseLayer<T> first, DenseLayer<T> second, bool training, Random random)
    {
        var hidden = Drop(_engine.Swish(first.Forward(x)), training, random);
        return Drop(second.Forward(hidden), training, random);
    }

    private Tensor<T> Attention(Tensor<T> x, Tensor<T> cos, Tensor<T> sin, bool training, Random random)
    {
        int time = x.Shape[0], heads = _c.Heads, headDim = _c.HeadDim;
        Tensor<T> Heads(Tensor<T> t) =>
            _engine.TensorPermute(_engine.Reshape(t, new[] { time, heads, headDim }), new[] { 1, 0, 2 });           // [H, T, d]
        var keyValue = ToKeyValue.Forward(x);                                                                     // [T, 2·inner]
        var q = Rotate(Heads(ToQuery.Forward(x)), cos, sin);
        var k = Rotate(Heads(_engine.TensorSlice(keyValue, new[] { 0, 0 }, new[] { time, _inner })), cos, sin);
        var v = Heads(_engine.TensorSlice(keyValue, new[] { 0, _inner }, new[] { time, _inner }));
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })),
            MathHelper.GetNumericOperations<T>().FromDouble(1.0 / Math.Sqrt(headDim)));
        var weights = Drop(_engine.Softmax(scores, axis: 2), training, random);
        var context = _engine.BatchMatMul(weights, v);                                                           // [H, T, d]
        var merged = _engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { time, _inner });
        return ToOut.Forward(merged);
    }

    // apply_rotary_pos_emb: t · cos + rotate_half(t) · sin, rotate_half(t) = [−t₂, t₁] over the halves of the head.
    private Tensor<T> Rotate(Tensor<T> t, Tensor<T> cos, Tensor<T> sin)
    {
        int heads = t.Shape[0], time = t.Shape[1], d = t.Shape[2], halfDim = d / 2;
        var first = _engine.TensorSlice(t, new[] { 0, 0, 0 }, new[] { heads, time, halfDim });
        var second = _engine.TensorSlice(t, new[] { 0, 0, halfDim }, new[] { heads, time, d - halfDim });
        var rotated = _engine.TensorConcatenate(new[] { _engine.TensorNegate(second), first }, 2);
        var cosTiled = _engine.TensorTile(_engine.Reshape(cos, new[] { 1, time, d }), new[] { heads, 1, 1 });
        var sinTiled = _engine.TensorTile(_engine.Reshape(sin, new[] { 1, time, d }), new[] { heads, 1, 1 });
        return _engine.TensorAdd(_engine.TensorMultiply(t, cosTiled), _engine.TensorMultiply(rotated, sinTiled));
    }

    private Tensor<T> Convolution(Tensor<T> x, bool training, Random random)
    {
        int time = x.Shape[0], k = _c.ConvKernel;
        var normed = ConvNorm.Forward(x);
        var channels = _engine.Reshape(_engine.TensorTranspose(normed), new[] { 1, _c.Dim, time });                // [1, C, T]
        var expanded = ConvIn.Forward(channels);                                                                  // [1, 2I, T]
        var output = _engine.TensorSlice(expanded, new[] { 0, 0, 0 }, new[] { 1, _convInner, time });
        var gate = _engine.TensorSlice(expanded, new[] { 0, _convInner, 0 }, new[] { 1, _convInner, time });
        var glu = _engine.TensorMultiply(output, _engine.Sigmoid(gate));
        // calc_same_padding: (k / 2, k / 2 − (k + 1) % 2) zeros before and after.
        int left = k / 2, right = k / 2 - (k + 1) % 2;
        var parts = new List<Tensor<T>>();
        if (left > 0) parts.Add(new Tensor<T>(new[] { 1, _convInner, left }));
        parts.Add(glu);
        if (right > 0) parts.Add(new Tensor<T>(new[] { 1, _convInner, right }));
        var padded = parts.Count == 1 ? glu : _engine.TensorConcatenate(parts.ToArray(), 2);
        var depthwise = _engine.Swish(Depthwise.Forward(padded));
        var projected = ConvOut.Forward(ConvChannelNorm.Forward(depthwise));                                    // [1, C, T]
        var back = _engine.TensorTranspose(_engine.Reshape(projected, new[] { _c.Dim, time }));                     // [T, C]
        return Drop(back, training, random);
    }
}

/// <summary>A stack of <see cref="SoundStormConformerBlock{T}"/> sharing one rotary embedding (θ = 10000 over the head
/// width), as SoundStorm's and Pheme's <c>Conformer</c> run it (no key padding mask).</summary>
internal sealed class SoundStormConformer<T>
{
    public SoundStormConformer(IEngine engine, List<LayerBase<T>> layers, SoundStormConformerConfiguration configuration)
    {
        Configuration = configuration;
        for (int i = 0; i < configuration.Layers; i++) Blocks.Add(new SoundStormConformerBlock<T>(engine, layers, configuration));
    }

    public SoundStormConformerConfiguration Configuration { get; }
    public List<SoundStormConformerBlock<T>> Blocks { get; } = new();

    /// <summary>The blocks on <c>[time, dim]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x, bool training, Random random)
    {
        var (cos, sin) = Rotary(x.Shape[0], Configuration.HeadDim);
        foreach (var block in Blocks) x = block.Forward(x, cos, sin, training, random);
        return x;
    }

    // RotaryEmbedding: inv_freq = θ^(−2i/d); freqs = [t · inv_freq, t · inv_freq] over the head width.
    private static (Tensor<T> Cos, Tensor<T> Sin) Rotary(int time, int headDim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var cos = new Tensor<T>(new[] { time, headDim });
        var sin = new Tensor<T>(new[] { time, headDim });
        int half = headDim / 2;
        for (int t = 0; t < time; t++)
            for (int i = 0; i < half; i++)
            {
                // The reference computes inv_freq in single precision when the module is built, and the angles in the
                // model's precision.
                float inverseFrequency = 1.0f / (float)Math.Pow(10000.0, (float)(2 * i) / headDim);
                double angle = t * (double)inverseFrequency;
                cos[t, i] = cos[t, i + half] = ops.FromDouble(Math.Cos(angle));
                sin[t, i] = sin[t, i + half] = ops.FromDouble(Math.Sin(angle));
            }
        return (cos, sin);
    }

    /// <summary>Loads the Conformer's parameters from a PyTorch state dict under <paramref name="prefix"/>
    /// (for example "model.conformer"); <paramref name="read"/> checks each tensor's shape.</summary>
    public void LoadTorchWeights(IEngine engine, string prefix, Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int inner = c.Heads * c.HeadDim, hidden = c.Dim * c.FeedForwardMultiplier, convInner = c.Dim * c.ConvExpansion;
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = prefix.Length == 0 ? $"layers.{i}" : $"{prefix}.layers.{i}";
            void Norm(string name, LayerNormalizationLayer<T> layer) =>
                TorchParameters.LayerNorm(engine, layer, read(name + ".weight", new[] { c.Dim }), read(name + ".bias", new[] { c.Dim }));
            void Dense(string name, DenseLayer<T> layer, int inputs, int outputs) =>
                TorchParameters.Linear(engine, layer, inputs, outputs, read(name + ".weight", new[] { outputs, inputs }), read(name + ".bias", new[] { outputs }));
            Norm($"{p}.ff1.fn.norm", b.Ff1Norm);
            Dense($"{p}.ff1.fn.fn.net.0", b.Ff1In, c.Dim, hidden);
            Dense($"{p}.ff1.fn.fn.net.3", b.Ff1Out, hidden, c.Dim);
            Norm($"{p}.attn.norm", b.AttentionNorm);
            TorchParameters.Linear(b.ToQuery, c.Dim, inner, read($"{p}.attn.fn.to_q.weight", new[] { inner, c.Dim }));
            TorchParameters.Linear(b.ToKeyValue, c.Dim, 2 * inner, read($"{p}.attn.fn.to_kv.weight", new[] { 2 * inner, c.Dim }));
            Dense($"{p}.attn.fn.to_out", b.ToOut, inner, c.Dim);
            Norm($"{p}.conv.net.0", b.ConvNorm);
            b.ConvIn.LoadTorchWeights(read($"{p}.conv.net.2.weight", new[] { 2 * convInner, c.Dim, 1 }), null, read($"{p}.conv.net.2.bias", new[] { 2 * convInner }));
            b.Depthwise.LoadTorchWeights(read($"{p}.conv.net.4.conv.weight", new[] { convInner, 1, c.ConvKernel }), null,
                read($"{p}.conv.net.4.conv.bias", new[] { convInner }));
            var gamma = read($"{p}.conv.net.6.gamma", new[] { 1, convInner, 1 });
            var ops = MathHelper.GetNumericOperations<T>();
            for (int k = 0; k < convInner; k++) b.ConvChannelNorm.Gamma[k] = ops.FromDouble(gamma[k]);
            engine.InvalidatePersistentTensor(b.ConvChannelNorm.Gamma);
            b.ConvOut.LoadTorchWeights(read($"{p}.conv.net.7.weight", new[] { c.Dim, convInner, 1 }), null, read($"{p}.conv.net.7.bias", new[] { c.Dim }));
            Norm($"{p}.ff2.fn.norm", b.Ff2Norm);
            Dense($"{p}.ff2.fn.fn.net.0", b.Ff2In, c.Dim, hidden);
            Dense($"{p}.ff2.fn.fn.net.3", b.Ff2Out, hidden, c.Dim);
            Norm($"{p}.post_norm", b.PostNorm);
        }
    }
}
