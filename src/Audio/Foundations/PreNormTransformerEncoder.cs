using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// fairseq's wav2vec 2.0 <c>TransformerEncoder</c> with <c>layer_norm_first</c>: the weight-normalized grouped
/// convolutional position embedding (GELU, added to the input), dropout, pre-norm Transformer layers with GELU feed-forward
/// blocks and LayerDrop, and a closing layer norm.
/// </summary>
/// <remarks>
/// Used by SPIRAL (twice, one per block) and AV-HuBERT, whose encoders are this stack over their own front ends. It works
/// on one sequence <c>[L, width]</c>; its layers are added to the given list.
/// </remarks>
internal sealed class PreNormTransformerEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Layer(LayerNormalizationLayer<T> AttentionNorm, DenseLayer<T> Query, DenseLayer<T> Key,
        DenseLayer<T> Value, DenseLayer<T> Output, LayerNormalizationLayer<T> FeedForwardNorm, DenseLayer<T> Intermediate,
        DenseLayer<T> FeedForwardOutput);

    /// <param name="engine">The tensor engine.</param>
    /// <param name="layers">Receives the stack's layers.</param>
    /// <param name="width">The model width.</param>
    /// <param name="count">The Transformer layers.</param>
    /// <param name="heads">The attention heads.</param>
    /// <param name="feedForward">The feed-forward width.</param>
    /// <param name="positionalKernel">The position convolution's kernel (wav2vec 2.0: 128).</param>
    /// <param name="positionalGroups">The position convolution's groups (wav2vec 2.0: 16).</param>
    public PreNormTransformerEncoder(IEngine engine, List<LayerBase<T>> layers, int width, int count, int heads, int feedForward,
        int positionalKernel = 128, int positionalGroups = 16)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        if (width % heads != 0) throw new ArgumentException($"The width {width} must divide into {heads} heads.");
        if (width % positionalGroups != 0) throw new ArgumentException($"The width {width} must divide into {positionalGroups} positional groups.");
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        DenseLayer<T> Linear(int outputs) => Own(new DenseLayer<T>(outputs, identity));
        Width = width;
        Heads = heads;
        PositionalConv = Own(new Wav2Vec2PositionalConvLayer<T>(width, positionalKernel, positionalGroups));
        for (int i = 0; i < count; i++)
            Layers.Add(new Layer(Own(new LayerNormalizationLayer<T>(width)), Linear(width), Linear(width), Linear(width),
                Linear(width), Own(new LayerNormalizationLayer<T>(width)), Linear(feedForward), Linear(width)));
        FinalNorm = Own(new LayerNormalizationLayer<T>(width));
    }

    public int Width { get; }
    public int Heads { get; }
    public Wav2Vec2PositionalConvLayer<T> PositionalConv { get; }
    public List<Layer> Layers { get; } = new();
    public LayerNormalizationLayer<T> FinalNorm { get; }

    /// <summary>
    /// Initializes the weight-normalized position convolution as fairseq does: v ~ N(0, 2·√(1/(K·C))), g = ‖v‖ per tap,
    /// zero bias. Left at zero, w = g·v/‖v‖ is 0/0.
    /// </summary>
    public void InitializePositionalConvolution(Random random)
    {
        var (v, g, b) = PositionalConv.WeightNormParameters;
        int channels = v.Shape[0], perGroup = v.Shape[1], kernel = v.Shape[2];
        double std = 2 * Math.Sqrt(1.0 / ((double)kernel * channels));
        var direction = new double[v.Length];
        for (int i = 0; i < direction.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            direction[i] = std * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            v[i] = NumOps.FromDouble(direction[i]);
        }
        for (int t = 0; t < kernel; t++)
        {
            double sum = 0;
            for (int o = 0; o < channels * perGroup; o++) sum += direction[o * kernel + t] * direction[o * kernel + t];
            g[t] = NumOps.FromDouble(Math.Sqrt(sum));
        }
        for (int i = 0; i < b.Length; i++) b[i] = NumOps.Zero;
        _engine.InvalidatePersistentTensor(v);
        _engine.InvalidatePersistentTensor(g);
        _engine.InvalidatePersistentTensor(b);
    }

    /// <summary>
    /// The encoder output <c>[L, width]</c> of <c>[L, width]</c>. In training, dropout (<paramref name="dropout"/> on the
    /// embedded input, the residual branches and the attention weights; <paramref name="activationDropout"/> inside the
    /// feed-forward block) and LayerDrop apply; <paramref name="states"/>, when given, receives the state after the position
    /// embedding and after every layer.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> x, bool training, Random random, double dropout, double activationDropout, double layerDrop,
        List<Tensor<T>>? states = null)
    {
        int length = x.Shape[0];
        var positional = _engine.Reshape(PositionalConv.Forward(_engine.Reshape(x, new[] { 1, length, Width })), new[] { length, Width });
        x = Drop(_engine.TensorAdd(x, ActivationHelper.ExactGelu(positional, _engine)), dropout, training, random);
        states?.Add(x);
        foreach (var layer in Layers)
        {
            if (!(training && random.NextDouble() < layerDrop)) x = Encode(layer, x, training, random, dropout, activationDropout);
            states?.Add(x);
        }
        return FinalNorm.Forward(x);
    }

    private Tensor<T> Drop(Tensor<T> x, double rate, bool training, Random random)
    {
        if (!training || rate <= 0) return x;
        var mask = new Tensor<T>(x._shape);
        var keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        for (int i = 0; i < mask.Length; i++) mask[i] = random.NextDouble() < rate ? NumOps.Zero : keep;
        return _engine.TensorMultiply(x, mask);
    }

    private Tensor<T> Encode(Layer l, Tensor<T> x, bool training, Random random, double dropout, double activationDropout)
    {
        int length = x.Shape[0], e = Width / Heads;
        var normed = l.AttentionNorm.Forward(x);
        Tensor<T> Split(DenseLayer<T> projection) =>
            _engine.TensorPermute(_engine.Reshape(projection.Forward(normed), new[] { length, Heads, e }), new[] { 1, 0, 2 });
        var q = Split(l.Query);
        var k = Split(l.Key);
        var v = Split(l.Value);
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(e)));
        var weights = Drop(_engine.Softmax(scores, axis: 2), dropout, training, random);
        var context = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(weights, v), new[] { 1, 0, 2 }), new[] { length, Width });
        x = _engine.TensorAdd(x, Drop(l.Output.Forward(context), dropout, training, random));
        var hidden = Drop(ActivationHelper.ExactGelu(l.Intermediate.Forward(l.FeedForwardNorm.Forward(x)), _engine), activationDropout, training, random);
        return _engine.TensorAdd(x, Drop(l.FeedForwardOutput.Forward(hidden), dropout, training, random));
    }
}
