using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// FunASR's <c>ParaformerSANMDecoder</c> without its embedding and output layers: SAN-M blocks reading a memory sequence,
/// a last feed-forward block, and the after-norm.
/// </summary>
/// <remarks>
/// Each block is a feed-forward (with a norm on its hidden layer) feeding an FSMN memory (a depthwise convolution over
/// time plus its input) added to the block input, then cross-attention over the memory. The last block is the
/// feed-forward alone, without a residual. Paraformer-large decodes the encoder output this way; SeACo-Paraformer's bias
/// decoder reads the encoded hotwords this way.
/// </remarks>
internal sealed class SanmDecoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _width;
    private readonly int _heads;
    private readonly int _kernel;

    internal sealed record Block(LayerNormalizationLayer<T> Norm1, DenseLayer<T> FeedForward1, LayerNormalizationLayer<T> FeedForwardNorm,
        DenseLayer<T> FeedForward2, LayerNormalizationLayer<T>? Norm2, NormedConv1DLayer<T>? Fsmn, LayerNormalizationLayer<T>? Norm3,
        DenseLayer<T>? CrossQuery, DenseLayer<T>? CrossKeyValue, DenseLayer<T>? CrossOutput);

    /// <param name="engine">The tensor engine.</param>
    /// <param name="layers">Receives every layer the decoder owns, in creation order.</param>
    /// <param name="width">The model width, shared with the memory.</param>
    /// <param name="heads">Cross-attention heads.</param>
    /// <param name="feedForward">The feed-forward's hidden width.</param>
    /// <param name="blocks">Blocks with cross-attention, before the last feed-forward block.</param>
    /// <param name="kernel">The FSMN memories' kernel.</param>
    /// <param name="epsilon">The layer norms' epsilon.</param>
    public SanmDecoder(IEngine engine, List<LayerBase<T>> layers, int width, int heads, int feedForward, int blocks, int kernel, double epsilon)
    {
        if (width % heads != 0) throw new ArgumentException("The width must divide by the heads.", nameof(heads));
        if (blocks < 1) throw new ArgumentException("The decoder needs at least one block.", nameof(blocks));
        _engine = engine;
        _width = width;
        _heads = heads;
        _kernel = kernel;
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        DenseLayer<T> Linear(int outputs, bool bias = true) => Own(new DenseLayer<T>(outputs, identity, biasMode: bias ? BiasMode.Always : BiasMode.Never));
        LayerNormalizationLayer<T> Norm(int size) => Own(new LayerNormalizationLayer<T>(size, epsilon));
        for (int i = 0; i < blocks; i++)
            Blocks.Add(new Block(Norm(width), Linear(feedForward), Norm(feedForward), Linear(width, bias: false), Norm(width),
                Own(new NormedConv1DLayer<T>(width, width, kernel, 1, 1, width, 0, false, ConvolutionNormalization.None, useBias: false)),
                Norm(width), Linear(width), Linear(2 * width), Linear(width)));
        Blocks.Add(new Block(Norm(width), Linear(feedForward), Norm(feedForward), Linear(width, bias: false), null, null, null, null, null, null));
        AfterNorm = Norm(width);
    }

    public List<Block> Blocks { get; } = new();
    public LayerNormalizationLayer<T> AfterNorm { get; }

    /// <summary>The hidden states <c>[n, width]</c> (after the after-norm) of the inputs <c>[n, width]</c> reading the
    /// memory <c>[m, width]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> memory, Tensor<T> inputs) => Run(memory, inputs, null);

    /// <summary>The attention each memory row receives in the last cross-attention block, summed over the heads and the
    /// inputs (SeACo-Paraformer's attention score filtering, section 3.2).</summary>
    public double[] LastAttentionTotals(Tensor<T> memory, Tensor<T> inputs)
    {
        var totals = new double[memory.Shape[0]];
        Run(memory, inputs, totals);
        return totals;
    }

    private Tensor<T> Run(Tensor<T> memory, Tensor<T> inputs, double[]? lastAttention)
    {
        var x = inputs;
        int d = _width, m = memory.Shape[0];
        for (int index = 0; index < Blocks.Count; index++)
        {
            var b = Blocks[index];
            var residual = x;
            var h = b.FeedForward2.Forward(b.FeedForwardNorm.Forward(_engine.ReLU(b.FeedForward1.Forward(b.Norm1.Forward(x)))));
            if (b.Norm2 is null || b.Fsmn is null || b.Norm3 is null || b.CrossQuery is null || b.CrossKeyValue is null || b.CrossOutput is null)
            {
                x = h;                                                                              // the last block: no residual
                continue;
            }
            // forward_fsmn with sanm_shfit 0: (k - 1)/2 zeros each side, the depthwise convolution, plus its input.
            var normed = b.Norm2.Forward(h);
            int left = (_kernel - 1) / 2;
            x = _engine.TensorAdd(residual, _engine.TensorAdd(Convolve(_engine, b.Fsmn, normed, left, _kernel - 1 - left), normed));
            var q = b.CrossQuery.Forward(b.Norm3.Forward(x));
            var kv = b.CrossKeyValue.Forward(memory);
            if (lastAttention is not null && index == Blocks.Count - 2)
            {
                var weights = Weights(_engine, q, _engine.TensorSlice(kv, new[] { 0, 0 }, new[] { m, d }), _heads);
                for (int i = 0; i < weights.Length; i++) lastAttention[i % m] += NumOps.ToDouble(weights.GetFlat(i));
                return x;
            }
            x = _engine.TensorAdd(x, b.CrossOutput.Forward(Attend(_engine, q, _engine.TensorSlice(kv, new[] { 0, 0 }, new[] { m, d }),
                _engine.TensorSlice(kv, new[] { 0, d }, new[] { m, d }), _heads)));
        }
        return AfterNorm.Forward(x);
    }

    /// <summary>[L, d] features through a 1-D convolution over time with <paramref name="left"/> and
    /// <paramref name="right"/> zero frames, back to [L, d].</summary>
    internal static Tensor<T> Convolve(IEngine engine, NormedConv1DLayer<T> convolution, Tensor<T> x, int left, int right)
    {
        int length = x.Shape[0], d = x.Shape[1];
        var channels = engine.Reshape(engine.TensorTranspose(x), new[] { 1, d, length });
        var padded = engine.TensorConcatenate(new[] { new Tensor<T>(new[] { 1, d, left }), channels, new Tensor<T>(new[] { 1, d, right }) }, 2);
        return engine.TensorTranspose(engine.Reshape(convolution.Forward(padded), new[] { d, length }));
    }

    /// <summary>Scaled dot-product attention of the queries <c>[n, d]</c> over the keys and values <c>[m, d]</c>,
    /// split into <paramref name="heads"/> heads.</summary>
    internal static Tensor<T> Attend(IEngine engine, Tensor<T> q, Tensor<T> k, Tensor<T> v, int heads)
    {
        int rows = q.Shape[0], keys = k.Shape[0], width = q.Shape[1], dk = width / heads;
        var context = engine.BatchMatMul(Weights(engine, q, k, heads), Heads(engine, v, keys, heads, dk));
        return engine.Reshape(engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { rows, width });
    }

    // The attention weights [heads, n, m].
    private static Tensor<T> Weights(IEngine engine, Tensor<T> q, Tensor<T> k, int heads)
    {
        int rows = q.Shape[0], keys = k.Shape[0], dk = q.Shape[1] / heads;
        var scores = engine.BatchMatMul(engine.TensorMultiplyScalar(Heads(engine, q, rows, heads, dk), NumOps.FromDouble(Math.Pow(dk, -0.5))),
            engine.TensorPermute(Heads(engine, k, keys, heads, dk), new[] { 0, 2, 1 }));
        return engine.Softmax(scores, axis: 2);
    }

    private static Tensor<T> Heads(IEngine engine, Tensor<T> x, int n, int heads, int dk) =>
        engine.TensorPermute(engine.Reshape(x, new[] { n, heads, dk }), new[] { 1, 0, 2 });
}
