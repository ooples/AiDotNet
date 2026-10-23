using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>
/// A stack of Conformer blocks (Gulati et al. 2020) as wenet's <c>ConformerEncoderLayer</c> builds them — StyleTTS-ZS's
/// modules are all conformer stacks, and its paper names Gulati's block without its code, so the block follows wenet's
/// reproduction: the input scaled by √d with ESPnet's relative position table (<c>EspnetRelPositionalEncoding</c>), then
/// per block a half-step macaron feed-forward, relative-position self-attention (Transformer-XL's u and v biases),
/// the convolution module (pointwise to 2d, GLU, depthwise of the block's kernel, BatchNorm, Swish, pointwise), a
/// half-step feed-forward and a final LayerNorm, each sub-module pre-norm and residual (norms ε 1e-12; Swish
/// feed-forwards).
/// </summary>
internal sealed class StyleTtsZsConformer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Block(int Kernel,
        LayerNormalizationLayer<T> NormMacaron, DenseLayer<T> Macaron1, DenseLayer<T> Macaron2,
        LayerNormalizationLayer<T> NormAttention, DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Output,
        BiasFreeLinearLayer<T> PositionProjection, TiedEmbeddingLayer<T> BiasU, TiedEmbeddingLayer<T> BiasV,
        LayerNormalizationLayer<T> NormConv, NormedConv1DLayer<T> Pointwise1, NormedConv1DLayer<T> Depthwise, BatchNormalizationLayer<T> ConvNorm,
        NormedConv1DLayer<T> Pointwise2,
        LayerNormalizationLayer<T> NormFeedForward, DenseLayer<T> FeedForward1, DenseLayer<T> FeedForward2, LayerNormalizationLayer<T> NormFinal);

    public StyleTtsZsConformer(IEngine engine, List<LayerBase<T>> layers, int dim, int heads, int feedForward, IReadOnlyList<int> kernels,
        double dropout, Random initialization)
    {
        if (dim % heads != 0) throw new ArgumentException("The width must divide by the heads.", nameof(heads));
        _engine = engine;
        Dim = dim;
        Heads = heads;
        FeedForwardDim = feedForward;
        DropoutRate = dropout;
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> => StyleTtsOps.Own(layers, layer);
        DenseLayer<T> Linear(int outputs) => Own(StyleTtsOps.Linear<T>(outputs));
        LayerNormalizationLayer<T> Norm() => Own(new LayerNormalizationLayer<T>(dim, 1e-12));
        NormedConv1DLayer<T> Conv(int input, int output, int kernel, int groups) =>
            Own(new NormedConv1DLayer<T>(input, output, kernel, 1, 1, groups, (kernel - 1) / 2, false, ConvolutionNormalization.None));
        int dk = dim / heads;
        double xavier = Math.Sqrt(6.0 / (heads + dk));
        foreach (int kernel in kernels)
        {
            if (kernel % 2 == 0) throw new ArgumentException("A symmetric convolution module needs an odd kernel.", nameof(kernels));
            var biasU = Own(new TiedEmbeddingLayer<T>(1, dim));
            biasU.Reinitialize(() => xavier * (2 * initialization.NextDouble() - 1));
            var biasV = Own(new TiedEmbeddingLayer<T>(1, dim));
            biasV.Reinitialize(() => xavier * (2 * initialization.NextDouble() - 1));
            Blocks.Add(new Block(kernel,
                Norm(), Linear(feedForward), Linear(dim),
                Norm(), Linear(dim), Linear(dim), Linear(dim), Linear(dim), Own(new BiasFreeLinearLayer<T>(dim, dim)), biasU, biasV,
                Norm(), Conv(dim, 2 * dim, 1, 1), Conv(dim, dim, kernel, dim), Own(new BatchNormalizationLayer<T>(dim, 1e-5, 0.9)), Conv(dim, dim, 1, 1),
                Norm(), Linear(feedForward), Linear(dim), Norm()));
        }
    }

    public int Dim { get; }
    public int Heads { get; }
    public int FeedForwardDim { get; }
    public double DropoutRate { get; }
    public List<Block> Blocks { get; } = new();

    private Tensor<T> Dropout(Tensor<T> x, bool training, Random random) =>
        training && DropoutRate > 0 ? T5Seq2Seq<T>.Dropout(_engine, x, DropoutRate, random) : x;

    /// <summary>ESPnet's relative position table <c>[2L − 1, d]</c> over relative positions L−1 … −(L−1).</summary>
    private Tensor<T> PositionTable(int length)
    {
        var table = new Tensor<T>(new[] { 2 * length - 1, Dim });
        for (int r = 0; r < 2 * length - 1; r++)
        {
            double position = length - 1 - r;
            for (int k = 0; k < Dim; k += 2)
            {
                double frequency = Math.Exp(k * -(Math.Log(10000.0) / Dim));
                table[r, k] = NumOps.FromDouble(Math.Sin(position * frequency));
                if (k + 1 < Dim) table[r, k + 1] = NumOps.FromDouble(Math.Cos(position * frequency));
            }
        }
        return table;
    }

    private Tensor<T> FeedForwardOf(DenseLayer<T> first, DenseLayer<T> second, Tensor<T> x, bool training, Random random) =>
        second.Forward(Dropout(_engine.Swish(first.Forward(x)), training, random));

    /// <summary>The stack's output <c>[L, d]</c> for inputs <c>[L, d]</c> (one sequence, every position attended); each
    /// block's output is added to <paramref name="states"/> when given.</summary>
    public Tensor<T> Forward(Tensor<T> input, bool training, Random random, List<Tensor<T>>? states = null)
    {
        int length = input.Shape[0], d = Dim, heads = Heads, dk = d / heads;
        var x = Dropout(_engine.TensorMultiplyScalar(input, NumOps.FromDouble(Math.Sqrt(d))), training, random);
        var positions = Dropout(PositionTable(length), training, random);
        var shift = new Tensor<int>(new[] { length * length });
        for (int i = 0; i < length; i++)
            for (int j = 0; j < length; j++) shift[i * length + j] = i * (2 * length - 1) + (length - 1 - i + j);
        Tensor<T> Split(Tensor<T> t, int rows) => _engine.TensorPermute(_engine.Reshape(t, new[] { rows, heads, dk }), new[] { 1, 0, 2 });
        Tensor<T> Row(TiedEmbeddingLayer<T> holder) => _engine.Reshape(holder.Forward(new Tensor<T>(new[] { 1 })), new[] { heads, 1, dk });
        var half = NumOps.FromDouble(0.5);
        foreach (var b in Blocks)
        {
            x = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(Dropout(FeedForwardOf(b.Macaron1, b.Macaron2, b.NormMacaron.Forward(x), training, random),
                training, random), half));

            var normed = b.NormAttention.Forward(x);
            var q = Split(b.Query.Forward(normed), length);
            var k = Split(b.Key.Forward(normed), length);
            var v = Split(b.Value.Forward(normed), length);
            var p = Split(b.PositionProjection.Forward(positions), 2 * length - 1);
            var ac = _engine.BatchMatMul(_engine.TensorAdd(q, _engine.TensorBroadcastTo(Row(b.BiasU), new[] { heads, length, dk })),
                _engine.TensorPermute(k, new[] { 0, 2, 1 }));
            var bdFull = _engine.BatchMatMul(_engine.TensorAdd(q, _engine.TensorBroadcastTo(Row(b.BiasV), new[] { heads, length, dk })),
                _engine.TensorPermute(p, new[] { 0, 2, 1 }));
            var bd = _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(bdFull, new[] { heads, length * (2 * length - 1) }), shift, 1),
                new[] { heads, length, length });
            var scores = _engine.TensorMultiplyScalar(_engine.TensorAdd(ac, bd), NumOps.FromDouble(1.0 / Math.Sqrt(dk)));
            var weights = Dropout(_engine.Softmax(scores, axis: 2), training, random);
            var context = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(weights, v), new[] { 1, 0, 2 }), new[] { length, d });
            x = _engine.TensorAdd(x, Dropout(b.Output.Forward(context), training, random));

            var conv = _engine.Reshape(_engine.TensorTranspose(b.NormConv.Forward(x)), new[] { 1, d, length });
            var gated = b.Pointwise1.Forward(conv);                                                               // [1, 2d, L]
            var glu = _engine.TensorMultiply(_engine.TensorSlice(gated, new[] { 0, 0, 0 }, new[] { 1, d, length }),
                _engine.Sigmoid(_engine.TensorSlice(gated, new[] { 0, d, 0 }, new[] { 1, d, length })));
            var depth = b.Depthwise.Forward(glu);
            var normedConv = _engine.Reshape(b.ConvNorm.Forward(_engine.Reshape(depth, new[] { 1, d, length, 1 })), new[] { 1, d, length });
            var convOut = b.Pointwise2.Forward(_engine.Swish(normedConv));
            x = _engine.TensorAdd(x, Dropout(_engine.TensorTranspose(_engine.Reshape(convOut, new[] { d, length })), training, random));

            x = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(Dropout(FeedForwardOf(b.FeedForward1, b.FeedForward2, b.NormFeedForward.Forward(x), training, random),
                training, random), half));
            x = b.NormFinal.Forward(x);
            states?.Add(x);
        }
        return x;
    }

    /// <summary>Loads wenet <c>ConformerEncoderLayer</c>s named <c>{prefix}{i}.*</c>.</summary>
    public void Load(TorchReader read, string prefix)
    {
        int d = Dim, f = FeedForwardDim, heads = Heads;
        void Dense(DenseLayer<T> layer, string name, int inputs, int outputs) => StyleTtsOps.LoadLinear(_engine, layer, read, name, inputs, outputs);
        void Norm(LayerNormalizationLayer<T> norm, string name) =>
            TorchParameters.LayerNorm(_engine, norm, read(name + ".weight", new[] { d }), read(name + ".bias", new[] { d }));
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = prefix + i;
            Norm(b.NormMacaron, p + ".norm_ff_macaron");
            Dense(b.Macaron1, p + ".feed_forward_macaron.w_1", d, f);
            Dense(b.Macaron2, p + ".feed_forward_macaron.w_2", f, d);
            Norm(b.NormAttention, p + ".norm_mha");
            Dense(b.Query, p + ".self_attn.linear_q", d, d);
            Dense(b.Key, p + ".self_attn.linear_k", d, d);
            Dense(b.Value, p + ".self_attn.linear_v", d, d);
            Dense(b.Output, p + ".self_attn.linear_out", d, d);
            TorchParameters.Linear(b.PositionProjection, d, d, read(p + ".self_attn.linear_pos.weight", new[] { d, d }));
            b.BiasU.LoadTable(read(p + ".self_attn.pos_bias_u", new[] { heads, d / heads }));
            b.BiasV.LoadTable(read(p + ".self_attn.pos_bias_v", new[] { heads, d / heads }));
            Norm(b.NormConv, p + ".norm_conv");
            StyleTtsOps.LoadPlain(b.Pointwise1, read, p + ".conv_module.pointwise_conv1");
            StyleTtsOps.LoadPlain(b.Depthwise, read, p + ".conv_module.depthwise_conv");
            TorchParameters.BatchNorm(b.ConvNorm, read(p + ".conv_module.norm.weight", new[] { d }), read(p + ".conv_module.norm.bias", new[] { d }),
                read(p + ".conv_module.norm.running_mean", new[] { d }), read(p + ".conv_module.norm.running_var", new[] { d }));
            StyleTtsOps.LoadPlain(b.Pointwise2, read, p + ".conv_module.pointwise_conv2");
            Norm(b.NormFeedForward, p + ".norm_ff");
            Dense(b.FeedForward1, p + ".feed_forward.w_1", d, f);
            Dense(b.FeedForward2, p + ".feed_forward.w_2", f, d);
            Norm(b.NormFinal, p + ".norm_final");
        }
    }
}
