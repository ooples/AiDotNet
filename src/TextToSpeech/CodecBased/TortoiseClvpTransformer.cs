using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The released CLVP2 x-transformers tower, including its Q/K/V rotary convention.</summary>
internal sealed class TortoiseClvpTower<T>
{
    private readonly IEngine _engine;
    private readonly int _dim;
    private readonly EmbeddingLayer<T> _embedding;
    private readonly List<Block> _blocks = new();
    private readonly LayerNormalizationLayer<T> _norm;
    private readonly BiasFreeLinearLayer<T> _projection;

    internal TortoiseClvpTower(IEngine engine, List<LayerBase<T>> owner, int vocabulary, int dim, int heads, int depth)
    {
        _engine = engine;
        _dim = dim;
        _embedding = TortoiseOps<T>.Own(owner, new EmbeddingLayer<T>(vocabulary, dim));
        for (int i = 0; i < depth; i++) _blocks.Add(new Block(engine, owner, dim, heads));
        _norm = TortoiseOps<T>.Own(owner, new LayerNormalizationLayer<T>(dim));
        _projection = TortoiseOps<T>.Own(owner, new BiasFreeLinearLayer<T>(dim, dim));
    }

    internal Tensor<T> Forward(IReadOnlyList<int> ids, bool training, double maskProbability, Random random)
    {
        if (ids.Count == 0) throw new ArgumentException("CLVP needs nonempty token sequences.", nameof(ids));
        var keep = new bool[ids.Count];
        for (int i = 0; i < keep.Length; i++) keep[i] = !training || random.NextDouble() > maskProbability;
        var x = _embedding.Forward(TortoiseOps<T>.Ids(ids));
        foreach (var block in _blocks) x = block.Forward(x, keep, training, random);
        x = _norm.Forward(x);
        var mask = new Tensor<T>(new[] { ids.Count, _dim });
        for (int i = 0; i < ids.Count; i++)
            for (int d = 0; d < _dim; d++) mask[i, d] = keep[i] ? TortoiseOps<T>.NumOps.One : TortoiseOps<T>.NumOps.Zero;
        x = _engine.TensorMultiply(x, mask);
        var pooled = _engine.TensorMultiplyScalar(_engine.ReduceSum(x, new[] { 0 }, true),
            TortoiseOps<T>.NumOps.FromDouble(1.0 / Math.Max(1, keep.Count(v => v))));
        var projected = _projection.Forward(pooled);
        var norm = _engine.TensorSqrt(_engine.ReduceSum(_engine.TensorMultiply(projected, projected), new[] { 1 }, true));
        norm = _engine.TensorClampMin(norm, TortoiseOps<T>.NumOps.FromDouble(1e-12));
        return _engine.TensorDivide(projected, _engine.TensorTile(norm, new[] { 1, _dim }));
    }

    internal void LoadTorchWeights(TorchTensorSource source, string embedding, string transformer, string projection, int vocabulary)
    {
        TorchParameters.Parameters(_embedding, source, embedding + ".weight", new[] { vocabulary, _dim });
        TorchParameters.LayerNorm(_engine, _norm, source, transformer + ".norm.weight", transformer + ".norm.bias");
        TorchParameters.Linear(_projection, _dim, _dim, source, projection + ".weight");
        for (int i = 0; i < _blocks.Count; i++) _blocks[i].Load(source, transformer + ".attn_layers.layers.", i);
    }

    private sealed class Block
    {
        private readonly IEngine _engine;
        private readonly int _dim, _heads;
        private const int HeadDim = 64, RotaryDim = 32;
        private readonly TiedEmbeddingLayer<T> _norm1, _norm2;
        private readonly BiasFreeLinearLayer<T> _query, _key, _value;
        private readonly DenseLayer<T> _out, _ffIn, _ffOut;

        internal Block(IEngine engine, List<LayerBase<T>> owner, int dim, int heads)
        {
            _engine = engine; _dim = dim; _heads = heads;
            var identity = new IdentityActivation<T>() as IActivationFunction<T>;
            _norm1 = TortoiseOps<T>.Own(owner, new TiedEmbeddingLayer<T>(1, dim));
            _norm2 = TortoiseOps<T>.Own(owner, new TiedEmbeddingLayer<T>(1, dim));
            _norm1.Reinitialize(() => 1); _norm2.Reinitialize(() => 1);
            _query = TortoiseOps<T>.Own(owner, new BiasFreeLinearLayer<T>(dim, heads * HeadDim));
            _key = TortoiseOps<T>.Own(owner, new BiasFreeLinearLayer<T>(dim, heads * HeadDim));
            _value = TortoiseOps<T>.Own(owner, new BiasFreeLinearLayer<T>(dim, heads * HeadDim));
            _out = TortoiseOps<T>.Own(owner, new DenseLayer<T>(dim, identity, biasMode: BiasMode.Always));
            _ffIn = TortoiseOps<T>.Own(owner, new DenseLayer<T>(4 * dim, identity, biasMode: BiasMode.Always));
            _ffOut = TortoiseOps<T>.Own(owner, new DenseLayer<T>(dim, identity, biasMode: BiasMode.Always));
        }

        private Tensor<T> Normalize(Tensor<T> x, TiedEmbeddingLayer<T> gain)
        {
            // Released RMSNorm clamps the RMS itself, rather than adding epsilon to the variance.
            var rms = _engine.TensorClampMin(_engine.TensorSqrt(_engine.ReduceMean(
                _engine.TensorMultiply(x, x), new[] { 1 }, true)), TortoiseOps<T>.NumOps.FromDouble(1e-8));
            var g = gain.Forward(new Tensor<T>(new[] { 1 }));
            return _engine.TensorMultiply(_engine.TensorDivide(x, _engine.TensorTile(rms, new[] { 1, _dim })),
                _engine.TensorTile(g, new[] { x.Shape[0], 1 }));
        }

        private Tensor<T> Heads(Tensor<T> x) => _engine.TensorPermute(
            _engine.Reshape(x, new[] { x.Shape[0], _heads, HeadDim }), new[] { 1, 0, 2 });

        private Tensor<T> Rotate(Tensor<T> x)
        {
            int n = x.Shape[1], half = RotaryDim / 2;
            var left = _engine.TensorNarrow(x, 2, 0, half);
            var right = _engine.TensorNarrow(x, 2, half, half);
            var cos = new Tensor<T>(new[] { _heads, n, half });
            var sin = new Tensor<T>(new[] { _heads, n, half });
            for (int h = 0; h < _heads; h++) for (int t = 0; t < n; t++) for (int d = 0; d < half; d++)
            {
                double angle = t * Math.Pow(10000, -2.0 * d / RotaryDim);
                cos[h, t, d] = TortoiseOps<T>.NumOps.FromDouble(Math.Cos(angle));
                sin[h, t, d] = TortoiseOps<T>.NumOps.FromDouble(Math.Sin(angle));
            }
            return _engine.TensorConcatenate(new[] {
                _engine.TensorSubtract(_engine.TensorMultiply(left, cos), _engine.TensorMultiply(right, sin)),
                _engine.TensorAdd(_engine.TensorMultiply(right, cos), _engine.TensorMultiply(left, sin)),
                _engine.TensorNarrow(x, 2, RotaryDim, HeadDim - RotaryDim) }, 2);
        }

        internal Tensor<T> Forward(Tensor<T> x, bool[] keep, bool training, Random random)
        {
            int n = x.Shape[0];
            var normalized = Normalize(x, _norm1);
            var q = Rotate(Heads(_query.Forward(normalized)));
            var k = Rotate(Heads(_key.Forward(normalized)));
            var v = Rotate(Heads(_value.Forward(normalized)));
            var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })),
                TortoiseOps<T>.NumOps.FromDouble(1.0 / Math.Sqrt(HeadDim)));
            var mask = new Tensor<T>(new[] { _heads, n, n });
            for (int h = 0; h < _heads; h++) for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
                if (!keep[i] || !keep[j]) mask[h, i, j] = TortoiseOps<T>.NumOps.FromDouble(-1e9);
            var weights = TortoiseOps<T>.Dropout(_engine, _engine.Softmax(_engine.TensorAdd(scores, mask), 2), 0.1, training, random);
            var context = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(weights, v), new[] { 1, 0, 2 }), new[] { n, _heads * HeadDim });
            x = _engine.TensorAdd(x, _out.Forward(context));
            var ff = _ffIn.Forward(Normalize(x, _norm2));
            var hidden = _engine.TensorMultiply(_engine.TensorNarrow(ff, 1, 0, 2 * _dim),
                ActivationHelper.ExactGelu(_engine.TensorNarrow(ff, 1, 2 * _dim, 2 * _dim), _engine));
            return _engine.TensorAdd(x, _ffOut.Forward(TortoiseOps<T>.Dropout(_engine, hidden, 0.1, training, random)));
        }

        internal void Load(TorchTensorSource source, string prefix, int index)
        {
            string a = prefix + (2 * index), f = prefix + (2 * index + 1);
            _norm1.SetParameters(new Vector<T>(source.Read(a + ".0.0.g", new[] { _dim }).Select(TortoiseOps<T>.NumOps.FromDouble)));
            _norm2.SetParameters(new Vector<T>(source.Read(f + ".0.0.g", new[] { _dim }).Select(TortoiseOps<T>.NumOps.FromDouble)));
            TorchParameters.Linear(_query, _dim, _heads * HeadDim, source, a + ".1.wrap.to_q.weight");
            TorchParameters.Linear(_key, _dim, _heads * HeadDim, source, a + ".1.wrap.to_k.weight");
            TorchParameters.Linear(_value, _dim, _heads * HeadDim, source, a + ".1.wrap.to_v.weight");
            TorchParameters.Linear(_engine, _out, _heads * HeadDim, _dim, source, a + ".1.wrap.to_out.weight", a + ".1.wrap.to_out.bias");
            TorchParameters.Linear(_engine, _ffIn, _dim, 4 * _dim, source, f + ".1.wrap.net.0.proj.weight", f + ".1.wrap.net.0.proj.bias");
            TorchParameters.Linear(_engine, _ffOut, 2 * _dim, _dim, source, f + ".1.wrap.net.3.weight", f + ".1.wrap.net.3.bias");
        }
    }
}
