using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.MetaLearning.Modules;

/// <summary>
/// The relation module <c>g</c> of Relation Networks in engine tensor ops, so a live tape differentiates it: a
/// relation score in (0, 1) for each (sample, query) pair of embeddings, over weights unpacked from one flat vector.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Every variant builds a <c>[pairs, hidden]</c> representation of the pair and ends as the paper's module does, with
/// a fully connected sigmoid unit (Sung et al. 2018, Figure 2): <c>r = sigmoid(w h + b)</c>.
/// </para>
/// <list type="bullet">
/// <item><see cref="RelationModuleType.Concatenate"/> - the paper: <c>h = relu(W [f(x_i); f(x_j)] + b)</c>, the
/// sample then the query, as <c>C(f(x_i), f(x_j))</c> concatenates them.</item>
/// <item><see cref="RelationModuleType.Convolution"/> - the vector form of the paper's convolutional relation module:
/// the two embeddings are two channels of a one-dimensional signal, convolved by <c>hidden</c> width-3 filters with
/// zero padding, ReLU, then averaged over positions.</item>
/// <item><see cref="RelationModuleType.Attention"/> - cross-attention of the query over the sample's positions: the
/// projected query attends to one key/value token per embedding position (the position's value times a learned
/// vector plus a learned position embedding); the read-out and the projected query go through a ReLU layer.</item>
/// <item><see cref="RelationModuleType.Transformer"/> - one self-attention block over the two tokens (projected sample
/// and query plus segment embeddings), with residual feed-forward layer, mean-pooled over the tokens. No layer
/// normalisation: the block is one layer deep.</item>
/// </list>
/// <para>
/// Weights are initialised as PyTorch initialises <c>nn.Linear</c>: U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
/// </para>
/// </remarks>
internal sealed class RelationFunction<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly List<Tensor<T>> _blocks = new List<Tensor<T>>();

    /// <summary>Unpacks one relation module's weights from <paramref name="packed"/> at <paramref name="offset"/>.</summary>
    /// <exception cref="ArgumentException">The vector is too short for a module of this size.</exception>
    internal RelationFunction(RelationModuleType type, Vector<T> packed, int offset, int width, int hidden)
    {
        if (packed is null) throw new ArgumentNullException(nameof(packed));
        int count = ParameterCount(type, width, hidden);
        if (offset < 0 || offset + count > packed.Length)
        {
            throw new ArgumentException(
                $"A {type} relation module of width {width} and hidden size {hidden} needs {count} weights from offset "
                + $"{offset}, but the vector holds {packed.Length}.", nameof(packed));
        }

        Type = type;
        Width = width;
        Hidden = hidden;
        int position = offset;
        foreach (var (shape, _) in Blocks(type, width, hidden))
        {
            var block = new Tensor<T>(shape);
            for (int i = 0; i < block.Length; i++) block[i] = packed[position + i];
            position += block.Length;
            _blocks.Add(block);
        }
    }

    /// <summary>The module's architecture.</summary>
    internal RelationModuleType Type { get; }

    /// <summary>Width of each embedding in the pair.</summary>
    internal int Width { get; }

    /// <summary>Width of the pair representation before the sigmoid unit.</summary>
    internal int Hidden { get; }

    /// <summary>The weight blocks in the flat layout's order: the tensors a tape differentiates against.</summary>
    internal IReadOnlyList<Tensor<T>> Leaves => _blocks;

    /// <summary>Number of weights one module reads from the flat vector.</summary>
    internal static int ParameterCount(RelationModuleType type, int width, int hidden)
    {
        int count = 0;
        foreach (var (shape, _) in Blocks(type, width, hidden))
        {
            int size = 1;
            foreach (int extent in shape) size *= extent;
            count += size;
        }

        return count;
    }

    /// <summary>Fills one module's weights with U(-1/sqrt(fan_in), 1/sqrt(fan_in)), block by block.</summary>
    internal static void Initialize(RelationModuleType type, Vector<T> packed, int offset, int width, int hidden, Random random)
    {
        int position = offset;
        foreach (var (shape, bound) in Blocks(type, width, hidden))
        {
            int size = 1;
            foreach (int extent in shape) size *= extent;
            for (int i = 0; i < size; i++) packed[position + i] = Ops.FromDouble((2.0 * random.NextDouble() - 1.0) * bound);
            position += size;
        }
    }

    /// <summary>Writes the leaves' gradients into <paramref name="into"/> at <paramref name="offset"/>.</summary>
    internal void CopyGradients(Dictionary<Tensor<T>, Tensor<T>> gradients, Vector<T> into, int offset)
    {
        int position = offset;
        foreach (var block in _blocks)
        {
            if (gradients.TryGetValue(block, out var gradient))
            {
                for (int i = 0; i < block.Length; i++) into[position + i] = gradient[i];
            }

            position += block.Length;
        }
    }

    /// <summary>
    /// Relation scores <c>[pairs, 1]</c> of each sample row with the query row beside it.
    /// </summary>
    /// <param name="sample">The class (or support-example) embeddings, <c>[pairs, Width]</c>.</param>
    /// <param name="query">The query embeddings, <c>[pairs, Width]</c>.</param>
    /// <param name="hiddenMask">An inverted-dropout mask for the pair representation, or null for none.</param>
    internal Tensor<T> Scores(Tensor<T> sample, Tensor<T> query, Tensor<T>? hiddenMask)
    {
        var engine = AiDotNetEngine.Current;
        var hidden = Type switch
        {
            RelationModuleType.Convolution => ConvolutionHidden(sample, query),
            RelationModuleType.Attention => AttentionHidden(sample, query),
            RelationModuleType.Transformer => TransformerHidden(sample, query),
            _ => ConcatenateHidden(sample, query),
        };
        if (hiddenMask is not null) hidden = engine.TensorMultiply(hidden, hiddenMask);

        var outputWeights = _blocks[_blocks.Count - 2];
        var outputBias = _blocks[_blocks.Count - 1];
        return engine.Sigmoid(engine.TensorAdd(
            engine.TensorMatMul(hidden, engine.TensorTranspose(outputWeights)), engine.Reshape(outputBias, new[] { 1, 1 })));
    }

    private Tensor<T> ConcatenateHidden(Tensor<T> sample, Tensor<T> query)
    {
        var engine = AiDotNetEngine.Current;
        var pair = Concatenate(sample, query);
        return engine.ReLU(engine.TensorAdd(
            engine.TensorMatMul(pair, engine.TensorTranspose(_blocks[0])), engine.Reshape(_blocks[1], new[] { 1, Hidden })));
    }

    private Tensor<T> ConvolutionHidden(Tensor<T> sample, Tensor<T> query)
    {
        var engine = AiDotNetEngine.Current;
        int pairs = sample.Shape[0], width = Width;

        // im2col: one row per (pair, position), one column per (channel, tap).
        Tensor<T>? columns = null;
        int column = 0;
        foreach (var channel in new[] { sample, query })
        {
            for (int shift = -1; shift <= 1; shift++, column++)
            {
                var shifted = engine.Reshape(engine.TensorMatMul(channel, Shift(width, shift)), new[] { pairs * width, 1 });
                var placed = engine.TensorMatMul(shifted, Unit(1, 6, 0, column));
                columns = columns is null ? placed : engine.TensorAdd(columns, placed);
            }
        }

        var filtered = engine.ReLU(engine.TensorAdd(
            engine.TensorMatMul(columns ?? new Tensor<T>(new[] { pairs * width, 6 }), engine.TensorTranspose(_blocks[0])),
            engine.Reshape(_blocks[1], new[] { 1, Hidden })));
        return engine.TensorMatMul(engine.Reshape(filtered, new[] { pairs, width * Hidden }), AveragePositions(width, Hidden));
    }

    private Tensor<T> AttentionHidden(Tensor<T> sample, Tensor<T> query)
    {
        var engine = AiDotNetEngine.Current;
        var queryWeights = _blocks[0];
        var keyVector = _blocks[1];
        var keyPositions = _blocks[2];
        var valueVector = _blocks[3];
        var valuePositions = _blocks[4];
        var outWeights = _blocks[5];
        var outBias = _blocks[6];

        // Token d of the sample: key k_d = x_d w_k + p_k[d], value v_d = x_d w_v + p_v[d]. With the projected query q,
        // q . k_d = x_d (q . w_k) + (q . p_k[d]) - two-dimensional ops for every pair at once.
        var projected = engine.TensorMatMul(query, engine.TensorTranspose(queryWeights));
        var scores = engine.TensorAdd(
            engine.TensorMultiply(sample, engine.TensorMatMul(projected, engine.TensorTranspose(keyVector))),
            engine.TensorMatMul(projected, engine.TensorTranspose(keyPositions)));
        var weights = SoftmaxRows(engine.TensorMultiplyScalar(scores, Ops.FromDouble(1.0 / Math.Sqrt(Hidden))));
        var valueScale = engine.ReduceSum(engine.TensorMultiply(weights, sample), new[] { 1 }, keepDims: true);
        var read = engine.TensorAdd(engine.TensorMatMul(valueScale, valueVector), engine.TensorMatMul(weights, valuePositions));
        return engine.ReLU(engine.TensorAdd(
            engine.TensorMatMul(Concatenate(read, projected), engine.TensorTranspose(outWeights)),
            engine.Reshape(outBias, new[] { 1, Hidden })));
    }

    private Tensor<T> TransformerHidden(Tensor<T> sample, Tensor<T> query)
    {
        var engine = AiDotNetEngine.Current;
        var tokenWeights = _blocks[0];
        var sampleSegment = _blocks[1];
        var querySegment = _blocks[2];
        var queryProjection = _blocks[3];
        var keyProjection = _blocks[4];
        var valueProjection = _blocks[5];
        var feedIn = _blocks[6];
        var feedInBias = _blocks[7];
        var feedOut = _blocks[8];
        var feedOutBias = _blocks[9];
        T scale = Ops.FromDouble(1.0 / Math.Sqrt(Hidden));

        var t0 = engine.TensorAdd(engine.TensorMatMul(sample, engine.TensorTranspose(tokenWeights)), sampleSegment);
        var t1 = engine.TensorAdd(engine.TensorMatMul(query, engine.TensorTranspose(tokenWeights)), querySegment);
        Tensor<T> Project(Tensor<T> token, Tensor<T> weights) => engine.TensorMatMul(token, engine.TensorTranspose(weights));
        Tensor<T> Dot(Tensor<T> a, Tensor<T> b)
            => engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(a, b), new[] { 1 }, keepDims: true), scale);

        var q0 = Project(t0, queryProjection);
        var q1 = Project(t1, queryProjection);
        var k0 = Project(t0, keyProjection);
        var k1 = Project(t1, keyProjection);
        var v0 = Project(t0, valueProjection);
        var v1 = Project(t1, valueProjection);

        // Softmax over two tokens is a sigmoid of the score difference.
        Tensor<T> Attend(Tensor<T> token, Tensor<T> q)
        {
            var toFirst = Dot(q, k0);
            var toSecond = Dot(q, k1);
            var first = engine.Sigmoid(engine.TensorAdd(toFirst, engine.TensorNegate(toSecond)));
            var second = engine.Sigmoid(engine.TensorAdd(toSecond, engine.TensorNegate(toFirst)));
            var attended = engine.TensorAdd(token, engine.TensorAdd(engine.TensorMultiply(first, v0), engine.TensorMultiply(second, v1)));
            var inner = engine.ReLU(engine.TensorAdd(
                engine.TensorMatMul(attended, engine.TensorTranspose(feedIn)), engine.Reshape(feedInBias, new[] { 1, Hidden })));
            return engine.TensorAdd(attended, engine.TensorAdd(
                engine.TensorMatMul(inner, engine.TensorTranspose(feedOut)), engine.Reshape(feedOutBias, new[] { 1, Hidden })));
        }

        return engine.TensorMultiplyScalar(engine.TensorAdd(Attend(t0, q0), Attend(t1, q1)), Ops.FromDouble(0.5));
    }

    /// <summary>The layout: each block's shape and its initialisation bound, in flat-vector order.</summary>
    private static List<(int[] Shape, double Bound)> Blocks(RelationModuleType type, int width, int hidden)
    {
        double Bound(int fanIn) => 1.0 / Math.Sqrt(fanIn);
        switch (type)
        {
            case RelationModuleType.Convolution:
                return new List<(int[] Shape, double Bound)>
                {
                    (new[] { hidden, 6 }, Bound(6)), (new[] { hidden }, Bound(6)),
                    (new[] { 1, hidden }, Bound(hidden)), (new[] { 1 }, Bound(hidden)),
                };
            case RelationModuleType.Attention:
                return new List<(int[] Shape, double Bound)>
                {
                    (new[] { hidden, width }, Bound(width)),
                    (new[] { 1, hidden }, 1.0), (new[] { width, hidden }, Bound(hidden)),
                    (new[] { 1, hidden }, 1.0), (new[] { width, hidden }, Bound(hidden)),
                    (new[] { hidden, 2 * hidden }, Bound(2 * hidden)), (new[] { hidden }, Bound(2 * hidden)),
                    (new[] { 1, hidden }, Bound(hidden)), (new[] { 1 }, Bound(hidden)),
                };
            case RelationModuleType.Transformer:
                return new List<(int[] Shape, double Bound)>
                {
                    (new[] { hidden, width }, Bound(width)),
                    (new[] { 1, hidden }, Bound(hidden)), (new[] { 1, hidden }, Bound(hidden)),
                    (new[] { hidden, hidden }, Bound(hidden)), (new[] { hidden, hidden }, Bound(hidden)),
                    (new[] { hidden, hidden }, Bound(hidden)),
                    (new[] { hidden, hidden }, Bound(hidden)), (new[] { hidden }, Bound(hidden)),
                    (new[] { hidden, hidden }, Bound(hidden)), (new[] { hidden }, Bound(hidden)),
                    (new[] { 1, hidden }, Bound(hidden)), (new[] { 1 }, Bound(hidden)),
                };
            default:
                return new List<(int[] Shape, double Bound)>
                {
                    (new[] { hidden, 2 * width }, Bound(2 * width)), (new[] { hidden }, Bound(2 * width)),
                    (new[] { 1, hidden }, Bound(hidden)), (new[] { 1 }, Bound(hidden)),
                };
        }
    }

    /// <summary><c>[a, b]</c> side by side, <c>[rows, a + b]</c>, through constant placement matrices.</summary>
    private static Tensor<T> Concatenate(Tensor<T> left, Tensor<T> right)
    {
        var engine = AiDotNetEngine.Current;
        int m = left.Shape[1], n = right.Shape[1];
        var placeLeft = new Tensor<T>(new[] { m, m + n });
        for (int i = 0; i < m; i++) placeLeft[i * (m + n) + i] = Ops.One;
        var placeRight = new Tensor<T>(new[] { n, m + n });
        for (int i = 0; i < n; i++) placeRight[i * (m + n) + m + i] = Ops.One;
        return engine.TensorAdd(engine.TensorMatMul(left, placeLeft), engine.TensorMatMul(right, placeRight));
    }

    /// <summary><c>[width, width]</c>: <c>(x S)[d] = x[d + shift]</c>, zero outside the signal.</summary>
    private static Tensor<T> Shift(int width, int shift)
    {
        var matrix = new Tensor<T>(new[] { width, width });
        for (int d = 0; d < width; d++)
        {
            int source = d + shift;
            if (source >= 0 && source < width) matrix[source * width + d] = Ops.One;
        }

        return matrix;
    }

    /// <summary><c>[width * hidden, hidden]</c>: the mean over positions of a <c>[position, channel]</c> row.</summary>
    private static Tensor<T> AveragePositions(int width, int hidden)
    {
        var matrix = new Tensor<T>(new[] { width * hidden, hidden });
        T share = Ops.FromDouble(1.0 / width);
        for (int d = 0; d < width; d++)
            for (int c = 0; c < hidden; c++) matrix[(d * hidden + c) * hidden + c] = share;
        return matrix;
    }

    private static Tensor<T> Unit(int rows, int columns, int row, int column)
    {
        var matrix = new Tensor<T>(new[] { rows, columns });
        matrix[row * columns + column] = Ops.One;
        return matrix;
    }

    private static Tensor<T> SoftmaxRows(Tensor<T> scores)
    {
        var engine = AiDotNetEngine.Current;
        var max = engine.ReduceMax(scores, new[] { 1 }, keepDims: true, out _);
        var exp = engine.TensorExp(engine.TensorAdd(scores, engine.TensorNegate(engine.StopGradient(max))));
        return engine.TensorDivide(exp, engine.ReduceSum(exp, new[] { 1 }, keepDims: true));
    }
}
