using System;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Engine;

/// <summary>
/// A correct, CPU reference implementation of a paged-attention causal language model: a small decoder-only
/// transformer that stores its key/value cache in the engine's paged blocks and decodes incrementally. It
/// implements the fast-path <see cref="ICausalLmRunner{T}"/> (paged prefill / single-token decode over block
/// tables) and, for verification, the universal <see cref="ICausalLmModel{T}"/> (full recompute). The two paths
/// produce identical logits — which is exactly the correctness contract a production (GPU-backed) paged runner
/// must satisfy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> this is the "real" fast-path model the serving engine was built for. Instead of
/// re-reading the whole conversation each token, it keeps its memory (the KV cache) in the paged blocks the
/// engine manages and only computes the new token's contribution each step. It is intentionally small and runs
/// on the CPU so it is easy to read and to test; a production runner would do the same math on a GPU. Because it
/// also offers a plain full-recompute path, tests can prove the paged path gives the same answer.</para>
/// </remarks>
/// <typeparam name="T">The numeric type exposed to the engine.</typeparam>
public sealed class ReferencePagedAttentionRunner<T> : ICausalLmRunner<T>, ICausalLmModel<T>
{
    private readonly int _vocab, _dModel, _numLayers, _numHeads, _headDim, _ffnDim, _blockSize, _maxBlocks;
    private readonly double _attnScale;

    // Weights (row-major [outDim][inDim]).
    private readonly double[][] _embed;      // [vocab][dModel]
    private readonly double[][] _posEnc;     // [maxPos][dModel]
    private readonly double[][][] _wq, _wk, _wv, _wo; // [layer][dModel][dModel]
    private readonly double[][][] _w1;       // [layer][ffnDim][dModel]
    private readonly double[][][] _w2;       // [layer][dModel][ffnDim]
    private readonly double[][] _outProj;    // [vocab][dModel]

    // Paged KV store, indexed by global slot = blockId * blockSize + slotInBlock.
    private readonly double[][][] _kStore;   // [layer][maxBlocks*blockSize][dModel]
    private readonly double[][][] _vStore;

    /// <summary>Builds a reference paged runner with deterministic (seeded) weights.</summary>
    public ReferencePagedAttentionRunner(
        int vocabularySize, int dModel = 32, int numLayers = 2, int numHeads = 4, int ffnDim = 64,
        int blockSize = 8, int maxBlocks = 256, int maxPositions = 2048, int? eosTokenId = null, int seed = 12345)
    {
        if (dModel % numHeads != 0) throw new ArgumentException("dModel must be divisible by numHeads.");
        _vocab = vocabularySize;
        _dModel = dModel;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _headDim = dModel / numHeads;
        _ffnDim = ffnDim;
        _blockSize = blockSize;
        _maxBlocks = maxBlocks;
        _attnScale = 1.0 / Math.Sqrt(_headDim);
        EosTokenId = eosTokenId;

        var rng = RandomHelper.CreateSeededRandom(seed);
        _embed = Matrix(rng, vocabularySize, dModel);
        _posEnc = SinusoidalPositions(maxPositions, dModel);
        _wq = Layers(rng, numLayers, dModel, dModel);
        _wk = Layers(rng, numLayers, dModel, dModel);
        _wv = Layers(rng, numLayers, dModel, dModel);
        _wo = Layers(rng, numLayers, dModel, dModel);
        _w1 = Layers(rng, numLayers, ffnDim, dModel);
        _w2 = Layers(rng, numLayers, dModel, ffnDim);
        _outProj = Matrix(rng, vocabularySize, dModel);

        _kStore = new double[numLayers][][];
        _vStore = new double[numLayers][][];
        int slots = maxBlocks * blockSize;
        for (int l = 0; l < numLayers; l++)
        {
            _kStore[l] = new double[slots][];
            _vStore[l] = new double[slots][];
        }
    }

    // ---- Contract properties ----
    /// <inheritdoc/>
    public int VocabularySize => _vocab;
    /// <inheritdoc/>
    public int NumLayers => _numLayers;
    /// <inheritdoc/>
    public int NumKvHeads => _numHeads;
    /// <inheritdoc/>
    public int HeadDim => _headDim;
    /// <inheritdoc/>
    public int BlockSize => _blockSize;
    /// <inheritdoc/>
    public int? EosTokenId { get; }

    // ---- Fast path: paged prefill / decode ----

    /// <inheritdoc/>
    public Tensor<T> Prefill(
        System.Collections.Generic.IReadOnlyList<System.Collections.Generic.IReadOnlyList<int>> tokenIdsPerSequence,
        System.Collections.Generic.IReadOnlyList<SequenceKvLayout> layouts,
        System.Collections.Generic.IReadOnlyList<int> tokenCounts)
    {
        var result = new Tensor<T>(new[] { tokenIdsPerSequence.Count, _vocab });
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int s = 0; s < tokenIdsPerSequence.Count; s++)
        {
            var tokens = tokenIdsPerSequence[s];
            var blockTable = layouts[s].BlockTable;
            int start = layouts[s].FilledTokens;
            int end = start + tokenCounts[s];
            double[] last = Array.Empty<double>();
            for (int pos = start; pos < end; pos++)
                last = ForwardPositionPaged(tokens[pos], pos, blockTable);
            for (int v = 0; v < _vocab; v++) result[s, v] = numOps.FromDouble(last[v]);
        }
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> DecodeStep(
        System.Collections.Generic.IReadOnlyList<int> lastTokenIds,
        System.Collections.Generic.IReadOnlyList<SequenceKvLayout> layouts)
    {
        var result = new Tensor<T>(new[] { lastTokenIds.Count, _vocab });
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int s = 0; s < lastTokenIds.Count; s++)
        {
            int pos = layouts[s].FilledTokens; // the position of the token being computed this step
            double[] logits = ForwardPositionPaged(lastTokenIds[s], pos, layouts[s].BlockTable);
            for (int v = 0; v < _vocab; v++) result[s, v] = numOps.FromDouble(logits[v]);
        }
        return result;
    }

    /// <inheritdoc/>
    public void CopyBlocks(System.Collections.Generic.IReadOnlyList<BlockCopy> copies)
    {
        foreach (var copy in copies)
        {
            int src = copy.Source * _blockSize, dst = copy.Destination * _blockSize;
            for (int l = 0; l < _numLayers; l++)
                for (int i = 0; i < _blockSize; i++)
                {
                    _kStore[l][dst + i] = _kStore[l][src + i] is { } k ? (double[])k.Clone() : null!;
                    _vStore[l][dst + i] = _vStore[l][src + i] is { } v ? (double[])v.Clone() : null!;
                }
        }
    }

    // Computes logits for one token at absolute position `pos`, writing its KV into the paged store and
    // attending causally over the sequence's cached KV (positions 0..pos via the block table).
    private double[] ForwardPositionPaged(int token, int pos, System.Collections.Generic.IReadOnlyList<int> blockTable)
    {
        var x = AddPos(_embed[token], pos);
        for (int l = 0; l < _numLayers; l++)
        {
            var q = MatVec(_wq[l], x);
            var k = MatVec(_wk[l], x);
            var v = MatVec(_wv[l], x);
            int slot = GlobalSlot(blockTable, pos);
            _kStore[l][slot] = k;
            _vStore[l][slot] = v;

            var attn = new double[_dModel];
            for (int h = 0; h < _numHeads; h++)
            {
                int off = h * _headDim;
                var scores = new double[pos + 1];
                for (int j = 0; j <= pos; j++)
                {
                    int js = GlobalSlot(blockTable, j);
                    scores[j] = DotHead(q, _kStore[l][js], off) * _attnScale;
                }
                Softmax(scores);
                for (int j = 0; j <= pos; j++)
                {
                    int js = GlobalSlot(blockTable, j);
                    var vj = _vStore[l][js];
                    for (int d = 0; d < _headDim; d++) attn[off + d] += scores[j] * vj[off + d];
                }
            }
            AddInPlace(x, MatVec(_wo[l], attn));  // attention residual
            AddInPlace(x, FeedForward(l, x));     // FFN residual
        }
        return MatVec(_outProj, x);
    }

    // ---- Universal path: full recompute (reference for correctness) ----

    /// <inheritdoc/>
    public Tensor<T> ForwardLogits(Tensor<T> tokenIds)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = tokenIds.Shape[tokenIds.Shape.Length - 1];
        var tokens = new int[n];
        for (int i = 0; i < n; i++) tokens[i] = (int)Math.Round(numOps.ToDouble(tokenIds[0, i]));

        // Fresh local KV (no paging): identical math, so identical logits to the paged path.
        var kLocal = new double[_numLayers][][];
        var vLocal = new double[_numLayers][][];
        for (int l = 0; l < _numLayers; l++) { kLocal[l] = new double[n][]; vLocal[l] = new double[n][]; }

        var result = new Tensor<T>(new[] { 1, n, _vocab });
        for (int pos = 0; pos < n; pos++)
        {
            var x = AddPos(_embed[tokens[pos]], pos);
            for (int l = 0; l < _numLayers; l++)
            {
                var q = MatVec(_wq[l], x);
                kLocal[l][pos] = MatVec(_wk[l], x);
                vLocal[l][pos] = MatVec(_wv[l], x);

                var attn = new double[_dModel];
                for (int h = 0; h < _numHeads; h++)
                {
                    int off = h * _headDim;
                    var scores = new double[pos + 1];
                    for (int j = 0; j <= pos; j++) scores[j] = DotHead(q, kLocal[l][j], off) * _attnScale;
                    Softmax(scores);
                    for (int j = 0; j <= pos; j++)
                        for (int d = 0; d < _headDim; d++) attn[off + d] += scores[j] * vLocal[l][j][off + d];
                }
                AddInPlace(x, MatVec(_wo[l], attn));
                AddInPlace(x, FeedForward(l, x));
            }
            var logits = MatVec(_outProj, x);
            for (int v = 0; v < _vocab; v++) result[0, pos, v] = numOps.FromDouble(logits[v]);
        }
        return result;
    }

    // ---- Math helpers ----

    private int GlobalSlot(System.Collections.Generic.IReadOnlyList<int> blockTable, int pos)
        => blockTable[pos / _blockSize] * _blockSize + (pos % _blockSize);

    private double[] FeedForward(int layer, double[] x)
    {
        var h = MatVec(_w1[layer], x);
        for (int i = 0; i < h.Length; i++) h[i] = Math.Max(0.0, h[i]); // ReLU
        return MatVec(_w2[layer], h);
    }

    private double[] AddPos(double[] embed, int pos)
    {
        var x = new double[_dModel];
        var pe = _posEnc[pos % _posEnc.Length];
        for (int i = 0; i < _dModel; i++) x[i] = embed[i] + pe[i];
        return x;
    }

    // Dot product of one head's slice ([off, off+headDim)) of two dModel-length vectors.
    private double DotHead(double[] a, double[] b, int off)
    {
        double s = 0.0;
        for (int d = 0; d < _headDim; d++) s += a[off + d] * b[off + d];
        return s;
    }

    private static double[] MatVec(double[][] w, double[] x)
    {
        var y = new double[w.Length];
        for (int o = 0; o < w.Length; o++)
        {
            double s = 0.0;
            var row = w[o];
            for (int i = 0; i < row.Length; i++) s += row[i] * x[i];
            y[o] = s;
        }
        return y;
    }

    private static void AddInPlace(double[] x, double[] delta)
    {
        for (int i = 0; i < x.Length; i++) x[i] += delta[i];
    }

    private static void Softmax(double[] s)
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < s.Length; i++) if (s[i] > max) max = s[i];
        double sum = 0.0;
        for (int i = 0; i < s.Length; i++) { s[i] = Math.Exp(s[i] - max); sum += s[i]; }
        if (sum <= 0.0) sum = 1.0;
        for (int i = 0; i < s.Length; i++) s[i] /= sum;
    }

    private static double[][] Matrix(Random rng, int rows, int cols)
    {
        var m = new double[rows][];
        for (int r = 0; r < rows; r++)
        {
            m[r] = new double[cols];
            for (int c = 0; c < cols; c++) m[r][c] = (rng.NextDouble() - 0.5) * 0.2; // ~U(-0.1, 0.1)
        }
        return m;
    }

    private static double[][][] Layers(Random rng, int layers, int rows, int cols)
    {
        var w = new double[layers][][];
        for (int l = 0; l < layers; l++) w[l] = Matrix(rng, rows, cols);
        return w;
    }

    private double[][] SinusoidalPositions(int maxPos, int dModel)
    {
        var pe = new double[maxPos][];
        for (int pos = 0; pos < maxPos; pos++)
        {
            pe[pos] = new double[dModel];
            for (int i = 0; i < dModel; i++)
            {
                double angle = pos / Math.Pow(10000.0, (2.0 * (i / 2)) / dModel);
                pe[pos][i] = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
            }
        }
        return pe;
    }
}
