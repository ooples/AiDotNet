using System;
using System.Linq;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// T5 attention as the paper and HF <c>T5Attention</c> define it: unscaled <c>q k^T</c> scores plus the
/// relative bias, independent head width (d_kv), causal masking for the decoder, and cross-attention.
/// </summary>
public class T5RelativeBiasAttentionBehaviourTests
{
    private static Tensor<double> Random2D(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(new[] { rows, cols });
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() - 0.5;
        return t;
    }

    private static double RowDiff(Tensor<double> a, Tensor<double> b, int row) =>
        Enumerable.Range(0, a.Shape[1]).Sum(c => Math.Abs(a[row, c] - b[row, c]));

    [Fact]
    public void Forward_IsUnscaledScoresPlusTheRelativeBias()
    {
        // hidden 4, 2 heads of d_kv 3 (inner 6, so d_kv is NOT hidden / heads), 8 buckets.
        const int hidden = 4, heads = 2, dkv = 3, inner = heads * dkv, buckets = 8, s = 3;
        using var layer = new T5RelativeBiasAttentionLayer<double>(hidden, heads, buckets, 16, bidirectional: true, seed: 5, keyValueDim: dkv);
        using var x = Random2D(s, hidden, 6);
        var output = layer.Forward(x);
        var p = layer.GetParameters();
        Assert.Equal((3 * hidden * inner) + (inner * hidden) + (buckets * heads), p.Length);
        double W(int offset, int cols, int r, int c) => p[offset + (r * cols) + c];
        int qOff = 0, kOff = hidden * inner, vOff = 2 * hidden * inner, oOff = 3 * hidden * inner, bOff = oOff + (inner * hidden);
        double Proj(int off, int row, int col) => Enumerable.Range(0, hidden).Sum(d => x[row, d] * W(off, inner, d, col));

        var merged = new double[s, inner];
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < s; i++)
            {
                var scores = Enumerable.Range(0, s).Select(j =>
                    Enumerable.Range(0, dkv).Sum(c => Proj(qOff, i, (h * dkv) + c) * Proj(kOff, j, (h * dkv) + c))
                    + p[bOff + (T5RelativeBiasAttentionLayer<double>.RelativePositionBucket(j - i, true, buckets, 16) * heads) + h]).ToArray();
                double max = scores.Max();
                var w = scores.Select(v => Math.Exp(v - max)).ToArray();
                double sum = w.Sum();
                for (int c = 0; c < dkv; c++)
                    merged[i, (h * dkv) + c] = Enumerable.Range(0, s).Sum(j => w[j] / sum * Proj(vOff, j, (h * dkv) + c));
            }
        for (int i = 0; i < s; i++)
            for (int d = 0; d < hidden; d++)
            {
                double expected = Enumerable.Range(0, inner).Sum(c => merged[i, c] * W(oOff, hidden, c, d));
                Assert.Equal(expected, output[i, d], 9);
            }
    }

    [Fact]
    public void UnidirectionalLayer_MasksFutureKeys()
    {
        using var decoder = new T5RelativeBiasAttentionLayer<double>(8, 2, bidirectional: false, seed: 1);
        using var encoder = new T5RelativeBiasAttentionLayer<double>(8, 2, bidirectional: true, seed: 1);
        using var a = Random2D(4, 8, 2);
        using var b = a.Clone();
        for (int d = 0; d < 8; d++) b[3, d] += 1.0;
        var da = decoder.Forward(a);
        var db = decoder.Forward(b);
        for (int row = 0; row < 3; row++)
            Assert.True(RowDiff(da, db, row) < 1e-12, $"decoder row {row} saw a later token: not causal");
        Assert.True(RowDiff(encoder.Forward(a), encoder.Forward(b), 0) > 1e-9, "The bidirectional layer should see the last token from row 0.");
    }

    [Fact]
    public void CrossAttention_ReadsTheMemory()
    {
        using var layer = new T5RelativeBiasAttentionLayer<double>(8, 2, seed: 3, usesExternalPositionBias: true);
        using var x = Random2D(2, 8, 4);
        using var memoryA = Random2D(5, 8, 5);
        using var memoryB = Random2D(5, 8, 6);
        var a = layer.Forward(x, memoryA, null, causal: false);
        var b = layer.Forward(x, memoryB, null, causal: false);
        Assert.Equal(new[] { 2, 8 }, a.Shape.ToArray());
        Assert.True(RowDiff(a, b, 0) > 1e-9, "Cross-attention ignored its memory.");
    }
}
