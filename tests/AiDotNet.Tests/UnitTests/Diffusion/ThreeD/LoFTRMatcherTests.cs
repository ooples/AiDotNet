using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.NeuralRadianceFields.NeuS;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.ThreeD;

/// <summary>
/// LoFTR (Sun et al. 2021, zju3dv/LoFTR as vendored in One-2-3-45 <c>elevation_estimate/loftr/</c>): parameter
/// count and state-dict names, linear attention, dual-softmax coarse matching with border removal and mutual
/// nearest, fine-window expectation, and a deterministic tiny end-to-end match.
/// </summary>
public class LoFTRMatcherTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>A tiny configuration: widths 8/12/16, 2 heads, one self/cross pair at each level.</summary>
    internal static LoFTROptions TinyOptions(int seed, int inputSize = 0) => new LoFTROptions
    {
        InitialDim = 8,
        BlockDims = new[] { 8, 12, 16 },
        CoarseHeads = 2,
        CoarseLayerPairs = 1,
        FineHeads = 2,
        FineLayerPairs = 1,
        MatchThreshold = 0.0,
        BorderRemoval = 1,
        InputSize = inputSize,
        Seed = seed,
    };

    // ------------------------------------------------------------------ parameters

    [Fact]
    public void ParameterCountMatchesHandDerivationOnTinyConfig()
    {
        // Stem: conv 7·7·1·8 = 392, bn 2·8 = 16                                              → 408
        // layer1 (8→8, stride 1, no shortcut), 2 blocks of 9·8·8 + 16 + 9·8·8 + 16 = 1184     → 2368
        // layer2 block 0 (8→12, stride 2): 9·8·12 + 24 + 9·12·12 + 24 + 1·8·12 + 24 = 2328
        //        block 1 (12→12): 1296 + 24 + 1296 + 24 = 2640                                → 4968
        // layer3 block 0 (12→16, stride 2): 1728 + 32 + 2304 + 32 + 192 + 32 = 4320
        //        block 1 (16→16): 2304 + 32 + 2304 + 32 = 4672                                → 8992
        // FPN: 16·16 + 12·16 + (9·16·16 + 32 + 9·16·12) + 8·12 + (9·12·12 + 24 + 9·12·8)
        //    = 256 + 192 + 4064 + 96 + 2184                                                   → 6792
        // Coarse encoder layer (d = 16): q, k, v, merge 4·d² + mlp (2d·2d + 2d·d) + 2 norms 4d = 10d² + 4d = 2624,
        // two layers                                                                           → 5248
        // Fine preprocess: down_proj 16·8 + 8 = 136, merge_feat 16·8 + 8 = 136                  → 272
        // Fine encoder layer (d = 8): 10·64 + 32 = 672, two layers                               → 1344
        // Total 408 + 2368 + 4968 + 8992 + 6792 + 5248 + 272 + 1344 = 30392.
        var matcher = new LoFTRMatcher<float>(TinyOptions(seed: 1));
        Assert.Equal(30392, matcher.ParameterCount);

        // The running statistics are buffers: 2 · channels for each of the 17 batch norms.
        var state = matcher.GetParameters();
        int buffers = state.Where(p => p.Key.EndsWith(".running_mean", StringComparison.Ordinal)
            || p.Key.EndsWith(".running_var", StringComparison.Ordinal)).Sum(p => p.Value.Length);
        // bn1 8, layer1 4·8, layer2 5·12, layer3 5·16, layer2_outconv2.1 16, layer1_outconv2.1 12 = 208 channels.
        Assert.Equal(2 * 208, buffers);
        Assert.Equal(30392 + 2 * 208, state.Sum(p => p.Value.Length));
    }

    [Fact]
    public void OfficialConfigurationHasThePublishedParameterCount()
    {
        // The same derivation at 128/196/256 with 4 coarse pairs gives 11,561,456, LoFTR's published 11.56 M.
        var matcher = new LoFTRMatcher<float>(new LoFTROptions { Seed = 1 });
        Assert.Equal(11561456, matcher.ParameterCount);
    }

    [Fact]
    public void StateDictUsesReferenceNamesAndLoadsCheckpoints()
    {
        var source = new LoFTRMatcher<float>(TinyOptions(seed: 1));
        var state = source.GetParameters();
        foreach (var name in new[]
        {
            "backbone.conv1.weight", "backbone.bn1.running_var", "backbone.layer1.0.conv1.weight",
            "backbone.layer1.1.bn2.bias", "backbone.layer2.0.downsample.0.weight",
            "backbone.layer2.0.downsample.1.running_mean", "backbone.layer3_outconv.weight",
            "backbone.layer2_outconv2.0.weight", "backbone.layer2_outconv2.1.weight", "backbone.layer2_outconv2.3.weight",
            "backbone.layer1_outconv2.3.weight", "loftr_coarse.layers.0.q_proj.weight", "loftr_coarse.layers.1.merge.weight",
            "loftr_coarse.layers.1.mlp.0.weight", "loftr_coarse.layers.1.mlp.2.weight", "loftr_coarse.layers.0.norm1.weight",
            "fine_preprocess.down_proj.weight", "fine_preprocess.down_proj.bias", "fine_preprocess.merge_feat.bias",
            "loftr_fine.layers.1.norm2.bias",
        })
            Assert.True(state.ContainsKey(name), name);

        // Bias-free convolutions and projections, and no shortcut on the stride-1 stage, as in the reference.
        Assert.False(state.ContainsKey("backbone.conv1.bias"));
        Assert.False(state.ContainsKey("loftr_coarse.layers.0.q_proj.bias"));
        Assert.False(state.ContainsKey("backbone.layer1.0.downsample.0.weight"));
        Assert.Equal(new[] { 8, 1, 7, 7 }, state["backbone.conv1.weight"].Shape.ToArray());
        Assert.Equal(new[] { 32, 32 }, state["loftr_coarse.layers.0.mlp.0.weight"].Shape.ToArray());
        Assert.Equal(new[] { 8, 16 }, state["fine_preprocess.down_proj.weight"].Shape.ToArray());

        // A training checkpoint: "matcher." prefixes and num_batches_tracked counters, as in indoor_ds_new.ckpt.
        var checkpoint = new Dictionary<string, Tensor<float>>();
        foreach (var p in state) checkpoint["matcher." + p.Key] = p.Value;
        checkpoint["matcher.backbone.bn1.num_batches_tracked"] = new Tensor<float>(new[] { 1 });

        var target = new LoFTRMatcher<float>(TinyOptions(seed: 2));
        var (image0, image1) = ShiftedPair(32, 2, 1, seed: 4);
        Assert.NotEqual(Flatten(state), Flatten(target.GetParameters()));
        target.LoadParameters(checkpoint);
        Assert.Equal(Flatten(state), Flatten(target.GetParameters()));
        Assert.Equal(Describe(source.Match(image0, image1)), Describe(target.Match(image0, image1)));

        Assert.Throws<ArgumentException>(() => target.LoadParameters(
            new Dictionary<string, Tensor<float>> { ["backbone.conv9.weight"] = new Tensor<float>(new[] { 1 }) }));
        Assert.Throws<ArgumentException>(() => target.LoadParameters(
            new Dictionary<string, Tensor<float>> { ["backbone.conv1.weight"] = new Tensor<float>(new[] { 3 }) }));
    }

    // ------------------------------------------------------------------ linear attention

    [Fact]
    public void LinearAttentionMatchesAHandLoop()
    {
        const int n = 2, l = 5, s = 4, h = 2, d = 3;
        var random = RandomHelper.CreateSeededRandom(11);
        double[] q = Draw(random, n * l * h * d), k = Draw(random, n * s * h * d), v = Draw(random, n * s * h * d);
        var actual = NeuSHost<double>.ToDoubles(LoFTRNetwork<double>.LinearAttention(Engine,
            NeuSHost<double>.FromDoubles(q, n, l, h, d), NeuSHost<double>.FromDoubles(k, n, s, h, d),
            NeuSHost<double>.FromDoubles(v, n, s, h, d)));

        static double Phi(double x) => x > 0 ? x + 1 : Math.Exp(x); // elu(x) + 1
        Assert.Equal(n * l * h * d, actual.Length);
        for (int b = 0; b < n; b++)
            for (int head = 0; head < h; head++)
                for (int i = 0; i < l; i++)
                {
                    var numerator = new double[d];
                    double denominator = 0;
                    for (int j = 0; j < s; j++)
                    {
                        double dot = 0;
                        for (int c = 0; c < d; c++)
                            dot += Phi(q[((b * l + i) * h + head) * d + c]) * Phi(k[((b * s + j) * h + head) * d + c]);
                        denominator += dot;
                        for (int c = 0; c < d; c++) numerator[c] += dot * v[((b * s + j) * h + head) * d + c] / s;
                    }

                    for (int c = 0; c < d; c++)
                    {
                        double expected = numerator[c] / (denominator + 1e-6) * s;
                        Assert.Equal(expected, actual[((b * l + i) * h + head) * d + c], 10);
                    }
                }
    }

    // ------------------------------------------------------------------ coarse matching

    [Fact]
    public void DualSoftmaxAndMutualNearestMatchAHandReference()
    {
        const int h0 = 4, w0 = 5, h1 = 4, w1 = 4, c = 6;
        int l = h0 * w0, s = h1 * w1;
        var random = RandomHelper.CreateSeededRandom(5);
        double[] f0 = Draw(random, l * c), f1 = Draw(random, s * c);
        const double temperature = 0.1;
        var confidence = NeuSHost<double>.ToDoubles(LoFTRNetwork<double>.DualSoftmax(Engine,
            NeuSHost<double>.FromDoubles(f0, l, c), NeuSHost<double>.FromDoubles(f1, s, c), temperature));

        // conf = softmax over L · softmax over S of ⟨f₀/√C, f₁/√C⟩ / τ.
        var sim = new double[l, s];
        for (int i = 0; i < l; i++)
            for (int j = 0; j < s; j++)
            {
                double dot = 0;
                for (int k = 0; k < c; k++) dot += f0[i * c + k] / Math.Sqrt(c) * (f1[j * c + k] / Math.Sqrt(c));
                sim[i, j] = dot / temperature;
            }

        var expected = new double[l * s];
        for (int i = 0; i < l; i++)
            for (int j = 0; j < s; j++)
            {
                double row = 0, column = 0;
                for (int jj = 0; jj < s; jj++) row += Math.Exp(sim[i, jj]);
                for (int ii = 0; ii < l; ii++) column += Math.Exp(sim[ii, j]);
                expected[i * s + j] = Math.Exp(sim[i, j]) / column * (Math.Exp(sim[i, j]) / row);
            }

        for (int t = 0; t < expected.Length; t++) Assert.Equal(expected[t], confidence[t], 10);

        // Without threshold or border the global maximum is always a mutual nearest match.
        Assert.NotEmpty(LoFTRNetwork<double>.CoarseMatches(confidence, h0, w0, h1, w1, 0.0, 0));
        foreach (var (threshold, border) in new[] { (0.0, 0), (0.0, 1), (0.05, 0), (0.0, 2) })
        {
            var reference = ReferenceMatches(confidence, h0, w0, h1, w1, threshold, border);
            var actual = LoFTRNetwork<double>.CoarseMatches(confidence, h0, w0, h1, w1, threshold, border);
            Assert.Equal(reference, actual.Select(m => (m.I, m.J)).ToList());
            foreach (var m in actual) Assert.Equal(confidence[m.I * s + m.J], m.Confidence);
        }
    }

    [Fact]
    public void CoarseMatchingRecoversAPermutation()
    {
        // f1 is a permutation of distinct, nearly orthogonal features: every cell is its partner's mutual nearest.
        const int side = 4, c = 16;
        int l = side * side;
        var perm = Enumerable.Range(0, l).Select(i => (i * 5 + 3) % l).ToArray();
        var f0 = new double[l * c];
        var f1 = new double[l * c];
        for (int i = 0; i < l; i++) f0[i * c + i] = 4.0;
        for (int j = 0; j < l; j++) f1[j * c + perm[j]] = 4.0;
        var confidence = NeuSHost<double>.ToDoubles(LoFTRNetwork<double>.DualSoftmax(Engine,
            NeuSHost<double>.FromDoubles(f0, l, c), NeuSHost<double>.FromDoubles(f1, l, c), 0.1));
        var inverse = new int[l];
        for (int j = 0; j < l; j++) inverse[perm[j]] = j;
        var all = LoFTRNetwork<double>.CoarseMatches(confidence, side, side, side, side, 0.2, 0);
        Assert.Equal(Enumerable.Range(0, l).Select(i => (i, inverse[i])).ToList(), all.Select(m => (m.I, m.J)).ToList());

        var matches = LoFTRNetwork<double>.CoarseMatches(confidence, side, side, side, side, 0.2, 1);

        // Border 1 on a 4 × 4 grid keeps the 2 × 2 interior of both grids.
        var interior = new[] { 5, 6, 9, 10 };
        var expected = interior.SelectMany(i => Enumerable.Range(0, l).Where(j => perm[j] == i && interior.Contains(j))
            .Select(j => (i, j))).ToList();
        Assert.Equal(expected, matches.Select(m => (m.I, m.J)).ToList());
        Assert.All(matches, m => Assert.True(m.Confidence > 0.2));
    }

    // ------------------------------------------------------------------ fine refinement

    [Fact]
    public void FineExpectationRecoversAKnownShift()
    {
        // One-hot fine features per pixel; the second map is the first moved by (+1, −1) fine pixels, so the center
        // of every window reappears one pixel right and one up, i.e. at the normalized offset (+0.5, −0.5) for W = 5.
        const int hf = 12, wf = 12, window = 5, coarseWidth = 3, stride = 4;
        int c = hf * wf, ww = window * window;
        const int sx = 1, sy = -1;
        var rows0 = new double[(hf * wf + 1) * c];
        var rows1 = new double[(hf * wf + 1) * c];
        for (int y = 0; y < hf; y++)
            for (int x = 0; x < wf; x++)
            {
                rows0[(y * wf + x) * c + y * wf + x] = 30.0;
                int ox = x - sx, oy = y - sy;
                if (ox >= 0 && ox < wf && oy >= 0 && oy < hf) rows1[(y * wf + x) * c + oy * wf + ox] = 30.0;
            }

        var ids = new[] { 4, 5, 7 }; // interior coarse cells of the 3 × 3 grid: fine centers (4, 4), (4, 8), (8, 4)
        var index = LoFTRNetwork<double>.WindowRows(ids, coarseWidth, stride, window, hf, wf, hf * wf);
        Assert.Equal(4 * wf + 4, index[ww / 2]);
        Assert.Equal(2 * wf + 2, index[0]);
        var feat0 = Gather(rows0, index, ids.Length, ww, c);
        var feat1 = Gather(rows1, index, ids.Length, ww, c);
        var coords = NeuSHost<double>.ToDoubles(LoFTRNetwork<double>.FineExpectation(Engine, feat0, feat1, window));

        // Hand reference: softmax(⟨center, f1_r⟩ / √C) over the 25 window positions, expectation over the grid.
        var f0 = NeuSHost<double>.ToDoubles(feat0);
        var f1 = NeuSHost<double>.ToDoubles(feat1);
        for (int m = 0; m < ids.Length; m++)
        {
            var logits = new double[ww];
            for (int r = 0; r < ww; r++)
            {
                double dot = 0;
                for (int k = 0; k < c; k++) dot += f0[(m * ww + ww / 2) * c + k] * f1[(m * ww + r) * c + k];
                logits[r] = dot / Math.Sqrt(c);
            }

            double max = logits.Max(), sum = logits.Sum(z => Math.Exp(z - max));
            double ex = 0, ey = 0;
            for (int r = 0; r < ww; r++)
            {
                double p = Math.Exp(logits[r] - max) / sum;
                ex += p * (-1.0 + 2.0 * (r % window) / (window - 1));
                ey += p * (-1.0 + 2.0 * (r / window) / (window - 1));
            }

            Assert.Equal(ex, coords[2 * m], 10);
            Assert.Equal(ey, coords[2 * m + 1], 10);
            Assert.Equal(sx / 2.0, coords[2 * m], 6);
            Assert.Equal(sy / 2.0, coords[2 * m + 1], 6);
            // get_fine_match: offset · (W // 2) · (H_image / H_fine) image pixels: 0.5 · 2 · 2 = 2, one fine pixel.
            Assert.Equal(sx * 2.0, coords[2 * m] * (window / 2) * 2.0, 6);
        }
    }

    // ------------------------------------------------------------------ end to end

    [Fact]
    public void TinyMatcherIsDeterministicEndToEnd()
    {
        var (image0, image1) = ShiftedPair(64, 3, 2, seed: 8);
        var first = new LoFTRMatcher<float>(TinyOptions(seed: 5)).Match(image0, image1);
        var second = new LoFTRMatcher<float>(TinyOptions(seed: 5)).Match(image0, image1);
        Assert.NotEmpty(first);
        Assert.Equal(Describe(first), Describe(second));
        foreach (var m in first)
        {
            // Coarse keypoints of image 0 sit on the 8-pixel grid inside the 1-cell border; image 1's are refined
            // by at most (W // 2) · 2 = 4 pixels around its grid point.
            Assert.Equal(0.0, m.X0 % 8, 9);
            Assert.Equal(0.0, m.Y0 % 8, 9);
            Assert.InRange(m.X0, 8, 48);
            Assert.InRange(m.Y0, 8, 48);
            Assert.InRange(m.X1, 8 - 4 - 1e-6, 48 + 4 + 1e-6);
            Assert.InRange(m.Y1, 8 - 4 - 1e-6, 48 + 4 + 1e-6);
            Assert.True(m.Confidence > 0 && m.Confidence <= 1.0, "The confidence is a product of probabilities.");
        }

        // Resizing to the network size and back (elev_est_api's 480 × 480): keypoints come back in input pixels.
        var (small0, small1) = ShiftedPair(32, 1, 1, seed: 8);
        var resized = new LoFTRMatcher<float>(TinyOptions(seed: 5, inputSize: 64)).Match(small0, small1);
        Assert.NotEmpty(resized);
        Assert.All(resized, m => Assert.Equal(0.0, m.X0 % 4, 9));
        Assert.All(resized, m => Assert.InRange(m.X0, 4, 24));
    }

    // ------------------------------------------------------------------ helpers

    private static List<(int I, int J)> ReferenceMatches(double[] conf, int h0, int w0, int h1, int w1, double thr, int b)
    {
        // get_coarse_match: mask = conf > thr; mask_border on the [h0, w0, h1, w1] view; mutual nearest; mask.max(dim=2).
        int l = h0 * w0, s = h1 * w1;
        var mask = new bool[l, s];
        for (int i = 0; i < l; i++)
            for (int j = 0; j < s; j++)
            {
                int r0 = i / w0, c0 = i % w0, r1 = j / w1, c1 = j % w1;
                bool border = b > 0 && (r0 < b || r0 >= h0 - b || c0 < b || c0 >= w0 - b
                    || r1 < b || r1 >= h1 - b || c1 < b || c1 >= w1 - b);
                double rowMax = Enumerable.Range(0, s).Max(jj => conf[i * s + jj]);
                double columnMax = Enumerable.Range(0, l).Max(ii => conf[ii * s + j]);
                mask[i, j] = conf[i * s + j] > thr && !border && conf[i * s + j] == rowMax && conf[i * s + j] == columnMax;
            }

        var result = new List<(int, int)>();
        for (int i = 0; i < l; i++)
            for (int j = 0; j < s; j++)
                if (mask[i, j]) { result.Add((i, j)); break; }
        return result;
    }

    private static Tensor<double> Gather(double[] rows, int[] index, int m, int ww, int c)
    {
        var values = new double[index.Length * c];
        for (int r = 0; r < index.Length; r++) Array.Copy(rows, index[r] * c, values, r * c, c);
        return NeuSHost<double>.FromDoubles(values, m, ww, c);
    }

    private static (Tensor<float>, Tensor<float>) ShiftedPair(int size, int dx, int dy, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        int big = size + 8;
        var texture = new double[big, big];
        for (int y = 0; y < big; y++)
            for (int x = 0; x < big; x++)
                texture[y, x] = 0.5 + 0.3 * Math.Sin(x * 0.7 + y * 0.3) + 0.2 * random.NextDouble();
        var a = new double[size * size * 3];
        var b = new double[size * size * 3];
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
                for (int ch = 0; ch < 3; ch++)
                {
                    a[(y * size + x) * 3 + ch] = texture[y + 4, x + 4];
                    b[(y * size + x) * 3 + ch] = texture[y + 4 - dy, x + 4 - dx];
                }

        return (NeuSHost<float>.FromDoubles(a, size, size, 3), NeuSHost<float>.FromDoubles(b, size, size, 3));
    }

    private static double[] Draw(Random random, int count)
        => Enumerable.Range(0, count).Select(_ => random.NextDouble() * 2 - 1).ToArray();

    private static double[] Flatten(IReadOnlyDictionary<string, Tensor<float>> state)
        => state.SelectMany(p => NeuSHost<float>.ToDoubles(p.Value)).ToArray();

    private static List<(double, double, double, double, double)> Describe(IReadOnlyList<FeatureCorrespondence> matches)
        => matches.Select(m => (m.X0, m.Y0, m.X1, m.Y1, m.Confidence)).ToList();
}
