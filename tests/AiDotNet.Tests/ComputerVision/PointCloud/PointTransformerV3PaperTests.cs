using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ComputerVision.Segmentation.PointCloud;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ComputerVision.PointCloud;

/// <summary>
/// Point Transformer V3 (Wu et al., CVPR 2024) against Pointcept's <c>point_transformer_v3m1_base.py</c>
/// and <c>serialization/</c>: curve codes, serialized patch attention, grid pooling and unpooling, and
/// the parameter count of the published base configuration.
/// </summary>
public class PointTransformerV3PaperTests
{
    private const double Tolerance = 1e-9;

    // Pointcept serialization.encode(grid, depth=3, order) for every cell of a 4x4x4 grid enumerated
    // x-major, then y, then z (index = 16x + 4y + z); produced with Pointcept's z_order.py / hilbert.py.
    private static readonly long[] ZCodes =
    {
        0, 1, 8, 9, 2, 3, 10, 11, 16, 17, 24, 25, 18, 19, 26, 27, 4, 5, 12, 13, 6, 7, 14, 15, 20, 21, 28, 29, 22, 23, 30, 31,
        32, 33, 40, 41, 34, 35, 42, 43, 48, 49, 56, 57, 50, 51, 58, 59, 36, 37, 44, 45, 38, 39, 46, 47, 52, 53, 60, 61, 54, 55, 62, 63
    };

    private static readonly long[] ZTransCodes =
    {
        0, 1, 8, 9, 4, 5, 12, 13, 32, 33, 40, 41, 36, 37, 44, 45, 2, 3, 10, 11, 6, 7, 14, 15, 34, 35, 42, 43, 38, 39, 46, 47,
        16, 17, 24, 25, 20, 21, 28, 29, 48, 49, 56, 57, 52, 53, 60, 61, 18, 19, 26, 27, 22, 23, 30, 31, 50, 51, 58, 59, 54, 55, 62, 63
    };

    private static readonly long[] HilbertCodes =
    {
        0, 3, 60, 63, 7, 4, 59, 56, 8, 11, 52, 55, 9, 10, 53, 54, 1, 2, 61, 62, 6, 5, 58, 57, 15, 12, 51, 48, 14, 13, 50, 49,
        30, 31, 32, 33, 25, 24, 39, 38, 16, 23, 40, 47, 17, 22, 41, 46, 29, 28, 35, 34, 26, 27, 36, 37, 19, 20, 43, 44, 18, 21, 42, 45
    };

    private static readonly long[] HilbertTransCodes =
    {
        0, 3, 60, 63, 1, 2, 61, 62, 30, 31, 32, 33, 29, 28, 35, 34, 7, 4, 59, 56, 6, 5, 58, 57, 25, 24, 39, 38, 26, 27, 36, 37,
        8, 11, 52, 55, 15, 12, 51, 48, 16, 23, 40, 47, 19, 20, 43, 44, 9, 10, 53, 54, 14, 13, 50, 49, 17, 22, 41, 46, 18, 21, 42, 45
    };

    public static IEnumerable<object[]> GridCurves()
    {
        yield return new object[] { PointSerializationOrder.ZOrder, ZCodes };
        yield return new object[] { PointSerializationOrder.ZOrderTransposed, ZTransCodes };
        yield return new object[] { PointSerializationOrder.Hilbert, HilbertCodes };
        yield return new object[] { PointSerializationOrder.HilbertTransposed, HilbertTransCodes };
    }

    [Theory]
    [MemberData(nameof(GridCurves))]
    public void Curve_CodesMatchPointceptOnFourCubedGrid(PointSerializationOrder order, long[] expected)
    {
        var curve = SpaceFillingCurves.Get(order);
        var actual = new long[64];
        for (int x = 0; x < 4; x++)
            for (int y = 0; y < 4; y++)
                for (int z = 0; z < 4; z++)
                    actual[x * 16 + y * 4 + z] = curve.Encode(x, y, z, 3);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void Curve_CodesMatchPointceptAtDepthSixteen()
    {
        int[][] points = { new[] { 0, 0, 0 }, new[] { 5, 9, 3 }, new[] { 300, 17, 1023 }, new[] { 65535, 0, 12345 }, new[] { 1000, 2000, 3000 } };
        long[] z = { 0, 1295, 220642123, 161461319162661, 11810814464 };
        long[] hilbert = { 0, 2011, 170222766, 279423209514058, 63505582080 };
        for (int i = 0; i < points.Length; i++)
        {
            Assert.Equal(z[i], SpaceFillingCurves.ZOrder.Encode(points[i][0], points[i][1], points[i][2], 16));
            Assert.Equal(hilbert[i], SpaceFillingCurves.Hilbert.Encode(points[i][0], points[i][1], points[i][2], 16));
        }
    }

    [Fact]
    public void Serialization_PrefixesBatchAndUsesPointceptDepth()
    {
        // Pointcept: depth = bit_length(grid.max() + 1); code = batch << (3 * depth) | curve(grid).
        var grid = new[] { 0, 0, 0, 3, 1, 2, 2, 2, 2, 1, 0, 3 };
        var batch = new[] { 0, 0, 1, 1 };
        int depth = PointSerialization.DepthFor(grid);
        Assert.Equal(3, depth);
        var result = PointSerialization.Serialize(grid, batch, depth, new[] { SpaceFillingCurves.ZOrder });
        for (int i = 0; i < 4; i++)
        {
            long expected = ((long)batch[i] << (3 * depth)) | ZOrderCode(grid[i * 3], grid[i * 3 + 1], grid[i * 3 + 2], depth);
            Assert.Equal(expected, result.Codes[0][i]);
            Assert.Equal(i, result.Orders[0][result.Inverses[0][i]]);
        }
    }

    [Fact]
    public void SerializedAttention_EqualsPlainMultiHeadAttentionWithinOnePatch()
    {
        // 12 points <= patch 16: Pointcept's flash path makes the whole cloud one patch.
        var ctx = new PtContext<double>(RandomHelper.CreateSeededRandom(3), truncatedNormalInit: false);
        var attention = new SerializedPatchAttention<double>(ctx, 8, 2, 16);
        var level = RandomLevel(12, 8, seed: 5);
        var output = attention.Forward(level, level.Features, 0).ToArray();
        var expected = ReferenceAttention(attention, level.Features.ToArray(), Enumerable.Range(0, 12).ToArray(), 8, 2);
        AssertClose(expected, output, 12 * 8);
    }

    [Fact]
    public void SerializedAttention_AttendsWithinConsecutiveSerializedPatches()
    {
        // 32 points, patch 16: two patches of 16 consecutive points in serialized order, each plain attention.
        var ctx = new PtContext<double>(RandomHelper.CreateSeededRandom(11), truncatedNormalInit: false);
        var attention = new SerializedPatchAttention<double>(ctx, 8, 2, 16);
        var level = RandomLevel(32, 8, seed: 13);
        var output = attention.Forward(level, level.Features, 1).ToArray();
        var order = level.Serialization.Orders[1];
        var x = level.Features.ToArray();
        var expected = new double[32 * 8];
        foreach (var patch in new[] { order.Take(16).ToArray(), order.Skip(16).ToArray() })
        {
            var rows = ReferenceAttention(attention, x, patch, 8, 2);
            for (int i = 0; i < patch.Length; i++) Array.Copy(rows, patch[i] * 8, expected, patch[i] * 8, 8);
        }

        AssertClose(expected, output, 32 * 8);
    }

    [Fact]
    public void GridPooling_MaxPoolsEachCoarseCellAndUnpoolingRestoresEveryPoint()
    {
        var ctx = new PtContext<double>(RandomHelper.CreateSeededRandom(17), truncatedNormalInit: false);
        var pooling = new PointTransformerV3Pooling<double>(ctx, 4, 6, 2, PointTransformerV3Revision.M2Sonata, "down");
        var unpooling = new PointTransformerV3Unpooling<double>(ctx, 6, 4, 4, PointTransformerV3Revision.M2Sonata, "up");
        var level = RandomLevel(40, 4, seed: 19);
        var coarse = pooling.Forward(level, new[] { SpaceFillingCurves.ZOrder }, shuffleOrders: false, training: false);

        // Clusters are the stride-2 parent cells.
        var inverse = coarse.PoolingInverse ?? throw new InvalidOperationException("No cluster map.");
        var cells = new HashSet<(int, int, int)>();
        for (int i = 0; i < level.Count; i++)
        {
            var parent = (level.Grid[i * 3] / 2, level.Grid[i * 3 + 1] / 2, level.Grid[i * 3 + 2] / 2);
            cells.Add(parent);
            Assert.Equal(parent, (coarse.Grid[inverse[i] * 3], coarse.Grid[inverse[i] * 3 + 1], coarse.Grid[inverse[i] * 3 + 2]));
        }

        Assert.Equal(cells.Count, coarse.Count);

        // feat = GELU(LN(segment_max(x W + b))), coord = segment_mean(coord).
        var w = ctx.Created[0].ToArray();
        var b = ctx.Created[1].ToArray();
        var x = level.Features.ToArray();
        var pooled = new double[coarse.Count * 6];
        for (int i = 0; i < pooled.Length; i++) pooled[i] = double.NegativeInfinity;
        var sums = new double[coarse.Count * 3];
        var counts = new int[coarse.Count];
        for (int i = 0; i < level.Count; i++)
        {
            for (int o = 0; o < 6; o++)
            {
                double v = b[o];
                for (int c = 0; c < 4; c++) v += x[i * 4 + c] * w[c * 6 + o];
                pooled[inverse[i] * 6 + o] = Math.Max(pooled[inverse[i] * 6 + o], v);
            }

            for (int a = 0; a < 3; a++) sums[inverse[i] * 3 + a] += level.Coordinates[i * 3 + a];
            counts[inverse[i]]++;
        }

        var normalized = LayerNormRows(pooled, 6);
        var expected = AiDotNetEngine.Current.TensorGELU(new Tensor<double>(normalized, new[] { coarse.Count, 6 })).ToArray();
        AssertClose(expected, coarse.Features.ToArray(), expected.Length);
        for (int k = 0; k < coarse.Count; k++)
            for (int a = 0; a < 3; a++)
                Assert.Equal(sums[k * 3 + a] / counts[k], coarse.Coordinates[k * 3 + a], 12);

        // Unpooling: one row per original point, parent = skip(parent) + up(coarse)[inverse].
        var restored = unpooling.Forward(coarse, level, training: false);
        Assert.Equal(new[] { level.Count, 4 }, restored.Shape.ToArray());
        var noSkip = new PointCloudLevel<double>(new Tensor<double>(new[] { level.Count, 4 }), level.Coordinates,
            level.OriginCoordinates, level.Grid, level.Batch, level.BatchCount, level.Serialization);
        var upOnly = unpooling.Forward(coarse, noSkip, training: false).ToArray();
        for (int i = 0; i < level.Count; i++)
        {
            for (int j = 0; j < level.Count; j++)
            {
                if (inverse[i] != inverse[j]) continue;
                for (int c = 0; c < 4; c++) Assert.Equal(upOnly[i * 4 + c], upOnly[j * 4 + c], 12);
            }
        }
    }

    [Fact]
    public void DefaultConfiguration_HasPointceptPTv3m1BaseParameterCount()
    {
        // Pointcept PT-v3m1 base backbone (no seg head): 38,672,640 encoder + 7,493,632 decoder = 46,166,272,
        // the paper's 46.2M. Block = 40C^2 + 17C; stem SubMConv3d(6, 32, k=5, bias=False) + BN = 24,064.
        var encoder = new PointTransformerV3Encoder<float>(seed: 1);
        long count = encoder.GetTrainableParameters().Sum(t => (long)t.Length);
        Assert.Equal(46_166_272L, count);
        Assert.Equal(64, encoder.OutputChannels);
    }

    [Fact]
    public void Model_DefaultsAreThePaperConfiguration()
    {
        var options = new PointTransformerV3Options();
        Assert.Equal(new[] { 2, 2, 2, 6, 2 }, options.EncoderDepths);
        Assert.Equal(new[] { 32, 64, 128, 256, 512 }, options.EncoderChannels);
        Assert.Equal(new[] { 2, 4, 8, 16, 32 }, options.EncoderNumHeads);
        Assert.Equal(new[] { 2, 2, 2, 2 }, options.DecoderDepths);
        Assert.Equal(new[] { 64, 64, 128, 256 }, options.DecoderChannels);
        Assert.All(options.EncoderPatchSizes, p => Assert.Equal(1024, p));
        Assert.Equal(new[] { PointSerializationOrder.ZOrder, PointSerializationOrder.ZOrderTransposed, PointSerializationOrder.Hilbert, PointSerializationOrder.HilbertTransposed }
            .Select(SpaceFillingCurves.Get), options.SerializationCurves);
        Assert.Equal(new[] { "z", "z-trans", "hilbert", "hilbert-trans" }, options.SerializationCurves.Select(c => c.Name));
        Assert.Equal(4.0, options.MlpRatio);
        Assert.Equal(0.3, options.DropPathRate);
        Assert.True(options.EnableConditionalPositionalEncoding);
        Assert.True(options.PreNorm);
    }

    [Fact]
    public void Model_PredictsOneLogitRowPerPoint()
    {
        var arch = new NeuralNetworkArchitecture<float>(InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
            NetworkComplexity.Deep, 0, 32, 32, 3, 0);
        var model = new PointTransformerV3<float>(arch, options: PointCloudTestOptions.SmallPtv3(5));
        var cloud = PointCloudTestOptions.Cloud(48, 9, seed: 23);
        var logits = model.Predict(cloud);
        Assert.Equal(new[] { 48, 5 }, logits.Shape.ToArray());
        Assert.Equal(new[] { 48 }, model.GetClassMap(cloud).Shape.ToArray());
    }

    [Fact]
    public void CustomCurve_PassedThroughOptions_DrivesTheSerializationOrder()
    {
        var curve = new LexicographicCurve("test-lexicographic-options");
        var options = PointCloudTestOptions.SmallPtv3(5);
        options.SerializationCurves = new ISpaceFillingCurve[] { curve };
        var model = new PointTransformerV3<float>(PointCloudArchitecture(), options: options);
        var encoder = model.Encoder ?? throw new InvalidOperationException("The native model has an encoder.");
        Assert.Same(curve, Assert.Single(encoder.Curves));

        var level = encoder.PrepareInput(PointCloudTestOptions.Cloud(48, 9, seed: 29), training: false).Level;
        var serialization = level.Serialization;
        int depth = serialization.Depth;
        var order = serialization.Orders[0];
        for (int i = 0; i < level.Count; i++)
        {
            long expected = ((long)level.Batch[i] << (3 * depth))
                | curve.Encode(level.Grid[i * 3], level.Grid[i * 3 + 1], level.Grid[i * 3 + 2], depth);
            Assert.Equal(expected, serialization.Codes[0][i]);
        }

        for (int i = 1; i < order.Length; i++)
            Assert.True(serialization.Codes[0][order[i - 1]] <= serialization.Codes[0][order[i]], $"order breaks at {i}");

        // The same cloud under the paper's first curve (z) is ordered differently.
        var zOrder = PointSerialization.Serialize(level.Grid, level.Batch, depth, new[] { SpaceFillingCurves.ZOrder }).Orders[0];
        Assert.NotEqual(zOrder, order);
    }

    [Fact]
    public void RegisteredCustomCurve_SurvivesSaveAndLoad()
    {
        var curve = new LexicographicCurve("test-lexicographic-registered");
        SpaceFillingCurves.Register(curve);
        var options = PointCloudTestOptions.SmallPtv3(5);
        options.SerializationCurves = new ISpaceFillingCurve[] { curve, SpaceFillingCurves.Hilbert };
        var model = new PointTransformerV3<float>(PointCloudArchitecture(), options: options);
        var cloud = PointCloudTestOptions.Cloud(48, 9, seed: 31);
        var expected = model.Predict(cloud).ToArray();
        var saved = model.Serialize();

        var loaded = new PointTransformerV3<float>(PointCloudArchitecture(), options: PointCloudTestOptions.SmallPtv3(5));
        loaded.Deserialize(saved);
        var encoder = loaded.Encoder ?? throw new InvalidOperationException("The loaded model has an encoder.");
        Assert.Equal(new[] { "test-lexicographic-registered", "hilbert" }, encoder.Curves.Select(c => c.Name));
        Assert.Same(curve, encoder.Curves[0]);
        var actual = loaded.Predict(cloud).ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 5);
    }

    [Fact]
    public void UnregisteredCustomCurve_SaveThrowsNamingTheCurve()
    {
        var curve = new LexicographicCurve("test-lexicographic-unregistered");
        Assert.False(SpaceFillingCurves.IsRegistered(curve));
        var options = PointCloudTestOptions.SmallPtv3(5);
        options.SerializationCurves = new ISpaceFillingCurve[] { curve };
        var model = new PointTransformerV3<float>(PointCloudArchitecture(), options: options);

        var thrown = Assert.ThrowsAny<Exception>(() => model.Serialize());
        var cause = Unwrap(thrown).OfType<InvalidOperationException>().FirstOrDefault(e => e.Message.Contains("test-lexicographic-unregistered"));
        Assert.NotNull(cause);
        Assert.Contains(nameof(SpaceFillingCurves.Register), cause?.Message ?? string.Empty);
    }

    private static IEnumerable<Exception> Unwrap(Exception exception)
    {
        for (Exception? e = exception; e is not null; e = e.InnerException) yield return e;
    }

    private static NeuralNetworkArchitecture<float> PointCloudArchitecture()
        => new(InputType.ThreeDimensional, NeuralNetworkTaskType.Regression, NetworkComplexity.Deep, 0, 32, 32, 3, 0);

    /// <summary>x-major lexicographic order: a curve none of the built-in four produce.</summary>
    private sealed class LexicographicCurve : ISpaceFillingCurve
    {
        internal LexicographicCurve(string name) => Name = name;

        public string Name { get; }

        public long Encode(int x, int y, int z, int depth) => ((long)x << (2 * depth)) | ((long)y << depth) | (long)z;
    }

    private static PointCloudLevel<double> RandomLevel(int count, int channels, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var grid = new int[count * 3];
        var coordinates = new double[count * 3];
        var used = new HashSet<(int, int, int)>();
        for (int i = 0; i < count; i++)
        {
            (int, int, int) cell;
            do cell = (random.Next(8), random.Next(8), random.Next(8)); while (!used.Add(cell));
            grid[i * 3] = cell.Item1;
            grid[i * 3 + 1] = cell.Item2;
            grid[i * 3 + 2] = cell.Item3;
            for (int a = 0; a < 3; a++) coordinates[i * 3 + a] = (grid[i * 3 + a] + random.NextDouble()) * 0.02;
        }

        var features = new double[count * channels];
        for (int i = 0; i < features.Length; i++) features[i] = random.NextDouble() * 2 - 1;
        var batch = new int[count];
        var serialization = PointSerialization.Serialize(grid, batch, PointSerialization.DepthFor(grid), SpaceFillingCurves.PaperOrders());
        return new PointCloudLevel<double>(new Tensor<double>(features, new[] { count, channels }), coordinates, coordinates,
            grid, batch, 1, serialization);
    }

    /// <summary>Plain multi-head self-attention over the given rows, in double, with the layer's weights.</summary>
    private static double[] ReferenceAttention(SerializedPatchAttention<double> attention, double[] x, int[] rows, int c, int h)
    {
        var wq = attention.Qkv.Weight.ToArray();
        var bq = (attention.Qkv.Bias ?? throw new InvalidOperationException("qkv bias expected")).ToArray();
        var wo = attention.Projection.Weight.ToArray();
        var bo = (attention.Projection.Bias ?? throw new InvalidOperationException("proj bias expected")).ToArray();
        int l = rows.Length, d = c / h;
        var qkv = new double[l, 3 * c];
        for (int i = 0; i < l; i++)
            for (int o = 0; o < 3 * c; o++)
            {
                double v = bq[o];
                for (int k = 0; k < c; k++) v += x[rows[i] * c + k] * wq[k * 3 * c + o];
                qkv[i, o] = v;
            }

        var mixed = new double[l, c];
        double scale = 1.0 / Math.Sqrt(d);
        for (int head = 0; head < h; head++)
            for (int i = 0; i < l; i++)
            {
                var scores = new double[l];
                double max = double.NegativeInfinity;
                for (int j = 0; j < l; j++)
                {
                    double s = 0;
                    for (int k = 0; k < d; k++) s += qkv[i, head * d + k] * qkv[j, c + head * d + k];
                    scores[j] = s * scale;
                    max = Math.Max(max, scores[j]);
                }

                double total = 0;
                for (int j = 0; j < l; j++) total += scores[j] = Math.Exp(scores[j] - max);
                for (int k = 0; k < d; k++)
                {
                    double v = 0;
                    for (int j = 0; j < l; j++) v += scores[j] / total * qkv[j, 2 * c + head * d + k];
                    mixed[i, head * d + k] = v;
                }
            }

        var result = new double[x.Length];
        for (int i = 0; i < l; i++)
            for (int o = 0; o < c; o++)
            {
                double v = bo[o];
                for (int k = 0; k < c; k++) v += mixed[i, k] * wo[k * c + o];
                result[rows[i] * c + o] = v;
            }

        return result;
    }

    private static double[] LayerNormRows(double[] values, int width)
    {
        var result = new double[values.Length];
        for (int r = 0; r < values.Length / width; r++)
        {
            double mean = 0, variance = 0;
            for (int c = 0; c < width; c++) mean += values[r * width + c];
            mean /= width;
            for (int c = 0; c < width; c++) variance += Math.Pow(values[r * width + c] - mean, 2);
            variance /= width;
            for (int c = 0; c < width; c++) result[r * width + c] = (values[r * width + c] - mean) / Math.Sqrt(variance + 1e-5);
        }

        return result;
    }

    private static long ZOrderCode(int x, int y, int z, int depth)
    {
        long code = 0;
        for (int i = 0; i < depth; i++)
        {
            code |= (long)((x >> i) & 1) << (3 * i + 2);
            code |= (long)((y >> i) & 1) << (3 * i + 1);
            code |= (long)((z >> i) & 1) << (3 * i);
        }

        return code;
    }

    private static void AssertClose(double[] expected, double[] actual, int length)
    {
        Assert.Equal(length, actual.Length);
        for (int i = 0; i < length; i++) Assert.True(Math.Abs(expected[i] - actual[i]) < Tolerance, $"index {i}: expected {expected[i]}, got {actual[i]}");
    }
}

/// <summary>CI-scale PTv3 and Sonata configurations and synthetic clouds shared by the point-cloud tests.</summary>
internal static class PointCloudTestOptions
{
    /// <summary>Two stages, 8/16 channels, 16-point patches, 5 cm voxels.</summary>
    internal static PointTransformerV3Options SmallPtv3(int classes = 20) => new()
    {
        NumClasses = classes, InputChannels = 6, GridSize = 0.05, Strides = new[] { 2 }, EncoderDepths = new[] { 1, 1 },
        EncoderChannels = new[] { 8, 16 }, EncoderNumHeads = new[] { 2, 2 }, EncoderPatchSizes = new[] { 16, 16 },
        DecoderDepths = new[] { 1 }, DecoderChannels = new[] { 8 }, DecoderNumHeads = new[] { 2 },
        DecoderPatchSizes = new[] { 16 }, Seed = 7
    };

    /// <summary>Two stages, up-cast one level (24 channels), 16 prototypes.</summary>
    internal static SonataOptions SmallSonata(int classes = 20) => new()
    {
        NumClasses = classes, InputChannels = 9, GridSize = 0.05, Strides = new[] { 2 }, EncoderDepths = new[] { 1, 1 },
        EncoderChannels = new[] { 8, 16 }, EncoderNumHeads = new[] { 2, 2 }, EncoderPatchSizes = new[] { 16, 16 },
        UpCastLevel = 1, HeadHiddenChannels = 16, HeadEmbedChannels = 8, PrototypeCount = 16, Seed = 7
    };

    /// <summary>A <c>[points, width]</c> cloud in the unit cube: xyz, then values in [-1, 1].</summary>
    internal static Tensor<float> Cloud(int points, int width, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var data = new float[points * width];
        for (int i = 0; i < points; i++)
            for (int c = 0; c < width; c++)
                data[i * width + c] = c < 3 ? (float)random.NextDouble() : (float)(random.NextDouble() * 2 - 1);
        return new Tensor<float>(data, new[] { points, width });
    }
}
