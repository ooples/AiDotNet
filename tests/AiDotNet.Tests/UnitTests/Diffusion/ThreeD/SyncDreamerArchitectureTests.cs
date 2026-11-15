using System;
using System.Linq;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.VAE;
using AiDotNet.NeuralRadianceFields.NeuS;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.ThreeD;

/// <summary>
/// Reference tests for SyncDreamer (Liu et al. 2023): the paper's camera layout and sampler defaults, the spatial
/// volume's shape and its consistency under a permutation of the views, the frustum geometry, the depth-wise attention
/// against a hand-written reference, the DDIM schedule and one synchronized sampling step, seeded determinism, and
/// output that depends on the input image, on a tiny configuration.
/// </summary>
public class SyncDreamerArchitectureTests
{
    private static readonly IEngine Engine = AiDotNetEngine.Current;

    // ------------------------------------------------------------------ helpers

    private static SyncDreamerOptions Tiny() => new SyncDreamerOptions
    {
        ViewCount = 4,
        ImageSize = 64,
        FocalLength = 70,
        UNetBaseChannels = 8,
        UNetChannelMultipliers = new[] { 1, 2 },
        UNetNumResBlocks = 1,
        UNetAttentionLevels = new[] { 1 },
        UNetHeads = 2,
        CrossAttentionDim = 16,
        TimeEmbedDim = 8,
        TargetEncoderWidth = 8,
        TargetEncoderOutputDim = 4,
        NormGroups = 4,
        SpatialVolumeSize = 8,
        SpatialVolumeDims = new[] { 8, 8, 8, 8 },
        FrustumVolumeDepth = 8,
        FrustumVolumeDims = new[] { 8, 8, 8, 8 },
        DepthAttentionHeads = 2,
        ClipImageSize = 16,
        ClipPatchSize = 8,
        ClipWidth = 16,
        ClipLayers = 1,
        ClipHeads = 2,
        SamplingSteps = 2,
        ReconstructionIterations = 2,
        ReconstructionWarmUpIterations = 0,
        MeshResolution = 12,
        PointCount = 16,
        Seed = 11,
    };

    private static NeuSOptions TinyNeuS() => new NeuSOptions
    {
        SdfHiddenDim = 16,
        SdfHiddenLayers = 2,
        SdfFeatureDim = 8,
        SdfSkipLayers = Array.Empty<int>(),
        SdfFrequencies = 2,
        ColorHiddenDim = 16,
        ColorHiddenLayers = 1,
        ViewFrequencies = 1,
        Samples = 8,
        ImportanceSamples = 8,
        UpSampleSteps = 4,
        BatchSize = 16,
        MaskWeight = 0.0,
        WarmUpIterations = 0,
        Iterations = 2,
        GridChunkSize = 512,
        Seed = 5,
    };

    // A tiny model whose zero-initialized U-Net output has been replaced by small random weights, so its noise
    // prediction (and everything downstream) depends on the conditioning.
    private static SyncDreamerModel<double> TinyModel()
    {
        var vae = new StandardVAE<double>(inputChannels: 3, latentChannels: 4, baseChannels: 32,
            channelMultipliers: new[] { 1, 1, 1, 1 }, numResBlocksPerLevel: 1, latentScaleFactor: 0.18215, seed: 3);
        var model = new SyncDreamerModel<double>(Tiny(), vae: vae) { ReconstructionOptions = TinyNeuS() };
        Randomize(model.UNet.GetParameters, model.UNet.SetParameters, 21, 0.1);
        return model;
    }

    private static void Randomize(Func<Vector<double>> get, Action<Vector<double>> set, int seed, double scale)
    {
        int count = get().Length;
        var random = RandomHelper.CreateSeededRandom(seed);
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = (random.NextDouble() * 2 - 1) * scale;
        set(new Vector<double>(values));
    }

    private static Tensor<double> Noise(int[] shape, int seed, double scale = 1.0)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        int count = shape.Aggregate(1, (a, b) => a * b);
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = (random.NextDouble() * 2 - 1) * scale;
        return new Tensor<double>(values, shape);
    }

    private static Tensor<double> Image(int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var values = new double[3 * 64 * 64];
        for (int i = 0; i < values.Length; i++) values[i] = random.NextDouble();
        return new Tensor<double>(values, new[] { 1, 3, 64, 64 });
    }

    private static double[] Data(Tensor<double> tensor) => tensor.ToArray();

    private static double MaxDifference(Tensor<double> a, Tensor<double> b)
    {
        Assert.Equal(a.Shape.ToArray(), b.Shape.ToArray());
        var x = Data(a);
        var y = Data(b);
        double max = 0;
        for (int i = 0; i < x.Length; i++) max = Math.Max(max, Math.Abs(x[i] - y[i]));
        return max;
    }

    private static Tensor<double> Row(Tensor<double> t, int index)
    {
        var start = new int[t.Shape.ToArray().Length];
        var length = t.Shape.ToArray();
        start[0] = index;
        length[0] = 1;
        return Engine.TensorSlice(t, start, length);
    }

    // ------------------------------------------------------------------ camera layout

    [Fact]
    public void CameraLayout_MatchesThePaper()
    {
        var options = new SyncDreamerOptions();
        Assert.Equal(16, options.ViewCount);
        Assert.Equal(30.0, options.TargetElevationDegrees);
        Assert.Equal(1.5, options.CameraDistance);
        Assert.Equal(256, options.ImageSize);
        Assert.Equal(50, options.SamplingSteps);
        Assert.Equal(2.0, options.GuidanceScale);
        Assert.Equal(1.0, options.DdimEta);

        var layout = new SyncDreamerCameraLayout(options);
        Assert.Equal(16, layout.ViewCount);
        Assert.Equal(32, layout.LatentSize);
        for (int i = 0; i < 16; i++)
        {
            Assert.Equal(22.5 * i, layout.AzimuthDegrees[i], 12);
            var c = layout.Center(i);
            double distance = Math.Sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
            Assert.Equal(1.5, distance, 12);
            Assert.Equal(30.0, Math.Asin(c[2] / distance) * 180 / Math.PI, 9);
            double azimuth = Math.Atan2(c[1], c[0]) * 180 / Math.PI;
            if (azimuth < -1e-9) azimuth += 360;
            Assert.Equal(22.5 * i, azimuth, 9);

            // Every camera looks at the origin through K = [[280, 0, 128], [0, 280, 128], [0, 0, 1]], world up is up.
            var camera = layout.CameraForNeuS(i);
            Assert.Equal(280.0, camera.FocalX);
            var (u, v, depth) = camera.Project(0, 0, 0);
            Assert.Equal(128.0, u, 9);
            Assert.Equal(128.0, v, 9);
            Assert.Equal(1.5, depth, 9);
            Assert.True(camera.Project(0, 0, 0.1).V < 128.0);
        }

        // The viewpoint embedding: (d_elevation, sin d_azimuth, cos d_azimuth, 0) relative to the input view at
        // azimuth 0, with the azimuths wrapped into [-pi, pi) as the reference stores them.
        Assert.Equal(-Math.PI, layout.Azimuths[8], 12);
        var embedding = layout.ViewpointEmbedding();
        for (int i = 0; i < 16; i++)
        {
            double a = 22.5 * i * Math.PI / 180;
            Assert.Equal(0.0, embedding[i * 4], 12);
            Assert.Equal(Math.Sin(a), embedding[i * 4 + 1], 12);
            Assert.Equal(Math.Cos(a), embedding[i * 4 + 2], 12);
            Assert.Equal(0.0, embedding[i * 4 + 3], 12);
        }

        // A lower input view shifts the elevation difference: -30 deg - (-10 deg) in Zero123's negated convention.
        var lower = new SyncDreamerCameraLayout(new SyncDreamerOptions { InputElevationDegrees = 10 });
        Assert.Equal(-20.0 * Math.PI / 180, lower.ViewpointEmbedding()[4], 12);
    }

    [Fact]
    public void SpatialVolumeWarp_ProjectsEveryVoxelThroughTheViewCamera()
    {
        var options = new SyncDreamerOptions();
        var layout = new SyncDreamerCameraLayout(options);
        int v = options.SpatialVolumeSize, side = 32;
        double ratio = side / 256.0, half = (side - 1) / 2.0;
        var warp = layout.SpatialVolumeWarp(3, side);
        Assert.Equal(v * v * v * 2, warp.Length);
        var camera = layout.CameraForNeuS(3);
        foreach (var (i, j, k) in new[] { (0, 0, 0), (1, 2, 5), (31, 17, 9), (16, 16, 16) })
        {
            // Voxel [i, j, k] (depth, height, width) is the point (x, y, z) = (s_k, s_j, s_i).
            double S(int m) => -0.5 + m / 31.0;
            var (u, w, _) = camera.Project(S(k), S(j), S(i));
            int p = ((i * v + j) * v + k) * 2;
            Assert.Equal(u * ratio / half - 1, warp[p], 9);
            Assert.Equal(w * ratio / half - 1, warp[p + 1], 9);
        }
    }

    [Fact]
    public void FrustumGrid_UnprojectsEveryLatentPixelAtItsDepth()
    {
        var options = new SyncDreamerOptions();
        var layout = new SyncDreamerCameraLayout(options);
        int d = options.FrustumVolumeDepth, side = layout.LatentSize;
        var grid = layout.FrustumGrid(2);
        Assert.Equal(d * side * side * 3, grid.Length);
        Assert.Equal(1.5 - 0.86603, layout.FrustumDepth(0), 12);
        Assert.Equal(1.5 + 0.86603, layout.FrustumDepth(d - 1), 12);
        var camera = layout.CameraForNeuS(2);
        foreach (var (depth, row, col) in new[] { (0, 0, 0), (3, 4, 7), (47, 31, 31), (20, 16, 9) })
        {
            int p = ((depth * side + row) * side + col) * 3;
            // The grid is the world point divided by the spatial volume's half extent; project it back.
            var (u, v, z) = camera.Project(grid[p] * 0.5, grid[p + 1] * 0.5, grid[p + 2] * 0.5);
            Assert.Equal(col, u * side / 256.0, 9);
            Assert.Equal(row, v * side / 256.0, 9);
            Assert.Equal(layout.FrustumDepth(depth), z, 9);
        }
    }

    // ------------------------------------------------------------------ volumes

    [Fact]
    public void SpatialVolume_HasThePaperShape_AndPermutesWithTheViews()
    {
        var model = TinyModel();
        var network = model.Network;
        var geometry = model.Geometry();
        var noisy = Noise(new[] { 4, 4, 8, 8 }, 7);
        var time = network.EmbedTime(501);
        Assert.Equal(new[] { 1, 8 }, time.Shape.ToArray());

        var perView = network.UnprojectViews(noisy, time, geometry.Viewpoints, geometry.Warps);
        Assert.Equal(new[] { 4, 4, 8, 8, 8 }, perView.Shape.ToArray());
        var volume = network.ConstructSpatialVolume(noisy, time, geometry.Viewpoints, geometry.Warps);
        Assert.Equal(new[] { 1, 8, 8, 8, 8 }, volume.Shape.ToArray());
        Assert.All(Data(volume), value => Assert.False(double.IsNaN(value) || double.IsInfinity(value)));

        // Reordering the views (their latents, viewpoints and cameras together) reorders their unprojected volumes.
        var order = new[] { 2, 0, 3, 1 };
        var index = new Tensor<int>(order, new[] { 4 });
        var permuted = network.UnprojectViews(Engine.TensorIndexSelect(noisy, index, 0), time,
            Engine.TensorIndexSelect(geometry.Viewpoints, index, 0), Engine.TensorIndexSelect(geometry.Warps, index, 0));
        for (int k = 0; k < 4; k++)
            Assert.True(MaxDifference(Row(permuted, k), Row(perView, order[k])) < 1e-10);

        // The views see the volume differently, so the per-view volumes are not copies of one another.
        Assert.True(MaxDifference(Row(perView, 0), Row(perView, 1)) > 1e-6);

        // Each frustum volume is read at the four depth-attention scales.
        var viewpoint = Row(geometry.Viewpoints, 0);
        var frustum = network.ConstructFrustumVolume(volume, time, viewpoint, Row(geometry.Frustums, 0));
        Assert.Equal(4, frustum.Length);
        Assert.Equal(new[] { 1, 8, 8, 8, 8 }, frustum[0].Shape.ToArray());
        Assert.Equal(new[] { 1, 8, 4, 4, 4 }, frustum[1].Shape.ToArray());
        Assert.Equal(new[] { 1, 8, 2, 2, 2 }, frustum[2].Shape.ToArray());
        Assert.Equal(new[] { 1, 8, 1, 1, 1 }, frustum[3].Shape.ToArray());
    }

    [Fact]
    public void SpatialVolume_DependsOnTheNoisyLatents()
    {
        var model = TinyModel();
        var geometry = model.Geometry();
        var time = model.Network.EmbedTime(301);
        var a = model.Network.ConstructSpatialVolume(Noise(new[] { 4, 4, 8, 8 }, 1), time, geometry.Viewpoints, geometry.Warps);
        var b = model.Network.ConstructSpatialVolume(Noise(new[] { 4, 4, 8, 8 }, 2), time, geometry.Viewpoints, geometry.Warps);
        Assert.True(MaxDifference(a, b) > 1e-6);
    }

    // ------------------------------------------------------------------ depth-wise attention

    [Fact]
    public void DepthAttention_MatchesAHandWrittenReference()
    {
        const int query = 4, contextDim = 6, heads = 2, headDim = 3, depth = 5, h = 2, w = 3;
        var attention = new SdDepthAttention<double>(query, contextDim, heads, headDim, outputBias: true);
        var named = attention.NamedTensors(string.Empty).ToDictionary(p => p.Name, p => Data(p.Tensor));
        var wq = named["to_q.weight"];
        var wk = named["to_k.weight"];
        var wv = named["to_v.weight"];
        var wo = named["to_out.weight"];
        var bo = named["to_out.bias"];
        Assert.False(named.ContainsKey("to_q.bias"));

        var x = Noise(new[] { 1, query, h, w }, 31);
        var context = Noise(new[] { 1, contextDim, depth, h, w }, 32);
        var actual = Data(attention.Forward(x, context));
        var xs = Data(x);
        var cs = Data(context);
        int inner = heads * headDim;
        var expected = new double[query * h * w];
        for (int r = 0; r < h; r++)
        {
            for (int c = 0; c < w; c++)
            {
                var q = new double[inner];
                var k = new double[inner, depth];
                var v = new double[inner, depth];
                for (int o = 0; o < inner; o++)
                {
                    for (int i = 0; i < query; i++) q[o] += wq[o * query + i] * xs[(i * h + r) * w + c];
                    for (int dd = 0; dd < depth; dd++)
                    {
                        for (int j = 0; j < contextDim; j++)
                        {
                            double value = cs[((j * depth + dd) * h + r) * w + c];
                            k[o, dd] += wk[o * contextDim + j] * value;
                            v[o, dd] += wv[o * contextDim + j] * value;
                        }
                    }
                }

                var attended = new double[inner];
                for (int n = 0; n < heads; n++)
                {
                    var logits = new double[depth];
                    for (int dd = 0; dd < depth; dd++)
                    {
                        for (int e = 0; e < headDim; e++) logits[dd] += q[n * headDim + e] * k[n * headDim + e, dd];
                        logits[dd] /= Math.Sqrt(headDim);
                    }

                    double max = logits.Max();
                    var weights = logits.Select(l => Math.Exp(l - max)).ToArray();
                    double sum = weights.Sum();
                    for (int e = 0; e < headDim; e++)
                        for (int dd = 0; dd < depth; dd++)
                            attended[n * headDim + e] += v[n * headDim + e, dd] * weights[dd] / sum;
                }

                for (int o = 0; o < query; o++)
                {
                    double value = bo[o];
                    for (int m = 0; m < inner; m++) value += wo[o * inner + m] * attended[m];
                    expected[(o * h + r) * w + c] = value;
                }
            }
        }

        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 10);
    }

    [Fact]
    public void DepthAttention_AttendsOnlyAlongEachPixelsOwnRay()
    {
        var attention = new SdDepthAttention<double>(4, 6, 2, 3, outputBias: true);
        var x = Noise(new[] { 1, 4, 2, 3 }, 41);
        var context = Noise(new[] { 1, 6, 5, 2, 3 }, 42);
        var baseline = Data(attention.Forward(x, context));
        // Changing the frustum features of pixel (0, 0) changes that pixel's output and no other.
        var data = Data(context);
        for (int j = 0; j < 6; j++) for (int d = 0; d < 5; d++) data[((j * 5 + d) * 2 + 0) * 3 + 0] += 1.0;
        var changed = Data(attention.Forward(x, new Tensor<double>(data, new[] { 1, 6, 5, 2, 3 })));
        for (int o = 0; o < 4; o++)
        {
            for (int p = 0; p < 6; p++)
            {
                int i = o * 6 + p;
                if (p == 0) Assert.NotEqual(baseline[i], changed[i]);
                else Assert.Equal(baseline[i], changed[i], 12);
            }
        }
    }

    [Fact]
    public void DepthTransformer_StartsAsTheIdentity_WithAHandCountedParameterCount()
    {
        // dim 8, 2 heads of width 4 (inner 8), context 8, 4 groups:
        // proj_in conv 8*8 + 8, norm 2*8; proj_context bias-free conv 8*8, norm 2*8; to_q/to_k/to_v/to_out bias-free
        // 4 * 8*8; proj_out norm 2*8, bias-free 3x3 conv 8*8*9, norm 2*8, bias-free 3x3 conv 8*8*9.
        long expected = (64 + 8) + 16 + 64 + 16 + 4 * 64 + 16 + 576 + 16 + 576;
        var transformer = new SdDepthTransformer<double>(8, 2, 4, 8, 4);
        Assert.Equal(expected, transformer.ParameterCount);
        Assert.Equal(1608, expected);

        // The last projection is zero-initialized, as zero_module does, so the block passes its input through.
        var x = Noise(new[] { 1, 8, 2, 2 }, 51);
        var output = transformer.Forward(x, Noise(new[] { 1, 8, 3, 2, 2 }, 52));
        Assert.Equal(0.0, MaxDifference(output, x));
    }

    [Fact]
    public void Network_PlacesDepthAttentionAfterTheMiddleAndUpperDecoderBlocks()
    {
        var model = TinyModel();
        var sites = model.UNet.FeatureSites();
        // Two levels with one residual block each: middle, level-1 res, level-1 upsample, level-0 res.
        Assert.Equal(4, sites.Count);
        Assert.NotNull(model.Network.ConditionAt(0));
        Assert.Null(model.Network.ConditionAt(1));
        Assert.Null(model.Network.ConditionAt(2));
        Assert.NotNull(model.Network.ConditionAt(3));
        Assert.Equal(16, model.Network.MiddleCondition.Dim);
        Assert.Single(model.Network.OutputConditions);
        Assert.Equal(8, model.Network.OutputConditions[0].Dim);

        // The paper configuration: the middle block, then every decoder block above the deepest level.
        // Weights are lazy, so the paper-sized U-Net and network are cheap to build without a forward pass.
        var o = new SyncDreamerOptions();
        var unet = new UNetNoisePredictor<float>(inputChannels: 8, outputChannels: 4, baseChannels: o.UNetBaseChannels,
            channelMultipliers: o.UNetChannelMultipliers, numResBlocks: o.UNetNumResBlocks,
            attentionResolutions: o.UNetAttentionLevels, contextDim: o.CrossAttentionDim, numHeads: o.UNetHeads,
            inputHeight: 32, seed: 1);
        var paperSites = unet.FeatureSites();
        Assert.Equal(1 + 3 * 3 + 2, paperSites.Count);
        var paper = new SyncDreamerNetwork<float>(o, 4, paperSites, 4);
        int conditioned = 0;
        for (int i = 0; i < paperSites.Count; i++) if (paper.ConditionAt(i) is not null) conditioned++;
        // The middle block plus the 8 decoder blocks of levels 2, 1 and 0 (two residual blocks and an upsampling each,
        // none after the last level's residual blocks).
        Assert.Equal(1 + 3 + 3 + 2, conditioned);
        Assert.Equal(1280, paper.MiddleCondition.Dim);
        Assert.Equal(512, paper.MiddleCondition.ContextDim);
    }

    // ------------------------------------------------------------------ sampler

    [Fact]
    public void DdimSchedule_MatchesTheReferenceMakeSchedule()
    {
        var schedule = SyncDreamerDdimSchedule.Create(1000, 0.00085, 0.012, 50, 1.0);
        Assert.Equal(50, schedule.Count);
        Assert.Equal(1, schedule.Timesteps[0]);
        Assert.Equal(21, schedule.Timesteps[1]);
        Assert.Equal(981, schedule.Timesteps[49]);

        double Cumulative(int t)
        {
            double product = 1, a = Math.Sqrt(0.00085), b = Math.Sqrt(0.012);
            for (int i = 0; i <= t; i++)
            {
                double root = a + (b - a) * i / 999.0;
                product *= 1 - root * root;
            }

            return product;
        }

        Assert.Equal(Cumulative(981), schedule.Alphas[49], 12);
        Assert.Equal(Cumulative(961), schedule.AlphasPrevious[49], 12);
        Assert.Equal(Cumulative(0), schedule.AlphasPrevious[0], 12);
        double at = Cumulative(501), ap = Cumulative(481);
        Assert.Equal(Math.Sqrt((1 - ap) / (1 - at) * (1 - at / ap)), schedule.Sigmas[25], 12);
        Assert.Equal(0.0, SyncDreamerDdimSchedule.Create(1000, 0.00085, 0.012, 50, 0.0).Sigmas[25]);
    }

    [Fact]
    public void DdimStep_MatchesTheReferenceUpdate()
    {
        var schedule = SyncDreamerDdimSchedule.Create(1000, 0.00085, 0.012, 50, 1.0);
        var x = Noise(new[] { 2, 4, 2, 2 }, 61);
        var noise = Noise(new[] { 2, 4, 2, 2 }, 62);
        var fresh = Noise(new[] { 2, 4, 2, 2 }, 63);
        const int index = 10;
        var stepped = Data(schedule.Step(Engine, x, noise, index, fresh));
        var last = Data(schedule.Step(Engine, x, noise, 0, null));
        double a = schedule.Alphas[index], ap = schedule.AlphasPrevious[index], s = schedule.Sigmas[index];
        double a0 = schedule.Alphas[0], ap0 = schedule.AlphasPrevious[0], s0 = schedule.Sigmas[0];
        var xs = Data(x);
        var es = Data(noise);
        var zs = Data(fresh);
        for (int i = 0; i < xs.Length; i++)
        {
            double x0 = (xs[i] - Math.Sqrt(1 - a) * es[i]) / Math.Sqrt(a);
            double expected = Math.Sqrt(ap) * x0 + Math.Sqrt(Math.Max(1 - ap - s * s, 1e-7)) * es[i] + s * zs[i];
            Assert.Equal(expected, stepped[i], 10);
            double x00 = (xs[i] - Math.Sqrt(1 - a0) * es[i]) / Math.Sqrt(a0);
            Assert.Equal(Math.Sqrt(ap0) * x00 + Math.Sqrt(Math.Max(1 - ap0 - s0 * s0, 1e-7)) * es[i], last[i], 10);
        }
    }

    [Fact]
    public void OneSamplingStep_IsTheGuidedDdimUpdateOfTheSynchronizedNoise()
    {
        var model = TinyModel();
        var input = model.EncodeInputView(Image(1));
        Assert.Equal(new[] { 1, 16 }, input.Clip.Shape.ToArray());
        Assert.Equal(new[] { 1, 4, 8, 8 }, input.Latent.Shape.ToArray());
        var schedule = SyncDreamerDdimSchedule.Create(1000, 0.00085, 0.012, 2, 1.0);
        var x = Noise(new[] { 4, 4, 8, 8 }, 71);
        int t = schedule.Timesteps[0];

        var guided = model.PredictMultiviewNoise(x, input, t, 2.0);
        Assert.Equal(new[] { 4, 4, 8, 8 }, guided.Shape.ToArray());
        // Classifier-free guidance over the conditional and the all-zero unconditional branch.
        var conditional = Data(model.PredictMultiviewNoise(x, input, t, 1.0));
        var unconditional = Data(model.PredictMultiviewNoise(x, input, t, 0.0));
        var g = Data(guided);
        for (int i = 0; i < g.Length; i++) Assert.Equal(unconditional[i] + 2.0 * (conditional[i] - unconditional[i]), g[i], 9);
        Assert.True(conditional.Zip(unconditional, (c, u) => Math.Abs(c - u)).Max() > 1e-8);

        // The last step (index 0) adds no noise, so it is exactly the DDIM update of that noise.
        var stepped = model.DenoiseStep(x, input, schedule, 0, 2.0, RandomHelper.CreateSeededRandom(5));
        var expected = schedule.Step(Engine, x, guided, 0, null);
        Assert.True(MaxDifference(stepped, expected) < 1e-10);
        Assert.True(MaxDifference(stepped, x) > 1e-6);
    }

    [Fact]
    public void Noise_OfOneViewDependsOnTheOtherViews()
    {
        // Synchronization: the spatial volume carries every view's noisy latent into each view's prediction.
        var model = TinyModel();
        var output = model.Network.OutputConditions[0];
        var middle = model.Network.MiddleCondition;
        Randomize(output.GetParameters, output.SetParameters, 81, 0.2);
        Randomize(middle.GetParameters, middle.SetParameters, 82, 0.2);
        var input = model.EncodeInputView(Image(2));
        var x = Noise(new[] { 4, 4, 8, 8 }, 83);
        var data = Data(x);
        for (int i = 4 * 64; i < 2 * 4 * 64; i++) data[i] += 0.5;   // view 1 ([1, 4, 8, 8]) only
        var changed = new Tensor<double>(data, x.Shape.ToArray());
        var a = model.PredictMultiviewNoise(x, input, 501, 1.0);
        var b = model.PredictMultiviewNoise(changed, input, 501, 1.0);
        Assert.True(MaxDifference(Row(a, 0), Row(b, 0)) > 1e-9);
    }

    // ------------------------------------------------------------------ end to end

    [Fact]
    public void GenerateViews_IsDeterministicUnderASeed_AndDependsOnTheInput()
    {
        var model = TinyModel();
        var image = Image(3);
        var first = model.GenerateViews(image, 2, 2.0, seed: 9);
        var second = model.GenerateViews(image, 2, 2.0, seed: 9);
        Assert.Equal(4, first.Length);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(new[] { 1, 3, 64, 64 }, first[i].Shape.ToArray());
            Assert.Equal(0.0, MaxDifference(first[i], second[i]));
            Assert.All(Data(first[i]), value => Assert.InRange(value, 0.0, 1.0));
        }

        var otherSeed = model.GenerateViews(image, 2, 2.0, seed: 10);
        Assert.True(MaxDifference(first[0], otherSeed[0]) > 1e-9);
        var otherImage = model.GenerateViews(Image(4), 2, 2.0, seed: 9);
        Assert.True(MaxDifference(first[0], otherImage[0]) > 1e-9);
    }

    [Fact]
    public void ImageTo3D_ReconstructsAColoredNeuSMesh_Deterministically()
    {
        var model = TinyModel();
        var image = Image(5);
        var mesh = model.GenerateMesh(image, 1, 2.0, seed: 4);
        int vertices = mesh.Vertices.Shape[0];
        Assert.True(vertices > 0);
        Assert.Equal(3, mesh.Vertices.Shape[1]);
        Assert.True(mesh.Faces.GetLength(0) > 0);
        Assert.NotNull(mesh.Colors);
        var colors = mesh.Colors ?? throw new InvalidOperationException("No colors.");
        Assert.Equal(new[] { vertices, 3 }, colors.Shape.ToArray());
        Assert.All(Data(mesh.Vertices), value => Assert.InRange(value, -1.0, 1.0));
        Assert.All(Data(colors), value => Assert.InRange(value, 0.0, 1.0));

        var again = model.ImageTo3D(image, numViews: 4, numInferenceSteps: 1, guidanceScale: 2.0, seed: 4);
        Assert.Equal(0.0, MaxDifference(mesh.Vertices, again.Vertices));

        var cloud = model.GeneratePointCloud(image, numPoints: 16, numInferenceSteps: 1, guidanceScale: 2.0, seed: 4);
        Assert.Equal(new[] { 1, Math.Min(16, vertices), 6 }, cloud.Shape.ToArray());
        Assert.Equal(0.0, MaxDifference(cloud, model.SurfacePoints(mesh, 16)));
    }

    [Fact]
    public void TextEntryPoints_AreNotSupported()
    {
        var model = new SyncDreamerModel<float>(Tiny());
        Assert.Throws<NotSupportedException>(() => model.GenerateMesh("a teapot"));
        Assert.Throws<NotSupportedException>(() => model.GeneratePointCloud("a teapot"));
        Assert.Null(model.Conditioner);
    }
}
