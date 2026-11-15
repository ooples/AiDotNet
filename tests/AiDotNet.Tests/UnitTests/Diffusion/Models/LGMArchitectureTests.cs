using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.ThreeD.Meshing;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Reference tests for LGM (Tang et al., "LGM: Large Multi-View Gaussian Model", ECCV 2024, arXiv 2402.05054) against
/// the official 3DTopia/LGM code: the Plucker ray embedding of <c>prepare_default_rays</c>/<c>get_rays</c>, the
/// asymmetric U-Net's shapes and parameter count (<c>core/unet.py</c>), the Gaussian activations of
/// <c>forward_gaussians</c>, determinism, input dependence, and the Gaussian-to-mesh path.
/// </summary>
public class LGMArchitectureTests
{
    private const double Tolerance = 1e-9;

    /// <summary>
    /// A same-structure LGM small enough for a unit test: 16x16 input, down (8, 16, 16) with attention on the last,
    /// attention in the middle, up (16, 16) with attention on the first, one residual block per down block, 2 heads,
    /// GroupNorm(4). Splatter maps: 16 -> 8 -> 4 -> 8, so 4 views x 8 x 8 = 256 Gaussians.
    /// </summary>
    private static LGMConfig TinyConfig() => new LGMConfig
    {
        InputSize = 16,
        DownChannels = new[] { 8, 16, 16 },
        DownAttention = new[] { false, false, true },
        MidAttention = true,
        UpChannels = new[] { 16, 16 },
        UpAttention = new[] { true, false },
        LayersPerBlock = 1,
        AttentionHeads = 2,
        NormGroups = 4
    };

    private static MVDreamModel<double> TinyMultiView(int seed) => new MVDreamModel<double>(
        multiViewUNet: new MultiViewUNet<double>(
            inputChannels: 4, outputChannels: 4, baseChannels: 32, numViews: 4, contextDim: 1024, seed: seed),
        imageVAE: new StandardVAE<double>(
            inputChannels: 3, latentChannels: 4, baseChannels: 16,
            channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: seed),
        seed: seed);

    private static LGMModel<double> TinyModel(int seed)
        => new LGMModel<double>(config: TinyConfig(), multiViewModel: TinyMultiView(seed), seed: seed);

    private static Tensor<double> RandomViews(int seed, int size = 16)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var data = new double[4 * 3 * size * size];
        for (int i = 0; i < data.Length; i++) data[i] = rng.NextDouble();
        return new Tensor<double>(data, new[] { 4, 3, size, size });
    }

    // ---------------------------------------------------------------- Plucker rays

    [Fact]
    public void PluckerRays_MatchTheHandComputedOrbitCameras()
    {
        // 2x2 image, fovy 90 (focal = 1), radius 1.5, elevation 0. Pixel (0, 0) has camera direction
        // (-0.5, 0.5, -1) / sqrt(1.5). View 0 (azimuth 0) sits at (0, 0, 1.5) with an identity rotation; view 1
        // (azimuth 90) at (1.5, 0, 0) with right = (0, 0, -1), up = (0, 1, 0), forward = (1, 0, 0).
        var rays = LGMModel<double>.PluckerRays(4, 2, 90.0, 1.5, 0.0);
        Assert.Equal(4 * 6 * 2 * 2, rays.Length);
        double s = 1.0 / Math.Sqrt(1.5);

        // View 0, pixel (x 0, y 0): d = (-0.5, 0.5, -1) s, m = o x d = (-1.5 d_y, 1.5 d_x, 0).
        AssertRay(rays, view: 0, x: 0, y: 0,
            m: new[] { -1.5 * 0.5 * s, -1.5 * 0.5 * s, 0.0 },
            d: new[] { -0.5 * s, 0.5 * s, -s });

        // View 1, pixel (x 1, y 0): camera direction (0.5, 0.5, -1) -> world 0.5 right + 0.5 up - forward
        // = (-1, 0.5, -0.5); m = (1.5, 0, 0) x d = (0, -1.5 d_z, 1.5 d_y).
        AssertRay(rays, view: 1, x: 1, y: 0,
            m: new[] { 0.0, 1.5 * 0.5 * s, 1.5 * 0.5 * s },
            d: new[] { -s, 0.5 * s, -0.5 * s });

        // View 2 (azimuth 180) at (0, 0, -1.5), right = (-1, 0, 0): pixel (1, 1) has camera direction
        // (0.5, -0.5, -1) -> world (-0.5, -0.5, 1) s; m = (1.5 d_y, -1.5 d_x, 0).
        AssertRay(rays, view: 2, x: 1, y: 1,
            m: new[] { 1.5 * -0.5 * s, -1.5 * -0.5 * s, 0.0 },
            d: new[] { -0.5 * s, -0.5 * s, s });
    }

    [Fact]
    public void PluckerRays_AreUnitDirectionsWithMomentsAboutTheElevatedCamera()
    {
        const int size = 5;
        const double radius = 1.5, elevation = -30.0;
        var rays = LGMModel<double>.PluckerRays(4, size, 49.1, radius, elevation);
        double e = elevation * Math.PI / 180.0;
        for (int v = 0; v < 4; v++)
        {
            double a = v * Math.PI / 2.0;
            // kiui orbit_camera: negative elevation puts the camera above the object.
            double[] o = { radius * Math.Cos(e) * Math.Sin(a), -radius * Math.Sin(e), radius * Math.Cos(e) * Math.Cos(a) };
            Assert.True(o[1] > 0.0);
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    var (m, d) = Ray(rays, size, v, x, y);
                    Assert.Equal(1.0, Math.Sqrt(Dot(d, d)), 12);
                    var expected = Cross(o, d);
                    for (int k = 0; k < 3; k++) Assert.Equal(expected[k], m[k], 12);
                    // Every ray of the 49.1-degree frustum heads toward the object at the origin.
                    Assert.True(-Dot(d, o) / radius > Math.Cos(49.1 * Math.PI / 180.0));
                }
            }

            // The centre pixel looks straight at the origin.
            var (_, centre) = Ray(rays, size, v, 2, 2);
            for (int k = 0; k < 3; k++) Assert.Equal(-o[k] / radius, centre[k], 12);
        }
    }

    [Fact]
    public void UNetInput_IsImageNetNormalizedRgbFollowedByTheRays()
    {
        var model = TinyModel(3);
        var views = new Tensor<double>(Enumerable.Repeat(0.5, 4 * 3 * 16 * 16).ToArray(), new[] { 4, 3, 16, 16 });
        var input = model.BuildUNetInput(views);
        Assert.Equal(new[] { 4, 9, 16, 16 }, input.Shape.ToArray());

        var values = input.ToArray();
        var rays = LGMModel<double>.PluckerRays(4, 16, 49.1, 1.5, 0.0);
        double[] mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 };
        int plane = 16 * 16;
        for (int v = 0; v < 4; v++)
        {
            for (int c = 0; c < 9; c++)
            {
                for (int p = 0; p < plane; p++)
                {
                    double expected = c < 3 ? (0.5 - mean[c]) / std[c] : rays[(((v * 6) + (c - 3)) * plane) + p];
                    Assert.Equal(expected, values[(((v * 9) + c) * plane) + p], 12);
                }
            }
        }
    }

    [Fact]
    public void UNetInput_ResizesViewsToTheInputSize()
    {
        var model = TinyModel(3);
        var input = model.BuildUNetInput(RandomViews(5, size: 32));
        Assert.Equal(new[] { 4, 9, 16, 16 }, input.Shape.ToArray());
    }

    // ---------------------------------------------------------------- asymmetric U-Net

    [Fact]
    public void DefaultConfiguration_IsThePapersBigModel()
    {
        var config = LGMConfig.Default;
        Assert.Equal(256, config.InputSize);
        Assert.Equal(new[] { 64, 128, 256, 512, 1024, 1024 }, config.DownChannels);
        Assert.Equal(new[] { false, false, false, true, true, true }, config.DownAttention);
        Assert.Equal(new[] { 1024, 1024, 512, 256, 128 }, config.UpChannels);
        Assert.Equal(new[] { true, true, true, false, false }, config.UpAttention);
        Assert.True(config.MidAttention);
        Assert.Equal(2, config.LayersPerBlock);
        Assert.Equal(16, config.AttentionHeads);
        Assert.Equal(Math.Sqrt(0.5), config.SkipScale, 15);
        Assert.Equal(49.1, config.FieldOfViewY);
        Assert.Equal(1.5, config.CameraRadius);
        // 256 -> five stride-2 downsamples -> 8 -> four x2 upsamples -> 128: 4 x 128 x 128 Gaussians.
        Assert.Equal(4 * 128 * 128, LGMModel<double>.GaussianCountFor(config));
    }

    [Fact]
    public void UNet_ProducesHalfResolutionFourteenChannelMaps()
    {
        var model = TinyModel(11);
        Assert.Equal(8, model.SplatSize);
        Assert.Equal(256, model.DefaultPointCount);

        var maps = model.GaussianUNet.Forward(model.BuildUNetInput(RandomViews(1)));
        Assert.Equal(new[] { 4, 14, 8, 8 }, maps.Shape.ToArray());
        Assert.All(maps.ToArray(), value => Assert.True(IsFinite(value)));
    }

    [Fact]
    public void UNet_ParameterCountMatchesTheHandDerivedTotal()
    {
        // Per-block counts with PyTorch's layers: Conv2d(i, o, k) = k^2 i o + o, GroupNorm(C) = 2C,
        // ResnetBlock(i, o) = GN(i) + conv3(i, o) + GN(o) + conv3(o, o) [+ conv1(i, o) when i != o],
        // MVAttention(C) = GN(C) + qkv 3C^2 (no bias) + proj C^2 + C.
        static long Conv(int i, int o, int k) => (k * k * i * o) + o;
        static long Res(int i, int o) => (2 * i) + Conv(i, o, 3) + (2 * o) + Conv(o, o, 3) + (i != o ? Conv(i, o, 1) : 0);
        static long Attn(int c) => (2 * c) + (4 * c * c) + c;

        long expected =
            Conv(9, 8, 3)                                   // conv_in
            + Res(8, 8) + Conv(8, 8, 3)                     // down 0 + downsample
            + Res(8, 16) + Conv(16, 16, 3)                  // down 1 + downsample
            + Res(16, 16) + Attn(16)                        // down 2 (attention, no downsample)
            + Res(16, 16) + Attn(16) + Res(16, 16)          // middle
            + Res(16 + 16, 16) + Attn(16)                   // up 0, skip: down-2 residual
            + Res(16 + 16, 16) + Attn(16) + Conv(16, 16, 3) // up 0, skip: down-1 downsample; then upsample conv
            + Res(16 + 16, 16)                              // up 1, skip: down-1 residual
            + Res(16 + 8, 16)                               // up 1, skip: down-0 downsample
            + (2 * 16) + Conv(16, 14, 3);                   // norm_out + conv_out

        var model = TinyModel(2);
        Assert.Equal(60198L, expected);
        Assert.Equal(expected, model.GaussianUNet.ParameterCount);
        Assert.Equal(expected, model.GaussianUNet.GetParameters().Length);
        Assert.Equal(Conv(14, 14, 1), model.GaussianProjection.ParameterCount);
    }

    // ---------------------------------------------------------------- Gaussian activations

    [Fact]
    public void ActivateGaussians_ApplyThePaperActivationsToEachPixel()
    {
        double[][] raw =
        {
            new[] { 0.5, -2.0, 3.0, 0.3, -1.0, 0.0, 50.0, 1.0, 2.0, -2.0, 4.0, 0.2, -50.0, 1.0 },
            new[] { -0.1, 0.9, -1.5, -40.0, 2.0, -60.0, 0.7, 0.0, 0.0, 3.0, -4.0, -0.3, 0.0, 2.0 }
        };
        // [V = 1, 14, H = 1, W = 2]: channel c of pixel w at c * 2 + w.
        var data = new double[28];
        for (int w = 0; w < 2; w++)
            for (int c = 0; c < 14; c++)
                data[(c * 2) + w] = raw[w][c];

        var model = TinyModel(4);
        var gaussians = model.ActivateGaussians(new Tensor<double>(data, new[] { 1, 14, 1, 2 }));
        Assert.Equal(2, gaussians.Count);
        var positions = gaussians.Positions.ToArray();
        var opacities = gaussians.Opacities.ToArray();
        var scales = gaussians.Scales.ToArray();
        var rotations = gaussians.Rotations.ToArray();
        var colors = gaussians.Colors.ToArray();

        for (int n = 0; n < 2; n++)
        {
            var r = raw[n];
            for (int k = 0; k < 3; k++)
                Assert.Equal(Math.Max(-1.0, Math.Min(1.0, r[k])), positions[(3 * n) + k], 12);
            Assert.Equal(1.0 / (1.0 + Math.Exp(-r[3])), opacities[n], 12);
            for (int k = 0; k < 3; k++)
            {
                double x = r[4 + k];
                double softplus = Math.Max(x, 0.0) + Math.Log(1.0 + Math.Exp(-Math.Abs(x)));
                Assert.Equal(0.1 * softplus, scales[(3 * n) + k], 12);
            }

            double norm = Math.Sqrt((r[7] * r[7]) + (r[8] * r[8]) + (r[9] * r[9]) + (r[10] * r[10]));
            for (int k = 0; k < 4; k++)
                Assert.Equal(r[7 + k] / norm, rotations[(4 * n) + k], 12);
            for (int k = 0; k < 3; k++)
                Assert.Equal((0.5 * Math.Tanh(r[11 + k])) + 0.5, colors[(3 * n) + k], 12);
        }

        // softplus(50) = 50 exactly in double, not an overflow.
        Assert.Equal(5.0, scales[2], 9);
    }

    [Fact]
    public void ReconstructedGaussians_LieInThePapersRanges()
    {
        var model = TinyModel(21);
        var gaussians = model.ReconstructGaussians(RandomViews(8));
        Assert.Equal(256, gaussians.Count);
        Assert.Equal(new[] { 256, 14 }, gaussians.ToTensor().Shape.ToArray());

        Assert.All(gaussians.Positions.ToArray(), p => Assert.InRange(p, -1.0, 1.0));
        Assert.All(gaussians.Opacities.ToArray(), a => Assert.InRange(a, 0.0, 1.0));
        Assert.All(gaussians.Scales.ToArray(), s => Assert.True(s > 0.0 && IsFinite(s)));
        Assert.All(gaussians.Colors.ToArray(), c => Assert.InRange(c, 0.0, 1.0));
        var q = gaussians.Rotations.ToArray();
        for (int n = 0; n < gaussians.Count; n++)
        {
            double norm = Math.Sqrt((q[4 * n] * q[4 * n]) + (q[(4 * n) + 1] * q[(4 * n) + 1])
                + (q[(4 * n) + 2] * q[(4 * n) + 2]) + (q[(4 * n) + 3] * q[(4 * n) + 3]));
            Assert.Equal(1.0, norm, 9);
        }

        // The Gaussians are not a constant: the network output varies over pixels.
        Assert.True(gaussians.Positions.ToArray().Distinct().Count() > 100);
    }

    // ---------------------------------------------------------------- determinism and input dependence

    [Fact]
    public void Reconstruction_IsDeterministicUnderASeed()
    {
        var views = RandomViews(30);
        var first = TinyModel(17).ReconstructGaussians(views).ToTensor().ToArray();
        var second = TinyModel(17).ReconstructGaussians(views).ToTensor().ToArray();
        Assert.Equal(first, second);

        var other = TinyModel(18).ReconstructGaussians(views).ToTensor().ToArray();
        Assert.True(MaxAbsDifference(first, other) > 1e-6, "a different seed must give different weights");
    }

    [Fact]
    public void DifferentImages_GiveDifferentGaussians()
    {
        var model = TinyModel(9);
        var a = model.ReconstructGaussians(RandomViews(100)).ToTensor().ToArray();
        var b = model.ReconstructGaussians(RandomViews(101)).ToTensor().ToArray();
        var again = model.ReconstructGaussians(RandomViews(100)).ToTensor().ToArray();
        Assert.Equal(a, again);
        Assert.True(MaxAbsDifference(a, b) > 1e-6, "the Gaussians must depend on the input views");
    }

    // ---------------------------------------------------------------- outputs

    [Fact]
    public void PointCloud_IsTheVisibleCentresWithTheirColours()
    {
        var model = TinyModel(12);
        var gaussians = model.ReconstructGaussians(RandomViews(2));
        var cloud = model.GaussiansToPointCloud(gaussians);
        var positions = gaussians.Positions.ToArray();
        var colors = gaussians.Colors.ToArray();
        var opacities = gaussians.Opacities.ToArray();
        var visible = Enumerable.Range(0, gaussians.Count).Where(i => opacities[i] >= 0.005).ToArray();

        Assert.Equal(new[] { 1, visible.Length, 6 }, cloud.Shape.ToArray());
        var points = cloud.ToArray();
        for (int j = 0; j < visible.Length; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                Assert.Equal(positions[(3 * visible[j]) + k], points[(6 * j) + k], Tolerance);
                Assert.Equal(colors[(3 * visible[j]) + k], points[(6 * j) + 3 + k], Tolerance);
            }
        }

        // A cap keeps the most opaque, in Gaussian order.
        var capped = model.GaussiansToPointCloud(gaussians, maxPoints: 10).ToArray();
        var top = visible.OrderByDescending(i => opacities[i]).ThenBy(i => i).Take(10).OrderBy(i => i).ToArray();
        Assert.Equal(60, capped.Length);
        for (int j = 0; j < 10; j++)
            Assert.Equal(positions[3 * top[j]], capped[6 * j], Tolerance);
    }

    [Fact]
    public void GaussianSplattingScene_HoldsTheReconstructedGaussians()
    {
        var model = TinyModel(13);
        var gaussians = model.ReconstructGaussians(RandomViews(3));
        var scene = model.ToGaussianSplatting(gaussians);
        Assert.Equal(gaussians.Count, scene.GaussianCount);

        // The scene's own Gaussian cloud gives back the same centres, scales, rotations and opacities.
        var cloud = scene.ToGaussianCloud();
        AssertClose(gaussians.Positions.ToArray(), cloud.Positions.ToArray(), 1e-12);
        AssertClose(gaussians.Scales.ToArray(), cloud.Scales.ToArray(), 1e-12);
        AssertClose(gaussians.Rotations.ToArray(), cloud.Rotations.ToArray(), 1e-12);
        AssertClose(gaussians.Opacities.ToArray(), cloud.Opacities.ToArray(), 1e-9);
    }

    [Fact]
    public void SeededImage_GivesANonEmptyWatertightMesh()
    {
        // The released weights are not loaded, so the 1x1 Gaussian projection is set by hand to make the geometry
        // predictable while every Gaussian still comes from the U-Net: position = clamp(1000 * u_k) drives each
        // centre to a corner of the [-1, 1] cube chosen by the sign pattern of that pixel's U-Net output u, and the
        // biases give every Gaussian opacity sigmoid(4), scale 0.1 softplus(ln(e - 1)) = 0.1 and the identity
        // rotation. 256 Gaussians on 8 corners put at least 32 on one corner, whose summed density
        // (32 x 0.98 > 1) closes a sphere of radius 0.1 sqrt(2 ln(0.98 k)) around it.
        var model = TinyModel(5);
        var views = RandomViews(77);
        model.ReconstructGaussians(views); // materializes the lazy projection

        var kernel = new double[14 * 14]; // [out, in, 1, 1]
        for (int c = 0; c < 3; c++) kernel[(c * 14) + c] = 1000.0;
        double softplusInverseOfOne = Math.Log(Math.E - 1.0);
        double[] bias = { 0, 0, 0, 4.0, softplusInverseOfOne, softplusInverseOfOne, softplusInverseOfOne, 1, 0, 0, 0, 0, 0, 0 };
        SetProjection(model.GaussianProjection, kernel, bias);

        var gaussians = model.ReconstructGaussians(views);
        var mesh = model.GaussiansToMesh(gaussians, new GaussianMeshExtractionOptions
        {
            Resolution = 64,
            BlockCount = 8,
            NormalizedHalfExtent = 0.5
        });

        var surface = new MeshSurface(mesh.Vertices.ToArray(), Triangles(mesh.Faces), mesh.Normals?.ToArray());
        Assert.True(surface.TriangleCount > 0, "the mesh must not be empty");
        var (unmatched, nonManifold) = surface.EdgeDefects();
        Assert.Equal(0, unmatched);
        Assert.Equal(0, nonManifold);
        Assert.True(surface.SignedVolume() > 0.0);

        // Every vertex lies on a sphere around a cube corner.
        for (int v = 0; v < surface.VertexCount; v++)
        {
            for (int k = 0; k < 3; k++)
                Assert.InRange(Math.Abs(surface.Coordinates[(3 * v) + k]), 0.5, 1.5);
        }
    }

    [Fact]
    public void ImageTo3D_RejectsAViewCountOtherThanFour()
    {
        var model = TinyModel(6);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            model.ImageTo3D(new Tensor<double>(new[] { 1, 3, 16, 16 }), numViews: 6));
    }

    // ---------------------------------------------------------------- helpers

    private static void AssertRay(double[] rays, int view, int x, int y, double[] m, double[] d)
    {
        var (am, ad) = Ray(rays, 2, view, x, y);
        for (int k = 0; k < 3; k++)
        {
            Assert.Equal(m[k], am[k], 12);
            Assert.Equal(d[k], ad[k], 12);
        }
    }

    private static (double[] M, double[] D) Ray(double[] rays, int size, int view, int x, int y)
    {
        int plane = size * size;
        int pixel = (y * size) + x;
        var m = new double[3];
        var d = new double[3];
        for (int k = 0; k < 3; k++)
        {
            m[k] = rays[(((view * 6) + k) * plane) + pixel];
            d[k] = rays[(((view * 6) + 3 + k) * plane) + pixel];
        }

        return (m, d);
    }

    // net471 has no double.IsFinite.
    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

    private static double Dot(double[] a, double[] b) => (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);

    private static double[] Cross(double[] a, double[] b) => new[]
    {
        (a[1] * b[2]) - (a[2] * b[1]),
        (a[2] * b[0]) - (a[0] * b[2]),
        (a[0] * b[1]) - (a[1] * b[0])
    };

    private static double MaxAbsDifference(double[] a, double[] b)
    {
        Assert.Equal(a.Length, b.Length);
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    private static void AssertClose(double[] expected, double[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"index {i}: {expected[i]} vs {actual[i]}");
    }

    /// <summary>Writes a 1x1 convolution's kernel and bias through its parameter vector, reading the vector's layout
    /// (kernel then bias, or bias then kernel) off the layer's own tensors.</summary>
    private static void SetProjection(AiDotNet.NeuralNetworks.Layers.ConvolutionalLayer<double> layer, double[] kernel,
        double[] bias)
    {
        var current = layer.GetParameters().ToArray();
        var filters = layer.GetFilters().ToArray();
        Assert.Equal(kernel.Length + bias.Length, current.Length);
        Assert.Equal(kernel.Length, filters.Length);
        bool kernelFirst = current.Take(kernel.Length).SequenceEqual(filters);
        Assert.True(kernelFirst || current.Skip(bias.Length).SequenceEqual(filters));
        var updated = kernelFirst ? kernel.Concat(bias).ToArray() : bias.Concat(kernel).ToArray();
        layer.SetParameters(new Vector<double>(updated));
        Assert.Equal(kernel, layer.GetFilters().ToArray());
    }

    private static int[] Triangles(int[,] faces)
    {
        int count = faces.GetLength(0);
        var triangles = new int[3 * count];
        for (int f = 0; f < count; f++)
            for (int k = 0; k < 3; k++)
                triangles[(3 * f) + k] = faces[f, k];
        return triangles;
    }
}
