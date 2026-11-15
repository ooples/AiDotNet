using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields;

/// <summary>
/// Pins <see cref="GaussianSplatting{T}"/> to the numbers of the per-Gaussian (object-per-splat, double
/// arithmetic) implementation it replaced, on a tiny three-Gaussian scene.
/// </summary>
/// <remarks>
/// <para>
/// The expected values are not recorded outputs: each test re-derives them with an independent scalar
/// re-implementation of that implementation's maths -- the nearest-neighbour scale initialisation, the
/// world-to-camera transform (camera rotation is camera-to-world, so the transform uses its transpose),
/// the EWA projection with its 1e-6 regularisation, 3-sigma bounding boxes and visibility rules, the
/// depth-ordered alpha compositing with its 1e-4 transmittance stop, the -20 exponent cutoff and the 1e-6
/// alpha cutoff, the colour and opacity gradients of one image-space step, and the split/prune rules of
/// one densification pass.
/// </para>
/// <para>
/// Only public behaviour is observed (RenderImage, Train, GetGaussianDiagnostics and the densification
/// test hook), so the same tests run unchanged against either storage layout. The scene is rendered
/// without spherical harmonics, the mode whose colour path the tensor rewrite keeps bit-for-bit; the
/// spherical-harmonics direction, the position/scale/rotation gradients and the paper fixes are pinned
/// separately in <see cref="GaussianSplattingPaperGradientTests"/>.
/// </para>
/// </remarks>
public class GaussianSplattingMasterReferenceTests
{
    private const int Width = 12;
    private const int Height = 10;
    private const double Focal = 9.0;
    private const double Tolerance = 1e-9;

    // Three points in front of the camera at different depths, overlapping on screen.
    private static readonly double[,] Points =
    {
        { 0.00, 0.00, 4.0 },
        { 0.30, 0.10, 4.4 },
        { -0.20, -0.25, 4.9 },
    };

    private static readonly double[,] Colors =
    {
        { 0.90, 0.20, 0.10 },
        { 0.10, 0.80, 0.30 },
        { 0.20, 0.30, 0.70 },
    };

    // Camera at the origin turned 0.1 rad about +y; rows are the camera-to-world rotation.
    private static readonly double[] CameraPosition = { 0.05, -0.02, 0.0 };

    private static double[,] CameraRotation()
    {
        double c = Math.Cos(0.1), s = Math.Sin(0.1);
        return new double[,]
        {
            { c, 0.0, s },
            { 0.0, 1.0, 0.0 },
            { -s, 0.0, c },
        };
    }

    private static GaussianSplattingOptions SceneOptions(bool enableDensification = false) => new()
    {
        UseSphericalHarmonics = false,
        ShDegree = 0,
        EnableSpatialIndex = false,
        EnableDensification = enableDensification,
        MaxGaussians = 64,
        TileSize = 4,
    };

    private static GaussianSplatting<double> BuildModel(GaussianSplattingOptions options)
    {
        int n = Points.GetLength(0);
        var pts = new Matrix<double>(n, 3);
        var cols = new Matrix<double>(n, 3);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                pts[i, j] = Points[i, j];
                cols[i, j] = Colors[i, j];
            }
        }

        return new GaussianSplatting<double>(options, pts, cols, new MeanSquaredErrorLoss<double>());
    }

    [Fact]
    public void InitialScales_MatchNearestNeighbourRule()
    {
        var options = SceneOptions();
        var model = BuildModel(options);
        var expected = ReferenceInitialScales(options);
        var diagnostics = model.GetGaussianDiagnostics();

        Assert.Equal(expected.Length, diagnostics.Count);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], diagnostics[i].MaxScale, 12);
            Assert.Equal(0.5, diagnostics[i].Opacity, 12);
            Assert.Equal(Points[i, 0], diagnostics[i].PosX, 12);
            Assert.Equal(Points[i, 1], diagnostics[i].PosY, 12);
            Assert.Equal(Points[i, 2], diagnostics[i].PosZ, 12);
            Assert.Equal((Colors[i, 0] + Colors[i, 1] + Colors[i, 2]) / 3.0, diagnostics[i].MeanColor, 12);
        }
    }

    [Fact]
    public void RenderImage_MatchesReferenceCompositing()
    {
        var options = SceneOptions();
        var model = BuildModel(options);
        var scales = ReferenceInitialScales(options);
        var opacityLogits = new double[Points.GetLength(0)];
        var expected = ReferenceRender(scales, Colors, opacityLogits, clamp: true);

        var image = model.RenderImage(
            new Vector<double>(CameraPosition),
            ToMatrix(CameraRotation()),
            Width,
            Height,
            Focal);

        Assert.Equal(new[] { Height, Width, 3 }, image.Shape.ToArray());
        double maxValue = 0.0;
        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    Assert.Equal(expected[y, x, c], image[y, x, c], 9);
                    maxValue = Math.Max(maxValue, expected[y, x, c]);
                }
            }
        }

        // The fixture must actually splat something, or the comparison proves nothing.
        Assert.True(maxValue > 0.05, $"reference render is empty (max {maxValue})");
    }

    [Fact]
    public void OneImageGradientStep_MatchesReferenceColourAndOpacityUpdate()
    {
        var options = SceneOptions();
        var model = BuildModel(options);
        var scales = ReferenceInitialScales(options);
        int n = Points.GetLength(0);
        var opacityLogits = new double[n];
        var prediction = ReferenceRender(scales, Colors, opacityLogits, clamp: false);

        // A flat grey target so every covered pixel carries a non-zero residual.
        var target = new Tensor<double>(new[] { Height, Width, 3 });
        for (int y = 0; y < Height; y++)
            for (int x = 0; x < Width; x++)
                for (int c = 0; c < 3; c++)
                    target[y, x, c] = 0.35;

        var (colorGrad, opacityGrad) = ReferenceColourAndOpacityGradients(scales, prediction, target);
        Assert.True(opacityGrad.Any(g => Math.Abs(g) > 1e-6), "fixture produces no opacity gradient");

        model.Train(CameraInput(), target);

        var diagnostics = model.GetGaussianDiagnostics();
        Assert.Equal(n, diagnostics.Count);
        for (int i = 0; i < n; i++)
        {
            double newLogit = opacityLogits[i] - model.OpacityLearningRate * opacityGrad[i];
            Assert.Equal(Sigmoid(newLogit), diagnostics[i].Opacity, 10);

            double meanColor = 0.0;
            for (int c = 0; c < 3; c++)
            {
                meanColor += Clamp01(Colors[i, c] - model.ColorLearningRate * colorGrad[i, c]);
            }

            Assert.Equal(meanColor / 3.0, diagnostics[i].MeanColor, 10);
        }
    }

    [Fact]
    public void DensifyAndPrune_SplitsInStableGradientOrderUpToTheCap()
    {
        var options = SceneOptions(enableDensification: true);
        options.DensificationStartIteration = 1;
        options.DensificationEndIteration = 100;
        options.OpacityPruneThreshold = 0.4;      // 0.5 opacities survive
        options.GradientNormThreshold = 1e-9;
        options.MaxGaussianCount = 5;             // room for exactly two splits
        var model = BuildModel(options);
        var scales = ReferenceInitialScales(options);

        // Equal gradient magnitudes: the stable descending sort keeps index order, so Gaussians 0 and 1
        // split and Gaussian 2 is left alone once the cap is reached.
        model.RunDensifyAndPruneForTest(iteration: 10, gradientsMagnitude: 1.0);

        var d = model.GetGaussianDiagnostics();
        Assert.Equal(5, d.Count);

        var childScales = new SplitChildLearningRateScales();
        double jitter = options.SplitPositionJitter * childScales.Position;
        double scaleFactor = options.SplitScaleFactor / childScales.Scale;
        double splitOpacity = Math.Min(options.SplitOpacityMax, 0.5 * options.SplitOpacityFactor);

        // Sources keep their slots and move along +x (the normalised (1, 0, 0) gradient).
        for (int i = 0; i < 2; i++)
        {
            Assert.Equal(Points[i, 0] + jitter, d[i].PosX, 12);
            Assert.Equal(Points[i, 1], d[i].PosY, 12);
            Assert.Equal(Math.Max(options.MinScale, scales[i] * scaleFactor), d[i].MaxScale, 12);
            Assert.Equal(splitOpacity, d[i].Opacity, 9);
        }

        // The untouched Gaussian.
        Assert.Equal(Points[2, 0], d[2].PosX, 12);
        Assert.Equal(scales[2], d[2].MaxScale, 12);
        Assert.Equal(0.5, d[2].Opacity, 12);

        // Clones are appended in split order and move the other way.
        for (int k = 0; k < 2; k++)
        {
            Assert.Equal(Points[k, 0] - jitter, d[3 + k].PosX, 12);
            Assert.Equal(Points[k, 2], d[3 + k].PosZ, 12);
            Assert.Equal(Math.Max(options.MinScale, scales[k] * scaleFactor), d[3 + k].MaxScale, 12);
            Assert.Equal(splitOpacity, d[3 + k].Opacity, 9);
            Assert.Equal((Colors[k, 0] + Colors[k, 1] + Colors[k, 2]) / 3.0, d[3 + k].MeanColor, 12);
        }
    }

    [Fact]
    public void DensifyAndPrune_PrunesBeforeSplittingAndSkipsPrunedSources()
    {
        var options = SceneOptions(enableDensification: true);
        options.DensificationStartIteration = 1;
        options.DensificationEndIteration = 100;
        options.OpacityPruneThreshold = 0.6;      // every 0.5 opacity is pruned
        options.GradientNormThreshold = 1e-9;
        var model = BuildModel(options);

        model.RunDensifyAndPruneForTest(iteration: 10, gradientsMagnitude: 1.0);

        // Pruning runs first; the gradients then point at removed Gaussians and none may be split.
        Assert.Equal(0, model.GaussianCount);
    }

    private static Tensor<double> CameraInput()
    {
        var rot = CameraRotation();
        var data = new double[13];
        data[0] = CameraPosition[0];
        data[1] = CameraPosition[1];
        data[2] = CameraPosition[2];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                data[3 + r * 3 + c] = rot[r, c];
        data[12] = Focal;
        return new Tensor<double>(data, new[] { 1, 13 });
    }

    private static Matrix<double> ToMatrix(double[,] values)
    {
        var m = new Matrix<double>(3, 3);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                m[r, c] = values[r, c];
        return m;
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private static double Clamp01(double v) => v <= 0.0 ? 0.0 : v >= 1.0 ? 1.0 : v;

    /// <summary>The nearest-neighbour scale rule: grid cells of the mean spacing, 3x3x3 neighbourhood.</summary>
    private static double[] ReferenceInitialScales(GaussianSplattingOptions o)
    {
        int n = Points.GetLength(0);
        double minX = double.PositiveInfinity, minY = double.PositiveInfinity, minZ = double.PositiveInfinity;
        double maxX = double.NegativeInfinity, maxY = double.NegativeInfinity, maxZ = double.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            minX = Math.Min(minX, Points[i, 0]); maxX = Math.Max(maxX, Points[i, 0]);
            minY = Math.Min(minY, Points[i, 1]); maxY = Math.Max(maxY, Points[i, 1]);
            minZ = Math.Min(minZ, Points[i, 2]); maxZ = Math.Max(maxZ, Points[i, 2]);
        }

        double ex = maxX - minX, ey = maxY - minY, ez = maxZ - minZ;
        double volume = ex * ey * ez;
        double baseSpacing = volume > 0.0
            ? Math.Pow(volume / n, 1.0 / 3.0)
            : Math.Max(ex, Math.Max(ey, ez)) > 0.0 ? Math.Max(ex, Math.Max(ey, ez)) / Math.Sqrt(n) : 0.0;
        if (baseSpacing <= 0.0) baseSpacing = o.DefaultPointSpacing;

        double cellSize = Math.Max(baseSpacing, o.MinScale * o.InitialNeighborSearchScale);
        double inv = 1.0 / cellSize;
        int Cell(double v, double min) => (int)Math.Floor((v - min) * inv);

        var scales = new double[n];
        for (int i = 0; i < n; i++)
        {
            double nearestSq = double.PositiveInfinity;
            for (int j = 0; j < n; j++)
            {
                if (j == i) continue;
                if (Math.Abs(Cell(Points[i, 0], minX) - Cell(Points[j, 0], minX)) > 1) continue;
                if (Math.Abs(Cell(Points[i, 1], minY) - Cell(Points[j, 1], minY)) > 1) continue;
                if (Math.Abs(Cell(Points[i, 2], minZ) - Cell(Points[j, 2], minZ)) > 1) continue;
                double dx = Points[i, 0] - Points[j, 0];
                double dy = Points[i, 1] - Points[j, 1];
                double dz = Points[i, 2] - Points[j, 2];
                nearestSq = Math.Min(nearestSq, dx * dx + dy * dy + dz * dz);
            }

            double distance = nearestSq < double.PositiveInfinity ? Math.Sqrt(nearestSq) : baseSpacing;
            if (distance <= 0.0) distance = baseSpacing;
            scales[i] = Math.Max(o.MinScale, distance * o.InitialScaleMultiplier);
        }

        return scales;
    }

    private sealed class Splat
    {
        public int Index;
        public double MeanX, MeanY, Depth, InvA, InvB, InvC, Opacity, CamX, CamY;
        public double[] Color = new double[3];
        public int MinX, MaxX, MinY, MaxY;
    }

    /// <summary>Camera transform and EWA projection of the isotropic, unrotated scene Gaussians.</summary>
    private static List<Splat> ReferenceProject(double[] scales, double[,] colors, double[] opacityLogits)
    {
        var rot = CameraRotation();
        // Camera-to-world rotation: the world-to-camera transform is its transpose.
        double r00 = rot[0, 0], r10 = rot[0, 1], r20 = rot[0, 2];
        double r01 = rot[1, 0], r11 = rot[1, 1], r21 = rot[1, 2];
        double r02 = rot[2, 0], r12 = rot[2, 1], r22 = rot[2, 2];
        double cx = (Width - 1) * 0.5, cy = (Height - 1) * 0.5;
        var result = new List<Splat>();

        for (int i = 0; i < scales.Length; i++)
        {
            double s = Math.Max(1e-6, Math.Abs(scales[i]));
            double c00 = s * s + 1e-6, c11 = s * s + 1e-6, c22 = s * s + 1e-6;
            double c01 = 0.0, c02 = 0.0, c12 = 0.0;

            double px = Points[i, 0] - CameraPosition[0];
            double py = Points[i, 1] - CameraPosition[1];
            double pz = Points[i, 2] - CameraPosition[2];
            double x = r00 * px + r01 * py + r02 * pz;
            double y = r10 * px + r11 * py + r12 * pz;
            double z = r20 * px + r21 * py + r22 * pz;

            double t00 = r00 * c00 + r01 * c01 + r02 * c02;
            double t01 = r00 * c01 + r01 * c11 + r02 * c12;
            double t02 = r00 * c02 + r01 * c12 + r02 * c22;
            double t10 = r10 * c00 + r11 * c01 + r12 * c02;
            double t11 = r10 * c01 + r11 * c11 + r12 * c12;
            double t12 = r10 * c02 + r11 * c12 + r12 * c22;
            double t20 = r20 * c00 + r21 * c01 + r22 * c02;
            double t21 = r20 * c01 + r21 * c11 + r22 * c12;
            double t22 = r20 * c02 + r21 * c12 + r22 * c22;
            double s00 = t00 * r00 + t01 * r01 + t02 * r02;
            double s01 = t00 * r10 + t01 * r11 + t02 * r12;
            double s02 = t00 * r20 + t01 * r21 + t02 * r22;
            double s11 = t10 * r10 + t11 * r11 + t12 * r12;
            double s12 = t10 * r20 + t11 * r21 + t12 * r22;
            double s22 = t20 * r20 + t21 * r21 + t22 * r22;

            if (z <= 1e-6) continue;
            double invZ = 1.0 / z;
            double meanX = Focal * x * invZ + cx;
            double meanY = Focal * y * invZ + cy;
            double j00 = Focal * invZ, j02 = -Focal * x * invZ * invZ;
            double j11 = Focal * invZ, j12 = -Focal * y * invZ * invZ;
            double t0x = s00 * j00 + s02 * j02, t0y = s01 * j00 + s12 * j02, t0z = s02 * j00 + s22 * j02;
            double t1x = s01 * j11 + s02 * j12, t1y = s11 * j11 + s12 * j12, t1z = s12 * j11 + s22 * j12;
            double a = j00 * t0x + j02 * t0z + 1e-6;
            double b = j00 * t1x + j02 * t1z;
            double c = j11 * t1y + j12 * t1z + 1e-6;
            double det = a * c - b * b;
            if (det <= 1e-12) continue;

            double trace = a + c;
            double disc = Math.Max(0.0, trace * trace - 4.0 * det);
            double radius = Math.Max(1.0, 3.0 * Math.Sqrt(Math.Max(0.5 * (trace + Math.Sqrt(disc)), 0.0)));
            int minX = (int)Math.Floor(meanX - radius), maxX = (int)Math.Ceiling(meanX + radius);
            int minY = (int)Math.Floor(meanY - radius), maxY = (int)Math.Ceiling(meanY + radius);
            if (maxX < 0 || maxY < 0 || minX >= Width || minY >= Height) continue;

            var splat = new Splat
            {
                Index = i,
                MeanX = meanX,
                MeanY = meanY,
                Depth = z,
                InvA = c / det,
                InvB = -b / det,
                InvC = a / det,
                Opacity = Sigmoid(opacityLogits[i]),
                CamX = x,
                CamY = y,
                MinX = Math.Max(0, minX),
                MaxX = Math.Min(Width - 1, maxX),
                MinY = Math.Max(0, minY),
                MaxY = Math.Min(Height - 1, maxY),
            };
            for (int ch = 0; ch < 3; ch++) splat.Color[ch] = Clamp01(colors[i, ch]);
            result.Add(splat);
        }

        return result.OrderBy(sp => sp.Depth).ToList();
    }

    private static bool Covers(Splat s, int x, int y) => x >= s.MinX && x <= s.MaxX && y >= s.MinY && y <= s.MaxY;

    /// <summary>Front-to-back compositing per pixel in depth order with every cutoff of the renderer.</summary>
    private static double[,,] ReferenceRender(double[] scales, double[,] colors, double[] opacityLogits, bool clamp)
    {
        var splats = ReferenceProject(scales, colors, opacityLogits);
        var image = new double[Height, Width, 3];
        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                double t = 1.0;
                var acc = new double[3];
                foreach (var s in splats)
                {
                    if (!Covers(s, x, y) || t < 1e-4) continue;
                    double dx = x - s.MeanX, dy = y - s.MeanY;
                    double e = -0.5 * (s.InvA * dx * dx + 2.0 * s.InvB * dx * dy + s.InvC * dy * dy);
                    if (e < -20.0) continue;
                    double w = Math.Exp(e);
                    double alpha = s.Opacity * w;
                    if (alpha < 1e-6) continue;
                    for (int c = 0; c < 3; c++) acc[c] += t * alpha * s.Color[c];
                    t *= 1.0 - alpha;
                }

                for (int c = 0; c < 3; c++) image[y, x, c] = clamp ? Clamp01(acc[c]) : acc[c];
            }
        }

        return image;
    }

    /// <summary>
    /// Colour and opacity-logit gradients of one image-space step, accumulated Gaussian by Gaussian in depth
    /// order against the rendered totals (the "colour behind" term uses total - accumulated - own).
    /// </summary>
    private static (double[,] Color, double[] Opacity) ReferenceColourAndOpacityGradients(
        double[] scales, double[,,] prediction, Tensor<double> target)
    {
        var splats = ReferenceProject(scales, Colors, new double[scales.Length]);
        var predTensor = new Tensor<double>(new[] { Height, Width, 3 });
        for (int y = 0; y < Height; y++)
            for (int x = 0; x < Width; x++)
                for (int c = 0; c < 3; c++)
                    predTensor[y, x, c] = prediction[y, x, c];
        var lossGrad = new MeanSquaredErrorLoss<double>().ComputeGradient(predTensor, target);

        var transmittance = new double[Height, Width];
        var accum = new double[Height, Width, 3];
        for (int y = 0; y < Height; y++)
            for (int x = 0; x < Width; x++)
                transmittance[y, x] = 1.0;

        var colorGrad = new double[scales.Length, 3];
        var opacityGrad = new double[scales.Length];
        foreach (var s in splats)
        {
            for (int y = s.MinY; y <= s.MaxY; y++)
            {
                for (int x = s.MinX; x <= s.MaxX; x++)
                {
                    double t = transmittance[y, x];
                    if (t < 1e-4) continue;
                    double dx = x - s.MeanX, dy = y - s.MeanY;
                    double e = -0.5 * (s.InvA * dx * dx + 2.0 * s.InvB * dx * dy + s.InvC * dy * dy);
                    if (e < -20.0) continue;
                    double w = Math.Exp(e);
                    double alpha = s.Opacity * w;
                    if (alpha < 1e-6) continue;

                    double wa = t * alpha;
                    double dLdAlpha = 0.0;
                    var contrib = new double[3];
                    for (int c = 0; c < 3; c++)
                    {
                        double g = lossGrad[y, x, c];
                        colorGrad[s.Index, c] += wa * g;
                        contrib[c] = wa * s.Color[c];
                        double after = prediction[y, x, c] - accum[y, x, c] - contrib[c];
                        dLdAlpha += g * (t * s.Color[c] - after / Math.Max(1e-6, 1.0 - alpha));
                    }

                    opacityGrad[s.Index] += dLdAlpha * w * s.Opacity * (1.0 - s.Opacity);
                    for (int c = 0; c < 3; c++) accum[y, x, c] += contrib[c];
                    transmittance[y, x] = t * (1.0 - alpha);
                }
            }
        }

        return (colorGrad, opacityGrad);
    }
}
