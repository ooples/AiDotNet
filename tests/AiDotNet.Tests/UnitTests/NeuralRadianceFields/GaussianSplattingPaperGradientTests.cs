using System;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields;

/// <summary>
/// Pins the parts of <see cref="GaussianSplatting{T}"/> that follow Kerbl et al. 2023 ("3D Gaussian Splatting
/// for Real-Time Radiance Field Rendering") where the earlier implementation did not.
/// </summary>
/// <remarks>
/// <para>
/// <b>Gradients.</b> The published image-space gradient must be the true derivative of the rendered loss. A
/// central finite difference of the loss over every parameter checks it with a rotated camera and rotated,
/// anisotropic Gaussians -- the configuration that exposed the earlier errors: the position gradient was
/// rotated by R^T instead of R, the covariance gradient was used in camera space as if it were the world-space
/// gradient, the off-diagonal conic gradient was counted twice, the position gradient ignored the dependence
/// of the projected covariance on the camera-space mean (the EWA Jacobian), clamped spherical-harmonics
/// colours still passed gradient, and scales are now log-parameterised (the paper's exp activation), so the
/// published scale gradient is d/d(log s).
/// </para>
/// <para>
/// <b>Parameter layout.</b> The parameter vector is the five per-attribute tensors in order: positions [N, 3],
/// rotation quaternions [N, 4] (w, x, y, z), log-scales [N, 3], opacity logits [N] and spherical-harmonics
/// coefficients [N, K, 3].
/// </para>
/// </remarks>
public class GaussianSplattingPaperGradientTests
{
    private const int Width = 14;
    private const int Height = 12;
    private const double Focal = 10.0;
    private const double ShDc = 0.282095;

    private static readonly double[,] Positions =
    {
        { 0.10, 0.05, 3.0 },
        { -0.25, 0.15, 3.4 },
        { 0.20, -0.20, 3.8 },
    };

    private static readonly double[,] Scales =
    {
        { 0.12, 0.20, 0.08 },
        { 0.18, 0.10, 0.15 },
        { 0.10, 0.14, 0.22 },
    };

    private static readonly double[,] Quaternions =
    {
        { 0.9, 0.1, -0.2, 0.3 },
        { 0.8, -0.3, 0.1, 0.2 },
        { 1.0, 0.2, 0.3, -0.1 },
    };

    private static readonly double[] OpacityLogits = { 0.3, -0.2, 0.6 };

    // Rendered colours; the 1.3 red channel saturates the clamp, so its coefficient must get no gradient.
    private static readonly double[,] Rgb =
    {
        { 0.60, 0.30, 0.20 },
        { 0.25, 0.70, 0.40 },
        { 1.30, 0.50, 0.35 },
    };

    private static readonly double[] CameraPosition = { 0.03, -0.04, 0.1 };

    /// <summary>Camera-to-world rotation Ry(0.1) * Rx(0.07).</summary>
    private static double[,] CameraRotation()
    {
        double cy = Math.Cos(0.1), sy = Math.Sin(0.1), cx = Math.Cos(0.07), sx = Math.Sin(0.07);
        var ry = new[,] { { cy, 0.0, sy }, { 0.0, 1.0, 0.0 }, { -sy, 0.0, cy } };
        var rx = new[,] { { 1.0, 0.0, 0.0 }, { 0.0, cx, -sx }, { 0.0, sx, cx } };
        var r = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    r[i, j] += ry[i, k] * rx[k, j];
        return r;
    }

    private static GaussianSplatting<double> BuildModel(int shDegree)
    {
        var options = new GaussianSplattingOptions
        {
            UseSphericalHarmonics = true,
            ShDegree = shDegree,
            EnableSpatialIndex = false,
            EnableDensification = false,
            MaxGaussians = 64,
            TileSize = 4,
        };
        var cloud = new Matrix<double>(3, 3);
        var model = new GaussianSplatting<double>(options, cloud, cloud, new MeanSquaredErrorLoss<double>());
        model.PositionLearningRate = 0.0;
        model.ColorLearningRate = 0.0;
        model.OpacityLearningRate = 0.0;
        model.ScaleLearningRate = 0.0;
        model.RotationLearningRate = 0.0;
        return model;
    }

    private static void LoadScene(GaussianSplatting<double> model, int shDegree)
    {
        int n = Positions.GetLength(0);
        int k = (shDegree + 1) * (shDegree + 1);
        var positions = new Tensor<double>(new[] { n, 3 });
        var rotations = new Tensor<double>(new[] { n, 4 });
        var logScales = new Tensor<double>(new[] { n, 3 });
        var opacity = new Tensor<double>(new[] { n });
        var sh = new Tensor<double>(new[] { n, k, 3 });
        for (int i = 0; i < n; i++)
        {
            double norm = 0.0;
            for (int j = 0; j < 4; j++) norm += Quaternions[i, j] * Quaternions[i, j];
            norm = Math.Sqrt(norm);
            for (int j = 0; j < 4; j++) rotations[i, j] = Quaternions[i, j] / norm;
            for (int j = 0; j < 3; j++)
            {
                positions[i, j] = Positions[i, j];
                logScales[i, j] = Math.Log(Scales[i, j]);
                sh[i, 0, j] = Rgb[i, j] / ShDc;

                // View-dependent terms: small enough that only the saturated red channel of Gaussian 2 clamps.
                for (int b = 1; b < k; b++)
                    sh[i, b, j] = 0.1 * (((i + b + j) % 3) - 1);
            }

            opacity[i] = OpacityLogits[i];
        }

        model.SetGaussianStateForTest(positions, rotations, logScales, opacity, sh);
    }

    private static Tensor<double> CameraInput()
    {
        var rot = CameraRotation();
        var data = new double[13];
        for (int i = 0; i < 3; i++) data[i] = CameraPosition[i];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                data[3 + r * 3 + c] = rot[r, c];
        data[12] = Focal;
        return new Tensor<double>(data, new[] { 1, 13 });
    }

    private static Tensor<double> Target()
    {
        var target = new Tensor<double>(new[] { Height, Width, 3 });
        for (int y = 0; y < Height; y++)
            for (int x = 0; x < Width; x++)
                for (int c = 0; c < 3; c++)
                    target[y, x, c] = 0.2 + 0.01 * c;
        return target;
    }

    private static double Loss(GaussianSplatting<double> model, Tensor<double> target)
    {
        var rot = CameraRotation();
        var m = new Matrix<double>(3, 3);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                m[r, c] = rot[r, c];
        var image = model.RenderImage(new Vector<double>(CameraPosition), m, Width, Height, Focal);
        return new MeanSquaredErrorLoss<double>().CalculateLoss(image.ToVector(), target.ToVector());
    }

    [Fact]
    public void ImageGradient_MatchesFiniteDifferenceOfTheRenderedLoss()
    {
        AssertImageGradientMatchesFiniteDifference(shDegree: 0);
    }

    /// <summary>
    /// With view-dependent spherical harmonics the colour depends on the viewing direction (x - c) / |x - c|, so
    /// the position gradient also flows through the colour (Kerbl et al. 2023 back-propagate through the SH
    /// evaluation into the mean).
    /// </summary>
    [Fact]
    public void ImageGradient_WithViewDependentColour_MatchesFiniteDifferenceOfTheRenderedLoss()
    {
        AssertImageGradientMatchesFiniteDifference(shDegree: 1);
    }

    private static void AssertImageGradientMatchesFiniteDifference(int shDegree)
    {
        var model = BuildModel(shDegree);
        LoadScene(model, shDegree);
        int n = Positions.GetLength(0);
        int k = (shDegree + 1) * (shDegree + 1);
        Assert.Equal(n * (3 + 4 + 3 + 1 + 3 * k), model.ParameterCount);

        var target = Target();
        var rot = CameraRotation();
        var m = new Matrix<double>(3, 3);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                m[r, c] = rot[r, c];
        var image = model.RenderImage(new Vector<double>(CameraPosition), m, Width, Height, Focal);
        double maxPixel = image.ToVector().Max();
        Assert.True(maxPixel > 0.05 && maxPixel < 1.0, $"fixture must splat without clamping (max {maxPixel})");

        var parameters = model.GetParameters();
        model.Train(CameraInput(), target);
        var gradient = model.GetParameterGradients();
        Assert.Equal(parameters.Length, gradient.Length);

        // The zero learning rates leave the scene unchanged.
        var after = model.GetParameters();
        for (int i = 0; i < parameters.Length; i++)
            Assert.Equal(parameters[i], after[i], 12);

        const double h = 1e-6;
        var fd = new double[parameters.Length];
        for (int i = 0; i < parameters.Length; i++)
        {
            var plus = parameters.Clone();
            plus[i] += h;
            model.SetParameters(plus);
            double lp = Loss(model, target);
            var minus = parameters.Clone();
            minus[i] -= h;
            model.SetParameters(minus);
            double lm = Loss(model, target);
            fd[i] = (lp - lm) / (2.0 * h);
        }

        model.SetParameters(parameters);
        Assert.True(fd.Any(v => Math.Abs(v) > 1e-4), "fixture produces no gradient");

        int rotStart = 3 * n, scaleStart = 7 * n;
        for (int i = 0; i < parameters.Length; i++)
        {
            if (i >= rotStart && i < scaleStart) continue;
            AssertClose(fd[i], gradient[i], $"parameter {i}");
        }

        // Rotations are normalised before use, so the loss only sees the tangential part of the
        // (paper-convention, unnormalised) quaternion gradient: fd = (I - q q^T) g.
        for (int g = 0; g < n; g++)
        {
            var q = new double[4];
            var grad = new double[4];
            for (int j = 0; j < 4; j++)
            {
                q[j] = parameters[rotStart + g * 4 + j];
                grad[j] = gradient[rotStart + g * 4 + j];
            }

            double qg = q.Zip(grad, (a, b) => a * b).Sum();
            for (int j = 0; j < 4; j++)
                AssertClose(fd[rotStart + g * 4 + j], grad[j] - q[j] * qg, $"rotation {g}.{j}");
        }

        // The saturated red channel of Gaussian 2 contributes nothing.
        int shStart = 11 * n;
        for (int b = 0; b < k; b++)
            Assert.Equal(0.0, gradient[shStart + (2 * k + b) * 3 + 0], 12);
    }

    [Fact]
    public void RenderedColour_UsesTheWorldSpaceViewDirectionOfQueryField()
    {
        // One Gaussian whose projected mean lands exactly on pixel (5, 4): with identity camera rotation at the
        // origin, f = 10, W = 9 and H = 10 the mean of (0.3, -0.15, 3) is at (10 * 0.1 + 4, 10 * -0.05 + 4.5).
        var options = new GaussianSplattingOptions
        {
            UseSphericalHarmonics = true,
            ShDegree = 1,
            EnableSpatialIndex = false,
            EnableDensification = false,
            MaxGaussians = 4,
        };
        var model = new GaussianSplatting<double>(options, new Matrix<double>(1, 3), new Matrix<double>(1, 3));
        var position = new Tensor<double>(new[] { 1, 3 });
        position[0, 0] = 0.3; position[0, 1] = -0.15; position[0, 2] = 3.0;
        var rotation = new Tensor<double>(new[] { 1, 4 });
        rotation[0, 0] = 1.0;
        var logScale = new Tensor<double>(new[] { 1, 3 });
        for (int j = 0; j < 3; j++) logScale[0, j] = Math.Log(0.05);
        var opacity = new Tensor<double>(new[] { 1 });
        opacity[0] = 1.5;
        var sh = new Tensor<double>(new[] { 1, 4, 3 });
        for (int c = 0; c < 3; c++)
        {
            sh[0, 0, c] = 0.5 / ShDc;
            sh[0, 1, c] = 0.4 * (c + 1);   // Y_1^-1 ~ y
            sh[0, 2, c] = -0.3;            // Y_1^0  ~ z
            sh[0, 3, c] = 0.6;             // Y_1^1  ~ x
        }

        model.SetGaussianStateForTest(position, rotation, logScale, opacity, sh);

        var identity = new Matrix<double>(3, 3);
        for (int i = 0; i < 3; i++) identity[i, i] = 1.0;
        var image = model.RenderImage(new Vector<double>(3), identity, 9, 10, 10.0);

        // Viewing direction of the camera ray through the Gaussian: camera -> Gaussian, in world space.
        var direction = new Tensor<double>(new[] { 1, 3 });
        direction[0, 0] = 0.3; direction[0, 1] = -0.15; direction[0, 2] = 3.0;
        var (rgb, _) = model.QueryField(position, direction);

        double alpha = 1.0 / (1.0 + Math.Exp(-1.5));
        for (int c = 0; c < 3; c++)
        {
            Assert.True(rgb[0, c] > 0.0 && rgb[0, c] < 1.0, "fixture colour must not clamp");
            Assert.Equal(alpha * rgb[0, c], image[4, 5, c], 10);
        }
    }

    [Fact]
    public void OpacityReset_EveryThreeThousandIterationsInsideTheDensificationWindow()
    {
        var options = new GaussianSplattingOptions
        {
            UseSphericalHarmonics = false,
            ShDegree = 0,
            EnableSpatialIndex = false,
            EnableDensification = true,
            DensificationStartIteration = 500,
            DensificationEndIteration = 15000,
            MaxGaussians = 8,
        };
        var cloud = new Matrix<double>(2, 3);
        cloud[1, 0] = 1.0;
        var model = new GaussianSplatting<double>(options, cloud, cloud);

        // Not a reset iteration: opacities stay at the initial 0.5.
        model.RunOpacityResetForTest(2999);
        Assert.All(model.GetGaussianDiagnostics(), d => Assert.Equal(0.5, d.Opacity, 12));

        // Kerbl et al. 2023 section 5.2: every 3000 iterations alpha <- min(alpha, 0.01).
        model.RunOpacityResetForTest(3000);
        Assert.All(model.GetGaussianDiagnostics(), d => Assert.Equal(0.01, d.Opacity, 12));

        // After the densification window the cloud is frozen and no reset happens.
        var late = new GaussianSplatting<double>(options, cloud, cloud);
        late.RunOpacityResetForTest(18000);
        Assert.All(late.GetGaussianDiagnostics(), d => Assert.Equal(0.5, d.Opacity, 12));
    }

    private static GaussianSplatting<double> BuildDensifyModel(double[,] positions, double[] scales, int? accumulationWindow = null)
    {
        int n = scales.Length;
        var options = new GaussianSplattingOptions
        {
            UseSphericalHarmonics = false,
            ShDegree = 0,
            EnableSpatialIndex = false,
            EnableDensification = true,
            DensificationStartIteration = 1,
            DensificationEndIteration = 100,
            GradientNormThreshold = 0.2,
            GradientAccumulationWindow = accumulationWindow,
            MaxGaussians = 16,
        };
        var cloud = new Matrix<double>(n, 3);
        var model = new GaussianSplatting<double>(options, cloud, cloud);
        var position = new Tensor<double>(new[] { n, 3 });
        var rotation = new Tensor<double>(new[] { n, 4 });
        var logScale = new Tensor<double>(new[] { n, 3 });
        var opacity = new Tensor<double>(new[] { n });
        var sh = new Tensor<double>(new[] { n, 1, 3 });
        for (int i = 0; i < n; i++)
        {
            rotation[i, 0] = 1.0;
            for (int j = 0; j < 3; j++)
            {
                position[i, j] = positions[i, j];
                logScale[i, j] = Math.Log(scales[i]);
                sh[i, 0, j] = 0.5;
            }
        }

        model.SetGaussianStateForTest(position, rotation, logScale, opacity, sh);
        return model;
    }

    private static Tensor<double> XGradients(params double[] values)
    {
        var gradients = new Tensor<double>(new[] { values.Length, 2 });
        for (int i = 0; i < values.Length; i++) gradients[i, 0] = values[i];
        return gradients;
    }

    [Fact]
    public void Densification_ClonesSmallGaussiansAndSplitsLargeOnes()
    {
        // Kerbl et al. 2023 section 5.2: Gaussians no larger than 1% of the scene extent are cloned unchanged,
        // larger ones are split. A and B are tiny; C is large and far away, so the extent is about 7.3.
        var model = BuildDensifyModel(
            new[,] { { 0.0, 0.0, 0.0 }, { 0.001, 0.0, 0.0 }, { 10.0, 0.0, 0.0 } },
            new[] { 0.001, 0.001, 1.0 });

        model.RunDensifyAndPruneForTest(10, 1.0);

        var d = model.GetGaussianDiagnostics();
        Assert.Equal(6, d.Count);

        // Clone sources and clones are identical to the original small Gaussians.
        foreach (var (row, x) in new[] { (0, 0.0), (1, 0.001), (4, 0.0), (5, 0.001) })
        {
            Assert.Equal(x, d[row].PosX, 12);
            Assert.Equal(0.0, d[row].PosY, 12);
            Assert.Equal(0.0, d[row].PosZ, 12);
            Assert.Equal(0.001, d[row].MaxScale, 12);
            Assert.Equal(0.5, d[row].Opacity, 12);
        }

        // The large Gaussian is split: both children move apart along the gradient and shrink.
        Assert.True(d[2].PosX > 10.0, $"split source should move along +x (x = {d[2].PosX})");
        Assert.True(d[3].PosX < 10.0, $"split child should move along -x (x = {d[3].PosX})");
        Assert.Equal(d[2].MaxScale, d[3].MaxScale, 12);
        Assert.True(d[2].MaxScale < 1.0, "split children shrink");
    }

    [Fact]
    public void Densification_AveragesTheGradientOverTheAccumulationWindow()
    {
        var positions = new[,] { { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 } };
        var scales = new[] { 0.5, 0.5 };

        // Averages 0.15 and 0.15 are below the 0.2 threshold although the last step alone would split Gaussian 1.
        var averaged = BuildDensifyModel(positions, scales);
        averaged.AccumulateGradientsForTest(XGradients(0.3, 0.0));
        averaged.AccumulateGradientsForTest(XGradients(0.0, 0.3));
        averaged.RunDensifyFromAccumulatedForTest(10);
        Assert.Equal(2, averaged.GaussianCount);

        // Averages 0.3 and 0.15: only Gaussian 0 is split, although the first step alone would split both.
        var mixed = BuildDensifyModel(positions, scales);
        mixed.AccumulateGradientsForTest(XGradients(0.3, 0.3));
        mixed.AccumulateGradientsForTest(XGradients(0.3, 0.0));
        mixed.RunDensifyFromAccumulatedForTest(10);
        var d = mixed.GetGaussianDiagnostics();
        Assert.Equal(3, d.Count);
        Assert.True(Math.Abs(d[2].PosX) < 0.5, $"the split child belongs to Gaussian 0 (x = {d[2].PosX})");
        Assert.Equal(1.0, d[1].PosX, 12);

        // A one-step window restarts the statistics every step, so only the last step counts.
        var windowed = BuildDensifyModel(positions, scales, accumulationWindow: 1);
        windowed.AccumulateGradientsForTest(XGradients(0.3, 0.0));
        windowed.AccumulateGradientsForTest(XGradients(0.0, 0.3));
        windowed.RunDensifyFromAccumulatedForTest(10);
        var w = windowed.GetGaussianDiagnostics();
        Assert.Equal(3, w.Count);
        Assert.True(Math.Abs(w[2].PosX - 1.0) < 0.5, $"the split child belongs to Gaussian 1 (x = {w[2].PosX})");
        Assert.Equal(0.0, w[0].PosX, 12);
    }

    [Fact]
    public void DensificationStatistic_IsTheNdcGradientOfTheProjectedMean()
    {
        // Kerbl et al. 2023 section 5.2 / reference viewspace_point_tensor.grad[:, :2]: the statistic is the norm of
        // dL/d(projected mean) in normalised device coordinates, d(pixel)/d(ndc) = (W / 2, H / 2), and the default
        // threshold is the paper's 0.0002 on that quantity.
        Assert.Equal(0.0002, new GaussianSplattingOptions().SplitGradientThreshold, 12);

        var options = new GaussianSplattingOptions
        {
            UseSphericalHarmonics = false,
            ShDegree = 0,
            EnableSpatialIndex = false,
            EnableDensification = true,
            DensificationStartIteration = 1000,
            DensificationEndIteration = 2000,
            MaxGaussians = 4,
        };
        var cloud = new Matrix<double>(1, 3);
        var model = new GaussianSplatting<double>(options, cloud, cloud, new MeanSquaredErrorLoss<double>());
        Assert.Equal(0.0002, model.EffectiveGradientNormThreshold, 12);
        model.PositionLearningRate = 0.0;
        model.ColorLearningRate = 0.0;
        model.OpacityLearningRate = 0.0;
        model.ScaleLearningRate = 0.0;
        model.RotationLearningRate = 0.0;

        // One isotropic Gaussian on the optical axis: to first order a camera-space x/y shift moves only its
        // projected mean (dmu/dx = f / z) -- the projected covariance is stationary there -- so the finite difference
        // of the loss over the world position gives the pixel-space mean gradient.
        const double z = 3.0;
        var position = new Tensor<double>(new[] { 1, 3 });
        position[0, 2] = z;
        var rotation = new Tensor<double>(new[] { 1, 4 });
        rotation[0, 0] = 1.0;
        var logScale = new Tensor<double>(new[] { 1, 3 });
        for (int j = 0; j < 3; j++) logScale[0, j] = Math.Log(0.3);
        var opacity = new Tensor<double>(new[] { 1 });
        opacity[0] = 0.4;
        var sh = new Tensor<double>(new[] { 1, 1, 3 });
        for (int c = 0; c < 3; c++) sh[0, 0, c] = 0.3 + 0.1 * c;
        model.SetGaussianStateForTest(position, rotation, logScale, opacity, sh);

        var target = new Tensor<double>(new[] { Height, Width, 3 });
        for (int y = 0; y < Height; y++)
            for (int x = 0; x < Width; x++)
                for (int c = 0; c < 3; c++)
                    target[y, x, c] = 0.05 + 0.03 * x + 0.02 * y;

        var identity = new Matrix<double>(3, 3);
        for (int i = 0; i < 3; i++) identity[i, i] = 1.0;
        var input = new Tensor<double>(new double[] { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, Focal }, new[] { 1, 13 });
        double LossAt(Vector<double> p)
        {
            model.SetParameters(p);
            var image = model.RenderImage(new Vector<double>(3), identity, Width, Height, Focal);
            return new MeanSquaredErrorLoss<double>().CalculateLoss(image.ToVector(), target.ToVector());
        }

        var parameters = model.GetParameters();
        const double h = 1e-6;
        var pixelGrad = new double[2];
        for (int axis = 0; axis < 2; axis++)
        {
            var plus = parameters.Clone();
            plus[axis] += h;
            var minus = parameters.Clone();
            minus[axis] -= h;
            pixelGrad[axis] = (LossAt(plus) - LossAt(minus)) / (2.0 * h) * z / Focal;
        }

        model.SetParameters(parameters);
        model.Train(input, target);

        double gx = pixelGrad[0] * 0.5 * Width;
        double gy = pixelGrad[1] * 0.5 * Height;
        double expected = Math.Sqrt(gx * gx + gy * gy);
        Assert.True(expected > 1e-4, $"fixture produces no screen-space gradient ({expected})");
        AssertClose(expected, model.DensificationStatisticForTest()[0], "NDC densification statistic");
    }

    private static void AssertClose(double expected, double actual, string what)
    {
        double tolerance = 1e-6 + 1e-4 * Math.Abs(expected);
        Assert.True(Math.Abs(expected - actual) <= tolerance,
            $"{what}: finite difference {expected:R}, analytic {actual:R}");
    }
}
