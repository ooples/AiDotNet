using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ComputerVision.Segmentation.Salient;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.ThreeD.Meshing;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.ThreeD;

/// <summary>
/// DreamGaussian (Tang et al., ICLR 2024) against its official code: the SDS gradient, the differentiable Gaussian
/// rasterizer's backward pass, the reference-view loss, the densify/prune rules, the texture stage and the whole
/// pipeline on a tiny configuration.
/// </summary>
public class DreamGaussianTests
{
    // ---- Score distillation ----------------------------------------------------------------------------------------

    /// <summary>An epsilon predictor whose output depends on the latent and on the conditioning.</summary>
    private sealed class AffinePredictor : NoisePredictorBase<double>
    {
        public override int InputChannels => 4;
        public override int OutputChannels => 4;
        public override int BaseChannels => 1;
        public override int TimeEmbeddingDim => 1;
        public override bool SupportsCFG => true;
        public override bool SupportsCrossAttention => true;
        public override int ContextDimension => 1;

        public override Tensor<double> PredictNoise(Tensor<double> noisySample, int timestep, Tensor<double>? conditioning = null)
        {
            double shift = conditioning is null ? 0.0 : conditioning.ToArray()[0];
            var values = noisySample.ToArray();
            for (int i = 0; i < values.Length; i++)
                values[i] = (0.3 * values[i]) + shift + (0.001 * timestep);
            return new Tensor<double>(noisySample.Shape.ToArray(), new Vector<double>(values));
        }
    }

    private static Tensor<double> Filled(int[] shape, Func<int, double> value)
    {
        int length = shape.Aggregate(1, (a, b) => a * b);
        var data = new double[length];
        for (int i = 0; i < length; i++) data[i] = value(i);
        return new Tensor<double>(shape, new Vector<double>(data));
    }

    /// <summary>alpha_bar_t of Stable Diffusion's scaled-linear schedule, computed independently of the scheduler.</summary>
    private static double ScaledLinearAlphaBar(int t)
    {
        double start = Math.Sqrt(0.00085), end = Math.Sqrt(0.012);
        double product = 1.0;
        for (int i = 0; i <= t; i++)
        {
            double root = start + ((end - start) * i / 999.0);
            product *= 1.0 - (root * root);
        }

        return product;
    }

    private static StandardVAE<double> TinyVae() => new StandardVAE<double>(
        inputChannels: 3, latentChannels: 4, baseChannels: 32, channelMultipliers: new[] { 1, 2, 4 },
        numResBlocksPerLevel: 1, latentScaleFactor: 0.18215, seed: 42);

    [Fact]
    public void SdsLatentGradient_IsWeightTimesGuidedResidual()
    {
        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateStableDiffusion());
        var prompt = Filled(new[] { 1, 1, 1 }, _ => 0.25);
        var negative = Filled(new[] { 1, 1, 1 }, _ => -0.5);
        const double scale = 100.0;
        var guidance = new StableDiffusionScoreDistillationGuidance<double>(
            new AffinePredictor(), TinyVae(), scheduler, prompt, negative, scale, resolution: 16);

        var shape = new[] { 1, 4, 4, 4 };
        var latents = Filled(shape, i => Math.Sin(0.7 * i));
        var noise = Filled(shape, i => Math.Cos(1.3 * i));
        foreach (int t in new[] { 20, 500, 980 })
        {
            var gradient = guidance.LatentGradient(latents, noise, t).ToArray();
            double alphaBar = ScaledLinearAlphaBar(t);
            double weight = 1.0 - alphaBar;
            var z = latents.ToArray();
            var eps = noise.ToArray();
            for (int i = 0; i < z.Length; i++)
            {
                double noisy = (Math.Sqrt(alphaBar) * z[i]) + (Math.Sqrt(1.0 - alphaBar) * eps[i]);
                double conditional = (0.3 * noisy) + 0.25 + (0.001 * t);
                double unconditional = (0.3 * noisy) - 0.5 + (0.001 * t);
                double guided = unconditional + (scale * (conditional - unconditional));
                double expected = weight * (guided - eps[i]);
                Assert.True(Math.Abs(expected - gradient[i]) <= 1e-9 * Math.Max(1.0, Math.Abs(expected)),
                    $"t={t}, i={i}: {gradient[i]:G17} vs {expected:G17}");
            }
        }
    }

    [Fact]
    public void SdsTimestep_AnnealsLinearlyAndClipsToTheRange()
    {
        Assert.Equal(980, LatentScoreDistillation<double>.AnnealedTimestep(0.0, 1000, 20, 980));
        Assert.Equal(500, LatentScoreDistillation<double>.AnnealedTimestep(0.5, 1000, 20, 980));
        Assert.Equal(800, LatentScoreDistillation<double>.AnnealedTimestep(0.2, 1000, 20, 980));
        Assert.Equal(20, LatentScoreDistillation<double>.AnnealedTimestep(1.0, 1000, 20, 980));
    }

    [Fact]
    public void SdsImageGradient_IsTheEncoderVjpOfTheLatentResidual()
    {
        var vae = TinyVae();
        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateStableDiffusion());
        const int resolution = 16;
        const int timestep = 600;
        Func<Tensor<double>, int, Tensor<double>> predict = (latents, t) =>
            AiDotNetEngine.Current.TensorMultiplyScalar(latents, 0.3);
        var images = Filled(new[] { 1, 3, resolution, resolution }, i => 0.5 + (0.4 * Math.Sin(0.37 * i)));

        var gradient = LatentScoreDistillation<double>.ImageGradient(
            vae, scheduler, resolution, images, timestep, RandomHelper.CreateSeededRandom(5), predict).ToArray();

        // Replay the same draws: the posterior sample, then the noise.
        var replay = RandomHelper.CreateSeededRandom(5);
        var shape = LatentScoreDistillation<double>.LatentShape(vae, 1, resolution);
        var posteriorNoise = DiffusionNoiseHelper<double>.SampleGaussian(shape, replay);
        var noise = DiffusionNoiseHelper<double>.SampleGaussian(shape, replay);
        var latent = LatentScoreDistillation<double>.Encode(vae, images, posteriorNoise, scaled: true);
        var residual = LatentScoreDistillation<double>.LatentGradient(scheduler, latent, noise, timestep, predict).ToArray();

        double Objective(Tensor<double> x)
        {
            var z = LatentScoreDistillation<double>.Encode(vae, x, posteriorNoise, scaled: true).ToArray();
            double sum = 0.0;
            for (int i = 0; i < z.Length; i++) sum += z[i] * residual[i];
            return sum;
        }

        const double h = 1e-5;
        var baseValues = images.ToArray();
        double norm = Math.Sqrt(gradient.Sum(g => g * g));
        Assert.True(norm > 0.0, "The SDS gradient must reach the renders.");
        foreach (int index in new[] { 0, 77, 300, 3 * resolution * resolution - 1 })
        {
            var plus = (double[])baseValues.Clone();
            var minus = (double[])baseValues.Clone();
            plus[index] += h;
            minus[index] -= h;
            double numeric = (Objective(new Tensor<double>(images.Shape.ToArray(), new Vector<double>(plus)))
                - Objective(new Tensor<double>(images.Shape.ToArray(), new Vector<double>(minus)))) / (2.0 * h);
            Assert.True(Math.Abs(numeric - gradient[index]) <= 1e-6 + (1e-4 * Math.Abs(numeric)),
                $"pixel {index}: analytic {gradient[index]:G10} vs numeric {numeric:G10}");
        }
    }

    // ---- Differentiable Gaussian rasterizer ------------------------------------------------------------------------

    private static GaussianSplatting<double> SmallScene(int count, int seed)
        => GaussianSplatting<double>.CreateInSphere(
            new GaussianSplattingOptions { UseSphericalHarmonics = true, ShDegree = 0 }, count, 0.3, 0.6,
            RandomHelper.CreateSeededRandom(seed));

    [Fact]
    public void GaussianRasterizerBackward_MatchesFiniteDifferences()
    {
        var scene = SmallScene(6, 3);
        var camera = GaussianSplatCamera.FromOrbit(-10.0, 25.0, 2.0, 24, 24, 49.1);
        var background = new[] { 0.2, 0.3, 0.4 };
        var weightRandom = RandomHelper.CreateSeededRandom(11);
        var imageWeights = Enumerable.Range(0, 3 * 24 * 24).Select(_ => weightRandom.NextDouble() - 0.5).ToArray();
        var alphaWeights = Enumerable.Range(0, 24 * 24).Select(_ => weightRandom.NextDouble() - 0.5).ToArray();

        double Loss()
        {
            var r = scene.RenderDifferentiable(camera, background);
            double sum = 0.0;
            for (int i = 0; i < imageWeights.Length; i++) sum += imageWeights[i] * r.Image[i];
            for (int i = 0; i < alphaWeights.Length; i++) sum += alphaWeights[i] * r.Alpha[i];
            return sum;
        }

        var render = scene.RenderDifferentiable(camera, background);
        var gradients = scene.BackwardDifferentiable(render, imageWeights, alphaWeights);
        var original = scene.GetGenerativeParameters();

        void Check(string name, double analytic, Action<double[], double[], double[], double[], double[], double> perturb)
        {
            const double h = 1e-7;
            double Evaluate(double delta)
            {
                var p = (double[])original.Position.Clone();
                var s = (double[])original.LogScale.Clone();
                var q = (double[])original.Rotation.Clone();
                var o = (double[])original.OpacityLogit.Clone();
                var c = (double[])original.Color.Clone();
                perturb(p, s, q, o, c, delta);
                scene.SetGenerativeParameters(p, s, q, o, c);
                return Loss();
            }

            double numeric = (Evaluate(h) - Evaluate(-h)) / (2.0 * h);
            Assert.True(Math.Abs(numeric - analytic) <= 1e-5 + (1e-3 * Math.Abs(numeric)),
                $"{name}: analytic {analytic:G10} vs numeric {numeric:G10}");
        }

        for (int g = 0; g < 6; g += 2)
        {
            int i = g;
            Check($"x[{i}]", gradients.Position[3 * i], (p, s, q, o, c, d) => p[3 * i] += d);
            Check($"z[{i}]", gradients.Position[(3 * i) + 2], (p, s, q, o, c, d) => p[(3 * i) + 2] += d);
            Check($"logScale[{i}]", gradients.LogScale[(3 * i) + 1], (p, s, q, o, c, d) => s[(3 * i) + 1] += d);
            Check($"rotation[{i}]", gradients.Rotation[(4 * i) + 2], (p, s, q, o, c, d) => q[(4 * i) + 2] += d);
            Check($"opacity[{i}]", gradients.OpacityLogit[i], (p, s, q, o, c, d) => o[i] += d);
            Check($"color[{i}]", gradients.Color[(3 * i) + 1], (p, s, q, o, c, d) => c[(3 * i) + 1] += d);
        }
    }

    [Fact]
    public void ReferenceViewLoss_DecreasesOverTheFirstSteps()
    {
        var settings = TinySettings();
        settings.Iterations = 30;
        settings.ImageGuidanceWeight = 0.0;
        settings.ImageDensifyStartIteration = 1000;
        var (image, mask) = DiskReference(16, 0.9, 0.1, 0.1, 0.35);
        var pipeline = new DreamGaussianPipeline<double>(
            settings, DreamGaussianMode.ImageTo3D, new StubGuidance(16), image, mask, RandomHelper.CreateSeededRandom(1));
        pipeline.TrainGaussians();

        var history = pipeline.ReferenceLossHistory;
        Assert.Equal(30, history.Count);
        Assert.True(history[history.Count - 1] < 0.8 * history[0],
            $"Reference loss went from {history[0]:G6} to {history[history.Count - 1]:G6}.");
    }

    // ---- Adam schedule, densification and pruning ------------------------------------------------------------------

    [Fact]
    public void PositionLearningRate_IsTheExponentialSchedule()
    {
        var adam = new GaussianAdamSettings();
        Assert.Equal(0.001, adam.PositionLearningRate(0), 12);
        Assert.Equal(Math.Sqrt(0.001 * 0.00002), adam.PositionLearningRate(250), 12);
        Assert.Equal(0.00002, adam.PositionLearningRate(500), 12);
        Assert.Equal(0.00002, adam.PositionLearningRate(900), 12);
    }

    [Fact]
    public void DensifyAndPrune_FollowsTheOfficialThresholds()
    {
        var scene = SmallScene(6, 9);
        const double threshold = 0.01;
        var settings = new GaussianDensificationSettings();
        // percent_dense 0.01 * extent 4 = 0.04 separates clone from split; 0.1 * extent 4 = 0.4 is the world prune.
        double[] maxScale = { 0.02, 0.1, 0.02, 0.02, 0.5, 0.039 };
        double[] opacity = { 0.5, 0.5, 0.5, 0.005, 0.5, 0.5 };
        double[] grad = { 0.02, 0.02, 0.005, 0.02, 0.0, threshold };
        var position = new double[18];
        var logScale = new double[18];
        var rotation = new double[24];
        var logit = new double[6];
        var color = new double[18];
        for (int i = 0; i < 6; i++)
        {
            position[3 * i] = 0.1 * i;
            logScale[3 * i] = Math.Log(maxScale[i]);
            logScale[(3 * i) + 1] = Math.Log(maxScale[i] / 2.0);
            logScale[(3 * i) + 2] = Math.Log(maxScale[i] / 4.0);
            rotation[4 * i] = 1.0;
            logit[i] = Math.Log(opacity[i] / (1.0 - opacity[i]));
            color[3 * i] = i;
        }

        scene.SetGenerativeParameters(position, logScale, rotation, logit, color);
        // Two views each; Gaussian 2 is huge on screen, which the official order never prunes.
        scene.SetDensificationStatistics(grad.Select(g => 2.0 * g).ToArray(), new[] { 2, 2, 2, 2, 2, 2 },
            new[] { 0.0, 0.0, 100.0, 0.0, 0.0, 0.0 });

        var report = scene.DensifyAndPruneGenerative(settings, RandomHelper.CreateSeededRandom(4));

        Assert.Equal(3, report.Cloned);   // 0, 3 and 5 (grad >= 0.01, max scale <= 0.04); 3 is then pruned
        Assert.Equal(1, report.Split);    // 1 (grad >= 0.01, max scale > 0.04)
        Assert.Equal(3, report.Pruned);   // 3 and its clone (opacity < 0.01), 4 (max scale > 0.4)
        Assert.Equal(6 + 3 + 2 - 1 - 3, report.Count);

        var after = scene.GetGenerativeParameters();
        // Survivors in the reference's order: originals 0, 2, 5; clones of 0 and 5; the two children of 1.
        var expectedColor = new[] { 0.0, 2.0, 5.0, 0.0, 5.0, 1.0, 1.0 };
        Assert.Equal(expectedColor.Length, after.OpacityLogit.Length);
        for (int i = 0; i < expectedColor.Length; i++)
            Assert.Equal(expectedColor[i], after.Color[3 * i], 12);
        for (int k = 0; k < 3; k++)
        {
            // Clones copy their parent exactly; children shrink by 0.8 * N = 1.6.
            Assert.Equal(after.LogScale[k], after.LogScale[9 + k], 12);
            Assert.Equal(logScale[3 + k] - Math.Log(1.6), after.LogScale[15 + k], 12);
            Assert.Equal(logScale[3 + k] - Math.Log(1.6), after.LogScale[18 + k], 12);
        }

        // Children are sampled around the parent, not on it.
        Assert.NotEqual(0.1, after.Position[15]);
        Assert.NotEqual(after.Position[15], after.Position[18]);
    }

    [Fact]
    public void OpacityReset_CapsAtTheCeiling()
    {
        var scene = SmallScene(3, 2);
        var p = scene.GetGenerativeParameters();
        var logits = new[] { Math.Log(0.9 / 0.1), Math.Log(0.005 / 0.995), 0.0 };
        scene.SetGenerativeParameters(p.Position, p.LogScale, p.Rotation, logits, p.Color);
        scene.ResetGenerativeOpacity(0.01);
        var after = scene.GetGenerativeParameters().OpacityLogit;
        Assert.Equal(Math.Log(0.01 / 0.99), after[0], 12);
        Assert.Equal(logits[1], after[1], 12);
        Assert.Equal(Math.Log(0.01 / 0.99), after[2], 12);
    }

    // ---- Texture stage ---------------------------------------------------------------------------------------------

    private static (double[] Vertices, int[] Faces) Cube()
    {
        var vertices = new double[24];
        for (int i = 0; i < 8; i++)
        {
            vertices[3 * i] = (i & 1) == 0 ? -0.5 : 0.5;
            vertices[(3 * i) + 1] = (i & 2) == 0 ? -0.5 : 0.5;
            vertices[(3 * i) + 2] = (i & 4) == 0 ? -0.5 : 0.5;
        }

        // Outward-wound quads split into triangles.
        int[] faces =
        {
            0, 2, 3, 0, 3, 1, // -z
            4, 5, 7, 4, 7, 6, // +z
            0, 1, 5, 0, 5, 4, // -y
            2, 6, 7, 2, 7, 3, // +y
            0, 4, 6, 0, 6, 2, // -x
            1, 3, 7, 1, 7, 5, // +x
        };
        return (vertices, faces);
    }

    [Fact]
    public void UvAtlas_ChartsDoNotOverlapAndKeepPositions()
    {
        var (vertices, faces) = Cube();
        var atlas = UvAtlasUnwrapper.Unwrap(vertices, faces, new UvAtlasOptions { TextureResolution = 64, PaddingTexels = 2 });
        int faceCount = faces.Length / 3;
        Assert.Equal(faces.Length, atlas.Faces.Length);
        Assert.All(atlas.Uvs, uv => Assert.InRange(uv, 0.0, 1.0));
        for (int v = 0; v < atlas.SourceVertex.Length; v++)
        {
            for (int k = 0; k < 3; k++)
                Assert.Equal(vertices[(3 * atlas.SourceVertex[v]) + k], atlas.Vertices[(3 * v) + k]);
        }

        for (int f = 0; f < faceCount; f++)
        {
            for (int k = 0; k < 3; k++)
                Assert.Equal(faces[(3 * f) + k], atlas.SourceVertex[atlas.Faces[(3 * f) + k]]);
        }

        double[] Uv(int f)
        {
            var t = new double[6];
            for (int k = 0; k < 3; k++)
            {
                t[2 * k] = atlas.Uvs[2 * atlas.Faces[(3 * f) + k]];
                t[(2 * k) + 1] = atlas.Uvs[(2 * atlas.Faces[(3 * f) + k]) + 1];
            }

            return t;
        }

        for (int a = 0; a < faceCount; a++)
        {
            for (int b = a + 1; b < faceCount; b++)
                Assert.False(UvAtlasUnwrapper.TrianglesOverlap(Uv(a), Uv(b)), $"UV triangles {a} and {b} overlap.");
        }
    }

    private static double[] UvTriangle(UvAtlas atlas, int f)
    {
        var t = new double[6];
        for (int k = 0; k < 3; k++)
        {
            t[2 * k] = atlas.Uvs[2 * atlas.Faces[(3 * f) + k]];
            t[(2 * k) + 1] = atlas.Uvs[(2 * atlas.Faces[(3 * f) + k]) + 1];
        }

        return t;
    }

    private static double SignedUvArea(double[] t)
        => 0.5 * (((t[2] - t[0]) * (t[5] - t[1])) - ((t[4] - t[0]) * (t[3] - t[1])));

    /// <summary>
    /// The ratio of the singular values of the 3D-to-UV Jacobian of a face: 1 for a conformal (angle-preserving) map.
    /// </summary>
    private static double ConformalDistortion(UvAtlas atlas, int f)
    {
        double[] P(int k)
        {
            int v = atlas.Faces[(3 * f) + k];
            return new[] { atlas.Vertices[3 * v], atlas.Vertices[(3 * v) + 1], atlas.Vertices[(3 * v) + 2] };
        }

        double[] p0 = P(0), p1 = P(1), p2 = P(2);
        double[] e1 = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
        double[] e2 = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };
        double l1 = Math.Sqrt((e1[0] * e1[0]) + (e1[1] * e1[1]) + (e1[2] * e1[2]));
        double x2 = ((e1[0] * e2[0]) + (e1[1] * e2[1]) + (e1[2] * e2[2])) / l1;
        double e2Length2 = (e2[0] * e2[0]) + (e2[1] * e2[1]) + (e2[2] * e2[2]);
        double y2 = Math.Sqrt(e2Length2 - (x2 * x2));
        var t = UvTriangle(atlas, f);
        // J = D_uv * D_xy^-1 with D_xy = [[l1, x2], [0, y2]].
        double a = t[2] - t[0], b = t[4] - t[0], c = t[3] - t[1], d = t[5] - t[1];
        double j00 = a / l1, j01 = (b - (a * x2 / l1)) / y2;
        double j10 = c / l1, j11 = (d - (c * x2 / l1)) / y2;
        double frobenius = (j00 * j00) + (j01 * j01) + (j10 * j10) + (j11 * j11);
        double det = (j00 * j11) - (j01 * j10);
        double root = Math.Sqrt(Math.Max(0.0, (frobenius * frobenius) - (4.0 * det * det)));
        return Math.Sqrt((frobenius + root) / (frobenius - root));
    }

    private static (double[] Vertices, int[] Faces) UvSphere(int rings, int segments)
    {
        var vertices = new List<double> { 0.0, 0.0, 1.0 };
        for (int r = 1; r < rings; r++)
        {
            double theta = Math.PI * r / rings;
            for (int s = 0; s < segments; s++)
            {
                double phi = 2.0 * Math.PI * s / segments;
                vertices.Add(Math.Sin(theta) * Math.Cos(phi));
                vertices.Add(Math.Sin(theta) * Math.Sin(phi));
                vertices.Add(Math.Cos(theta));
            }
        }

        vertices.AddRange(new[] { 0.0, 0.0, -1.0 });
        int south = (vertices.Count / 3) - 1;
        int Ring(int r, int s) => 1 + ((r - 1) * segments) + (((s % segments) + segments) % segments);
        var faces = new List<int>();
        for (int s = 0; s < segments; s++)
        {
            faces.AddRange(new[] { 0, Ring(1, s), Ring(1, s + 1) });
            for (int r = 1; r < rings - 1; r++)
            {
                faces.AddRange(new[] { Ring(r, s), Ring(r + 1, s), Ring(r + 1, s + 1) });
                faces.AddRange(new[] { Ring(r, s), Ring(r + 1, s + 1), Ring(r, s + 1) });
            }

            faces.AddRange(new[] { Ring(rings - 1, s), south, Ring(rings - 1, s + 1) });
        }

        return (vertices.ToArray(), faces.ToArray());
    }

    [Fact]
    public void UvAtlas_PlanarQuad_IsOneConformalChart()
    {
        // A 2 x 0.71 rectangle in a tilted plane: LSCM maps it to a similar rectangle, so every face's distortion is 1.
        double[] origin = { 0.3, -0.2, 0.1 };
        double[] along = { 2.0 / Math.Sqrt(2.0), 2.0 / Math.Sqrt(2.0), 0.0 };
        double[] across = { -0.5 / Math.Sqrt(3.0), 0.5 / Math.Sqrt(3.0), 1.0 / Math.Sqrt(3.0) };
        var vertices = new double[12];
        for (int i = 0; i < 4; i++)
        {
            double s = (i & 1) == 0 ? 0.0 : 1.0, t = (i & 2) == 0 ? 0.0 : 1.0;
            for (int k = 0; k < 3; k++)
                vertices[(3 * i) + k] = origin[k] + (s * along[k]) + (t * across[k]);
        }

        int[] faces = { 0, 1, 3, 0, 3, 2 };
        var atlas = UvAtlasUnwrapper.Unwrap(vertices, faces, new UvAtlasOptions { TextureResolution = 256, PaddingTexels = 2 });
        Assert.Equal(1, atlas.ChartCount);
        Assert.Equal(4, atlas.SourceVertex.Length);
        for (int f = 0; f < 2; f++)
        {
            Assert.Equal(1.0, ConformalDistortion(atlas, f), 6);
            Assert.True(SignedUvArea(UvTriangle(atlas, f)) > 0.0);
        }

        // A similarity: both triangles keep the same UV-to-surface area ratio, so texel density is uniform.
        Assert.Equal(SignedUvArea(UvTriangle(atlas, 0)), SignedUvArea(UvTriangle(atlas, 1)), 9);
    }

    [Fact]
    public void UvAtlas_CurvedSurface_CoversEveryFaceWithoutFlipsOrOverlaps()
    {
        var (vertices, faces) = UvSphere(8, 12);
        int faceCount = faces.Length / 3;
        var atlas = UvAtlasUnwrapper.Unwrap(vertices, faces, new UvAtlasOptions { TextureResolution = 512, PaddingTexels = 2 });

        Assert.InRange(atlas.ChartCount, 2, faceCount - 1);
        Assert.Equal(faceCount, atlas.FaceChart.Length);
        Assert.All(atlas.FaceChart, c => Assert.InRange(c, 0, atlas.ChartCount - 1));
        Assert.All(atlas.Uvs, uv => Assert.InRange(uv, 0.0, 1.0));
        for (int f = 0; f < faceCount; f++)
        {
            // Covered with the input's orientation, and still close to conformal on a smooth surface.
            Assert.True(SignedUvArea(UvTriangle(atlas, f)) > 0.0, $"Face {f} is flipped or collapsed in UV.");
            Assert.InRange(ConformalDistortion(atlas, f), 1.0, 1.5);
            for (int k = 0; k < 3; k++)
                Assert.Equal(faces[(3 * f) + k], atlas.SourceVertex[atlas.Faces[(3 * f) + k]]);
        }

        for (int a = 0; a < faceCount; a++)
        {
            for (int b = a + 1; b < faceCount; b++)
                Assert.False(UvAtlasUnwrapper.TrianglesOverlap(UvTriangle(atlas, a), UvTriangle(atlas, b)), $"UV triangles {a} and {b} overlap.");
        }

        // The padding gutter: no UV closer than the padding to the square's border.
        double padding = 2.0 / 512.0;
        Assert.All(atlas.Uvs, uv => Assert.InRange(uv, padding - 1e-12, 1.0 - padding + 1e-12));
    }

    [Fact]
    public void UvAtlas_Segmentation_KeepsEachChartWithinItsNormalCone()
    {
        var (vertices, faces) = UvSphere(8, 12);
        var mesh = new UvAtlasUnwrapper.MeshData(vertices, faces);
        var all = Enumerable.Range(0, faces.Length / 3).ToArray();
        double cosLimit = Math.Cos(Math.PI / 4.0);
        var groups = UvAtlasUnwrapper.Segment(mesh, all, cosLimit);
        Assert.Equal(all, groups.SelectMany(g => g).OrderBy(f => f).ToArray());
        Assert.True(groups.Count > 1);
        foreach (var group in groups)
        {
            // Every face joined within the cone of the chart normal at the time, so none faces away from the final one.
            double sx = group.Sum(f => mesh.Normals[3 * f] * mesh.Areas[f]);
            double sy = group.Sum(f => mesh.Normals[(3 * f) + 1] * mesh.Areas[f]);
            double sz = group.Sum(f => mesh.Normals[(3 * f) + 2] * mesh.Areas[f]);
            double length = Math.Sqrt((sx * sx) + (sy * sy) + (sz * sz));
            foreach (int f in group)
            {
                double cos = ((mesh.Normals[3 * f] * sx) + (mesh.Normals[(3 * f) + 1] * sy) + (mesh.Normals[(3 * f) + 2] * sz)) / length;
                Assert.True(cos > 0.0, $"Face {f} faces away from its chart.");
            }
        }
    }

    [Fact]
    public void MeshRasterizer_SeesTheFrontFaceAtTheRightDepth()
    {
        var (vertices, faces) = Cube();
        // Azimuth 0 looks from +z towards the origin; the +z face is at distance 2 - 0.5.
        var camera = GaussianSplatCamera.FromOrbit(0.0, 0.0, 2.0, 15, 15, 49.1);
        var raster = MeshRasterizer.Rasterize(vertices, faces, camera);
        int center = (7 * 15) + 7;
        Assert.InRange(raster.Face[center], 2, 3);
        Assert.Equal(1.5, raster.Depth[center], 9);
        Assert.Equal(-1, raster.Face[0]);
    }

    [Fact]
    public void TextureBackward_MatchesFiniteDifferences()
    {
        var (vertices, faces) = Cube();
        var geometry = DreamGaussianTexturing.Prepare(vertices, faces, new UvAtlasOptions { TextureResolution = 16, PaddingTexels = 1 });
        var view = DreamGaussianTexturing.SampleView(geometry, GaussianSplatCamera.FromOrbit(-20.0, 30.0, 2.0, 12, 12, 49.1));
        var random = RandomHelper.CreateSeededRandom(8);
        var raw = Enumerable.Range(0, 16 * 16 * 3).Select(_ => random.NextDouble() - 0.5).ToArray();
        var weights = Enumerable.Range(0, 12 * 12 * 3).Select(_ => random.NextDouble() - 0.5).ToArray();
        var gradient = new double[raw.Length];
        DreamGaussianTexturing.Backward(view, raw, 16, weights, gradient);

        double Loss(double[] texture)
        {
            var image = DreamGaussianTexturing.Render(view, texture, 16, 1.0);
            return image.Select((v, i) => v * weights[i]).Sum();
        }

        int checkedTexels = 0;
        for (int i = 0; i < raw.Length; i++)
        {
            if (gradient[i] == 0.0 && i % 37 != 0) continue;
            var plus = (double[])raw.Clone();
            var minus = (double[])raw.Clone();
            plus[i] += 1e-6;
            minus[i] -= 1e-6;
            double numeric = (Loss(plus) - Loss(minus)) / 2e-6;
            Assert.True(Math.Abs(numeric - gradient[i]) <= 1e-7 + (1e-5 * Math.Abs(numeric)),
                $"texel value {i}: analytic {gradient[i]:G10} vs numeric {numeric:G10}");
            checkedTexels++;
        }

        Assert.True(checkedTexels > 10);
    }

    /// <summary>A <see cref="Random"/> whose NextDouble is fixed, to pin main2.py's ssaa draw.</summary>
    private sealed class FixedRandom : Random
    {
        private readonly double _value;

        public FixedRandom(double value) => _value = value;

        public override double NextDouble() => _value;
    }

    [Fact]
    public void Supersampling_DrawsAndSizesAsMain2()
    {
        // ssaa = min(2, max(0.125, 2 * random())).
        Assert.Equal(0.125, DreamGaussianTexturing.DrawSupersampling(new FixedRandom(0.01)), 12);
        Assert.Equal(0.6, DreamGaussianTexturing.DrawSupersampling(new FixedRandom(0.3)), 12);
        Assert.Equal(1.98, DreamGaussianTexturing.DrawSupersampling(new FixedRandom(0.99)), 12);
        // make_divisible(size * ssaa, 8) = ceil(size * ssaa / 8) * 8; ssaa = 1 keeps the size.
        Assert.Equal(312, DreamGaussianTexturing.SupersampledSize(512, 0.6));
        Assert.Equal(64, DreamGaussianTexturing.SupersampledSize(512, 0.125));
        Assert.Equal(1024, DreamGaussianTexturing.SupersampledSize(512, 2.0));
        Assert.Equal(136, DreamGaussianTexturing.SupersampledSize(100, 1.3));
        Assert.Equal(100, DreamGaussianTexturing.SupersampledSize(100, 1.0));
    }

    [Fact]
    public void ScaleImage_MinifiesWithoutAndMagnifiesWithAlignedCorners()
    {
        // Minification 8 -> 4, align_corners=False: output 0 reads source 0.5 (half of 0 and 1), output 3 reads 6.5.
        var down = DreamGaussianTexturing.ScaleImageTaps(8, 8, 4, 4);
        var ramp = Enumerable.Range(0, 64).Select(i => (double)(i % 8)).ToArray();
        var small = down.Apply(ramp, 1);
        Assert.Equal(0.5, small[0], 12);
        Assert.Equal(6.5, small[3], 12);
        // Magnification 4 -> 8, align_corners=True: output i reads source 3 i / 7.
        var up = DreamGaussianTexturing.ScaleImageTaps(4, 4, 8, 8);
        var coarse = Enumerable.Range(0, 16).Select(i => (double)(i % 4)).ToArray();
        var large = up.Apply(coarse, 1);
        for (int x = 0; x < 8; x++)
            Assert.Equal(3.0 * x / 7.0, large[x], 12);
        // The transpose is the adjoint: <A x, y> = <x, A^T y>.
        var random = RandomHelper.CreateSeededRandom(5);
        var y = Enumerable.Range(0, 16).Select(_ => random.NextDouble()).ToArray();
        double left = small.Select((v, i) => v * y[i]).Sum();
        var adjoint = down.Transpose(y, 1);
        double right = ramp.Select((v, i) => v * adjoint[i]).Sum();
        Assert.Equal(left, right, 9);
    }

    [Theory]
    [InlineData(1.5)]
    [InlineData(0.5)]
    public void SupersampledTextureBackward_MatchesFiniteDifferences(double ssaa)
    {
        var (vertices, faces) = Cube();
        var geometry = DreamGaussianTexturing.Prepare(vertices, faces, new UvAtlasOptions { TextureResolution = 16, PaddingTexels = 1 });
        var view = DreamGaussianTexturing.SampleSupersampledView(
            geometry, (w, h) => GaussianSplatCamera.FromOrbit(-20.0, 30.0, 2.0, w, h, 49.1), 12, 12, ssaa);
        Assert.Equal(DreamGaussianTexturing.SupersampledSize(12, ssaa), view.View.Width);
        Assert.NotNull(view.Taps);
        Assert.All(view.Alpha, a => Assert.InRange(a, 0.0, 1.0));
        var random = RandomHelper.CreateSeededRandom(9);
        var raw = Enumerable.Range(0, 16 * 16 * 3).Select(_ => random.NextDouble() - 0.5).ToArray();
        var weights = Enumerable.Range(0, 12 * 12 * 3).Select(_ => random.NextDouble() - 0.5).ToArray();
        var gradient = new double[raw.Length];
        DreamGaussianTexturing.Backward(view, raw, 16, weights, gradient);

        double Loss(double[] texture)
            => DreamGaussianTexturing.Render(view, texture, 16, 1.0).Select((v, i) => v * weights[i]).Sum();

        int checkedTexels = 0;
        for (int i = 0; i < raw.Length; i++)
        {
            if (gradient[i] == 0.0 && i % 37 != 0) continue;
            var plus = (double[])raw.Clone();
            var minus = (double[])raw.Clone();
            plus[i] += 1e-6;
            minus[i] -= 1e-6;
            double numeric = (Loss(plus) - Loss(minus)) / 2e-6;
            Assert.True(Math.Abs(numeric - gradient[i]) <= 1e-7 + (1e-5 * Math.Abs(numeric)),
                $"texel value {i}: analytic {gradient[i]:G10} vs numeric {numeric:G10}");
            checkedTexels++;
        }

        Assert.True(checkedTexels > 10);
    }

    [Fact]
    public void Mipmaps_AreBoxFilteredAndTheLevelIsTheFootprintsMajorAxis()
    {
        // 4 x 4 -> 2 x 2 -> 1 x 1, channel 0 = texel index.
        var texture = new double[48];
        for (int t = 0; t < 16; t++) texture[3 * t] = t;
        var (levels, sizes) = DreamGaussianTexturing.BuildMipmaps(texture, 4);
        Assert.Equal(new[] { 4, 2, 1 }, sizes);
        Assert.Equal((0 + 1 + 4 + 5) / 4.0, levels[1][0], 12);
        Assert.Equal((10 + 11 + 14 + 15) / 4.0, levels[1][3 * 3], 12);
        Assert.Equal(7.5, levels[2][0], 12);

        // Footprint (du/dx, du/dy, dv/dx, dv/dy) in uv per pixel on a 64-texel texture.
        double Level(double dudx, double dudy, double dvdx, double dvdy)
            => DreamGaussianTexturing.MipLevel(new[] { dudx / 64, dudy / 64, dvdx / 64, dvdy / 64 }, 0, 64, 6);
        Assert.Equal(2.0, Level(4, 0, 0, 4), 12);
        // Anisotropic: the major axis (4 texels) decides, not the minor one.
        Assert.Equal(2.0, Level(4, 0, 0, 1), 12);
        // Rotated: x-axis (2, 2), y-axis (-1, 1): A = 8, B = 2, C = 0, major^2 = 8, level 1.5.
        Assert.Equal(1.5, Level(2, -1, 2, 1), 12);
        // Magnification clamps to level 0 and a footprint wider than the texture to the last level.
        Assert.Equal(0.0, Level(0.25, 0, 0, 0.25), 12);
        Assert.Equal(6.0, Level(1000, 0, 0, 1000), 12);
    }

    [Fact]
    public void Render_MinifiedCheckerboard_IsFilteredByTheMipmap()
    {
        // A unit quad facing the camera covers ~9 pixels per side: ~7 texels of a 64-texel texture per pixel.
        double[] vertices = { -0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.5, 0.5, 0.0, -0.5, 0.5, 0.0 };
        int[] faces = { 0, 1, 2, 0, 2, 3 };
        var geometry = DreamGaussianTexturing.Prepare(vertices, faces, new UvAtlasOptions { TextureResolution = 64, PaddingTexels = 1 });
        var view = DreamGaussianTexturing.SampleView(geometry, GaussianSplatCamera.FromOrbit(0.0, 0.0, 2.0, 16, 16, 49.1));
        const int size = 64;
        var raw = new double[3 * size * size];
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                for (int c = 0; c < 3; c++)
                    raw[(3 * ((y * size) + x)) + c] = ((x + y) % 2 == 0) ? 3.0 : -3.0;
            }
        }

        var image = DreamGaussianTexturing.Render(view, raw, size, 1.0);
        int covered = 0;
        bool levelZeroAliases = false;
        var sample = new double[3];
        for (int p = 0; p < view.Covered.Length; p++)
        {
            if (!view.Covered[p]) continue;
            covered++;
            Assert.InRange(DreamGaussianTexturing.MipLevel(view.UvDerivatives, p, size, 6), 1.0, 6.0);
            // Every level from 1 up averages the checkerboard to 0, so the render is sigmoid(0) everywhere.
            Assert.Equal(0.5, image[3 * p], 12);
            DreamGaussianTexturing.SampleBilinear(raw, size, view.Uv[2 * p], view.Uv[(2 * p) + 1], sample);
            levelZeroAliases |= Math.Abs(sample[0]) > 0.5;
        }

        Assert.True(covered > 20);
        Assert.True(levelZeroAliases, "Plain bilinear sampling should alias on this checkerboard.");
    }

    /// <summary>A one- or two-triangle mesh with hand-placed screen positions over a 4 x 1 image.</summary>
    private static (UvTexturedMesh Mesh, MeshRasterization Raster) SyntheticRaster(
        double edgeX, double[]? neighbour, int[] pixelFaces, double[] pixelDepths)
    {
        var screen = new List<double[]> { new[] { -10.0, -10.0 }, new[] { edgeX, -10.0 }, new[] { edgeX, 10.0 } };
        var faceList = new List<int> { 0, 1, 2 };
        if (neighbour is not null)
        {
            screen.Add(neighbour);
            faceList.AddRange(new[] { 1, 3, 2 });
        }

        int vertexCount = screen.Count;
        var atlas = new UvAtlas(new double[3 * vertexCount], faceList.ToArray(), new double[2 * vertexCount],
            Enumerable.Range(0, vertexCount).ToArray(), new int[faceList.Count / 3], 1);
        var mesh = new UvTexturedMesh(atlas, new double[3 * vertexCount]);
        var raster = new MeshRasterization(4, 1)
        {
            ScreenX = screen.Select(v => v[0]).ToArray(),
            ScreenY = screen.Select(v => v[1]).ToArray(),
            VertexDepth = Enumerable.Repeat(1.0, vertexCount).ToArray(),
        };
        for (int p = 0; p < 4; p++)
        {
            raster.Face[p] = pixelFaces[p];
            raster.Depth[p] = pixelDepths[p];
        }

        return (mesh, raster);
    }

    [Fact]
    public void Antialias_BlendsAcrossSilhouettesByTheCrossingDistance()
    {
        // The vertical boundary edge x = 1.8 crosses between pixel 1 (inside) and pixel 2 at t = 0.8 from pixel 1:
        // pixel 2 takes 0.3 of pixel 1's colour.
        var (mesh, raster) = SyntheticRaster(1.8, null, new[] { 0, 0, -1, -1 }, new[] { 1.0, 1.0, 0.0, 0.0 });
        var blends = DreamGaussianTexturing.FindSilhouetteBlends(mesh, raster);
        Assert.Equal(new[] { 2 }, blends.Destination);
        Assert.Equal(new[] { 1 }, blends.Source);
        Assert.Equal(0.3, blends.Weight[0], 12);

        // At x = 1.3 the crossing is in pixel 1's half (t = 0.3): pixel 1 takes 0.2 of pixel 2's colour.
        (mesh, raster) = SyntheticRaster(1.3, null, new[] { 0, 0, -1, -1 }, new[] { 1.0, 1.0, 0.0, 0.0 });
        blends = DreamGaussianTexturing.FindSilhouetteBlends(mesh, raster);
        Assert.Equal(new[] { 1 }, blends.Destination);
        Assert.Equal(new[] { 2 }, blends.Source);
        Assert.Equal(0.2, blends.Weight[0], 12);

        // Rendered: pixel 1 = a + 0.2 (b - a), with a its own texel and b the uncovered pixel's lookup at uv (0, 0),
        // as main2.py antialiases before compositing over the background.
        double[] raw = { 1.0, 1.0, 1.0, -2.0, -2.0, -2.0, 0.5, 0.5, 0.5, 2.5, 2.5, 2.5 };
        double[] uv = { 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0 };
        var view = new MeshTextureView(4, 1, new[] { true, true, false, false }, uv, new double[4], new double[16], blends);
        var image = DreamGaussianTexturing.Render(view, raw, 2, 1.0);
        double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        double a = Sigmoid(1.0), b = Sigmoid((1.0 - 2.0 + 0.5 + 2.5) / 4.0);
        Assert.Equal(a, image[0], 12);
        Assert.Equal(a + (0.2 * (b - a)), image[3], 12);
        Assert.Equal(1.0, image[6], 12);
    }

    [Fact]
    public void Antialias_SkipsInteriorEdgesAndBlendsFolds()
    {
        // Two triangles sharing the edge x = 1.8 with the same winding: an interior edge, no blend.
        var (mesh, raster) = SyntheticRaster(1.8, new[] { 10.0, 0.0 }, new[] { 0, 0, 1, 1 }, new[] { 1.0, 1.0, 2.0, 2.0 });
        Assert.Empty(DreamGaussianTexturing.FindSilhouetteBlends(mesh, raster).Weight);

        // The neighbour folded back (opposite winding on screen): the shared edge is a silhouette of the nearer face.
        (mesh, raster) = SyntheticRaster(1.8, new[] { -5.0, 0.0 }, new[] { 0, 0, 1, 1 }, new[] { 1.0, 1.0, 2.0, 2.0 });
        var blends = DreamGaussianTexturing.FindSilhouetteBlends(mesh, raster);
        Assert.Equal(new[] { 2 }, blends.Destination);
        Assert.Equal(0.3, blends.Weight[0], 12);
    }

    [Fact]
    public void Inpaint_CopiesTheNearestBorderTexel()
    {
        const int size = 12;
        var albedo = new double[3 * size * size];
        var mask = new bool[size * size];
        for (int y = 3; y < 9; y++)
        {
            for (int x = 3; x < 9; x++)
            {
                int t = (y * size) + x;
                mask[t] = true;
                albedo[3 * t] = x;
                albedo[(3 * t) + 1] = y;
            }
        }

        var output = DreamGaussianTexturing.Inpaint(albedo, mask, size, dilation: 2, erosion: 1);
        // Two steps left of (3, 5) copies (3, 5); a texel three steps away stays empty.
        int left = (5 * size) + 1;
        Assert.Equal(3.0, output[3 * left]);
        Assert.Equal(5.0, output[(3 * left) + 1]);
        int far = (5 * size) + 0;
        Assert.Equal(0.0, output[3 * far]);
        // A diagonal texel within two 4-connected steps copies the nearest corner.
        int diagonal = (2 * size) + 2;
        Assert.Equal(3.0, output[3 * diagonal]);
        Assert.Equal(3.0, output[(3 * diagonal) + 1]);
        // Baked texels are untouched.
        int inside = (5 * size) + 5;
        Assert.Equal(5.0, output[3 * inside]);
    }

    [Fact]
    public void MipmapGridPut_SplatsBilinearlyAtFullResolution()
    {
        // One sample a quarter of the way between texels (1, 1) and (1, 2) of a 4 x 4 grid.
        double row = (1.0 / 3.0 * 2.0) - 1.0;
        double column = (1.25 / 3.0 * 2.0) - 1.0;
        var (result, count) = DreamGaussianTexturing.MipmapLinearGridPut(4, 4, new[] { row, column }, new[] { 1.0, 2.0, 3.0 }, 2);
        Assert.Equal(0.75, count[(1 * 4) + 1], 12);
        Assert.Equal(0.25, count[(1 * 4) + 2], 12);
        Assert.Equal(0.75 * 3.0, result[(3 * ((1 * 4) + 1)) + 2], 12);
    }

    // ---- End to end ------------------------------------------------------------------------------------------------

    /// <summary>A deterministic stand-in prior: pulls renders towards mid-grey and "refines" to the inverted render.</summary>
    private sealed class StubGuidance : IScoreDistillationGuidance<double>
    {
        public StubGuidance(int resolution) => Resolution = resolution;

        public int TrainTimesteps => 1000;
        public int MinStep => 20;
        public int MaxStep => 980;
        public int Resolution { get; }

        public Tensor<double> ComputeImageGradient(Tensor<double> images, IReadOnlyList<GuidanceViewOffset> views, int timestep, Random random)
        {
            Assert.Equal(images.Shape[0], views.Count);
            var values = images.ToArray();
            for (int i = 0; i < values.Length; i++)
                values[i] = 0.001 * (values[i] - 0.5) * timestep / 1000.0;
            return new Tensor<double>(images.Shape.ToArray(), new Vector<double>(values));
        }

        public Tensor<double> Refine(Tensor<double> images, IReadOnlyList<GuidanceViewOffset> views, double strength, int steps, Random random)
        {
            Assert.InRange(strength, 0.8, 0.95 + 1e-12);
            var resized = AiDotNetEngine.Current.Interpolate(images, new[] { Resolution, Resolution }, InterpolateMode.Bilinear, false);
            var values = resized.ToArray();
            for (int i = 0; i < values.Length; i++)
                values[i] = 1.0 - values[i];
            return new Tensor<double>(resized.Shape.ToArray(), new Vector<double>(values));
        }
    }

    private static DreamGaussianOptions TinySettings() => new DreamGaussianOptions
    {
        InitialGaussianCount = 300,
        InitialOpacity = 0.5,
        Iterations = 6,
        ReferenceSize = 16,
        RenderResolutions = new[] { 16 },
        RenderResolutionThresholds = new double[0],
        ImageGuidanceResolution = 16,
        ImageDensifyStartIteration = 2,
        ImageDensificationInterval = 3,
        OpacityResetInterval = 5,
        MeshExtraction = new GaussianMeshExtractionOptions { Resolution = 32, BlockCount = 4, DensityThreshold = 0.25 },
        TextureSize = 32,
        BakeRenderResolution = 16,
        BakeMinResolution = 8,
        InpaintDilation = 4,
        RefineIterations = 2,
        RefineRenderResolution = 16,
        RefineDenoisingSteps = 4,
    };

    private static (double[] Image, double[] Mask) DiskReference(int size, double red, double green, double blue, double radius)
    {
        var image = new double[3 * size * size];
        var mask = new double[size * size];
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                double u = ((x + 0.5) / size) - 0.5, v = ((y + 0.5) / size) - 0.5;
                int p = (y * size) + x;
                bool inside = (u * u) + (v * v) < radius * radius;
                mask[p] = inside ? 1.0 : 0.0;
                image[3 * p] = inside ? red : 1.0;
                image[(3 * p) + 1] = inside ? green : 1.0;
                image[(3 * p) + 2] = inside ? blue : 1.0;
            }
        }

        return (image, mask);
    }

    private static Tensor<double> RgbaInput(int size, double red, double green, double blue, double radius)
    {
        var (image, mask) = DiskReference(size, red, green, blue, radius);
        int pixels = size * size;
        var data = new double[4 * pixels];
        for (int p = 0; p < pixels; p++)
        {
            for (int c = 0; c < 3; c++) data[(c * pixels) + p] = image[(3 * p) + c];
            data[(3 * pixels) + p] = mask[p];
        }

        return new Tensor<double>(new[] { 1, 4, size, size }, new Vector<double>(data));
    }

    private static DreamGaussianModel<double> TinyModel()
    {
        var model = new DreamGaussianModel<double>(seed: 21) { Settings = TinySettings() };
        model.ImageGuidanceFactory = (reference, random) =>
        {
            Assert.Equal(new[] { 1, 3, 16, 16 }, reference.Shape.ToArray());
            return new StubGuidance(16);
        };
        return model;
    }

    [Fact]
    public void ImageTo3D_TinyConfig_IsDeterministicAndProducesATexturedMesh()
    {
        var input = RgbaInput(16, 0.9, 0.2, 0.1, 0.3);
        var first = TinyModel().RunImageTo3D(input, seed: 3);
        var second = TinyModel().RunImageTo3D(input, seed: 3);

        Assert.True(first.Mesh.Faces.GetLength(0) > 0, "The tiny run extracted no surface.");
        Assert.Equal(first.Mesh.Vertices.ToArray(), second.Mesh.Vertices.ToArray());
        Assert.Equal(first.PointCloud.ToArray(), second.PointCloud.ToArray());
        Assert.Equal(6, first.ReferenceLossHistory.Count);

        var texture = first.TexturedMesh?.TextureImage;
        var uvs = first.TexturedMesh?.UVCoordinates;
        var secondTexture = second.TexturedMesh?.TextureImage;
        if (first.TexturedMesh is null || texture is null || uvs is null || secondTexture is null)
        {
            Assert.True(false, "The texture stage produced no textured mesh.");
            return;
        }

        Assert.Equal(new[] { 32, 32, 3 }, texture.Shape.ToArray());
        Assert.Equal(texture.ToArray(), secondTexture.ToArray());
        Assert.Equal(first.TexturedMesh.Vertices.Shape[0], uvs.Shape[0]);
        Assert.All(texture.ToArray(), t => Assert.InRange(t, 0.0, 1.0));
        Assert.All(uvs.ToArray(), t => Assert.InRange(t, 0.0, 1.0));

        var points = first.PointCloud;
        Assert.Equal(3, points.Rank);
        Assert.Equal(6, points.Shape[2]);
        Assert.Equal(first.Gaussians.GenerativeCount, points.Shape[1]);
    }

    [Fact]
    public void ImageTo3D_DifferentInputs_GiveDifferentMeshes()
    {
        var red = TinyModel().RunImageTo3D(RgbaInput(16, 0.9, 0.2, 0.1, 0.3), seed: 3);
        var blue = TinyModel().RunImageTo3D(RgbaInput(16, 0.1, 0.2, 0.9, 0.45), seed: 3);
        Assert.NotEqual(red.Mesh.Vertices.ToArray(), blue.Mesh.Vertices.ToArray());
        Assert.NotEqual(red.PointCloud.ToArray(), blue.PointCloud.ToArray());
    }

    /// <summary>Returns a fixed matte and counts its calls.</summary>
    private sealed class RecordingSegmenter : IForegroundSegmenter<double>
    {
        public int Calls { get; private set; }

        public Tensor<double> Segment(Tensor<double> image)
        {
            Calls++;
            Assert.Equal(new[] { 1, 3, 4, 4 }, image.Shape.ToArray());
            return Matte(0.25);
        }
    }

    private static Tensor<double> Matte(double value)
        => new Tensor<double>(new[] { 4, 4 }, new Vector<double>(Enumerable.Repeat(value, 16).ToArray()));

    private static Tensor<double> Constant(int channels, double colour, double alpha)
    {
        var data = new double[channels * 16];
        for (int i = 0; i < 48; i++) data[i] = colour;
        for (int i = 48; i < data.Length; i++) data[i] = alpha;
        return new Tensor<double>(new[] { 1, channels, 4, 4 }, new Vector<double>(data));
    }

    [Fact]
    public void PrepareReference_TakesTheCallerMask_ThenTheAlpha_ThenTheSegmenter()
    {
        var settings = new DreamGaussianOptions { ReferenceSize = 4 };
        var segmenter = new RecordingSegmenter();

        // RGBA: the alpha is used and the segmenter is never run.
        var (rgbaImage, rgbaMask) = DreamGaussianModel<double>.PrepareReference(Constant(4, 0.2, 0.5), null, settings, () => segmenter);
        Assert.All(rgbaMask, a => Assert.Equal(0.5, a, 12));
        Assert.All(rgbaImage, v => Assert.Equal((0.2 * 0.5) + 0.5, v, 12));
        Assert.Equal(0, segmenter.Calls);

        // A caller mask wins over the alpha.
        var (_, callerMask) = DreamGaussianModel<double>.PrepareReference(Constant(4, 0.2, 0.5), Matte(0.75), settings, () => segmenter);
        Assert.All(callerMask, a => Assert.Equal(0.75, a, 12));
        Assert.Equal(0, segmenter.Calls);

        // RGB with a caller mask: still no segmentation.
        var (_, rgbCallerMask) = DreamGaussianModel<double>.PrepareReference(Constant(3, 0.2, 0.0), Matte(1.0), settings, () => segmenter);
        Assert.All(rgbCallerMask, a => Assert.Equal(1.0, a, 12));
        Assert.Equal(0, segmenter.Calls);

        // RGB alone: the segmenter's matte is the mask and the colour is composited onto white with it.
        var (rgbImage, segmentedMask) = DreamGaussianModel<double>.PrepareReference(Constant(3, 0.2, 0.0), null, settings, () => segmenter);
        Assert.Equal(1, segmenter.Calls);
        Assert.All(segmentedMask, a => Assert.Equal(0.25, a, 12));
        Assert.All(rgbImage, v => Assert.Equal((0.2 * 0.25) + 0.75, v, 12));
    }

    [Fact]
    public void ImageTo3D_RgbInput_RunsTheConfiguredSegmenter()
    {
        var model = TinyModel();
        var segmenter = new CountingSegmenter();
        model.ForegroundSegmenter = segmenter;
        var rgba = RgbaInput(16, 0.9, 0.2, 0.1, 0.3).ToArray();
        var rgb = new Tensor<double>(new[] { 1, 3, 16, 16 }, new Vector<double>(rgba.Take(3 * 256).ToArray()));
        model.RunImageTo3D(rgb, seed: 3);
        Assert.Equal(1, segmenter.Calls);
        model.RunImageTo3D(RgbaInput(16, 0.9, 0.2, 0.1, 0.3), seed: 3);
        Assert.Equal(1, segmenter.Calls);
    }

    private sealed class CountingSegmenter : IForegroundSegmenter<double>
    {
        public int Calls { get; private set; }

        public Tensor<double> Segment(Tensor<double> image)
        {
            Calls++;
            var (_, mask) = DiskReference(image.Shape[2], 0.0, 0.0, 0.0, 0.3);
            return new Tensor<double>(new[] { image.Shape[2], image.Shape[3] }, new Vector<double>(mask));
        }
    }

    [Fact]
    public void GeneratePointCloud_TextPathUsesTheTextGuidance()
    {
        var model = new DreamGaussianModel<double>(seed: 5) { Settings = TinySettings() };
        model.Settings.TextGuidanceResolution = 16;
        string? seenPrompt = null;
        model.TextGuidanceFactory = (prompt, negative) =>
        {
            seenPrompt = prompt;
            return new StubGuidance(16);
        };

        var cloud = model.GeneratePointCloud("a red apple", numPoints: 120, numInferenceSteps: 4, seed: 1);
        Assert.Equal("a red apple", seenPrompt);
        Assert.Equal(3, cloud.Rank);
        Assert.Equal(1, cloud.Shape[0]);
        Assert.Equal(6, cloud.Shape[2]);
        var values = cloud.ToArray();
        for (int i = 0; i < cloud.Shape[1]; i++)
        {
            for (int c = 3; c < 6; c++)
                Assert.InRange(values[(6 * i) + c], 0.0, 1.0);
        }
    }
}
