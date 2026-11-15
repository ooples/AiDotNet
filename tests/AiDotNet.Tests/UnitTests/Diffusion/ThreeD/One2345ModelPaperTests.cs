using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Interfaces;
using AiDotNet.NeuralRadianceFields.NeuS;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.ThreeD;

/// <summary>
/// One-2-3-45 (Liu et al. 2023, arXiv 2306.16928) against the official release (One-2-3-45/One-2-3-45): the Zero123
/// camera layout of <c>predict_stage1_gradio</c> / <c>zero123_infer</c> and the absolute poses of <c>get_poses</c>,
/// <c>calc_pose</c>, the elevation estimator's triangulation and hypothesis search (<c>elev_est_api.py</c>), the
/// scene normalization, and the full pipeline on a tiny configuration (an 8³ SparseNeuS volume) with stubbed
/// Zero123 views: it runs end to end, is deterministic under a seed, and depends on the input.
/// </summary>
public class One2345ModelPaperTests
{
    private const int ImageSize = 16;

    // ------------------------------------------------------------------ camera layout

    [Fact]
    public void Stage1DeltasMatchPredictStage1Gradio()
    {
        var o = new One2345Options();

        // delta_x_1_8 = [0]*4 + [30]*4 + [-30]*4; delta_y_1_8 = [0+90*(i%4) if i < 4 else 30+90*(i%4) ...]
        var below = One2345CameraLayout.Stage1(o, 60.0);
        double[] polarBelow = { 0, 0, 0, 0, 30, 30, 30, 30 };
        double[] azimuth = { 0, 90, 180, 270, 30, 120, 210, 300 };
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(polarBelow[i], below[i].PolarDelta);
            Assert.Equal(azimuth[i], below[i].AzimuthDelta);
        }

        // Above 75° polar, run.py switches to adjust_set 8..11: delta_x = -30.
        var above = One2345CameraLayout.Stage1(o, 80.0);
        for (int i = 4; i < 8; i++)
        {
            Assert.Equal(-30.0, above[i].PolarDelta);
            Assert.Equal(azimuth[i], above[i].AzimuthDelta);
        }

        Assert.Equal(30.0, One2345CameraLayout.Stage1(o, 75.0)[4].PolarDelta); // "if polar_angle <= 75"
    }

    [Fact]
    public void Stage2DeltasMatchZero123Infer()
    {
        var o = new One2345Options();
        var parent = new One2345ViewPose(0, 0, 70, 120);
        var nearby = One2345CameraLayout.Stage2(o, parent);

        // delta_x_2 = [-10, 10, 0, 0]; delta_y_2 = [0, 0, -10, 10]
        double[] dx = { -10, 10, 0, 0 }, dy = { 0, 0, -10, 10 };
        for (int j = 0; j < 4; j++)
        {
            Assert.Equal(dx[j], nearby[j].PolarDelta);
            Assert.Equal(dy[j], nearby[j].AzimuthDelta);
            Assert.Equal(70 + dx[j], nearby[j].Polar);
            Assert.Equal(120 + dy[j], nearby[j].Azimuth);
        }
    }

    [Theory]
    [InlineData(60.0)]
    [InlineData(100.0)]
    public void AbsolutePosesMatchGetPoses(double mid)
    {
        // utils.get_poses: elevations [mid]*4 + [low]*4 + [mid-10, mid+10, mid, mid]*4 + [low-10, low+10, low, low]*4,
        // azimuths overlook (30 + 90x) + eyelevel (60 + 90x) + each view's azimuth + [0, 0, -10, 10]; low = mid ± 30.
        var o = new One2345Options();
        double low = mid <= 75 ? mid + 30 : mid - 30;
        var polar = new List<double>();
        polar.AddRange(Enumerable.Repeat(mid, 4));
        polar.AddRange(Enumerable.Repeat(low, 4));
        for (int i = 0; i < 4; i++) polar.AddRange(new[] { mid - 10, mid + 10, mid, mid });
        for (int i = 0; i < 4; i++) polar.AddRange(new[] { low - 10, low + 10, low, low });
        var overlook = Enumerable.Range(0, 4).Select(x => 30.0 + x * 90).ToArray();
        var eyelevel = Enumerable.Range(0, 4).Select(x => 60.0 + x * 90).ToArray();
        var azimuth = new List<double>();
        azimuth.AddRange(overlook);
        azimuth.AddRange(eyelevel);
        foreach (double t in overlook) azimuth.AddRange(new[] { t, t, t - 10, t + 10 });
        foreach (double t in eyelevel) azimuth.AddRange(new[] { t, t, t - 10, t + 10 });

        var stage1 = One2345CameraLayout.Stage1(o, mid);
        var ours = stage1.ToList();
        foreach (var parent in stage1) ours.AddRange(One2345CameraLayout.Stage2(o, parent));

        Assert.Equal(40, ours.Count);
        for (int k = 0; k < 40; k++)
        {
            Assert.Equal(polar[k], ours[k].Polar, 12);
            Assert.Equal(azimuth[k], ours[k].Azimuth, 12);
        }

        Assert.Equal(32, ours.Count - 8); // 8 first-stage views × 4 neighbours: the 32 source views
    }

    [Fact]
    public void CalcPoseCenterAndOpenCvRotation()
    {
        // calc_pose: centers = (r sinθ sinφ, −r cosθ sinφ, r cosφ); at φ = 90°, θ = 30°, r = 1.2: (0.6, −1.0392, 0).
        var c = One2345CameraLayout.Center(90, 30, 1.2);
        Assert.Equal(0.6, c[0], 12);
        Assert.Equal(-1.2 * Math.Sqrt(3) / 2, c[1], 12);
        Assert.Equal(0.0, c[2], 12);

        var top = One2345CameraLayout.Center(60, 0, 1.2);
        Assert.Equal(0.0, top[0], 12);
        Assert.Equal(-1.2 * Math.Sin(Math.PI / 3), top[1], 12);
        Assert.Equal(0.6, top[2], 12);

        // OpenCV camera: looks at the origin (third column = −center/|center|), +y image-down is world −z here.
        var o = new One2345Options();
        var camera = One2345CameraLayout.Camera(o, new One2345ViewPose(0, 0, 60, 0), 256, 256);
        Assert.Equal(280.0, camera.FocalX, 12); // focal = 560 / 2 at 256 px
        Assert.Equal(128.0, camera.PrincipalX, 12); // w / 2
        var (u, v, depth) = camera.Project(0, 0, 0);
        Assert.Equal(128.0, u, 9);
        Assert.Equal(128.0, v, 9);
        Assert.Equal(1.2, depth, 9);
        var (_, vAbove, _) = camera.Project(0, 0, 0.1);
        Assert.True(vAbove < 128.0, "A point above the origin must appear higher in the image (smaller v).");
    }

    [Fact]
    public void NormalizationMapsCamerasBackToTheirWorldPoses()
    {
        var o = new One2345Options();
        var stage1 = One2345CameraLayout.Stage1(o, 70);
        var reference = One2345CameraLayout.Camera(o, stage1[0], 256, 256);
        var cameras = new List<NeuSCamera> { reference };
        foreach (var parent in stage1)
            foreach (var pose in One2345CameraLayout.Stage2(o, parent))
                cameras.Add(One2345CameraLayout.Camera(o, pose, 256, 256));

        var frame = One2345CameraLayout.Normalize(o, reference, cameras, 256, 256);
        Assert.Equal(33, frame.Cameras.Count);
        for (int i = 0; i < cameras.Count; i++)
        {
            var back = frame.ToWorld(frame.Cameras[i].Center);
            for (int k = 0; k < 3; k++) Assert.Equal(cameras[i].Center[k], back[k], 9);
        }

        for (int k = 0; k < 3; k++) Assert.Equal(frame.Cameras[0].Center[k], frame.QueryCenter[k], 12);
        // The reference camera is the identity rotation in its own frame.
        for (int i = 0; i < 9; i++) Assert.Equal(i % 4 == 0 ? 1.0 : 0.0, frame.Cameras[0].Rotation[i], 9);

        // cal_scale_mat: radius = half the largest bounding-box side × 1.1; every frustum corner lies within it.
        var corners = new List<double[]>();
        foreach (var camera in frame.Cameras)
            foreach (double d in new[] { o.Near, o.Far })
                foreach (double px in new[] { 0.0, 256.0 })
                    foreach (double py in new[] { 0.0, 256.0 })
                    {
                        var dir = new[] { (px - 128) * d / 280 / frame.Radius, (py - 128) * d / 280 / frame.Radius, d / frame.Radius };
                        var r = camera.Rotation;
                        corners.Add(new[]
                        {
                            r[0] * dir[0] + r[1] * dir[1] + r[2] * dir[2] + camera.Center[0],
                            r[3] * dir[0] + r[4] * dir[1] + r[5] * dir[2] + camera.Center[1],
                            r[6] * dir[0] + r[7] * dir[1] + r[8] * dir[2] + camera.Center[2],
                        });
                    }

        double largest = 0;
        for (int axis = 0; axis < 3; axis++)
        {
            double lo = corners.Min(p => p[axis]), hi = corners.Max(p => p[axis]);
            Assert.Equal(0.0, (lo + hi) / 2, 9); // centered
            largest = Math.Max(largest, hi - lo);
        }

        Assert.Equal(2.0 / 1.1, largest, 9);
    }

    // ------------------------------------------------------------------ elevation estimation

    [Fact]
    public void TriangulationRecoversAKnownPoint()
    {
        var o = new One2345Options();
        var cameras = One2345ElevationEstimator.Hypothesis(o, 70, 280, 128, 128);
        var point = new[] { 0.11, -0.07, 0.2 };
        var (ua, va, _) = cameras[0].Project(point[0], point[1], point[2]);
        var (ub, vb, _) = cameras[2].Project(point[0], point[1], point[2]);
        var x = One2345ElevationEstimator.Triangulate(cameras[0], cameras[2], ua, va, ub, vb);
        for (int k = 0; k < 3; k++) Assert.Equal(point[k], x[k], 8);
    }

    [Fact]
    public void HypothesisCamerasMatchGenPoseHypothesis()
    {
        // gen_pose_hypothesis(c): elevations [c−10, c+10, c, c], azimuths [30, 30, 20, 40], radius 1.2.
        var o = new One2345Options();
        var cameras = One2345ElevationEstimator.Hypothesis(o, 80, 280, 128, 128);
        double[] polar = { 70, 90, 80, 80 }, azimuth = { 30, 30, 20, 40 };
        for (int k = 0; k < 4; k++)
        {
            var expected = One2345CameraLayout.Center(polar[k], azimuth[k], 1.2);
            for (int a = 0; a < 3; a++) Assert.Equal(expected[a], cameras[k].Center[a], 12);
        }
    }

    [Fact]
    public void RangeMatchesNumpyArange()
    {
        Assert.Equal(new[] { 30.0, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140 }, One2345ElevationEstimator.Range(30, 150, 10));
        Assert.Equal(20, One2345ElevationEstimator.Range(60, 80, 1).Length);
        Assert.Equal(60.0, One2345ElevationEstimator.Range(60, 80, 1)[0]);
        Assert.Equal(79.0, One2345ElevationEstimator.Range(60, 80, 1)[19]);
    }

    [Theory]
    [InlineData(72.0)]
    [InlineData(95.0)]
    [InlineData(48.0)]
    public void EstimatorRecoversTheTruePolarAngleFromExactMatches(double truePolar)
    {
        var o = new One2345Options();
        var matches = ExactMatches(o, truePolar, seed: 3);

        // The true hypothesis reprojects exactly; a wrong one does not.
        Assert.Equal(0.0, One2345ElevationEstimator.Error(o, Symmetric(matches), truePolar, 280, 128, 128), 6);
        Assert.True(One2345ElevationEstimator.Error(o, Symmetric(matches), truePolar + 6, 280, 128, 128) > 0.1);

        Assert.Equal(truePolar, One2345ElevationEstimator.Estimate(o, matches, 280, 128, 128));
    }

    [Fact]
    public void EstimatorFallsBackTo90WhenAPairHasNoMatch()
    {
        var o = new One2345Options();
        var matches = ExactMatches(o, 72, seed: 3);
        matches[(1, 3)] = new List<FeatureCorrespondence>();
        Assert.Equal(90.0, One2345ElevationEstimator.Estimate(o, matches, 280, 128, 128));
    }

    [Fact]
    public void ReprojectionErrorIsConfidenceWeighted()
    {
        // ba_error_general: (|e| · conf).sum() / conf.sum() per third view.
        var o = new One2345Options();
        var cameras = One2345ElevationEstimator.Hypothesis(o, 70, 280, 128, 128);
        var points = new[] { new[] { 0.0, 0.0, 0.0 }, new[] { 0.1, 0.05, -0.05 } };
        IReadOnlyList<FeatureCorrespondence> Pair(int a, int b, double[] offsetsY, double[] confidences)
        {
            var list = new List<FeatureCorrespondence>();
            for (int p = 0; p < points.Length; p++)
            {
                var (ua, va, _) = cameras[a].Project(points[p][0], points[p][1], points[p][2]);
                var (ub, vb, _) = cameras[b].Project(points[p][0], points[p][1], points[p][2]);
                list.Add(new FeatureCorrespondence(ua, va, ub, vb + offsetsY[p], confidences[p]));
            }

            return list;
        }

        var batch = new[]
        {
            Pair(0, 1, new[] { 0.0, 0.0 }, new[] { 1.0, 1.0 }),
            Pair(0, 2, new[] { 2.0, 5.0 }, new[] { 3.0, 1.0 }),
            Pair(0, 3, new[] { 0.0, 0.0 }, new[] { 1.0, 1.0 }),
        };
        double error = One2345ElevationEstimator.BundleError(o, batch, cameras);
        Assert.Equal((2.0 * 3.0 + 5.0 * 1.0) / 4.0, error, 6);
    }

    [Fact]
    public void SmallestEigenvectorOfADiagonalMatrix()
    {
        var m = new double[16];
        m[0] = 4;
        m[5] = 1;
        m[10] = 0.25;
        m[15] = 9;
        var v = One2345ElevationEstimator.SmallestEigenvector(m);
        Assert.Equal(1.0, Math.Abs(v[2]), 12);
    }

    // ------------------------------------------------------------------ matcher

    [Fact]
    public void CorrelationMatcherFindsAKnownShift()
    {
        const int size = 40;
        var random = RandomHelper.CreateSeededRandom(5);
        var texture = new double[size + 4, size + 4];
        for (int y = 0; y < size + 4; y++)
            for (int x = 0; x < size + 4; x++) texture[y, x] = random.NextDouble();
        var a = new double[size * size * 3];
        var b = new double[size * size * 3];
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
                for (int c = 0; c < 3; c++)
                {
                    a[(y * size + x) * 3 + c] = texture[y + 2, x + 2];
                    b[(y * size + x) * 3 + c] = texture[y + 1, x]; // content moves by (+2, +1)
                }

        var matcher = new NormalizedCrossCorrelationMatcher<double>(patchRadius: 3, searchRadius: 4, stride: 5);
        var matches = matcher.Match(NeuSHost<double>.FromDoubles(a, size, size, 3), NeuSHost<double>.FromDoubles(b, size, size, 3));
        var interior = matches.Where(m => m.X0 >= 6 && m.X0 < size - 6 && m.Y0 >= 6 && m.Y0 < size - 6).ToList();
        Assert.NotEmpty(interior);
        foreach (var m in interior)
        {
            Assert.Equal(m.X0 + 2, m.X1);
            Assert.Equal(m.Y0 + 1, m.Y1);
            Assert.Equal(1.0, m.Confidence, 9);
        }
    }

    // ------------------------------------------------------------------ pipeline

    private static readonly Lazy<StubZero123> SharedStub = new Lazy<StubZero123>(() => new StubZero123());

    private static One2345Options TinyOptions() => new One2345Options
    {
        VolumeResolution = 8,
        MeshResolution = 16,
        SdfHiddenDim = 32,
        QueryChunkSize = 512,
        TextToImageSize = ImageSize,
        Stage1Steps = 1,
        Stage2Steps = 1,
    };

    /// <summary>
    /// A tiny pipeline. SparseNeuS's geometric initialization zeroes the SDF MLP's latent columns, so an untrained
    /// network's surface is the init sphere whatever the views; the SDF layers get seeded noise, standing in for
    /// trained weights, so the geometry depends on the encoded views as it does with the released checkpoint.
    /// </summary>
    private static One2345Model<float> TinyModel(int seed = 7)
    {
        var model = new One2345Model<float>(seed: seed, pipelineOptions: TinyOptions(), zero123: SharedStub.Value,
            textToImage: SharedStub.Value, featureMatcher: new LoFTRMatcher<float>(LoFTRMatcherTests.TinyOptions(seed, inputSize: 64)));
        var random = RandomHelper.CreateSeededRandom(seed + 1000);
        var perturbed = new Dictionary<string, Tensor<float>>();
        foreach (var p in model.GetReconstructionParameters())
        {
            if (!p.Key.StartsWith("sdf_network_lod0.sdf_layer", StringComparison.Ordinal)) continue;
            var values = NeuSHost<float>.ToDoubles(p.Value).Select(v => v + 0.1 * Gaussian(random)).ToArray();
            perturbed[p.Key] = NeuSHost<float>.FromDoubles(values, p.Value.Shape.ToArray());
        }

        Assert.NotEmpty(perturbed);
        model.LoadReconstructionParameters(perturbed);
        return model;
    }

    private static double Gaussian(Random random)
        => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());

    [Fact]
    public void PipelineRunsEndToEndWithThePaperLayout()
    {
        var model = TinyModel();
        var stub = SharedStub.Value;
        List<(double, double)> requests;
        One2345Result<float> result;
        lock (stub)
        {
            stub.Requests.Clear();
            result = model.Reconstruct(InputImage(0.2f), 1, 1, 3.0, 16, seed: 11);
            requests = stub.Requests.ToList();
        }

        // 4 first-stage views, view 0's 4 neighbours (for the elevation), the offset ring, then 7 × 4 neighbours.
        Assert.Equal(8 + 32, requests.Count);
        Assert.Equal(new[] { (0.0, 0.0), (0.0, 90.0), (0.0, 180.0), (0.0, 270.0) }, requests.Take(4));
        Assert.Equal(new[] { (-10.0, 0.0), (10.0, 0.0), (0.0, -10.0), (0.0, 10.0) }, requests.Skip(4).Take(4));
        double ring = result.Polar <= 75 ? 30 : -30;
        Assert.Equal(new[] { (ring, 30.0), (ring, 120.0), (ring, 210.0), (ring, 300.0) }, requests.Skip(8).Take(4));
        Assert.InRange(result.Polar, 20, 149);

        Assert.Equal(32, result.SourceViews.Count);
        foreach (var view in result.SourceViews)
        {
            Assert.Equal(ImageSize, view.Height);
            Assert.Equal(ImageSize, view.Width);
        }

        Assert.Equal(new[] { 16, 16, 16 }, result.SdfGrid.Shape.ToArray());
        var mesh = result.Mesh;
        int vertices = mesh.Vertices.Shape[0];
        Assert.True(vertices > 0, "The SDF must have a zero crossing.");
        Assert.True(mesh.Faces.GetLength(0) > 0);
        Assert.NotNull(mesh.Colors);
        var colors = NeuSHost<float>.ToDoubles(mesh.Colors ?? new Tensor<float>(new[] { 0, 3 }));
        Assert.Equal(vertices * 3, colors.Length);
        Assert.All(colors, c => Assert.InRange(c, -1e-5, 1 + 1e-5));
        Assert.All(NeuSHost<float>.ToDoubles(mesh.Vertices), x => Assert.True(!double.IsNaN(x) && !double.IsInfinity(x)));
        for (int f = 0; f < mesh.Faces.GetLength(0); f++)
            for (int k = 0; k < 3; k++) Assert.InRange(mesh.Faces[f, k], 0, vertices - 1);
    }

    [Fact]
    public void ImageTo3DRejectsViewCountsOtherThanTheFixedLayout()
    {
        var model = TinyModel(seed: 3);
        Assert.Throws<ArgumentException>(() => model.ImageTo3D(InputImage(0.3f), numViews: 4, seed: 21));
        Assert.Throws<ArgumentException>(() => model.ImageTo3D(InputImage(0.3f), numViews: 40, seed: 21));
        // Through the interface the default is 4, which is not the 8 × 4 layout either.
        I3DDiffusionModel<float> general = model;
        Assert.Throws<ArgumentException>(() => general.ImageTo3D(InputImage(0.3f), seed: 21));
    }

    [Fact]
    public void LoFTRIsTheDefaultElevationMatcher()
    {
        var model = new One2345Model<float>(seed: 3, pipelineOptions: TinyOptions(), zero123: SharedStub.Value);
        var loftr = Assert.IsType<LoFTRMatcher<float>>(model.FeatureMatcher);
        Assert.Equal(480, loftr.Options.InputSize);
        Assert.Equal(new[] { 128, 196, 256 }, loftr.Options.BlockDims);
        Assert.Equal(4, loftr.Options.CoarseLayerPairs);
    }

    [Fact]
    public void PipelineIsDeterministicUnderASeed()
    {
        var first = TinyModel(seed: 3).ImageTo3D(InputImage(0.3f), seed: 21);
        var second = TinyModel(seed: 3).ImageTo3D(InputImage(0.3f), seed: 21);
        Assert.Equal(NeuSHost<float>.ToDoubles(first.Vertices), NeuSHost<float>.ToDoubles(second.Vertices));
        Assert.Equal(first.Faces.Cast<int>().ToArray(), second.Faces.Cast<int>().ToArray());
        Assert.Equal(NeuSHost<float>.ToDoubles(first.Colors ?? new Tensor<float>(new[] { 0, 3 })),
            NeuSHost<float>.ToDoubles(second.Colors ?? new Tensor<float>(new[] { 0, 3 })));
    }

    [Fact]
    public void DifferentInputsGiveDifferentMeshes()
    {
        var model = TinyModel(seed: 3);
        var a = model.ImageTo3D(InputImage(0.1f), seed: 21);
        var b = model.ImageTo3D(InputImage(0.9f), seed: 21);
        var va = NeuSHost<float>.ToDoubles(a.Vertices);
        var vb = NeuSHost<float>.ToDoubles(b.Vertices);
        bool differ = va.Length != vb.Length || va.Zip(vb, (x, y) => Math.Abs(x - y)).Any(d => d > 1e-6);
        Assert.True(differ, "Different input images must give different meshes.");
    }

    [Fact]
    public void TextTo3DRendersThePromptThenReconstructs()
    {
        var model = TinyModel(seed: 3);
        var mesh = model.GenerateMesh("a red mug", resolution: 16, seed: 4);
        Assert.True(mesh.Vertices.Shape[0] > 0);

        var cloud = model.GeneratePointCloud("a red mug", numPoints: 64, seed: 4);
        Assert.Equal(new[] { 1, 64, 6 }, cloud.Shape.ToArray());
        var other = model.GeneratePointCloud("a blue chair", numPoints: 64, seed: 4);
        Assert.NotEqual(NeuSHost<float>.ToDoubles(cloud), NeuSHost<float>.ToDoubles(other));

        var withoutText = new One2345Model<float>(seed: 3, pipelineOptions: TinyOptions(), zero123: SharedStub.Value);
        Assert.Throws<InvalidOperationException>(() => withoutText.GenerateMesh("a red mug"));
    }

    [Fact]
    public void SurfaceSamplesLieOnTheTriangleWithInterpolatedColours()
    {
        var mesh = new Mesh3D<double>(NeuSHost<double>.FromDoubles(new double[] { 0, 0, 0, 2, 0, 0, 0, 2, 0 }, 3, 3), new[,] { { 0, 1, 2 } })
        {
            Colors = NeuSHost<double>.FromDoubles(new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 }, 3, 3),
        };
        var cloud = NeuSHost<double>.ToDoubles(One2345Model<double>.SampleSurface(mesh, 200, seed: 9));
        for (int p = 0; p < 200; p++)
        {
            double x = cloud[p * 6], y = cloud[p * 6 + 1], z = cloud[p * 6 + 2];
            Assert.Equal(0.0, z, 12);
            Assert.True(x >= -1e-12 && y >= -1e-12 && x + y <= 2 + 1e-12);
            // The barycentric weights are (1 − x/2 − y/2, x/2, y/2), so the colour is exactly them.
            Assert.Equal(1 - x / 2 - y / 2, cloud[p * 6 + 3], 9);
            Assert.Equal(x / 2, cloud[p * 6 + 4], 9);
            Assert.Equal(y / 2, cloud[p * 6 + 5], 9);
        }

        Assert.Equal(cloud, NeuSHost<double>.ToDoubles(One2345Model<double>.SampleSurface(mesh, 200, seed: 9)));
    }

    [Fact]
    public void ReconstructionWeightsUseReferenceNamesAndRoundTrip()
    {
        var model = TinyModel(seed: 3);
        var named = model.GetReconstructionParameters().ToList();
        Assert.Contains(named, p => p.Key.StartsWith("pyramid_feature_network_geometry_lod0.", StringComparison.Ordinal));
        Assert.Contains(named, p => p.Key.StartsWith("sdf_network_lod0.", StringComparison.Ordinal));
        Assert.Contains(named, p => p.Key.StartsWith("rendering_network_lod0.", StringComparison.Ordinal));
        Assert.Contains(named, p => p.Key == "variance_network_lod0.variance");

        // A second model with other initial weights becomes identical after loading the first's weights.
        var other = TinyModel(seed: 99);
        Assert.NotEqual(Flatten(named), Flatten(other.GetReconstructionParameters()));
        other.LoadReconstructionParameters(named.ToDictionary(p => p.Key, p => p.Value));
        Assert.Equal(Flatten(named), Flatten(other.GetReconstructionParameters()));
        Assert.Throws<ArgumentException>(() => other.LoadReconstructionParameters(
            new Dictionary<string, Tensor<float>> { ["not_a_parameter"] = new Tensor<float>(new[] { 1 }) }));
    }

    // ------------------------------------------------------------------ helpers

    private static double[] Flatten(IEnumerable<KeyValuePair<string, Tensor<float>>> parameters)
        => parameters.SelectMany(p => NeuSHost<float>.ToDoubles(p.Value)).ToArray();

    private static Dictionary<(int, int), IReadOnlyList<FeatureCorrespondence>> ExactMatches(One2345Options o, double polar, int seed)
    {
        var cameras = One2345ElevationEstimator.Hypothesis(o, polar, 280, 128, 128);
        var random = RandomHelper.CreateSeededRandom(seed);
        var points = Enumerable.Range(0, 40)
            .Select(_ => new[] { random.NextDouble() * 0.6 - 0.3, random.NextDouble() * 0.6 - 0.3, random.NextDouble() * 0.6 - 0.3 })
            .ToList();
        var matches = new Dictionary<(int, int), IReadOnlyList<FeatureCorrespondence>>();
        for (int i = 0; i < 4; i++)
            for (int j = i + 1; j < 4; j++)
            {
                var list = new List<FeatureCorrespondence>();
                foreach (var p in points)
                {
                    var (ui, vi, _) = cameras[i].Project(p[0], p[1], p[2]);
                    var (uj, vj, _) = cameras[j].Project(p[0], p[1], p[2]);
                    list.Add(new FeatureCorrespondence(ui, vi, uj, vj, 1.0));
                }

                matches[(i, j)] = list;
            }

        return matches;
    }

    private static Dictionary<(int, int), IReadOnlyList<FeatureCorrespondence>> Symmetric(
        Dictionary<(int, int), IReadOnlyList<FeatureCorrespondence>> forward)
    {
        var all = new Dictionary<(int, int), IReadOnlyList<FeatureCorrespondence>>(forward);
        foreach (var entry in forward) all[(entry.Key.Item2, entry.Key.Item1)] = entry.Value.Select(m => m.Reversed()).ToList();
        return all;
    }

    /// <summary>A white image with a textured, tinted disk, in [−1, 1].</summary>
    private static Tensor<float> InputImage(float tint)
    {
        var values = new float[3 * ImageSize * ImageSize];
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < ImageSize; y++)
                for (int x = 0; x < ImageSize; x++)
                {
                    double dx = x - ImageSize / 2.0 + 0.5, dy = y - ImageSize / 2.0 + 0.5;
                    bool inside = dx * dx + dy * dy < 25;
                    double v = inside ? tint * (0.6 + 0.4 * Math.Sin(x * 1.3 + c) * Math.Cos(y * 0.9)) : 1.0;
                    values[(c * ImageSize + y) * ImageSize + x] = (float)(v * 2 - 1);
                }

        return new Tensor<float>(new[] { 1, 3, ImageSize, ImageSize }, new Vector<float>(values));
    }

    /// <summary>
    /// Zero123 with its sampler replaced by a deterministic renderer of the requested pose: a disk tinted by the
    /// source image's mean, its texture phase set by the pose, plus seeded noise. Text prompts render the same way,
    /// tinted by the prompt.
    /// </summary>
    private sealed class StubZero123 : Zero123Model<float>
    {
        public StubZero123()
            : base(
                unet: new UNetNoisePredictor<float>(inputChannels: 4, outputChannels: 4, baseChannels: 64,
                    channelMultipliers: new[] { 1, 2, 4 }, numResBlocks: 1, attentionResolutions: new[] { 1, 2 }, contextDim: 0,
                    numHeads: 4, inputHeight: 16, seed: 42),
                vae: new StandardVAE<float>(inputChannels: 3, latentChannels: 4, baseChannels: 32,
                    channelMultipliers: new[] { 1, 2, 4 }, numResBlocksPerLevel: 1, latentScaleFactor: 0.18215, seed: 42),
                seed: 42)
        {
        }

        public List<(double Polar, double Azimuth)> Requests { get; } = new List<(double, double)>();

        public override Tensor<float> GenerateNovelView(Tensor<float> inputImage, double polarAngle, double azimuthAngle,
            double radius = 1.0, int numInferenceSteps = 50, double? guidanceScale = null, int? seed = null)
        {
            Requests.Add((polarAngle, azimuthAngle));
            var source = inputImage.ToVector().ToArray();
            double mean = source.Average(v => (double)v);
            return Render(mean, polarAngle, azimuthAngle, seed, inputImage.Shape[2], inputImage.Shape[3]);
        }

        public override Tensor<float> GenerateFromText(string prompt, string? negativePrompt = null, int width = 512,
            int height = 512, int numInferenceSteps = 50, double? guidanceScale = null, int? seed = null)
        {
            double tint = prompt.Sum(ch => (int)ch) % 97 / 97.0 * 2 - 1;
            return Render(tint, 0, 0, seed, height, width);
        }

        private static Tensor<float> Render(double tint, double polar, double azimuth, int? seed, int h, int w)
        {
            var noise = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSeededRandom(0);
            double shift = azimuth / 90.0;
            var values = new float[3 * h * w];
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        double dx = x - w / 2.0 + 0.5 - shift, dy = y - h / 2.0 + 0.5 + polar / 30.0;
                        bool inside = dx * dx + dy * dy < w * w / 9.0;
                        double v = inside
                            ? 0.5 + 0.3 * tint + 0.2 * Math.Sin(x * 1.1 + azimuth / 25.0 + c) * Math.Cos(y * 0.8 + polar / 25.0)
                                + 0.05 * noise.NextDouble()
                            : 1.0;
                        values[(c * h + y) * w + x] = (float)(Math.Max(0.0, Math.Min(1.0, v)) * 2 - 1);
                    }

            return new Tensor<float>(new[] { 1, 3, h, w }, new Vector<float>(values));
        }
    }
}
