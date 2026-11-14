using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.ThreeD;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Reference tests for the DreamFusion, MVDream and Shap-E geometry: the SDS weighting
/// w(t) = 1 - alpha_bar_t (Poole et al., 2022, eq. 3), the gradient-driven NeRF update, MVDream's camera-to-world
/// camera embedding (Shi et al., 2023), Shap-E's latent-parameterized implicit MLP (Jun &amp; Nichol, 2023) and the
/// marching-tetrahedra surface extraction they share.
/// </summary>
public class ThreeDGeometryTests
{
    private const double Tolerance = 1e-9;

    // ---------------------------------------------------------------- DreamFusion

    [Theory]
    [InlineData(0)]
    [InlineData(20)]
    [InlineData(500)]
    [InlineData(999)]
    public void SdsWeight_IsOneMinusScaledLinearAlphaBar(int timestep)
    {
        // Stable Diffusion's scaled-linear schedule: beta_i = (sqrt(0.00085) + i/999 (sqrt(0.012) - sqrt(0.00085)))^2.
        double alphaBar = 1.0;
        for (int i = 0; i <= timestep; i++)
        {
            double sqrtBeta = Math.Sqrt(0.00085) + (i / 999.0 * (Math.Sqrt(0.012) - Math.Sqrt(0.00085)));
            alphaBar *= 1.0 - (sqrtBeta * sqrtBeta);
        }

        Assert.Equal(alphaBar, DreamFusionModel<double>.GetAlphasCumprod(timestep), 12);
        Assert.Equal(1.0 - alphaBar, DreamFusionModel<double>.ComputeSDSWeight(timestep), 12);
    }

    [Fact]
    public void SdsWeight_GrowsWithNoiseLevel()
    {
        Assert.True(DreamFusionModel<double>.ComputeSDSWeight(100) < DreamFusionModel<double>.ComputeSDSWeight(500));
        Assert.True(DreamFusionModel<double>.ComputeSDSWeight(500) < DreamFusionModel<double>.ComputeSDSWeight(900));
    }

    [Fact]
    public void MarchingTetrahedra_SphereSurfaceLiesOnTheRadius()
    {
        const int resolution = 17;
        const double radius = 0.6;
        double step = 2.0 / (resolution - 1);
        var grid = new double[resolution, resolution, resolution];
        for (int x = 0; x < resolution; x++)
        {
            for (int y = 0; y < resolution; y++)
            {
                for (int z = 0; z < resolution; z++)
                {
                    double px = -1 + (x * step), py = -1 + (y * step), pz = -1 + (z * step);
                    grid[x, y, z] = radius - Math.Sqrt((px * px) + (py * py) + (pz * pz)); // positive inside
                }
            }
        }

        var (coordinates, triangles) = MarchingTetrahedra.Extract(grid, 0.0, -1.0, step);

        Assert.NotEmpty(triangles);
        Assert.Equal(0, triangles.Count % 3);
        int vertexCount = coordinates.Count / 3;
        Assert.All(triangles, index => Assert.InRange(index, 0, vertexCount - 1));
        for (int v = 0; v < vertexCount; v++)
        {
            double r = Math.Sqrt(
                (coordinates[3 * v] * coordinates[3 * v]) +
                (coordinates[(3 * v) + 1] * coordinates[(3 * v) + 1]) +
                (coordinates[(3 * v) + 2] * coordinates[(3 * v) + 2]));
            // Linear interpolation of a distance field is exact along axes and within a fraction of a cell otherwise.
            Assert.InRange(r, radius - (0.25 * step), radius + (0.25 * step));
        }
    }

    [Fact]
    public void MarchingTetrahedra_PlaneCrossingIsInterpolatedExactly()
    {
        const int resolution = 4;
        double step = 2.0 / (resolution - 1);
        var grid = new double[resolution, resolution, resolution];
        for (int x = 0; x < resolution; x++)
        {
            for (int y = 0; y < resolution; y++)
            {
                for (int z = 0; z < resolution; z++)
                    grid[x, y, z] = 0.25 - (-1 + (x * step)); // inside where x < 0.25
            }
        }

        var (coordinates, triangles) = MarchingTetrahedra.Extract(grid, 0.0, -1.0, step);

        Assert.NotEmpty(triangles);
        for (int v = 0; v < coordinates.Count; v += 3)
            Assert.Equal(0.25, coordinates[v], 12);
    }

    [Fact]
    public void MarchingTetrahedra_EveryTriangleNormalPointsOutOfTheSphere()
    {
        // The six cube tetrahedra alternate in handedness, so emitting crossings in table order gave half the
        // faces inward normals. A closed surface needs every normal to face outward (towards decreasing values).
        const int resolution = 17;
        const double radius = 0.6;
        double step = 2.0 / (resolution - 1);
        var grid = new double[resolution, resolution, resolution];
        for (int x = 0; x < resolution; x++)
        {
            for (int y = 0; y < resolution; y++)
            {
                for (int z = 0; z < resolution; z++)
                {
                    double px = -1 + (x * step), py = -1 + (y * step), pz = -1 + (z * step);
                    grid[x, y, z] = radius - Math.Sqrt((px * px) + (py * py) + (pz * pz));
                }
            }
        }

        var (c, triangles) = MarchingTetrahedra.Extract(grid, 0.0, -1.0, step);

        Assert.NotEmpty(triangles);
        for (int t = 0; t < triangles.Count; t += 3)
        {
            int a = 3 * triangles[t], b = 3 * triangles[t + 1], d = 3 * triangles[t + 2];
            double ux = c[b] - c[a], uy = c[b + 1] - c[a + 1], uz = c[b + 2] - c[a + 2];
            double vx = c[d] - c[a], vy = c[d + 1] - c[a + 1], vz = c[d + 2] - c[a + 2];
            double nx = (uy * vz) - (uz * vy), ny = (uz * vx) - (ux * vz), nz = (ux * vy) - (uy * vx);
            double cx = (c[a] + c[b] + c[d]) / 3, cy = (c[a + 1] + c[b + 1] + c[d + 1]) / 3;
            double cz = (c[a + 2] + c[b + 2] + c[d + 2]) / 3;
            Assert.True((nx * cx) + (ny * cy) + (nz * cz) > 0, $"Triangle {t / 3} faces inward.");
        }
    }

    [Fact]
    public void ShapE_FaceTensor_Half_RejectsIndicesHalfCannotHoldExactly()
    {
        // Half stores integers exactly only up to 2048 (2049 rounds to 2048, 3001 to 3000), so a face holding them
        // would silently point at the wrong vertex.
        var exact = ShapEModel<Half>.FaceTensor(new List<int> { 0, 1, 2048 }).ToArray();
        Assert.Equal(new[] { 0.0, 1.0, 2048.0 }, exact.Select(h => (double)h).ToArray());

        Assert.Throws<InvalidOperationException>(() => ShapEModel<Half>.FaceTensor(new List<int> { 0, 1, 2049 }));
        Assert.Throws<InvalidOperationException>(() => ShapEModel<Half>.FaceTensor(new List<int> { 3001, 1, 2 }));
    }

    [Fact]
    public void NeRF_SingleRayLooksAtTheTarget_AndTopRowLooksUp()
    {
        var nerf = new NeRFNetwork<double>(hiddenDim: 8, numLayers: 1, seed: 1);
        var pose = new CameraPose { Azimuth = 90, Elevation = 0, Radius = 2, FocalLength = 1 };

        var (origin, single) = nerf.BuildRays(pose, 1);
        var o = origin.ToArray();
        Assert.Equal(0.0, o[0], 12);
        Assert.Equal(0.0, o[1], 12);
        Assert.Equal(2.0, o[2], 12);
        Assert.Equal(new[] { 0.0, 0.0, -1.0 }, single.ToArray().Select(value => Math.Round(value, 12)).ToArray());

        // Camera on +Z looking at the origin: the top-left pixel ray points up (+Y) and left (-X).
        var (_, rays) = nerf.BuildRays(pose, 4);
        var d = rays.ToArray();
        Assert.True(d[1] > 0, "the top row must look up");
        Assert.True(d[0] < 0, "the left column must look left");
        int last = (4 * 4) - 1;
        Assert.True(d[(3 * last) + 1] < 0, "the bottom row must look down");
        Assert.True(d[3 * last] > 0, "the right column must look right");
    }

    [Fact]
    public void NeRF_BatchedDensityMatchesPointQuery_AndRenderIsAnRgbImage()
    {
        var nerf = new NeRFNetwork<double>(hiddenDim: 8, numLayers: 2, seed: 1);
        var points = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(new[] { 0.1, -0.2, 0.3, 0.5, 0.5, -0.5, -0.9, 0.0, 0.4 }));

        var batched = nerf.QueryDensities(points).ToArray();
        Assert.Equal(nerf.QueryDensity(0.1, -0.2, 0.3), batched[0], 12);
        Assert.Equal(nerf.QueryDensity(0.5, 0.5, -0.5), batched[1], 12);
        Assert.Equal(nerf.QueryDensity(-0.9, 0.0, 0.4), batched[2], 12);

        var image = nerf.Render(new CameraPose { Azimuth = 30, Elevation = 10, Radius = 1.5 }, 3);
        Assert.Equal(new[] { 3, 3, 3 }, image.Shape.ToArray());
        Assert.All(image.ToArray(), value => Assert.InRange(value, 0.0, 1.0));
    }

    [Fact]
    public void NeRF_SdsUpdate_DescendsTheImageSpaceGradient()
    {
        // With the SDS residual g held fixed, the update must lower <render(theta), g>: the old update perturbed
        // the weights at random and ignored g.
        var nerf = new NeRFNetwork<double>(hiddenDim: 16, numLayers: 2, seed: 7);
        var pose = new CameraPose { Azimuth = 45, Elevation = 15, Radius = 1.2 };
        const int resolution = 4;
        var gradient = new Tensor<double>(new[] { resolution, resolution, 3 },
            new Vector<double>(Enumerable.Repeat(1.0, resolution * resolution * 3).ToArray()));

        double before = nerf.Render(pose, resolution).ToArray().Sum();
        var parametersBefore = nerf.GetParameters().ToArray();
        nerf.UpdateParameters(gradient, pose, 1e-3);
        double after = nerf.Render(pose, resolution).ToArray().Sum();

        Assert.NotEqual(parametersBefore, nerf.GetParameters().ToArray());
        Assert.True(after < before, $"the SDS step must lower the gradient-weighted render ({before} -> {after})");
    }

    [Fact]
    public void NeRF_ParameterRoundTrip()
    {
        var nerf = new NeRFNetwork<double>(hiddenDim: 8, numLayers: 2, seed: 1);
        var parameters = nerf.GetParameters();
        Assert.Equal(nerf.ParameterCount, parameters.Length);

        var shifted = new Vector<double>(parameters.ToArray().Select(p => p + 0.5).ToArray());
        nerf.SetParameters(shifted);
        Assert.Equal(shifted.ToArray(), nerf.GetParameters().ToArray());
    }

    // ---------------------------------------------------------------- MVDream

    [Theory]
    [InlineData(0.0, 0.0, 2.0)]
    [InlineData(1.2, 0.4, 1.5)]
    [InlineData(-2.5, -0.7, 3.0)]
    public void CameraToWorld_IsARigidTransformLookingAtTheOrigin(double azimuth, double elevation, double radius)
    {
        var m = CameraEmbedding<double>.CameraToWorld(azimuth, elevation, radius);
        Assert.Equal(CameraEmbedding<double>.CameraDim, m.Length);

        double[] Column(int c) => new[] { m[c], m[4 + c], m[8 + c] };
        double Dot(double[] a, double[] b) => (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
        var right = Column(0);
        var up = Column(1);
        var back = Column(2);
        var position = Column(3);

        Assert.Equal(1.0, Dot(right, right), 12);
        Assert.Equal(1.0, Dot(up, up), 12);
        Assert.Equal(1.0, Dot(back, back), 12);
        Assert.Equal(0.0, Dot(right, up), 12);
        Assert.Equal(0.0, Dot(right, back), 12);
        Assert.Equal(0.0, Dot(up, back), 12);
        Assert.Equal(radius, Math.Sqrt(Dot(position, position)), 12);
        // The camera looks along -back, towards the origin: back is the unit position vector.
        for (int i = 0; i < 3; i++)
            Assert.Equal(position[i] / radius, back[i], 12);
        Assert.True(up[1] >= 0, "the camera's up vector must not point below the horizon");
        Assert.Equal(new[] { 0.0, 0.0, 0.0, 1.0 }, new[] { m[12], m[13], m[14], m[15] });
    }

    [Fact]
    public void CameraEmbedding_IsTwoLayerMlpOfTheExtrinsics()
    {
        const int dim = 6;
        var embedding = new CameraEmbedding<double>(dim, seed: 3);
        Assert.Equal((dim * 16) + dim + (dim * dim) + dim, embedding.ParameterCount);

        var e = embedding.Embed(0.7, 0.2, 2.0);
        Assert.Equal(new[] { 1, dim }, e.Shape.ToArray());

        // Reference: W2 SiLU(W1 vec(E) + b1) + b2 from the flat parameter layout.
        var p = embedding.GetParameters().ToArray();
        var extrinsic = CameraEmbedding<double>.CameraToWorld(0.7, 0.2, 2.0);
        var hidden = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            double sum = p[(dim * 16) + i];
            for (int k = 0; k < 16; k++)
                sum += p[(i * 16) + k] * extrinsic[k];
            hidden[i] = sum / (1.0 + Math.Exp(-sum));
        }

        int w2 = (dim * 16) + dim;
        var actual = e.ToArray();
        for (int i = 0; i < dim; i++)
        {
            double sum = p[w2 + (dim * dim) + i];
            for (int k = 0; k < dim; k++)
                sum += p[w2 + (i * dim) + k] * hidden[k];
            Assert.Equal(sum, actual[i], 10);
        }

        Assert.NotEqual(actual, embedding.Embed(2.0, 0.2, 2.0).ToArray());
    }

    [Fact]
    public void ScoreDistillationGradient_IsWeightedGuidedResidual()
    {
        var cond = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 1.0, -2.0 }));
        var uncond = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 0.5, 1.0 }));
        var noise = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 0.25, 0.0 }));

        var g = MVDreamModel<double>.ScoreDistillationGradient(cond, uncond, noise, 7.5, 0.64).ToArray();

        // (1 - 0.64) * (uncond + 7.5 (cond - uncond) - noise)
        Assert.Equal(0.36 * (0.5 + (7.5 * 0.5) - 0.25), g[0], 12);
        Assert.Equal(0.36 * (1.0 + (7.5 * -3.0)), g[1], 12);
    }

    [Fact]
    public void MultiViewAttention_IsResidualCrossViewAttention()
    {
        const int channels = 3;
        var attention = new MultiViewAttention<double>(channels, numViews: 2, seed: 5);
        var views = new[]
        {
            new Tensor<double>(new[] { 1, channels, 1, 2 }, new Vector<double>(new[] { 1.0, 2.0, 0.5, -1.0, 0.0, 3.0 })),
            new Tensor<double>(new[] { 1, channels, 1, 2 }, new Vector<double>(new[] { -1.0, 0.0, 1.5, 2.0, 1.0, -2.0 }))
        };

        // Zero output projection: the residual returns the views unchanged.
        var p = attention.GetParameters().ToArray();
        int block = channels * channels;
        var zeroOut = p.ToArray();
        for (int i = 3 * block; i < 4 * block; i++)
            zeroOut[i] = 0.0;
        attention.SetParameters(new Vector<double>(zeroOut));
        var identity = attention.Apply(views);
        Assert.Equal(views[0].ToArray(), identity[0].ToArray());
        Assert.Equal(views[1].ToArray(), identity[1].ToArray());

        // Reference attention at every position over the two views' tokens.
        attention.SetParameters(new Vector<double>(p));
        var result = attention.Apply(views);
        double[] W(int index, int row) => p.Skip((index * block) + (row * channels)).Take(channels).ToArray();
        double[] Project(int index, double[] x) =>
            Enumerable.Range(0, channels).Select(r => W(index, r).Zip(x, (w, v) => w * v).Sum()).ToArray();

        for (int position = 0; position < 2; position++)
        {
            var tokens = views.Select(view => Enumerable.Range(0, channels)
                .Select(c => view.ToArray()[(c * 2) + position]).ToArray()).ToArray();
            for (int v = 0; v < 2; v++)
            {
                var q = Project(0, tokens[v]);
                var scores = tokens.Select(t => Project(1, t).Zip(q, (a, b) => a * b).Sum() / Math.Sqrt(channels)).ToArray();
                double max = scores.Max();
                var weights = scores.Select(s => Math.Exp(s - max)).ToArray();
                double total = weights.Sum();
                var mixed = new double[channels];
                for (int u = 0; u < 2; u++)
                {
                    var value = Project(2, tokens[u]);
                    for (int c = 0; c < channels; c++)
                        mixed[c] += weights[u] / total * value[c];
                }

                var output = Project(3, mixed);
                for (int c = 0; c < channels; c++)
                    Assert.Equal(tokens[v][c] + output[c], result[v].ToArray()[(c * 2) + position], 10);
            }
        }
    }

    // ---------------------------------------------------------------- Shap-E

    private static ShapEModel<double> SmallShapE() => new ShapEModel<double>(
        latentPredictor: new DiTNoisePredictor<double>(
            inputChannels: 1024, hiddenSize: 32, numLayers: 1, numHeads: 2,
            patchSize: 1, contextDim: 1024, latentSpatialSize: 1, seed: 1));

    private const int Hidden = 22;
    private static int W1(int input, int unit) => (input * Hidden) + unit;
    private static int B1(int unit) => (15 * Hidden) + unit;
    private static int W2(int from, int to) => (16 * Hidden) + (from * Hidden) + to;
    private static int W3(int from, int output) => (17 * Hidden) + (Hidden * Hidden) + (from * 5) + output;
    private static int B3(int output) => (22 * Hidden) + (Hidden * Hidden) + output;

    /// <summary>A latent whose SDF is a monotone function of x vanishing at x = 0 (inside for x &lt; 0).</summary>
    private static Tensor<double> PlaneLatent()
    {
        var values = new double[1024];
        values[W1(0, 0)] = 1.0;
        values[W2(0, 0)] = 1.0;
        values[W3(0, 0)] = 1.0;
        return new Tensor<double>(new[] { 1, 1, 1024 }, new Vector<double>(values));
    }

    [Fact]
    public void ShapEImplicitFunction_LayoutFillsTheLatent()
    {
        Assert.Equal(15, ShapEImplicitFunction<double>.EncodedDim);
        Assert.Equal(Hidden, ShapEImplicitFunction<double>.HiddenWidth(1024));
        Assert.True(ShapEImplicitFunction<double>.ParameterCount(Hidden) <= 1024);
        Assert.True(ShapEImplicitFunction<double>.ParameterCount(Hidden + 1) > 1024);
        Assert.Equal(B3(4) + 1, ShapEImplicitFunction<double>.ParameterCount(Hidden));
    }

    [Fact]
    public void ShapEImplicitFunction_MatchesScalarMlp()
    {
        var latentValues = Enumerable.Range(0, 1024).Select(i => Math.Sin(i * 0.37)).ToArray();
        var field = new ShapEImplicitFunction<double>(
            new Tensor<double>(new[] { 1, 1, 1024 }, new Vector<double>(latentValues)));
        var point = new[] { 0.3, -0.6, 0.1 };
        var outputs = field.Evaluate(new Tensor<double>(new[] { 1, 3 }, new Vector<double>(point))).ToArray();

        var encoded = new List<double>(point);
        for (int k = 0; k < 2; k++)
        {
            foreach (var x in point)
            {
                encoded.Add(Math.Sin(Math.Pow(2, k) * Math.PI * x));
                encoded.Add(Math.Cos(Math.Pow(2, k) * Math.PI * x));
            }
        }

        double SiLU(double z) => z / (1.0 + Math.Exp(-z));
        var h1 = new double[Hidden];
        for (int j = 0; j < Hidden; j++)
        {
            double sum = latentValues[B1(j)] / Math.Sqrt(15);
            for (int i = 0; i < 15; i++)
                sum += latentValues[W1(i, j)] / Math.Sqrt(15) * encoded[i];
            h1[j] = SiLU(sum);
        }

        var h2 = new double[Hidden];
        for (int j = 0; j < Hidden; j++)
        {
            double sum = latentValues[(16 * Hidden) + (Hidden * Hidden) + j] / Math.Sqrt(Hidden);
            for (int i = 0; i < Hidden; i++)
                sum += latentValues[W2(i, j)] / Math.Sqrt(Hidden) * h1[i];
            h2[j] = SiLU(sum);
        }

        for (int o = 0; o < 5; o++)
        {
            double sum = latentValues[B3(o)] / Math.Sqrt(Hidden);
            for (int i = 0; i < Hidden; i++)
                sum += latentValues[W3(i, o)] / Math.Sqrt(Hidden) * h2[i];
            Assert.Equal(sum, outputs[o], 10);
        }
    }

    [Fact]
    public void ShapE_ExtractMesh_FollowsTheLatentSdfZeroSet()
    {
        var model = SmallShapE();
        var (vertices, faces) = model.ExtractMesh(PlaneLatent(), resolution: 5);

        Assert.True(faces.Shape[0] > 1, "the plane must be triangulated");
        var v = vertices.ToArray();
        for (int i = 0; i < v.Length; i += 3)
            Assert.Equal(0.0, v[i], 9);
        int vertexCount = vertices.Shape[0];
        Assert.All(faces.ToArray(), index => Assert.InRange(index, 0, vertexCount - 1));
    }

    [Fact]
    public void ShapE_SamplePointCloud_KeepsInsidePointsColouredByTheMlp()
    {
        var model = SmallShapE();
        var cloud = model.SamplePointCloud(PlaneLatent(), numPoints: 50, seed: 4);

        Assert.Equal(new[] { 1, 50, 6 }, cloud.Shape.ToArray());
        var values = cloud.ToArray();
        for (int p = 0; p < 50; p++)
        {
            Assert.True(values[p * 6] < 0, "accepted points have SDF < 0, i.e. x < 0");
            for (int c = 3; c < 6; c++)
                Assert.Equal(0.5, values[(p * 6) + c], 12); // sigmoid(0)
        }

        Assert.Equal(values, model.SamplePointCloud(PlaneLatent(), numPoints: 50, seed: 4).ToArray());
    }

    [Fact]
    public void ShapE_RenderView_IsUprightAndUnmirrored()
    {
        // Opaque field whose red grows with y and green with x, seen from +Z looking at the origin.
        var values = new double[1024];
        values[W1(0, 0)] = 1.0;
        values[W2(0, 0)] = 1.0;
        values[W3(0, 3)] = 100.0; // green from x
        values[W1(1, 1)] = 1.0;
        values[W2(1, 1)] = 1.0;
        values[W3(1, 2)] = 100.0; // red from y
        values[B3(1)] = 50.0;     // dense everywhere
        var latent = new Tensor<double>(new[] { 1, 1, 1024 }, new Vector<double>(values));

        const int size = 4;
        var image = SmallShapE().RenderView(latent, (0, 0, 2), (0, 0, 0), imageSize: size, numSamples: 16);
        Assert.Equal(new[] { 1, 3, size, size }, image.Shape.ToArray());

        var pixels = image.ToArray();
        double Red(int y, int x) => pixels[(0 * size * size) + (y * size) + x];
        double Green(int y, int x) => pixels[(1 * size * size) + (y * size) + x];
        Assert.True(Red(0, 1) > Red(size - 1, 1), "the top of the image must show +Y");
        Assert.True(Green(1, size - 1) > Green(1, 0), "the right of the image must show +X");
    }
}
