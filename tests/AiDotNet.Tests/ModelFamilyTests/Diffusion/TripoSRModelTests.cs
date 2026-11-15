using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Enums;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// Reference tests for TripoSR (Tochilkin et al. 2024): triplane sampling against a hand computation, parameter
/// counts derived by hand, shapes, determinism, input dependence, and the geometry built on the dense density grid.
/// </summary>
/// <remarks>
/// TripoSR is feed-forward (a DINO ViT, a triplane transformer and a triplane NeRF), so it is tested here directly
/// rather than through the denoising invariants of <c>DiffusionModelTestBase</c>, which do not apply to it.
/// </remarks>
public class TripoSRModelTests
{
    private const double Tolerance = 1e-9;

    // A tiny configuration: 32x32 image, 16x16 patches (a 2x2 grid), width-8 encoder and transformer, 2x2 token
    // planes upsampled to 4-channel 4x4 triplanes, and a decoder of two hidden layers of width 8.
    private static TripoSROptions Tiny(int seed = 42) => new()
    {
        ImageSize = 32,
        PatchSize = 16,
        EncoderLayers = 1,
        EncoderHeads = 2,
        EncoderHeadDim = 4,
        EncoderMlpRatio = 4,
        EncoderPositionGridSize = 2,
        EncoderLayerNormEpsilon = 1e-6,
        TriplaneTokenResolution = 2,
        TriplaneTokenChannels = 8,
        TransformerLayers = 1,
        TransformerHeads = 2,
        TransformerHeadDim = 4,
        TransformerNormGroups = 2,
        TriplaneChannels = 4,
        DecoderHiddenWidth = 8,
        DecoderHiddenLayers = 2,
        SamplesPerRay = 8,
        QueryChunkSize = 64,
        GridResolution = 8,
        RenderResolution = 4,
        Seed = seed
    };

    private static Tensor<double> Image(int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var values = new double[3 * 32 * 32];
        for (int i = 0; i < values.Length; i++) values[i] = random.NextDouble();
        return new Tensor<double>(new[] { 1, 3, 32, 32 }, new Vector<double>(values));
    }

    private static double[] Values(Tensor<double> tensor) => tensor.ToArray();

    // ---------------------------------------------------------------------------------------------
    // Parameter counts, derived by hand from the official module definitions.
    // ---------------------------------------------------------------------------------------------

    [Fact(Timeout = 120000)]
    public async Task TinyConfiguration_ParameterCount_MatchesHandDerivation()
    {
        await Task.Yield();
        var model = new TripoSRModel<double>(Tiny());

        // DINO ViT, width D = 8, patch 16, a 2x2 position grid, one layer with an MLP of 32:
        //   patch conv 8*3*16*16 + 8 = 6152, CLS 8, positions (4 + 1) * 8 = 40,
        //   layer: 2 LN (16 + 16), q/k/v 3 * (64 + 8) = 216, out 72, fc1 8*32 + 32 = 288, fc2 32*8 + 8 = 264 -> 872,
        //   final LN 16.                                                                      encoder = 7088
        // Triplane tokens 3 * 8 * 2 * 2.                                                      tokens  = 96
        // Transformer1D, C = 8, inner 2 * 4 = 8, context 8, one block:
        //   GroupNorm 16, proj_in 64 + 8, proj_out 64 + 8,
        //   block: 3 LN 48, attn1 q/k/v 3 * 64 (no bias) + out 72, attn2 q 64 + k/v 2 * 64 + out 72,
        //          GEGLU 8 * 64 + 64 = 576, ff out 32 * 8 + 8 = 264 -> 1416.                  backbone = 1576
        // Upsampler ConvTranspose2d(8, 4, 2): 8 * 4 * 2 * 2 + 4.                              upsampler = 132
        // NeRF MLP over 3 * 4 = 12: 12*8 + 8 = 104, 8*8 + 8 = 72, 8*4 + 4 = 36.                decoder = 212
        Assert.Equal(7088 + 96 + 1576 + 132 + 212, model.ParameterCount);
        Assert.Equal(9104L, model.ParameterCount);
        Assert.Equal(model.ParameterCount, model.GetParameters().Length);
    }

    [Fact(Timeout = 120000)]
    public async Task OfficialConfiguration_ParameterCount_MatchesHandDerivation()
    {
        await Task.Yield();
        // Lazily allocated: counting the paper-size network does not materialize its weights.
        var model = new TripoSRModel<float>();

        long d = 768, inner = 1024, c = 1024;
        long encoderLayer = (4 * d) + (3 * ((d * d) + d)) + ((d * d) + d) + ((d * 4 * d) + (4 * d)) + ((4 * d * d) + d);
        long encoder = ((d * 3 * 16 * 16) + d) + d + (197 * d) + (12 * encoderLayer) + (2 * d);
        long tokens = 3 * c * 32 * 32;
        long block = (6 * inner) + (3 * inner * inner) + ((inner * inner) + inner)
            + (inner * inner) + (2 * d * inner) + ((inner * inner) + inner)
            + ((inner * 8 * inner) + (8 * inner)) + ((4 * inner * inner) + inner);
        long backbone = (2 * c) + ((c * inner) + inner) + ((inner * c) + c) + (16 * block);
        long upsampler = (c * 40 * 4) + 40;
        long decoder = ((120 * 64) + 64) + (8 * ((64 * 64) + 64)) + ((64 * 4) + 4);

        Assert.Equal(85_798_656L, encoder);   // DINO ViT-B/16 without its pooler
        Assert.Equal(329_535_488L, backbone);
        Assert.Equal(encoder + tokens + backbone + upsampler + decoder, model.ParameterCount);
        Assert.Equal(418_685_036L, model.ParameterCount);
    }

    // ---------------------------------------------------------------------------------------------
    // Shapes.
    // ---------------------------------------------------------------------------------------------

    [Fact(Timeout = 120000)]
    public async Task Shapes_FollowTheConfiguration()
    {
        await Task.Yield();
        var model = new TripoSRModel<double>(Tiny());
        var image = Image(1);

        Assert.Equal(new[] { 1, 3, 4, 4, 4 }, model.EncodeSceneCodes(image).Shape.ToArray());
        Assert.Equal(new[] { 1, 3, 4, 4, 4 }, model.Predict(image).Shape.ToArray());
        Assert.Equal(new[] { 1, 3, 32, 32 }, model.GetInputShape());
        Assert.Equal(new[] { 1, 3, 4, 4, 4 }, model.GetOutputShape());

        var field = model.Reconstruct(image);
        Assert.Equal(new[] { 3, 4, 4, 4 }, field.Planes.Shape.ToArray());
        Assert.Equal(new[] { 6, 6, 6 }, field.SampleDensityGrid(6, 0.87).Shape.ToArray());
        Assert.Equal(new[] { 8, 8, 8 }, model.SampleDensityGrid(image).Shape.ToArray());

        var views = model.SynthesizeNovelViews(image, new[] { (0.0, 0.0), (Math.PI / 2, Math.PI / 6) });
        Assert.Equal(2, views.Length);
        foreach (var view in views) Assert.Equal(new[] { 1, 3, 4, 4 }, view.Shape.ToArray());

        // A batch of two images gives two scene codes; a non-RGB image is rejected.
        var batch = new Tensor<double>(new[] { 2, 3, 32, 32 });
        Assert.Equal(new[] { 2, 3, 4, 4, 4 }, model.EncodeSceneCodes(batch).Shape.ToArray());
        Assert.Throws<ArgumentException>(() => model.EncodeSceneCodes(new Tensor<double>(new[] { 1, 4, 32, 32 })));

        // An image of another size is resized to ImageSize; position embeddings stored on another grid are resized
        // (bicubic) to the 2x2 patch grid.
        var resizedGrid = Tiny();
        resizedGrid.EncoderPositionGridSize = 3;
        var resizing = new TripoSRModel<double>(resizedGrid);
        Assert.Equal(new[] { 1, 3, 4, 4, 4 }, resizing.EncodeSceneCodes(new Tensor<double>(new[] { 1, 3, 48, 48 })).Shape.ToArray());
    }

    // ---------------------------------------------------------------------------------------------
    // Triplane sampling against a hand computation.
    // ---------------------------------------------------------------------------------------------

    private const int PlaneChannels = 2;
    private const int PlaneSize = 4;
    private const double Radius = 0.87;

    private static Tensor<double> HandPlanes()
    {
        var random = RandomHelper.CreateSeededRandom(7);
        var values = new double[3 * PlaneChannels * PlaneSize * PlaneSize];
        for (int i = 0; i < values.Length; i++) values[i] = (2.0 * random.NextDouble()) - 1.0;
        return new Tensor<double>(new[] { 3, PlaneChannels, PlaneSize, PlaneSize }, new Vector<double>(values));
    }

    // torch grid_sample(mode="bilinear", padding_mode="zeros", align_corners=False) at one (u, v) in [-1, 1].
    private static double Bilinear(double[] planes, int plane, int channel, double u, double v)
    {
        double ix = (((u + 1.0) * PlaneSize) - 1.0) / 2.0;
        double iy = (((v + 1.0) * PlaneSize) - 1.0) / 2.0;
        int x0 = (int)Math.Floor(ix);
        int y0 = (int)Math.Floor(iy);
        double fx = ix - x0;
        double fy = iy - y0;
        double Read(int x, int y) => x < 0 || y < 0 || x >= PlaneSize || y >= PlaneSize
            ? 0.0
            : planes[(((plane * PlaneChannels) + channel) * PlaneSize + y) * PlaneSize + x];
        return (Read(x0, y0) * (1 - fx) * (1 - fy)) + (Read(x0 + 1, y0) * fx * (1 - fy))
            + (Read(x0, y0 + 1) * (1 - fx) * fy) + (Read(x0 + 1, y0 + 1) * fx * fy);
    }

    // Points inside the cube, one off-centre, and one near the +x face so a bilinear corner falls outside the plane.
    private static readonly double[,] HandPoints =
    {
        { 0.0, 0.0, 0.0 },
        { 0.31, -0.42, 0.17 },
        { 0.86, 0.25, -0.6 }
    };

    private static Tensor<double> HandPointTensor()
    {
        var values = new double[HandPoints.Length];
        for (int p = 0; p < HandPoints.GetLength(0); p++)
            for (int a = 0; a < 3; a++) values[(3 * p) + a] = HandPoints[p, a];
        return new Tensor<double>(new[] { HandPoints.GetLength(0), 3 }, new Vector<double>(values));
    }

    // The features of each plane per TriplaneNeRFRenderer.query_triplane: planes (x, y), (x, z), (y, z).
    private static double[][] HandPlaneFeatures(double[] planes, int point)
    {
        double x = HandPoints[point, 0] / Radius, y = HandPoints[point, 1] / Radius, z = HandPoints[point, 2] / Radius;
        var uv = new[] { (x, y), (x, z), (y, z) };
        var features = new double[3][];
        for (int p = 0; p < 3; p++)
        {
            features[p] = new double[PlaneChannels];
            for (int c = 0; c < PlaneChannels; c++) features[p][c] = Bilinear(planes, p, c, uv[p].Item1, uv[p].Item2);
        }

        return features;
    }

    private static TriplaneField<double> HandField(TriplaneFeatureReduction reduction, TsrNeRFDecoder<double> decoder)
    {
        var options = new TripoSROptions
        {
            Radius = Radius,
            FeatureReduction = reduction,
            DensityActivation = TriplaneDensityActivation.Exp,
            DensityBias = -1.0,
            QueryChunkSize = 2
        };
        return new TriplaneField<double>(HandPlanes(), decoder, options);
    }

    [Fact(Timeout = 120000)]
    public async Task SampleFeatures_Concat_IsBilinearPerPlaneConcatenatedPlaneMajor()
    {
        await Task.Yield();
        var planes = HandPlanes().ToArray();
        var field = HandField(TriplaneFeatureReduction.Concat, new TsrNeRFDecoder<double>(6, 2, 1, ActivationFunction.Identity));
        var features = field.SampleFeatures(HandPointTensor());

        Assert.Equal(new[] { 3, 3 * PlaneChannels }, features.Shape.ToArray());
        var actual = features.ToArray();
        for (int point = 0; point < 3; point++)
        {
            var expected = HandPlaneFeatures(planes, point);
            for (int p = 0; p < 3; p++)
                for (int c = 0; c < PlaneChannels; c++)
                    Assert.Equal(expected[p][c], actual[(point * 3 * PlaneChannels) + (p * PlaneChannels) + c], Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SampleFeatures_Mean_AveragesThePlanes()
    {
        await Task.Yield();
        var planes = HandPlanes().ToArray();
        var field = HandField(TriplaneFeatureReduction.Mean, new TsrNeRFDecoder<double>(2, 2, 1, ActivationFunction.Identity));
        var actual = field.SampleFeatures(HandPointTensor()).ToArray();
        for (int point = 0; point < 3; point++)
        {
            var expected = HandPlaneFeatures(planes, point);
            for (int c = 0; c < PlaneChannels; c++)
                Assert.Equal((expected[0][c] + expected[1][c] + expected[2][c]) / 3.0, actual[(point * PlaneChannels) + c], Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Query_DecodesExpDensityAndSigmoidColour_InChunks()
    {
        await Task.Yield();
        // A decoder with known weights: Linear(6, 2) then Identity then Linear(2, 4).
        var decoder = new TsrNeRFDecoder<double>(6, 2, 1, ActivationFunction.Identity);
        double[] w0 = { 0.5, -0.25, 0.75, 0.1, -0.6, 0.3, -0.2, 0.4, 0.15, -0.35, 0.05, 0.9 };
        double[] b0 = { 0.1, -0.2 };
        double[] w1 = { 1.2, -0.7, 0.3, 0.8, -0.5, 0.2, 0.9, -1.1 };
        double[] b1 = { 0.4, -0.1, 0.25, 0.05 };
        decoder.SetParameters(new Vector<double>(w0.Concat(b0).Concat(w1).Concat(b1).ToArray()));

        var planes = HandPlanes().ToArray();
        var field = HandField(TriplaneFeatureReduction.Concat, decoder);
        var output = field.Query(HandPointTensor());
        Assert.Equal(new[] { 3, 4 }, output.Shape.ToArray());
        var actual = output.ToArray();

        for (int point = 0; point < 3; point++)
        {
            var perPlane = HandPlaneFeatures(planes, point);
            var f = perPlane.SelectMany(v => v).ToArray();
            var h = new double[2];
            for (int o = 0; o < 2; o++)
            {
                h[o] = b0[o];
                for (int i = 0; i < 6; i++) h[o] += w0[(o * 6) + i] * f[i];
            }

            var raw = new double[4];
            for (int o = 0; o < 4; o++) raw[o] = b1[o] + (w1[o * 2] * h[0]) + (w1[(o * 2) + 1] * h[1]);

            Assert.Equal(Math.Exp(raw[0] - 1.0), actual[point * 4], Tolerance);
            for (int k = 1; k < 4; k++)
                Assert.Equal(1.0 / (1.0 + Math.Exp(-raw[k])), actual[(point * 4) + k], Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SampleDensityGrid_IsTheQueryOnLinspaceIjGrid()
    {
        await Task.Yield();
        var field = new TripoSRModel<double>(Tiny()).Reconstruct(Image(3));
        const int n = 4;
        const double bounds = 0.5;
        var grid = field.SampleDensityGrid(n, bounds).ToArray();

        // (i, j, k) -> (x_i, y_j, z_k) with x = linspace(-0.5, 0.5, 4).
        var axis = new[] { -0.5, -0.5 + (1.0 / 3.0), -0.5 + (2.0 / 3.0), 0.5 };
        var points = new double[n * n * n * 3];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                {
                    int row = 3 * ((((i * n) + j) * n) + k);
                    points[row] = axis[i];
                    points[row + 1] = axis[j];
                    points[row + 2] = axis[k];
                }

        var query = field.Query(new Tensor<double>(new[] { n * n * n, 3 }, new Vector<double>(points))).ToArray();
        for (int index = 0; index < n * n * n; index++)
            Assert.Equal(query[4 * index], grid[index], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task RenderRays_CompositesTheQueriedSamplesAndBlanksMisses()
    {
        await Task.Yield();
        var field = new TripoSRModel<double>(Tiny()).Reconstruct(Image(4));
        const int samples = 8;   // Tiny().SamplesPerRay
        double[] origins = { 0.0, -1.9, 0.1, 0.0, -1.9, 5.0 };
        double[] directions = { 0.0, 1.0, 0.0, 0.0, 1.0, 0.0 };
        var colours = field.RenderRays(origins, directions).ToArray();

        // Ray 1 passes above the cube: black.
        for (int a = 0; a < 3; a++) Assert.Equal(0.0, colours[3 + a]);

        // Ray 0 enters the 0.999 * 0.87 cube at y = -0.86913 and leaves at +0.86913.
        double bound = 0.999 * 0.87;
        double near = 1.9 - bound, far = 1.9 + bound;
        var positions = new double[samples * 3];
        for (int s = 0; s < samples; s++)
        {
            double mid = (s + 0.5) / samples;
            positions[(3 * s) + 1] = -1.9 + (near * (1 - mid)) + (far * mid);
            positions[(3 * s) + 2] = 0.1;
        }

        var q = field.Query(new Tensor<double>(new[] { samples, 3 }, new Vector<double>(positions))).ToArray();
        double transmittance = 1.0;
        var expected = new double[3];
        for (int s = 0; s < samples; s++)
        {
            double alpha = 1.0 - Math.Exp(-q[4 * s] / samples);   // the renderer's interval width is 1 / S
            for (int a = 0; a < 3; a++) expected[a] += transmittance * alpha * q[(4 * s) + 1 + a];
            transmittance *= 1.0 - alpha;
        }

        for (int a = 0; a < 3; a++) Assert.Equal(expected[a], colours[a], 1e-9);
    }

    // ---------------------------------------------------------------------------------------------
    // Determinism and input dependence.
    // ---------------------------------------------------------------------------------------------

    [Fact(Timeout = 120000)]
    public async Task SameSeed_GivesIdenticalWeightsAndFields()
    {
        await Task.Yield();
        var a = new TripoSRModel<double>(Tiny(seed: 11));
        var b = new TripoSRModel<double>(Tiny(seed: 11));
        var c = new TripoSRModel<double>(Tiny(seed: 12));
        var image = Image(5);

        Assert.Equal(Values(a.EncodeSceneCodes(image)), Values(b.EncodeSceneCodes(image)));
        Assert.Equal(Values(a.SampleDensityGrid(image, 5)), Values(b.SampleDensityGrid(image, 5)));
        Assert.NotEqual(a.GetParameters().ToArray(), c.GetParameters().ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task DifferentImages_GiveDifferentDensityGrids()
    {
        await Task.Yield();
        var model = new TripoSRModel<double>(Tiny());
        var first = Values(model.SampleDensityGrid(Image(21), 6));
        var second = Values(model.SampleDensityGrid(Image(22), 6));

        double largest = 0;
        for (int i = 0; i < first.Length; i++) largest = Math.Max(largest, Math.Abs(first[i] - second[i]));
        Assert.True(largest > 1e-6, $"The density grid did not depend on the image (largest difference {largest}).");
        Assert.All(first, value => Assert.True(value > 0 && !double.IsInfinity(value), $"exp density {value} is not positive and finite."));
    }

    // ---------------------------------------------------------------------------------------------
    // Geometry from the dense grid.
    // ---------------------------------------------------------------------------------------------

    private static double Median(double[] values)
    {
        var sorted = values.OrderBy(v => v).ToArray();
        return sorted[sorted.Length / 2];
    }

    [Fact(Timeout = 120000)]
    public async Task GeneratePointCloud_IsTheOccupiedGridSamplesWithDecodedColour()
    {
        await Task.Yield();
        var model = new TripoSRModel<double>(Tiny());
        var image = Image(8);
        var grid = Values(model.SampleDensityGrid(image));
        double threshold = Median(grid);
        int occupied = grid.Count(v => v > threshold);

        var cloud = model.GeneratePointCloud(image, threshold: threshold);
        Assert.Equal(new[] { 1, occupied, 6 }, cloud.Shape.ToArray());
        Assert.True(occupied > 0);

        // Every point sits on the linspace(-0.87, 0.87, 8) grid at an occupied sample, with the decoder's colour.
        var values = cloud.ToArray();
        double step = 2 * 0.87 / 7;
        var field = model.Reconstruct(image);
        var xyz = new double[occupied * 3];
        for (int p = 0; p < occupied; p++)
        {
            var index = new int[3];
            for (int a = 0; a < 3; a++)
            {
                double coordinate = values[(6 * p) + a];
                index[a] = (int)Math.Round((coordinate + 0.87) / step);
                Assert.Equal(-0.87 + (index[a] * step), coordinate, 1e-12);
                xyz[(3 * p) + a] = coordinate;
            }

            Assert.True(grid[(((index[0] * 8) + index[1]) * 8) + index[2]] > threshold);
        }

        var decoded = field.Query(new Tensor<double>(new[] { occupied, 3 }, new Vector<double>(xyz))).ToArray();
        for (int p = 0; p < occupied; p++)
            for (int k = 0; k < 3; k++)
                Assert.Equal(decoded[(4 * p) + 1 + k], values[(6 * p) + 3 + k], Tolerance);

        // A budget keeps the densest samples.
        var capped = model.GeneratePointCloud(image, maxPoints: 3, threshold: threshold);
        Assert.Equal(new[] { 1, Math.Min(3, occupied), 6 }, capped.Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task GenerateMesh_SeededImage_GivesNonEmptyMeshWithDecodedVertexColours()
    {
        await Task.Yield();
        var model = new TripoSRModel<double>(Tiny());
        var image = Image(9);
        double threshold = Median(Values(model.SampleDensityGrid(image)));

        var mesh = model.GenerateMesh(image, threshold: threshold);
        int vertices = mesh.Vertices.Shape[0];
        Assert.True(vertices > 0, "The mesh has no vertices.");
        Assert.True(mesh.Faces.GetLength(0) > 0, "The mesh has no faces.");
        Assert.NotNull(mesh.Colors);
        var colours = mesh.Colors ?? new Tensor<double>(new[] { 0, 3 });
        Assert.Equal(new[] { vertices, 3 }, colours.Shape.ToArray());

        // Vertex colours are the decoder's sigmoid colour at the vertices, and the vertices lie inside the cube.
        var decoded = model.Reconstruct(image).Query(mesh.Vertices).ToArray();
        var actual = colours.ToArray();
        var positions = mesh.Vertices.ToArray();
        for (int v = 0; v < vertices; v++)
        {
            for (int k = 0; k < 3; k++)
            {
                Assert.Equal(decoded[(4 * v) + 1 + k], actual[(3 * v) + k], Tolerance);
                Assert.InRange(actual[(3 * v) + k], 0.0, 1.0);
                Assert.InRange(positions[(3 * v) + k], -0.87 - 1e-9, 0.87 + 1e-9);
            }
        }

        // ImageTo3D at a threshold no sample reaches is an empty mesh, not placeholder geometry.
        var unreachable = Tiny();
        unreachable.DensityThreshold = 1e300;
        var empty = new TripoSRModel<double>(unreachable).ImageTo3D(image);
        Assert.Equal(0, empty.Vertices.Shape[0]);

        // ImageTo3D at the configured threshold is the same mesh GenerateMesh gives.
        var configured = Tiny();
        configured.DensityThreshold = threshold;
        var viaImageTo3D = new TripoSRModel<double>(configured).ImageTo3D(image);
        Assert.Equal(positions, viaImageTo3D.Vertices.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task TextEntryPoints_AreNotSupported()
    {
        await Task.Yield();
        var model = new TripoSRModel<double>(Tiny());
        Assert.Throws<NotSupportedException>(() => model.GenerateMesh("a chair"));
        Assert.Throws<NotSupportedException>(() => model.GeneratePointCloud("a chair"));
        Assert.Throws<NotSupportedException>(() => model.NoisePredictor);
        Assert.Throws<NotSupportedException>(() => model.VAE);
    }

    // ---------------------------------------------------------------------------------------------
    // Weight loading by the official state-dict names.
    // ---------------------------------------------------------------------------------------------

    [Fact(Timeout = 120000)]
    public async Task LoadTorchWeights_ReadsEveryWeightByItsOfficialName()
    {
        await Task.Yield();
        var shapes = new Dictionary<string, int[]>();
        double[] Read(string name, int[] shape)
        {
            shapes[name] = shape;
            long count = shape.Aggregate(1L, (x, y) => x * y);
            var values = new double[count];
            int hash = name.Aggregate(17, (h, ch) => unchecked((h * 31) + ch));
            for (int i = 0; i < values.Length; i++) values[i] = Math.Sin(hash + i) * 0.1;
            return values;
        }

        var a = new TripoSRModel<double>(Tiny(seed: 1));
        var b = new TripoSRModel<double>(Tiny(seed: 2));
        a.LoadTorchWeights(Read);
        b.LoadTorchWeights(Read);

        Assert.Equal(a.ParameterCount, shapes.Values.Sum(s => s.Aggregate(1L, (x, y) => x * y)));
        Assert.Equal(new[] { 1, 5, 8 }, shapes["image_tokenizer.model.embeddings.position_embeddings"]);
        Assert.Equal(new[] { 8, 3, 16, 16 }, shapes["image_tokenizer.model.embeddings.patch_embeddings.projection.weight"]);
        Assert.Equal(new[] { 8, 8 }, shapes["image_tokenizer.model.encoder.layer.0.attention.attention.query.weight"]);
        Assert.Equal(new[] { 32, 8 }, shapes["image_tokenizer.model.encoder.layer.0.intermediate.dense.weight"]);
        Assert.Equal(new[] { 3, 8, 2, 2 }, shapes["tokenizer.embeddings"]);
        Assert.Equal(new[] { 8 }, shapes["backbone.norm.weight"]);
        Assert.Equal(new[] { 8, 8 }, shapes["backbone.transformer_blocks.0.attn2.to_k.weight"]);
        Assert.Equal(new[] { 64, 8 }, shapes["backbone.transformer_blocks.0.ff.net.0.proj.weight"]);
        Assert.Equal(new[] { 8, 4, 2, 2 }, shapes["post_processor.upsample.weight"]);
        Assert.Equal(new[] { 4, 8 }, shapes["decoder.layers.4.weight"]);
        Assert.False(shapes.ContainsKey("backbone.transformer_blocks.0.attn1.to_q.bias"));

        // Every weight was overwritten: models built from different seeds now agree exactly.
        var image = Image(6);
        Assert.Equal(a.GetParameters().ToArray(), b.GetParameters().ToArray());
        Assert.Equal(Values(a.EncodeSceneCodes(image)), Values(b.EncodeSceneCodes(image)));
    }
}
