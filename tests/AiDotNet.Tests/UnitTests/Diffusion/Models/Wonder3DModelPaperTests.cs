using AiDotNet.Diffusion.Attention;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Enums;
using AiDotNet.NeuralRadianceFields.NeuS;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Diffusion;
using AiDotNet.VisionLanguage.Interfaces;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Reference tests for Wonder3D (Long et al. 2023): the domain switcher, the multi-view and cross-domain attention,
/// the orthographic cameras and front-frame normals of the reconstruction, the outlier-dropping geometry-aware normal
/// loss, and the full image-to-3D path on a tiny configuration.
/// </summary>
public class Wonder3DModelPaperTests
{
    private const double Tolerance = 1e-9;

    private static Tensor<double> TensorOf(int[] shape, double[] values) => new(shape, new Vector<double>(values));

    private static double[] Pattern(int count, double scale, double frequency)
    {
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = scale * Math.Sin(frequency * (i + 1));
        return values;
    }

    private static double[] Values(Tensor<double> tensor)
    {
        var values = new double[tensor.Length];
        for (int i = 0; i < values.Length; i++) values[i] = tensor[i];
        return values;
    }

    /// <summary>A stand-in image encoder: a fixed projection of the image's mean, so the embedding follows it.</summary>
    private sealed class MeanImageEncoder : IVisualEncoder<double>
    {
        public MeanImageEncoder(int width) => EmbeddingDimension = width;

        public int EmbeddingDimension { get; }

        public int ImageSize => 0;

        public int ImageChannels => 3;

        public Tensor<double> EncodeImage(Tensor<double> image)
        {
            double mean = 0;
            for (int i = 0; i < image.Length; i++) mean += image[i];
            mean /= image.Length;
            var embedding = new double[EmbeddingDimension];
            for (int i = 0; i < embedding.Length; i++) embedding[i] = mean * Math.Cos(0.7 * (i + 1));
            return TensorOf(new[] { 1, EmbeddingDimension }, embedding);
        }
    }

    private static Wonder3DModel<double> TinyModel(int seed = 42)
    {
        var options = Wonder3DModelTests.TinyOptions(seed);
        return new Wonder3DModel<double>(options, imageEncoder: new MeanImageEncoder(options.CrossAttentionDim));
    }

    // Gives the U-Net fixed non-zero weights: its output convolution and the cross-domain projections start at
    // zero (as in diffusers and Wonder3D), so an untrained U-Net predicts zero noise whatever it is given.
    private static void RandomizeUNet(Wonder3DModel<double> model)
    {
        var unet = (UNetNoisePredictor<double>)model.NoisePredictor;
        int count = unet.GetParameters().Length;
        unet.SetParameters(new Vector<double>(Pattern(count, 0.05, 0.37)));
    }

    private static Tensor<double> Image(double phase)
    {
        const int size = 16;
        var values = new double[3 * size * size];
        for (int i = 0; i < values.Length; i++) values[i] = 0.5 + 0.4 * Math.Sin(phase + 0.13 * i);
        return TensorOf(new[] { 1, 3, size, size }, values);
    }

    // ------------------------------------------------------------------ domain switcher

    [Fact]
    public void DomainEncoding_IsSinThenCosOfCameraAndOneHot()
    {
        var normal = Wonder3DDomainEmbedding.Encode(0.0, -0.2362, 0.8125, Wonder3DDomain.Normal);
        var color = Wonder3DDomainEmbedding.Encode(0.0, -0.2362, 0.8125, Wonder3DDomain.Color);
        var expectedNormal = new[]
        {
            0.0, Math.Sin(-0.2362), Math.Sin(0.8125), Math.Sin(1.0), 0.0,
            1.0, Math.Cos(-0.2362), Math.Cos(0.8125), Math.Cos(1.0), 1.0,
        };
        var expectedColor = new[]
        {
            0.0, Math.Sin(-0.2362), Math.Sin(0.8125), 0.0, Math.Sin(1.0),
            1.0, Math.Cos(-0.2362), Math.Cos(0.8125), 1.0, Math.Cos(1.0),
        };
        Assert.Equal(10, normal.Length);
        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(expectedNormal[i], normal[i], Tolerance);
            Assert.Equal(expectedColor[i], color[i], Tolerance);
        }
    }

    [Fact]
    public void DomainSwitcher_GivesEachDomainItsOwnEmbedding()
    {
        var model = TinyModel();
        int d = ((UNetNoisePredictor<double>)model.NoisePredictor).TimeEmbeddingDim;
        var embeddings = model.ClassEmbeddings(1);
        Assert.Equal(new[] { 12, d }, embeddings.Shape.ToArray());

        // Linear(10, d) and Linear(d, d), each with a bias.
        Assert.Equal(10 * d + d, model.DomainLinear1.GetParameters().Length);
        Assert.Equal(d * d + d, model.DomainLinear2.GetParameters().Length);

        var values = Values(embeddings);
        for (int view = 0; view < 6; view++)
        {
            double difference = 0;
            for (int j = 0; j < d; j++)
                difference = Math.Max(difference, Math.Abs(values[view * d + j] - values[(6 + view) * d + j]));
            Assert.True(difference > 1e-6, $"View {view}'s normal and color embeddings coincide.");
        }

        // Views of one domain differ by their cameras.
        double cameraDifference = 0;
        for (int j = 0; j < d; j++) cameraDifference = Math.Max(cameraDifference, Math.Abs(values[j] - values[d + j]));
        Assert.True(cameraDifference > 1e-6);
    }

    [Fact]
    public void DomainSwitcher_MatchesLinearSiluLinearByHand()
    {
        var model = TinyModel();
        int d = ((UNetNoisePredictor<double>)model.NoisePredictor).TimeEmbeddingDim;
        model.ClassEmbeddings(1);
        var w1 = Pattern(10 * d + d, 0.3, 0.41);
        var w2 = Pattern(d * d + d, 0.2, 0.29);
        model.DomainLinear1.SetParameters(new Vector<double>(w1));
        model.DomainLinear2.SetParameters(new Vector<double>(w2));

        var encoded = Wonder3DDomainEmbedding.Encode(0.0, 0.5220, 3.1406, Wonder3DDomain.Color);
        var actual = Values(model.EmbedDomains(TensorOf(new[] { 1, 10 }, encoded)));

        // DenseLayer packs its [in, out] weights row-major (x · W), then the bias.
        var hidden = new double[d];
        for (int o = 0; o < d; o++)
        {
            double sum = w1[10 * d + o];
            for (int i = 0; i < 10; i++) sum += w1[i * d + o] * encoded[i];
            hidden[o] = sum / (1.0 + Math.Exp(-sum));
        }

        for (int o = 0; o < d; o++)
        {
            double sum = w2[d * d + o];
            for (int i = 0; i < d; i++) sum += w2[i * d + o] * hidden[i];
            Assert.Equal(sum, actual[o], 1e-9);
        }
    }

    // ------------------------------------------------------------------ attention

    [Fact]
    public void CrossDomainAttention_MatchesHandReference()
    {
        // One view: a normal half and a color half, 2 channels, 1x2 tokens each.
        var inner = new DiffusionAttention<double>(2, 1, 2);
        var block = new Wonder3DAttentionBlock<double>(inner, 2, 1, 1, seed: 3);
        var x = new[] { 0.3, -0.8, 1.1, 0.4, -0.5, 0.9, 0.2, -1.3 }; // [2, 2, 1, 2]: n, c, w
        var input = TensorOf(new[] { 2, 2, 1, 2 }, x);
        var untouched = Values(block.PostProcess(input));
        for (int i = 0; i < x.Length; i++)
            Assert.Equal(x[i], untouched[i], Tolerance); // zero-initialized output projection

        var p = Pattern(4 * 4 + 2, 0.6, 0.53);
        block.CrossDomainAttention.SetParameters(new Vector<double>(p));
        var actual = Values(block.PostProcess(input));

        // Tokens in sequence order: normal w0, normal w1, color w0, color w1, each LayerNorm'd over channels.
        var tokens = new double[4][];
        for (int n = 0; n < 2; n++)
            for (int w = 0; w < 2; w++)
            {
                double a = x[(n * 2) * 2 + w], b = x[(n * 2 + 1) * 2 + w];
                double mean = (a + b) / 2, variance = ((a - mean) * (a - mean) + (b - mean) * (b - mean)) / 2;
                double scale = 1.0 / Math.Sqrt(variance + 1e-5);
                tokens[n * 2 + w] = new[] { (a - mean) * scale, (b - mean) * scale };
            }

        double[] Project(double[] row, int offset) => new[]
        {
            row[0] * p[offset] + row[1] * p[offset + 2],
            row[0] * p[offset + 1] + row[1] * p[offset + 3],
        };

        var q = tokens.Select(t => Project(t, 0)).ToArray();
        var k = tokens.Select(t => Project(t, 4)).ToArray();
        var v = tokens.Select(t => Project(t, 8)).ToArray();
        for (int s = 0; s < 4; s++)
        {
            var logits = new double[4];
            for (int t = 0; t < 4; t++) logits[t] = (q[s][0] * k[t][0] + q[s][1] * k[t][1]) / Math.Sqrt(2.0);
            double max = logits.Max();
            var weights = logits.Select(l => Math.Exp(l - max)).ToArray();
            double total = weights.Sum();
            var attended = new double[2];
            for (int t = 0; t < 4; t++)
                for (int c = 0; c < 2; c++) attended[c] += weights[t] / total * v[t][c];
            var output = Project(attended, 12);
            int n = s / 2, w = s % 2;
            for (int c = 0; c < 2; c++)
            {
                int index = (n * 2 + c) * 2 + w;
                Assert.Equal(x[index] + output[c] + p[16 + c], actual[index], 1e-9);
            }
        }
    }

    [Fact]
    public void CrossDomainAttention_PairsEachViewWithItsOwnOtherDomainView()
    {
        var inner = new DiffusionAttention<double>(4, 1, 2);
        var block = new Wonder3DAttentionBlock<double>(inner, 4, 2, 1, seed: 5);
        var x = Pattern(4 * 4 * 2 * 2, 0.8, 0.31); // [N0, N1, C0, C1]
        block.PostProcess(TensorOf(new[] { 4, 4, 2, 2 }, x));
        int count = block.CrossDomainAttention.GetParameters().Length;
        block.CrossDomainAttention.SetParameters(new Vector<double>(Pattern(count, 0.4, 0.23)));

        var baseline = Values(block.PostProcess(TensorOf(new[] { 4, 4, 2, 2 }, x)));
        var perturbed = (double[])x.Clone();
        int perView = 16;
        perturbed[3 * perView + 5] += 0.7; // color view 1
        var changed = Values(block.PostProcess(TensorOf(new[] { 4, 4, 2, 2 }, perturbed)));

        double MaxDifference(int view)
        {
            double m = 0;
            for (int i = view * perView; i < (view + 1) * perView; i++) m = Math.Max(m, Math.Abs(baseline[i] - changed[i]));
            return m;
        }

        Assert.Equal(0.0, MaxDifference(0), Tolerance); // normal view 0 sees only color view 0
        Assert.True(MaxDifference(1) > 1e-6);           // normal view 1 sees color view 1
        Assert.Equal(0.0, MaxDifference(2), Tolerance); // color view 0 sees only normal view 0
    }

    [Fact]
    public void MultiViewAttention_SharesKeysAcrossViewsOfOneDomainOnly()
    {
        var inner = new DiffusionAttention<double>(4, 1, 2);
        var x = Pattern(4 * 4 * 2 * 2, 0.8, 0.17); // [N0, N1, C0, C1]
        inner.ForwardJointTokens(TensorOf(new[] { 4, 4, 2, 2 }, x), 2);
        int count = inner.GetParameters().Length;
        inner.SetParameters(new Vector<double>(Pattern(count, 0.4, 0.43)));

        var baseline = Values(inner.ForwardJointTokens(TensorOf(new[] { 4, 4, 2, 2 }, x), 2));
        var perturbed = (double[])x.Clone();
        int perView = 16;
        perturbed[1 * perView + 2] += 0.9; // normal view 1
        var changed = Values(inner.ForwardJointTokens(TensorOf(new[] { 4, 4, 2, 2 }, perturbed), 2));

        double MaxDifference(int view)
        {
            double m = 0;
            for (int i = view * perView; i < (view + 1) * perView; i++) m = Math.Max(m, Math.Abs(baseline[i] - changed[i]));
            return m;
        }

        Assert.True(MaxDifference(0) > 1e-6);           // normal view 0 attends to normal view 1
        Assert.Equal(0.0, MaxDifference(2), Tolerance); // the color views do not
        Assert.Equal(0.0, MaxDifference(3), Tolerance);
    }

    [Fact]
    public void UNet_WrapsEverySelfAttentionBlock()
    {
        var model = TinyModel();
        Assert.True(model.AttentionBlockCount > 0);
    }

    // ------------------------------------------------------------------ orthographic cameras and normals

    [Fact]
    public void OrthographicCamera_CastsParallelRaysFromAPixelGrid()
    {
        const int size = 4;
        var camera = NeuSCamera.Orthographic(new[] { 0.0, -1.3, 0.0 }, new[] { 0.0, 0.0, 0.0 }, new[] { 0.0, 0.0, 1.0 },
            1.0, size, size);
        var origin = new double[3];
        var direction = new double[3];
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                camera.Ray(x, y, origin, direction);
                // Pixel centres at ((i + 0.5) / W - 0.5) * 2: right is world +x, image down is world -z.
                double u = ((x + 0.5) / size - 0.5) * 2, v = ((y + 0.5) / size - 0.5) * 2;
                Assert.Equal(u, origin[0], Tolerance);
                Assert.Equal(-1.3, origin[1], Tolerance);
                Assert.Equal(-v, origin[2], Tolerance);
                Assert.Equal(0.0, direction[0], Tolerance);
                Assert.Equal(1.0, direction[1], Tolerance);
                Assert.Equal(0.0, direction[2], Tolerance);

                var (pu, pv, depth) = camera.Project(origin[0] + 0.8 * direction[0], origin[1] + 0.8 * direction[1],
                    origin[2] + 0.8 * direction[2]);
                Assert.Equal(x, pu, 1e-9);
                Assert.Equal(y, pv, 1e-9);
                Assert.Equal(0.8, depth, 1e-9);
            }
    }

    [Fact]
    public void ViewCameras_CircleTheObjectAtTheReleasedAzimuths()
    {
        var model = TinyModel();
        var azimuths = new[] { 0.0, 45.0, 90.0, 180.0, 270.0, 315.0 };
        for (int view = 0; view < 6; view++)
        {
            var camera = model.ViewCamera(view, 8, 8);
            double a = azimuths[view] * Math.PI / 180.0;
            Assert.Equal(1.3 * Math.Sin(a), camera.Center[0], 1e-9);
            Assert.Equal(-1.3 * Math.Cos(a), camera.Center[1], 1e-9);
            Assert.Equal(0.0, camera.Center[2], 1e-9);
            var direction = new double[3];
            camera.RayDirection(3, 3, direction);
            Assert.Equal(-Math.Sin(a), direction[0], 1e-9);
            Assert.Equal(Math.Cos(a), direction[1], 1e-9);
        }
    }

    [Fact]
    public void FrontFrameNormals_RotateToWorld()
    {
        var model = TinyModel();
        var front = model.ViewCamera(0, 8, 8).Rotation;
        // OpenGL front-camera axes: +z faces the viewer (world -y), +y is up (world +z), +x is right (world +x).
        var toward = Wonder3DModel<double>.FrontNormalToWorld(front, 0, 0, 1);
        var up = Wonder3DModel<double>.FrontNormalToWorld(front, 0, 1, 0);
        var right = Wonder3DModel<double>.FrontNormalToWorld(front, 2, 0, 0);
        Assert.Equal(new[] { 0.0, -1.0, 0.0 }, toward.Select(c => Math.Round(c, 12) + 0.0).ToArray());
        Assert.Equal(new[] { 0.0, 0.0, 1.0 }, up.Select(c => Math.Round(c, 12) + 0.0).ToArray());
        Assert.Equal(new[] { 1.0, 0.0, 0.0 }, right.Select(c => Math.Round(c, 12) + 0.0).ToArray());
    }

    // ------------------------------------------------------------------ outlier-dropping normal loss

    [Fact]
    public void KeepSmallest_KeepsIntRatioTimesCandidatesSmallestErrors()
    {
        var errors = new[] { 0.5, 0.1, 0.9, 0.3, 0.7 };
        var candidates = new[] { 1.0, 1.0, 1.0, 0.0, 1.0 };
        // int(0.8 * 4) = 3 of the four candidates: 0.1, 0.5, 0.7; the excluded 0.3 stays out, the outlier 0.9 drops.
        var keep = NeuSReconstructor<double>.KeepSmallest(errors, candidates, 0.8, out int kept);
        Assert.Equal(3, kept);
        Assert.Equal(new[] { 1.0, 1.0, 0.0, 0.0, 1.0 }, keep);

        var ties = NeuSReconstructor<double>.KeepSmallest(new[] { 0.2, 0.2, 0.2 }, new[] { 1.0, 1.0, 1.0 }, 0.5, out int tied);
        Assert.Equal(1, tied);
        Assert.Equal(new[] { 1.0, 0.0, 0.0 }, ties);

        var all = NeuSReconstructor<double>.KeepSmallest(errors, candidates, 1.0, out int everything);
        Assert.Equal(4, everything);
        Assert.Equal(candidates, all);
    }

    [Fact]
    public void NormalSupervision_DropsGrazingRaysAndWeightsByExpAbsCosine()
    {
        var directions = new[] { 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0 };
        var normals = new[] { 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.6, 0.8, 0.0, -1.0, 0.0 };
        var mask = new[] { 1.0, 1.0, 1.0, 0.0 };
        var (rays, weights) = NeuSReconstructor<double>.NormalSupervision(directions, normals, mask, -0.1, true);
        Assert.Equal(new[] { 1.0, 0.0, 1.0, 0.0 }, rays);
        Assert.NotNull(weights);
        if (weights is null) return;
        double total = 2 * Math.E + 1.0 + Math.Exp(0.6);
        Assert.Equal(Math.E / total, weights[0], Tolerance);
        Assert.Equal(1.0 / total, weights[1], Tolerance); // cos 0 > -0.1, zeroed
        Assert.Equal(Math.Exp(0.6) / total, weights[2], Tolerance);
        Assert.Equal(Math.E / total, weights[3], Tolerance);

        var (plain, none) = NeuSReconstructor<double>.NormalSupervision(directions, normals, mask, null, false);
        Assert.Equal(new[] { 1.0, 1.0, 1.0, 0.0 }, plain);
        Assert.Null(none);
    }

    // ------------------------------------------------------------------ the full model

    [Fact]
    public void GenerateViews_IsSeededAndFollowsTheInput()
    {
        var first = TinyModel();
        var second = TinyModel();
        Assert.Equal(first.GetParameters().ToArray(), second.GetParameters().ToArray());
        RandomizeUNet(first);
        RandomizeUNet(second);

        var image = Image(0.0);
        var a = first.GenerateViews(image, seed: 7);
        var b = second.GenerateViews(image, seed: 7);
        Assert.Equal(new[] { 12, 3, 16, 16 }, a.Shape.ToArray());
        Assert.Equal(Values(a), Values(b));
        Assert.All(Values(a), value => Assert.InRange(value, 0.0, 1.0));

        var other = Values(first.GenerateViews(Image(1.7), seed: 7));
        Assert.Contains(Values(a).Zip(other, (p, q) => Math.Abs(p - q)), difference => difference > 1e-9);
    }

    [Fact]
    public void ImageConditioning_WithoutAnImageEncoderExplainsWhatIsNeeded()
    {
        var model = new Wonder3DModel<double>(Wonder3DModelTests.TinyOptions());
        var error = Assert.Throws<InvalidOperationException>(() => model.GenerateViews(Image(0.0)));
        Assert.Contains("CLIP", error.Message);
    }

    [Fact]
    public void ReconstructionViews_CarryCamerasMasksWorldNormalsAndViewWeights()
    {
        var model = TinyModel();
        const int size = 4;
        var values = new double[12 * 3 * size * size];
        for (int i = 0; i < values.Length; i++) values[i] = 1.0; // white background everywhere
        // Front view, pixel (1, 2): an OpenGL normal facing the viewer, (0, 0, 1) encoded as (0.5, 0.5, 1).
        int pixel = 2 * size + 1;
        values[(0 * size * size) + pixel] = 0.5;
        values[(1 * size * size) + pixel] = 0.5;
        values[(2 * size * size) + pixel] = 1.0;
        // Its color (front color view is index 6).
        values[((6 * 3 + 0) * size * size) + pixel] = 0.25;
        values[((6 * 3 + 1) * size * size) + pixel] = 0.5;
        values[((6 * 3 + 2) * size * size) + pixel] = 0.75;

        var views = model.ReconstructionViews(TensorOf(new[] { 12, 3, size, size }, values));
        Assert.Equal(6, views.Count);
        var expectedWeights = new[] { 1.0, 0.8, 0.2, 1.0, 0.4, 0.7 };
        for (int k = 0; k < 6; k++) Assert.Equal(expectedWeights[k], views[k].Weight, Tolerance);

        var front = views[0];
        var mask = front.Mask;
        var normals = front.Normals;
        Assert.NotNull(mask);
        Assert.NotNull(normals);
        if (mask is null || normals is null) return;
        Assert.Equal(1.0, mask[2, 1], Tolerance);
        Assert.Equal(0.0, mask[0, 0], Tolerance);
        Assert.Equal(0.0, normals[2, 1, 0], 1e-12);
        Assert.Equal(-1.0, normals[2, 1, 1], 1e-12);
        Assert.Equal(0.0, normals[2, 1, 2], 1e-12);
        Assert.Equal(0.25, front.Image[2, 1, 0], Tolerance);
        Assert.Equal(0.75, front.Image[2, 1, 2], Tolerance);
        Assert.Equal(0.0, front.Image[0, 0, 0], Tolerance); // composited onto black
    }

    [Fact]
    public void ImageTo3D_ProducesAColoredMeshAndPointCloud()
    {
        var model = TinyModel();
        RandomizeUNet(model);
        var mesh = model.ImageTo3D(Image(0.3), numInferenceSteps: 2, seed: 11);
        Assert.True(mesh.Vertices.Shape[0] > 0, "The reconstruction has no surface.");
        Assert.Equal(3, mesh.Vertices.Shape[1]);
        var colors = mesh.Colors;
        Assert.NotNull(colors);
        if (colors is null) return;
        Assert.Equal(mesh.Vertices.Shape[0], colors.Shape[0]);

        var cloud = model.GeneratePointCloudFromImage(Image(0.3), numPoints: 64, numInferenceSteps: 2, seed: 11);
        Assert.Equal(new[] { 1, 64, 6 }, cloud.Shape.ToArray());
        Assert.All(Values(cloud), value => Assert.False(double.IsNaN(value)));
    }

    [Fact]
    public void TextOverloads_AreNotSupported()
    {
        var model = TinyModel();
        Assert.Throws<NotSupportedException>(() => model.GenerateMesh("an owl"));
        Assert.Throws<NotSupportedException>(() => model.GeneratePointCloud("an owl"));
    }
}
