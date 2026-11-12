using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Metrics;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Metrics;

/// <summary>
/// Reference values for <see cref="CLIPScore{T}"/> (Hessel et al. 2021, torchmetrics CLIPScore: 100 * max(cos, 0);
/// RefCLIPScore: harmonic mean with max(max_r cos(c, r), 0)) and the LAION linear aesthetic head of
/// <see cref="AestheticScore{T}"/> (score = w . e / ||e|| + b).
/// </summary>
public class CLIPScoreReferenceTests
{
    private const double Tolerance = 1e-10;

    /// <summary>
    /// Embeds text and images by a fixed table; similarity is the cosine of the embeddings, as CLIP's is.
    /// </summary>
    private sealed class TableClip : IMultimodalEmbedding<double>
    {
        private readonly Dictionary<string, double[]> _text;
        private readonly Func<double[], double[]> _image;

        public TableClip(Dictionary<string, double[]> text, Func<double[], double[]> image)
        {
            _text = text;
            _image = image;
        }

        public int EmbeddingDimension => 3;
        public int MaxSequenceLength => 77;
        public int ImageSize => 1;

        public Vector<double> EncodeText(string text) => new Vector<double>(_text[text]);

        public Vector<double> EncodeImage(double[] imageData) => new Vector<double>(_image(imageData));

        public Matrix<double> EncodeTextBatch(IEnumerable<string> texts) =>
            Matrix<double>.FromRows(texts.Select(t => _text[t]).ToArray());

        public Matrix<double> EncodeImageBatch(IEnumerable<double[]> imageDataBatch) =>
            Matrix<double>.FromRows(imageDataBatch.Select(_image).ToArray());

        public double ComputeSimilarity(Vector<double> embedding1, Vector<double> embedding2) =>
            Cosine(embedding1.ToArray(), embedding2.ToArray());

        public Dictionary<string, double> ZeroShotClassify(double[] imageData, IEnumerable<string> labels) =>
            labels.ToDictionary(l => l, l => ComputeSimilarity(EncodeImage(imageData), EncodeText(l)));
    }

    private static double Cosine(double[] a, double[] b)
    {
        double dot = a.Zip(b, (x, y) => x * y).Sum();
        return dot / (Math.Sqrt(a.Sum(x => x * x)) * Math.Sqrt(b.Sum(x => x * x)));
    }

    private static readonly double[] ImageEmbedding = { 3.0, 4.0, 0.0 };

    private static TableClip CreateClip() => new TableClip(
        new Dictionary<string, double[]>
        {
            ["cat"] = new[] { 4.0, 3.0, 0.0 },
            ["opposite"] = new[] { -3.0, -4.0, 0.0 },
            ["ref a"] = new[] { 1.0, 0.0, 0.0 },
            ["ref b"] = new[] { 0.0, 1.0, 1.0 },
        },
        image => image[0] > 0 ? ImageEmbedding : new[] { 0.0, 0.0, 1.0 });

    [Fact(Timeout = 60000)]
    public async Task TextImageScore_Is100TimesClampedCosine()
    {
        await Task.Yield();
        var score = new CLIPScore<double>(CreateClip());

        // cos([3,4,0],[4,3,0]) = 24/25.
        Assert.Equal(100.0 * 24.0 / 25.0, score.ComputeTextImageScore(new[] { 1.0 }, "cat"), Tolerance);
        // Negative cosine clamps to zero.
        Assert.Equal(0.0, score.ComputeTextImageScore(new[] { 1.0 }, "opposite"), Tolerance);
    }

    [Fact(Timeout = 60000)]
    public async Task TextImageScoreBatch_IsMeanOfPairScores()
    {
        await Task.Yield();
        var score = new CLIPScore<double>(CreateClip());

        double result = score.ComputeTextImageScoreBatch(
            new List<double[]> { new[] { 1.0 }, new[] { 1.0 } }, new List<string> { "cat", "opposite" });

        Assert.Equal((96.0 + 0.0) / 2.0, result, Tolerance);
    }

    [Fact(Timeout = 60000)]
    public async Task CaptionScore_IsHarmonicMeanWithBestReferenceSimilarity()
    {
        await Task.Yield();
        var score = new CLIPScore<double>(CreateClip());

        double clipS = 100.0 * Cosine(ImageEmbedding, new[] { 4.0, 3.0, 0.0 });
        double refS = 100.0 * Math.Max(
            Cosine(new[] { 4.0, 3.0, 0.0 }, new[] { 1.0, 0.0, 0.0 }),
            Cosine(new[] { 4.0, 3.0, 0.0 }, new[] { 0.0, 1.0, 1.0 }));
        double expected = 2.0 * clipS * refS / (clipS + refS);

        double result = score.ComputeCaptionScore(new[] { 1.0 }, "cat", new[] { "ref a", "ref b" });

        Assert.Equal(expected, result, Tolerance);
    }

    [Fact(Timeout = 60000)]
    public async Task DirectionalSimilarity_MapsDirectionCosineToZeroToHundred()
    {
        await Task.Yield();
        var score = new CLIPScore<double>(CreateClip());

        // Image direction: [3,4,0] - [0,0,1] = [3,4,-1]; text direction: [4,3,0] - [1,0,0] = [3,3,0].
        double cos = Cosine(new[] { 3.0, 4.0, -1.0 }, new[] { 3.0, 3.0, 0.0 });

        double result = score.ComputeDirectionalSimilarity(new[] { -1.0 }, new[] { 1.0 }, "ref a", "cat");

        Assert.Equal((cos + 1.0) * 50.0, result, Tolerance);
    }

    [Fact(Timeout = 60000)]
    public async Task AestheticScore_TrainedWeights_IsLinearHeadOnNormalizedEmbedding()
    {
        await Task.Yield();
        var weights = new Vector<double>(new[] { 5.0, 10.0, 2.0 });
        var aesthetic = new AestheticScore<double>(CreateClip(), weights, aestheticBias: 0.5);

        // LAION predictor: w . (e / ||e||) + b = (15 + 40 + 0) / 5 + 0.5.
        Assert.Equal(11.5, aesthetic.Compute(new[] { 1.0 }), Tolerance);
    }

    [Fact(Timeout = 60000)]
    public async Task AestheticScore_ComputeBatchAndRank_FollowPerImageScores()
    {
        await Task.Yield();
        var weights = new Vector<double>(new[] { 1.0, 1.0, 0.0 });
        var aesthetic = new AestheticScore<double>(CreateClip(), weights);
        var images = new List<double[]> { new[] { -1.0 }, new[] { 1.0 } };

        var scores = aesthetic.ComputeBatch(images);

        Assert.Equal(0.0, scores[0], Tolerance);
        Assert.Equal(7.0 / 5.0, scores[1], Tolerance);
        Assert.Equal(new[] { 1, 0 }, aesthetic.RankByAesthetic(images));
    }
}
