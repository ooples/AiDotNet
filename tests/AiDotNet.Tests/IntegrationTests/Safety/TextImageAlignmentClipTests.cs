using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Safety;
using AiDotNet.Safety.Multimodal;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// The CLIP path of <see cref="TextImageAlignmentChecker{T}"/>: zero-shot comparison of the image embedding with its
/// caption and the unsafe-concept prompts.
/// </summary>
public class TextImageAlignmentClipTests
{
    /// <summary>
    /// A stand-in CLIP with a four-dimensional embedding space: one axis per unsafe concept and one for everything
    /// else. Images whose pixels average above 0.5 embed on the violence axis, others on the neutral axis.
    /// </summary>
    private sealed class FakeClip : IMultimodalEmbedding<double>
    {
        public int EmbeddingDimension => 4;
        public int MaxSequenceLength => 77;
        public int ImageSize => 2;

        public Vector<double> EncodeText(string text)
        {
            if (text.Contains("violence")) return Axis(0);
            if (text.Contains("sexually")) return Axis(1);
            if (text.Contains("hateful")) return Axis(2);
            return Axis(3);
        }

        public Vector<double> EncodeImage(double[] imageData) => Axis(imageData.Average() > 0.5 ? 0 : 3);

        public Matrix<double> EncodeTextBatch(IEnumerable<string> texts) => Stack(texts.Select(EncodeText).ToList());

        public Matrix<double> EncodeImageBatch(IEnumerable<double[]> imageDataBatch) => Stack(imageDataBatch.Select(EncodeImage).ToList());

        public double ComputeSimilarity(Vector<double> embedding1, Vector<double> embedding2)
        {
            double dot = 0;
            for (int i = 0; i < embedding1.Length; i++) dot += embedding1[i] * embedding2[i];
            return dot;
        }

        public Dictionary<string, double> ZeroShotClassify(double[] imageData, IEnumerable<string> labels) =>
            labels.ToDictionary(l => l, l => ComputeSimilarity(EncodeImage(imageData), EncodeText(l)));

        private static Vector<double> Axis(int axis)
        {
            var v = new Vector<double>(4);
            v[axis] = 1.0;
            return v;
        }

        private static Matrix<double> Stack(IReadOnlyList<Vector<double>> rows)
        {
            var m = new Matrix<double>(rows.Count, 4);
            for (int r = 0; r < rows.Count; r++)
                for (int c = 0; c < 4; c++) m[r, c] = rows[r][c];
            return m;
        }
    }

    private static Tensor<double> Image(double value) => new Tensor<double>(Enumerable.Repeat(value, 12).ToArray(), new[] { 3, 2, 2 });

    [Fact(Timeout = 60000)]
    public async Task SafeCaption_OnAViolentImage_IsReportedAsMisaligned()
    {
        await Task.Yield();
        var checker = new TextImageAlignmentChecker<double>(clipModel: new FakeClip());
        var finding = Assert.Single(checker.EvaluateAlignment("a cute puppy in a garden", Image(0.9)));
        Assert.Equal(SafetyCategory.Manipulated, finding.Category);
        Assert.True(finding.Confidence > 0.99, $"Zero-shot probability {finding.Confidence:F4}.");
        Assert.Contains("violence", finding.Description);
    }

    [Fact(Timeout = 60000)]
    public async Task MatchingCaption_AndHonestUnsafeCaption_AreNotReported()
    {
        await Task.Yield();
        var checker = new TextImageAlignmentChecker<double>(clipModel: new FakeClip());
        Assert.Empty(checker.EvaluateAlignment("a cute puppy in a garden", Image(0.1)));
        // The caption names the unsafe content itself, so caption and image agree.
        Assert.Empty(checker.EvaluateAlignment("a bloody fight with a knife", Image(0.9)));
    }

    [Fact(Timeout = 60000)]
    public async Task ImageOfTheWrongSize_IsRejected()
    {
        await Task.Yield();
        var checker = new TextImageAlignmentChecker<double>(clipModel: new FakeClip());
        var wrong = new Tensor<double>(new double[27], new[] { 3, 3, 3 });
        Assert.Throws<ArgumentException>(() => checker.EvaluateAlignment("a cute puppy", wrong));
    }
}
