using System;
using System.Linq;
using AiDotNet.ComputerVision.Segmentation.Common;
using AiDotNet.ComputerVision.Segmentation.OpenVocabulary;
using AiDotNet.ComputerVision.Segmentation.Referring;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Hand-derived references for the text-queried segmentation path: pixel embeddings scored against
/// CLIP-style text embeddings by cosine similarity over a temperature (open-vocabulary: softmax over
/// names; referring: sigmoid mask over one expression).
/// </summary>
public class TextQueriedSegmentationReferenceTests
{
    private const double Tolerance = 1e-9;

    private static Tensor<double> T(int[] shape, params double[] values)
        => new Tensor<double>(shape, new Vector<double>(values));

    [Fact]
    public void Classify_IsCosineSimilarityOverTemperature()
    {
        // Pixel embeddings [D=2, H=1, W=2]: p0 = (3, 4) -> (0.6, 0.8); p1 = (1, 0).
        var pixels = T(new[] { 2, 1, 2 }, 3, 1, 4, 0);
        // Text rows: t0 = (1, 0); t1 = (0, 2) -> (0, 1).
        var text = TextEmbeddingClassifier.NormalizeRows(T(new[] { 2, 2 }, 1, 0, 0, 2));

        var logits = TextEmbeddingClassifier.Classify(pixels, text, 100.0);

        Assert.Equal(new[] { 2, 1, 2 }, logits.Shape.ToArray());
        Assert.Equal(60.0, logits[0, 0, 0], 1e-6);   // 100 * 0.6
        Assert.Equal(100.0, logits[0, 0, 1], 1e-6);  // 100 * 1
        Assert.Equal(80.0, logits[1, 0, 0], 1e-6);   // 100 * 0.8
        Assert.Equal(0.0, logits[1, 0, 1], 1e-6);    // 100 * 0
    }

    [Fact]
    public void Classify_BatchedInput_KeepsTheBatchAxis()
    {
        var pixels = T(new[] { 1, 2, 1, 2 }, 3, 1, 4, 0);
        var text = TextEmbeddingClassifier.NormalizeRows(T(new[] { 1, 2 }, 0, 5));

        var logits = TextEmbeddingClassifier.Classify(pixels, text, 2.0);

        Assert.Equal(new[] { 1, 1, 1, 2 }, logits.Shape.ToArray());
        Assert.Equal(1.6, logits[0, 0, 0, 0], 1e-9); // 2 * 0.8
        Assert.Equal(0.0, logits[0, 0, 0, 1], 1e-9);
    }

    [Fact]
    public void Classify_RejectsAPixelWidthThatIsNotTheTextWidth()
    {
        var pixels = new Tensor<double>(new[] { 3, 2, 2 });
        var text = TextEmbeddingClassifier.NormalizeRows(T(new[] { 1, 2 }, 1, 0));
        Assert.Throws<ArgumentException>(() => TextEmbeddingClassifier.Classify(pixels, text, 1.0));
    }

    [Fact]
    public void OpenVocabularyResult_IsSoftmaxArgmaxWithMeanProbabilityScores()
    {
        // Logits [K=3, H=1, W=3]. Pixel 0: (ln 3, 0, -100) -> p = (3/4, 1/4, ~0), label 0.
        // Pixel 1: (0, ln 4, -100) -> (1/5, 4/5, ~0), label 1. Pixel 2: (ln 2, 0, -100) -> (2/3, 1/3, ~0), label 0.
        var logits = T(new[] { 3, 1, 3 },
            Math.Log(3), 0, Math.Log(2),
            0, Math.Log(4), 0,
            -100, -100, -100);
        var numOps = MathHelper.GetNumericOperations<double>();

        var result = OpenVocabSegmentationBase<double>.BuildResult(AiDotNetEngine.Current, numOps, logits, new[] { "a", "b", "c" });

        Assert.Equal(new[] { 3, 1, 3 }, result.Masks.Shape.ToArray());
        Assert.Equal(new[] { 1.0, 0.0, 1.0 }, new[] { result.Masks[0, 0, 0], result.Masks[0, 0, 1], result.Masks[0, 0, 2] });
        Assert.Equal(new[] { 0.0, 1.0, 0.0 }, new[] { result.Masks[1, 0, 0], result.Masks[1, 0, 1], result.Masks[1, 0, 2] });
        Assert.Equal(new[] { 0.0, 0.0, 0.0 }, new[] { result.Masks[2, 0, 0], result.Masks[2, 0, 1], result.Masks[2, 0, 2] });
        Assert.Equal((0.75 + 2.0 / 3.0) / 2.0, result.Scores[0], Tolerance);
        Assert.Equal(0.8, result.Scores[1], Tolerance);
        Assert.Equal(0.0, result.Scores[2], Tolerance); // a name that wins no pixel scores zero
        Assert.Equal(new[] { 0.0, 1.0, 0.0 }, new[] { result.SemanticMap[0, 0], result.SemanticMap[0, 1], result.SemanticMap[0, 2] });
        Assert.Equal(new[] { "a", "b", "c" }, result.ClassNames);
    }

    [Fact]
    public void ReferringResult_ThresholdsTheSigmoidStrictlyAndBoxesTheMask()
    {
        // Logits [1, H=2, W=3]: sigmoid -> [[0.5, 0.75, 0.25], [0.8, 0.2689, 0.5]].
        // LISA keeps logit > 0, i.e. probability > 0.5, so the two 0.5 pixels are out.
        var logits = T(new[] { 1, 2, 3 }, 0, Math.Log(3), -Math.Log(3), Math.Log(4), -1, 0);
        var numOps = MathHelper.GetNumericOperations<double>();

        var result = ReferringSegmentationBase<double>.BuildResult(AiDotNetEngine.Current, numOps, logits, 0.5, "the cat");

        var mask = result.Masks;
        Assert.Equal(new[] { 0.0, 1.0, 0.0, 1.0, 0.0, 0.0 },
            new[] { mask[0, 0, 0], mask[0, 0, 1], mask[0, 0, 2], mask[0, 1, 0], mask[0, 1, 1], mask[0, 1, 2] });
        Assert.Equal((0.75 + 0.8) / 2.0, result.Confidence, Tolerance);
        // Pixels (y=0, x=1) and (y=1, x=0): XYXY box (0, 0, 1, 1), inclusive like torchvision.masks_to_boxes.
        Assert.Equal(new[] { 0.0, 0.0, 1.0, 1.0 },
            new[] { result.BoundingBoxes[0, 0], result.BoundingBoxes[0, 1], result.BoundingBoxes[0, 2], result.BoundingBoxes[0, 3] });
        Assert.Contains("with 2 pixels", result.TextResponse);
    }

    [Fact]
    public void ReferringResult_EmptyMask_HasZeroConfidenceAndBox()
    {
        var logits = T(new[] { 1, 1, 2 }, -2, -3);
        var numOps = MathHelper.GetNumericOperations<double>();

        var result = ReferringSegmentationBase<double>.BuildResult(AiDotNetEngine.Current, numOps, logits, 0.5, "nothing");

        Assert.Equal(0.0, result.Confidence, Tolerance);
        Assert.Equal(new[] { 0.0, 0.0, 0.0, 0.0 },
            new[] { result.BoundingBoxes[0, 0], result.BoundingBoxes[0, 1], result.BoundingBoxes[0, 2], result.BoundingBoxes[0, 3] });
        Assert.StartsWith("No region found", result.TextResponse);
    }

    private static NeuralNetworkArchitecture<double> Arch()
        => new(InputType.ThreeDimensional, NeuralNetworkTaskType.MultiClassClassification,
               NetworkComplexity.Deep, 0, 64, 64, 3, 3);

    [Fact(Timeout = 120000)]
    public void SegmentWithText_ScoresTheEncodersEmbeddingsNotAHashOfTheText()
    {
        // Two encoders that swap the embeddings of "cat" and "dog" must swap the two names' maps:
        // the model only ever sees the encoder's vectors, never the characters of the names.
        var a = new DeterministicTextEncoder<double>(4).With("cat", 1, 0, 0, 0).With("dog", 0, 1, 0, 0);
        var b = new DeterministicTextEncoder<double>(4).With("cat", 0, 1, 0, 0).With("dog", 1, 0, 0, 0);
        var options = new CATSegOptions { NumClasses = 3, DropRate = 0.0, Seed = 7 };
        using var modelA = new CATSeg<double>(Arch(), a, options: new CATSegOptions(options));
        using var modelB = new CATSeg<double>(Arch(), b, options: new CATSegOptions(options));
        var image = new Tensor<double>(new[] { 3, 64, 64 });
        var rng = RandomHelper.CreateSeededRandom(3);
        for (int i = 0; i < image.Length; i++) image[i] = rng.NextDouble();

        int before = a.EncodeCount;
        var ra = modelA.SegmentWithText(image, new[] { "cat", "dog" });
        var rb = modelB.SegmentWithText(image, new[] { "cat", "dog" });

        Assert.Equal(before + 2, a.EncodeCount);
        Assert.Equal(ra.Scores[0], rb.Scores[1], 1e-9);
        Assert.Equal(ra.Scores[1], rb.Scores[0], 1e-9);
        int h = ra.SemanticMap.Shape[0], w = ra.SemanticMap.Shape[1];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                Assert.Equal(1.0, ra.Masks[0, y, x] + ra.Masks[1, y, x], Tolerance); // each pixel has one name
                Assert.Equal(ra.Masks[0, y, x], rb.Masks[1, y, x], Tolerance);
            }
    }

    [Fact(Timeout = 120000)]
    public void Predict_HasOneChannelPerClosedSetName_AndTemperatureScalesTheLogits()
    {
        var encoder = new DeterministicTextEncoder<double>(8);
        using var sharp = new LISA<double>(Arch(), encoder,
            options: new LISAOptions { NumClasses = 3, DropRate = 0.0, Seed = 11, Temperature = 0.01 });
        using var soft = new LISA<double>(Arch(), encoder,
            options: new LISAOptions { NumClasses = 3, DropRate = 0.0, Seed = 11, Temperature = 0.02 });
        var image = new Tensor<double>(new[] { 3, 64, 64 });
        var rng = RandomHelper.CreateSeededRandom(5);
        for (int i = 0; i < image.Length; i++) image[i] = rng.NextDouble();

        var a = sharp.Predict(image);
        var b = soft.Predict(image);

        Assert.Equal(3, a.Shape[0]);
        for (int i = 0; i < a.Length; i++)
        {
            Assert.InRange(Math.Abs(a[i]), 0.0, 100.0 + 1e-9); // |cos| <= 1, scaled by 1 / 0.01
            Assert.Equal(a[i], 2.0 * b[i], 1e-9);
        }
    }

    [Fact]
    public void ClassNames_MustMatchNumClasses()
    {
        var encoder = new DeterministicTextEncoder<double>(4);
        Assert.Throws<ArgumentException>(() => new SAN<double>(Arch(), encoder,
            options: new SANOptions { NumClasses = 3, ClassNames = new[] { "sky", "road" } }));
    }
}
