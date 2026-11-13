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

    private static Tensor<Half> H(int[] shape, params double[] values)
        => new Tensor<Half>(shape, new Vector<Half>(Array.ConvertAll(values, v => (Half)(float)v)));

    private static Tensor<Half> HalfFilled(int[] shape, double value)
    {
        int total = 1;
        foreach (int s in shape) total *= s;
        return new Tensor<Half>(shape, new Vector<Half>(Enumerable.Repeat((Half)(float)value, total).ToArray()));
    }

    [Fact]
    public void OpenVocabularyResult_SingleName_IsASigmoidMaskNotTheWholeImage()
    {
        // A softmax over one name is 1 at every pixel; SegmentWithPrompt's single name is decided by its sigmoid.
        // Logits (ln 3, -1, 0) -> sigmoid (0.75, 0.2689, 0.5): only the first pixel is strictly above 0.5.
        var logits = T(new[] { 1, 1, 3 }, Math.Log(3), -1, 0);
        var numOps = MathHelper.GetNumericOperations<double>();

        var result = OpenVocabSegmentationBase<double>.BuildResult(AiDotNetEngine.Current, numOps, logits, new[] { "a cat" });

        Assert.Equal(new[] { 1.0, 0.0, 0.0 }, new[] { result.Masks[0, 0, 0], result.Masks[0, 0, 1], result.Masks[0, 0, 2] });
        Assert.Equal(0.75, result.Scores[0], Tolerance);
        Assert.Equal(new[] { 0.0, -1.0, -1.0 }, new[] { result.SemanticMap[0, 0], result.SemanticMap[0, 1], result.SemanticMap[0, 2] });
        Assert.Equal(new[] { "a cat" }, result.ClassNames);
    }

    [Fact]
    public void OpenVocabularyResult_Half_ScoresStayExactBeyondTheHalfRange()
    {
        // 300 x 300 = 90000 pixels, all won by name 0 with probability 3/4: a Half sum of the mask or of the
        // probabilities overflows (Half's largest finite value is 65504), which zeroed or corrupted the scores.
        var first = HalfFilled(new[] { 1, 300, 300 }, Math.Log(3));
        var second = HalfFilled(new[] { 1, 300, 300 }, 0);
        var logits = AiDotNetEngine.Current.TensorConcatenate(new[] { first, second }, 0);
        var numOps = MathHelper.GetNumericOperations<Half>();

        var result = OpenVocabSegmentationBase<Half>.BuildResult(AiDotNetEngine.Current, numOps, logits, new[] { "sky", "road" });

        Assert.Equal(0.75, (double)(float)result.Scores[0], 2e-3);
        Assert.Equal(0.0, (double)(float)result.Scores[1], Tolerance);
    }

    [Fact]
    public void ReferringResult_Half_CountsTheAreaInInt()
    {
        var logits = HalfFilled(new[] { 1, 300, 300 }, Math.Log(3));
        var numOps = MathHelper.GetNumericOperations<Half>();

        var result = ReferringSegmentationBase<Half>.BuildResult(
            AiDotNetEngine.Current, numOps, logits, 0.5, "the road");

        Assert.Contains("with 90000 pixels", result.TextResponse);
        Assert.Equal(0.75, (double)(float)result.Confidence, 2e-3);
    }

    [Fact]
    public void ReferringResult_NaNLogitOutsideTheMask_DoesNotReachTheConfidence()
    {
        // sigmoid(NaN) is NaN, which fails the threshold: the pixel is outside the mask and must not turn the
        // confidence into NaN (0 * NaN did).
        var logits = T(new[] { 1, 1, 3 }, Math.Log(3), double.NaN, -1);
        var numOps = MathHelper.GetNumericOperations<double>();

        var result = ReferringSegmentationBase<double>.BuildResult(AiDotNetEngine.Current, numOps, logits, 0.5, "the cat");

        Assert.Equal(0.75, result.Confidence, Tolerance);
        Assert.Contains("with 1 pixels", result.TextResponse);
    }

    [Fact]
    public void Classify_Half_LargeAndZeroEmbeddingsStayFinite()
    {
        // Pixel 0 = (300, 400): its squares overflow Half, which made the norm infinite and the cosine 0.
        // Pixel 1 = (0, 0): 1e-24 flushes to 0 in Half, which made its norm 0 and the cosine 0 / 0 = NaN.
        var pixels = H(new[] { 2, 1, 2 }, 300, 0, 400, 0);
        var text = TextEmbeddingClassifier.NormalizeRows(H(new[] { 2, 2 }, 1, 0, 0, 1));

        var logits = TextEmbeddingClassifier.Classify(pixels, text, (Half)1f);

        Assert.Equal(0.6, (double)(float)logits[0, 0, 0], 2e-3);
        Assert.Equal(0.8, (double)(float)logits[1, 0, 0], 2e-3);
        Assert.Equal(0.0, (double)(float)logits[0, 0, 1], Tolerance);
        Assert.Equal(0.0, (double)(float)logits[1, 0, 1], Tolerance);
    }

    [Fact]
    public void ReferringResult_DecidesTheMaskOnTheLogit()
    {
        // LISA keeps logit > 0. sigmoid(1e-17) rounds to exactly 0.5 in double, so a threshold on the rounded
        // probability dropped a pixel that is above the threshold.
        var logits = T(new[] { 1, 1, 2 }, 1e-17, -1e-17);
        var numOps = MathHelper.GetNumericOperations<double>();

        var result = ReferringSegmentationBase<double>.BuildResult(AiDotNetEngine.Current, numOps, logits, 0.5, "the cat");

        Assert.Equal(new[] { 1.0, 0.0 }, new[] { result.Masks[0, 0, 0], result.Masks[0, 0, 1] });
        Assert.Contains("with 1 pixels", result.TextResponse);
    }

    [Fact]
    public void ReferringResult_Half_DoesNotRoundTheThresholdIntoHalf()
    {
        // Threshold 0.7 is logit ln(7/3) = 0.84730. The Half logits on either side of it are 0.84717 and 0.84766;
        // sigmoid(0.84766) = 0.700075 is above 0.7, but Half rounds it and the threshold to the same 0.70020, so a
        // comparison of rounded probabilities dropped it.
        var logits = H(new[] { 1, 1, 2 }, 0.84765625, 0.84716796875);
        var numOps = MathHelper.GetNumericOperations<Half>();

        var result = ReferringSegmentationBase<Half>.BuildResult(AiDotNetEngine.Current, numOps, logits, 0.7, "the road");

        Assert.Equal(new[] { 1f, 0f }, new[] { (float)result.Masks[0, 0, 0], (float)result.Masks[0, 0, 1] });
    }

    [Fact]
    public void OpenVocabularyResult_SingleName_DecidesTheMaskOnTheLogit()
    {
        var logits = T(new[] { 1, 1, 2 }, 1e-17, -1e-17);
        var numOps = MathHelper.GetNumericOperations<double>();

        var result = OpenVocabSegmentationBase<double>.BuildResult(AiDotNetEngine.Current, numOps, logits, new[] { "a cat" });

        Assert.Equal(new[] { 1.0, 0.0 }, new[] { result.Masks[0, 0, 0], result.Masks[0, 0, 1] });
        Assert.Equal(new[] { 0.0, -1.0 }, new[] { result.SemanticMap[0, 0], result.SemanticMap[0, 1] });
    }

    [Fact]
    public void MaskGreaterThan_Half_CountsTheRoundedUpThresholdAsBelow()
    {
        // Half(0.7) = 0.70020 is above the real 0.7, so it is above the threshold; 0.69971 is not.
        var values = H(new[] { 2 }, 0.7001953125, 0.69970703125);

        var mask = MaskStatistics.GreaterThan(values, 0.7);

        Assert.Equal(new[] { 1f, 0f }, new[] { (float)mask[0], (float)mask[1] });
    }

    [Fact]
    public void OpenVocabularyResult_IndependentSigmoid_ScoresEachNameByItsOwnSigmoid()
    {
        // One pixel, logits (ln 3, ln 1.5): the softmax gives name 0 probability 3 / 4.5 = 2/3, its sigmoid 3/4.
        var logits = T(new[] { 2, 1, 1 }, Math.Log(3), Math.Log(1.5));
        var numOps = MathHelper.GetNumericOperations<double>();

        var sigmoid = OpenVocabSegmentationBase<double>.BuildResult(AiDotNetEngine.Current, numOps, logits, new[] { "a", "b" }, true);
        var softmax = OpenVocabSegmentationBase<double>.BuildResult(AiDotNetEngine.Current, numOps, logits, new[] { "a", "b" });

        Assert.Equal(0.75, sigmoid.Scores[0], Tolerance);
        Assert.Equal(2.0 / 3.0, softmax.Scores[0], Tolerance);
        Assert.Equal(0.0, sigmoid.SemanticMap[0, 0]);
    }

    /// <summary>
    /// The scores <c>SegmentWithText</c> must report for <paramref name="logits"/> <c>[K, H, W]</c>: each pixel goes to its
    /// argmax name, and a name scores the mean probability (sigmoid or softmax) of the pixels it wins.
    /// </summary>
    private static double[] ExpectedScores(Tensor<double> logits, bool sigmoid)
    {
        int k = logits.Shape[0], h = logits.Shape[1], w = logits.Shape[2];
        var sum = new double[k];
        var count = new int[k];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int best = 0;
                double denominator = 0;
                for (int c = 0; c < k; c++)
                {
                    if (logits[c, y, x] > logits[best, y, x]) best = c;
                    denominator += Math.Exp(logits[c, y, x]);
                }
                double top = logits[best, y, x];
                sum[best] += sigmoid ? 1.0 / (1.0 + Math.Exp(-top)) : Math.Exp(top) / denominator;
                count[best]++;
            }
        return Enumerable.Range(0, k).Select(c => count[c] > 0 ? sum[c] / count[c] : 0.0).ToArray();
    }

    private static void AssertScoresFollowThePaper(OpenVocabSegmentationBase<double> model, bool sigmoid)
    {
        var image = new Tensor<double>(new[] { 3, 64, 64 });
        var rng = RandomHelper.CreateSeededRandom(13);
        for (int i = 0; i < image.Length; i++) image[i] = rng.NextDouble();

        var logits = SegmentationTensorOps.EnsureUnbatched(model.Predict(image));
        var result = model.SegmentWithText(image, new[] { "cat", "dog" });

        var expected = ExpectedScores(logits, sigmoid);
        Assert.Equal(expected[0], result.Scores[0], 1e-9);
        Assert.Equal(expected[1], result.Scores[1], 1e-9);
    }

    private static DeterministicTextEncoder<double> CatDog()
        => new DeterministicTextEncoder<double>(4).With("cat", 1, 0, 0, 0).With("dog", 0, 1, 0, 0);

    [Fact(Timeout = 120000)]
    public void CATSeg_ScoresEachNameByItsSigmoid_AsTheOfficialInference()
    {
        // CAT-Seg trains with binary cross-entropy and reads outputs.sigmoid(); the closed-set names equal the query
        // names, so Predict's logits are the ones SegmentWithText classifies.
        using var model = new CATSeg<double>(Arch(), CatDog(),
            options: new CATSegOptions { NumClasses = 2, ClassNames = new[] { "cat", "dog" }, DropRate = 0.0, Seed = 7 });
        AssertScoresFollowThePaper(model, sigmoid: true);
    }

    [Fact(Timeout = 120000)]
    public void SED_ScoresEachNameByItsSigmoid_AsTheOfficialInference()
    {
        using var model = new SED<double>(Arch(), CatDog(),
            options: new SEDOptions { NumClasses = 2, ClassNames = new[] { "cat", "dog" }, DropRate = 0.0, Seed = 7 });
        AssertScoresFollowThePaper(model, sigmoid: true);
    }

    [Fact(Timeout = 120000)]
    public void SAN_ScoresTheNamesBySoftmax()
    {
        using var model = new SAN<double>(Arch(), CatDog(),
            options: new SANOptions { NumClasses = 2, ClassNames = new[] { "cat", "dog" }, DropRate = 0.0, Seed = 7 });
        AssertScoresFollowThePaper(model, sigmoid: false);
    }
}
