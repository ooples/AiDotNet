using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Losses;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// The R-CNN family objectives against independent evaluations of Ren et al. 2015 (Eq. 1), Girshick 2015 (Eq. 1-3)
/// and Cai and Vasconcelos 2018 (Eq. 8). Sample sizes exceed the candidate counts, so no draw is random.
/// </summary>
public sealed class TwoStageDetectionLossTests
{
    public TwoStageDetectionLossTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void PublishedRecipe_IsTheDefault()
    {
        var options = new TwoStageDetectionLossOptions();
        Assert.Equal((0.7, 0.3, 256, 0.5, 10.0), (options.RpnPositiveIoU, options.RpnNegativeIoU,
            options.RpnBatchSizePerImage, options.RpnPositiveFraction, options.RpnRegressionWeight));
        Assert.Equal((64, 0.25, 0.1, 1.0), (options.RoiBatchSizePerImage, options.RoiForegroundFraction,
            options.RoiBackgroundIoULow, options.RoiRegressionWeight));
        Assert.Equal(new[] { 0.5, 0.6, 0.7 }, options.StageForegroundIoU);
        Assert.Equal(new[] { 1.0, 1.0, 1.0 }, options.StageLossWeights);
    }

    [Fact]
    public void EncodeDeltas_IsTheRcnnParameterization()
    {
        var reference = new[] { 10.0, 20.0, 30.0, 60.0 };  // center (20, 40), size 20 x 40
        var gold = new[] { 14.0, 16.0, 54.0, 36.0 };       // center (34, 26), size 40 x 20
        var delta = TwoStageDetectionLoss<double>.EncodeDeltas(reference, gold);
        Assert.Equal((34 - 20) / 20.0, delta[0], 12);
        Assert.Equal((26 - 40) / 40.0, delta[1], 12);
        Assert.Equal(Math.Log(40 / 20.0), delta[2], 12);
        Assert.Equal(Math.Log(20 / 40.0), delta[3], 12);
    }

    [Fact]
    public void ProposalLoss_LabelsByIoU_IgnoresTheMiddleBand_AndNormalizesRegressionByAnchorLocations()
    {
        var anchors = Boxes(
            new[] { 0.0, 0, 10, 10 },   // IoU 1.0: positive
            new[] { 5.0, 5, 15, 15 },   // IoU 0.143: negative
            new[] { 20.0, 20, 30, 30 }, // IoU 0: negative
            new[] { 0.0, 0, 20, 20 },   // IoU 0.25: negative
            new[] { 40.0, 40, 50, 50 }, // IoU 0: negative
            new[] { 1.0, 1, 11, 11 });  // IoU 0.681: ignored
        var gold = new[] { new[] { 0.0, 0, 10, 10 } };
        double[] logitValues = { 0.2, -0.4, 1.1, 0.3, -0.7, 0.9, 0.5, 0.5, 0.0, -1.2, 0.8, 0.1 };
        double[] deltaValues =
        {
            0.3, -1.7, 0.2, 2.5,  0.4, 0.4, 0.4, 0.4,  -0.3, 0.1, 0.9, -2.0,
            1.0, 1.0, 1.0, 1.0,   0.6, -0.6, 0.0, 0.2,  0.05, 0.05, 0.05, 0.05
        };
        var objectness = new Tensor<double>((double[])logitValues.Clone(), new[] { 6, 2 });
        var deltas = new Tensor<double>((double[])deltaValues.Clone(), new[] { 6, 4 });
        var loss = new TwoStageDetectionLoss<double>(1, 1, new TwoStageDetectionLossOptions());

        using var tape = new GradientTape<double>();
        var objective = loss.ComputeProposalLoss(objectness, deltas, anchors, gold, anchorsPerLocation: 3,
            RandomHelper.CreateSeededRandom(7));
        var gradients = tape.ComputeGradients(objective, new[] { objectness, deltas });

        int[] labels = { 1, 0, 0, 0, 0, -1 };
        const int sampled = 5;
        double classification = 0;
        for (int a = 0; a < 6; a++)
            if (labels[a] >= 0) classification += -LogSoftmax(logitValues[a * 2], logitValues[a * 2 + 1])[labels[a]];
        double regression = Enumerable.Range(0, 4).Sum(k => SmoothL1(deltaValues[k] - 0.0));
        double expected = classification / sampled + 10.0 / (6 / 3.0) * regression;
        Near(expected, objective[0], 1e-10);

        Assert.True(gradients.TryGetValue(objectness, out var objectnessGradient));
        Assert.True(gradients.TryGetValue(deltas, out var deltaGradient));
        Assert.NotNull(objectnessGradient);
        Assert.NotNull(deltaGradient);
        for (int a = 0; a < 6; a++)
        {
            var probabilities = LogSoftmax(logitValues[a * 2], logitValues[a * 2 + 1]).Select(Math.Exp).ToArray();
            for (int c = 0; c < 2; c++)
            {
                double expectedGradient = labels[a] < 0 ? 0 : (probabilities[c] - (labels[a] == c ? 1 : 0)) / sampled;
                Near(expectedGradient, objectnessGradient[a, c], 1e-10);
            }
            for (int k = 0; k < 4; k++)
            {
                double expectedGradient = a == 0 ? 5.0 * Math.Max(-1, Math.Min(1, deltaValues[k])) : 0;
                Near(expectedGradient, deltaGradient[a, k], 1e-10);
            }
        }
    }

    [Fact]
    public void ProposalLoss_AnObjectsBestAnchorIsPositiveEvenBelowThePositiveThreshold()
    {
        var anchors = Boxes(new[] { 0.0, 0, 10, 10 }, new[] { 30.0, 30, 40, 40 });
        var gold = new[] { new[] { 0.0, 0, 16, 16 } }; // Best IoU is 100 / 256 = 0.39 < 0.7.
        var objectness = new Tensor<double>(new[] { 0.0, 0, 0, 0 }, new[] { 2, 2 });
        var deltas = new Tensor<double>(new double[8], new[] { 2, 4 });
        var loss = new TwoStageDetectionLoss<double>(1, 1, new TwoStageDetectionLossOptions());
        using var tape = new GradientTape<double>();
        var objective = loss.ComputeProposalLoss(objectness, deltas, anchors, gold, 1, RandomHelper.CreateSeededRandom(7));
        var gradient = tape.ComputeGradients(objective, new[] { objectness })[objectness];
        Assert.True(gradient[0, 1] < 0, "The best anchor must be pushed toward the object class.");
        Assert.True(gradient[1, 0] < 0, "The disjoint anchor must be pushed toward background.");
    }

    [Theory]
    [InlineData(0, new[] { 0, 1, 2 }, new[] { 3 })]
    [InlineData(2, new[] { 0, 1 }, new[] { 2, 3 })]
    public void StageLoss_UsesItsStageThreshold_TheBackgroundInterval_AndTheObjectClassColumns(
        int stage, int[] foreground, int[] background)
    {
        var proposals = new[]
        {
            new[] { 0.0, 0, 10, 10 },   // IoU 1.0
            new[] { 0.0, 0, 10, 14 },   // IoU 0.714
            new[] { 0.0, 0, 10, 18 },   // IoU 0.556
            new[] { 5.0, 0, 15, 10 },   // IoU 0.333
            new[] { 40.0, 40, 50, 50 }  // IoU 0: below the background interval, excluded
        };
        var gold = new[] { new[] { 0.0, 0, 10, 10 } };
        const int goldClass = 1;
        const int width = 3; // background plus two foreground classes
        double[] logitValues = Enumerable.Range(0, 5 * width).Select(i => Math.Sin(i * 0.9) * 1.4).ToArray();
        double[] deltaValues = Enumerable.Range(0, 5 * width * 4).Select(i => Math.Cos(i * 0.37) * 1.3).ToArray();
        var logits = new Tensor<double>((double[])logitValues.Clone(), new[] { 5, width });
        var deltas = new Tensor<double>((double[])deltaValues.Clone(), new[] { 5, width * 4 });
        var boxes = new Tensor<double>(proposals.SelectMany(box => box).ToArray(), new[] { 5, 4 });
        var loss = new TwoStageDetectionLoss<double>(2, 3, new TwoStageDetectionLossOptions());

        using var tape = new GradientTape<double>();
        var objective = loss.ComputeStageLoss(logits, deltas, boxes, gold, new[] { goldClass }, stage, RandomHelper.CreateSeededRandom(7));
        var gradients = tape.ComputeGradients(objective, new[] { logits, deltas });

        int sampled = foreground.Length + background.Length;
        double classification = 0;
        foreach (int r in foreground) classification -= LogSoftmax(Row(logitValues, r, width))[goldClass + 1];
        foreach (int r in background) classification -= LogSoftmax(Row(logitValues, r, width))[0];
        double regression = 0;
        foreach (int r in foreground)
        {
            var target = TwoStageDetectionLoss<double>.EncodeDeltas(proposals[r], gold[0]);
            for (int k = 0; k < 4; k++)
                regression += SmoothL1(deltaValues[r * width * 4 + (goldClass + 1) * 4 + k] - target[k]);
        }
        Near((classification + regression) / sampled, objective[0], 1e-10);

        var deltaGradient = gradients[deltas];
        var logitGradient = gradients[logits];
        for (int c = 0; c < width; c++) Near(0, logitGradient[4, c], 1e-12); // Excluded proposal.
        for (int r = 0; r < 5; r++)
            for (int column = 0; column < width * 4; column++)
            {
                bool regressed = foreground.Contains(r) && column / 4 == goldClass + 1;
                if (!regressed) Near(0, deltaGradient[r, column], 1e-12);
            }
    }

    [Theory]
    [InlineData(nameof(TwoStageDetectionLossOptions.RpnPositiveIoU), 1.2)]
    [InlineData(nameof(TwoStageDetectionLossOptions.RpnRegressionWeight), -1.0)]
    [InlineData(nameof(TwoStageDetectionLossOptions.RoiForegroundFraction), double.NaN)]
    public void InvalidOptions_AreRejected(string property, double value)
    {
        var options = new TwoStageDetectionLossOptions();
        var info = typeof(TwoStageDetectionLossOptions).GetProperty(property);
        Assert.NotNull(info);
        info.SetValue(options, value);
        Assert.Equal(property, Assert.Throws<ArgumentOutOfRangeException>(() => new TwoStageDetectionLoss<double>(2, 1, options)).ParamName);
    }

    [Fact]
    public void MoreStagesThanThresholds_AreRejected()
    {
        Assert.Throws<ArgumentException>(() => new TwoStageDetectionLoss<double>(2, 4, new TwoStageDetectionLossOptions()));
    }

    private static List<BoundingBox<double>> Boxes(params double[][] boxes) =>
        boxes.Select(box => new BoundingBox<double>(box[0], box[1], box[2], box[3])).ToList();

    private static double[] Row(double[] values, int row, int width) => values.Skip(row * width).Take(width).ToArray();

    private static double[] LogSoftmax(params double[] logits)
    {
        double max = logits.Max();
        double logSum = max + Math.Log(logits.Sum(value => Math.Exp(value - max)));
        return logits.Select(value => value - logSum).ToArray();
    }

    private static double SmoothL1(double x) => Math.Abs(x) < 1 ? 0.5 * x * x : Math.Abs(x) - 0.5;

    private static void Near(double expected, double actual, double tolerance) =>
        Assert.True(!double.IsNaN(actual) && !double.IsInfinity(actual) && Math.Abs(expected - actual) <= tolerance,
            $"Expected {expected:R}; actual {actual:R}; tolerance {tolerance:R}.");
}
