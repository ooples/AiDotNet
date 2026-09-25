using AiDotNet.ComputerVision.Detection;
using AiDotNet.ComputerVision.Detection.Losses;
using AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Independent values and derivatives for the sigmoid focal (DINO) and varifocal (RT-DETR) forms of
/// the shared set prediction objective, computed from the published formulas rather than the code.
/// </summary>
public sealed class DetrFamilySetLossTests
{
    private const int Queries = 2;
    private const int Classes = 2;
    private static readonly double[] Logits = { 0.3, -0.2, 1.1, 0.4 };
    private static readonly double[] Boxes = { 0.5, 0.5, 0.4, 0.4, 0.2, 0.25, 0.2, 0.3 };
    private static readonly double[] Gold = { 0.52, 0.48, 0.38, 0.42 };
    private const int GoldClass = 1;

    public DetrFamilySetLossTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void PublishedRecipes_AreTheFactoryDefaults()
    {
        var detr = DetrSetLossOptions.ForDetr();
        Assert.Equal(SetPredictionClassificationLoss.SoftmaxCrossEntropy, detr.ClassificationLoss);
        Assert.Equal((1.0, 5.0, 2.0, 1.0, 5.0, 2.0, 0.1),
            (detr.ClassCostWeight, detr.L1CostWeight, detr.GIoUCostWeight, detr.ClassLossWeight, detr.L1LossWeight, detr.GIoULossWeight, detr.NoObjectWeight));

        // DINO Table 8: set cost class/bbox/giou 2/5/2, loss coef 1/5/2, focal alpha 0.25 (gamma 2 in Sec. 4.1).
        var dino = DetrSetLossOptions.ForDino();
        Assert.Equal(SetPredictionClassificationLoss.SigmoidFocal, dino.ClassificationLoss);
        Assert.Equal((2.0, 5.0, 2.0, 1.0, 5.0, 2.0, 0.25, 2.0),
            (dino.ClassCostWeight, dino.L1CostWeight, dino.GIoUCostWeight, dino.ClassLossWeight, dino.L1LossWeight, dino.GIoULossWeight, dino.FocalAlpha, dino.FocalGamma));

        // RT-DETR Table A: class/bbox/GIoU cost 2/5/2, loss weights 1/5/2, alpha 0.75 and gamma 2.0 in class loss.
        var rtDetr = DetrSetLossOptions.ForRtDetr();
        Assert.Equal(SetPredictionClassificationLoss.VariFocal, rtDetr.ClassificationLoss);
        Assert.Equal((2.0, 5.0, 2.0, 1.0, 5.0, 2.0, 0.75, 2.0),
            (rtDetr.ClassCostWeight, rtDetr.L1CostWeight, rtDetr.GIoUCostWeight, rtDetr.ClassLossWeight, rtDetr.L1LossWeight, rtDetr.GIoULossWeight, rtDetr.FocalAlpha, rtDetr.FocalGamma));
    }

    [Fact]
    public void SigmoidFocal_MatchesTheFocalLossValueAndDerivatives()
    {
        var options = DetrSetLossOptions.ForDino();
        int query = ExpectedQuery(options);
        Func<double[], double[], double> objective = (logits, boxes) =>
            FocalClassification(logits, query, options) + BoxObjective(boxes, query, options);

        var (value, logitGradient, boxGradient) = EvaluateOnTape(options);
        Near(objective(Logits, Boxes), value, 1e-10);
        for (int index = 0; index < Logits.Length; index++)
            Near(FiniteDifference(Logits, index, logits => objective(logits, Boxes)), logitGradient[index], 1e-6);
        for (int index = 0; index < Boxes.Length; index++)
            Near(FiniteDifference(Boxes, index, boxes => objective(Logits, boxes)), boxGradient[index], 2e-5);
    }

    [Fact]
    public void VariFocal_TrainsTheMatchedClassTowardTheDetachedBoxIoU()
    {
        var options = DetrSetLossOptions.ForRtDetr();
        int query = ExpectedQuery(options);
        double quality = IoU(Slice(Boxes, query), Gold);
        Assert.InRange(quality, 0.05, 0.95); // A soft target, distinguishable from the focal one-hot target.

        var weights = new double[Logits.Length];
        var targets = new double[Logits.Length];
        for (int index = 0; index < Logits.Length; index++)
        {
            bool positive = index == query * Classes + GoldClass;
            targets[index] = positive ? quality : 0;
            weights[index] = positive ? quality : options.FocalAlpha * Math.Pow(Sigmoid(Logits[index]), options.FocalGamma);
        }

        double classification = 0;
        for (int index = 0; index < Logits.Length; index++)
            classification += weights[index] * (Softplus(Logits[index]) - targets[index] * Logits[index]);
        var (value, logitGradient, boxGradient) = EvaluateOnTape(options);
        Near(classification + BoxObjective(Boxes, query, options), value, 1e-10);

        // Weights and IoU targets are detached, so d/dx = w (sigmoid(x) - q) and the IoU target
        // contributes nothing to the box gradient.
        for (int index = 0; index < Logits.Length; index++)
            Near(weights[index] * (Sigmoid(Logits[index]) - targets[index]), logitGradient[index], 1e-10);
        for (int index = 0; index < Boxes.Length; index++)
            Near(FiniteDifference(Boxes, index, boxes => BoxObjective(boxes, query, options)), boxGradient[index], 2e-5);
    }

    [Theory]
    [InlineData(SetPredictionClassificationLoss.SigmoidFocal)]
    [InlineData(SetPredictionClassificationLoss.VariFocal)]
    public void EmptyImages_TrainEveryClassAsBackgroundAndLeaveBoxesUntouched(SetPredictionClassificationLoss form)
    {
        var options = form == SetPredictionClassificationLoss.SigmoidFocal ? DetrSetLossOptions.ForDino() : DetrSetLossOptions.ForRtDetr();
        var loss = new DETRSetLoss<double>(Classes, options);
        var logits = new Tensor<double>((double[])Logits.Clone(), new[] { 1, Queries, Classes });
        var boxes = new Tensor<double>((double[])Boxes.Clone(), new[] { 1, Queries, 4 });
        var empty = new DetectionTrainingBatch<double>(new[] { Array.Empty<DetectionTrainingTarget<double>>() });
        using var tape = new GradientTape<double>();
        var objective = loss.ComputeTapeLoss(logits, boxes, empty);
        var gradients = tape.ComputeGradients(objective, new[] { logits, boxes });

        double expected = 0;
        foreach (double logit in Logits)
        {
            double p = Sigmoid(logit);
            // Focal: (1 - alpha) p^gamma (-log(1 - p)). Varifocal negative: alpha p^gamma (-log(1 - p)).
            double alpha = form == SetPredictionClassificationLoss.SigmoidFocal ? 1 - options.FocalAlpha : options.FocalAlpha;
            expected += alpha * Math.Pow(p, options.FocalGamma) * Softplus(logit);
        }
        Near(expected, objective[0], 1e-10); // Normalized by max(1, target count).
        Assert.True(gradients.TryGetValue(boxes, out var boxGradient));
        Assert.NotNull(boxGradient);
        Assert.All(boxGradient.ToArray(), value => Assert.Equal(0.0, value));
    }

    [Fact]
    public void SigmoidForms_RejectANoObjectColumnTarget()
    {
        var loss = new DETRSetLoss<double>(Classes, DetrSetLossOptions.ForDino());
        var logits = new Tensor<double>((double[])Logits.Clone(), new[] { 1, Queries, Classes });
        var boxes = new Tensor<double>((double[])Boxes.Clone(), new[] { 1, Queries, 4 });
        var outOfRange = new DetectionTrainingBatch<double>(new[] { new[] { new DetectionTrainingTarget<double>(Classes, 0.5, 0.5, 0.2, 0.2) } });
        Assert.Throws<ArgumentException>(() => loss.CalculateLoss(logits, boxes, outOfRange));
    }

    [Theory]
    [InlineData(nameof(DetrSetLossOptions.FocalAlpha), 1.5)]
    [InlineData(nameof(DetrSetLossOptions.FocalGamma), -1.0)]
    [InlineData(nameof(DetrSetLossOptions.ClassCostWeight), double.NaN)]
    [InlineData(nameof(DetrSetLossOptions.NoObjectWeight), -0.1)]
    public void InvalidOptions_AreRejectedAtConstruction(string property, double value)
    {
        var options = DetrSetLossOptions.ForDino();
        typeof(DetrSetLossOptions).GetProperty(property)?.SetValue(options, value);
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => new DETRSetLoss<double>(Classes, options));
        Assert.Equal(property, error.ParamName);
    }

    [Fact]
    public void Options_AreCopiedSoLaterMutationCannotChangeATrainedObjective()
    {
        var options = DetrSetLossOptions.ForDino();
        var loss = new DETRSetLoss<double>(Classes, options);
        var logits = new Tensor<double>((double[])Logits.Clone(), new[] { 1, Queries, Classes });
        var boxes = new Tensor<double>((double[])Boxes.Clone(), new[] { 1, Queries, 4 });
        double before = loss.CalculateLoss(logits, boxes, GoldBatch());
        options.ClassLossWeight = 100;
        Assert.Equal(before, loss.CalculateLoss(logits, boxes, GoldBatch()));
    }

    [Fact]
    public void Detectors_RejectAClassificationFormTheirHeadCannotRepresent()
    {
        var sigmoid = new ObjectDetectionOptions<double>
        {
            InputSize = new[] { 64, 64 }, Size = ModelSize.Nano, NumClasses = 2, SetPredictionLoss = DetrSetLossOptions.ForDino()
        };
        var softmax = new ObjectDetectionOptions<double>
        {
            InputSize = new[] { 64, 64 }, Size = ModelSize.Nano, NumClasses = 2, SetPredictionLoss = DetrSetLossOptions.ForDetr()
        };
        Assert.Equal("options", Assert.Throws<ArgumentException>(() => new DETR<double>(sigmoid)).ParamName);
        Assert.Equal("options", Assert.Throws<ArgumentException>(() => new DINO<double>(softmax)).ParamName);
        Assert.Equal("options", Assert.Throws<ArgumentException>(() => new RTDETR<double>(softmax)).ParamName);
    }

    private static (double Value, double[] LogitGradient, double[] BoxGradient) EvaluateOnTape(DetrSetLossOptions options)
    {
        var loss = new DETRSetLoss<double>(Classes, options);
        var logits = new Tensor<double>((double[])Logits.Clone(), new[] { 1, Queries, Classes });
        var boxes = new Tensor<double>((double[])Boxes.Clone(), new[] { 1, Queries, 4 });
        using var tape = new GradientTape<double>();
        var objective = loss.ComputeTapeLoss(logits, boxes, GoldBatch());
        var gradients = tape.ComputeGradients(objective, new[] { logits, boxes });
        Assert.True(gradients.TryGetValue(logits, out var logitGradient));
        Assert.True(gradients.TryGetValue(boxes, out var boxGradient));
        Assert.NotNull(logitGradient);
        Assert.NotNull(boxGradient);
        Near(objective[0], loss.CalculateLoss(logits, boxes, GoldBatch()), 1e-12);
        return (objective[0], logitGradient.ToArray(), boxGradient.ToArray());
    }

    private static DetectionTrainingBatch<double> GoldBatch() =>
        new(new[] { new[] { new DetectionTrainingTarget<double>(GoldClass, Gold[0], Gold[1], Gold[2], Gold[3]) } });

    /// <summary>Deformable DETR matcher: focal class cost plus weighted L1 and negative GIoU.</summary>
    private static int ExpectedQuery(DetrSetLossOptions options)
    {
        var costs = new double[Queries];
        for (int query = 0; query < Queries; query++)
        {
            double logit = Logits[query * Classes + GoldClass];
            double p = Sigmoid(logit);
            double classCost = options.MatchingFocalAlpha * Math.Pow(1 - p, options.MatchingFocalGamma) * Softplus(-logit)
                - (1 - options.MatchingFocalAlpha) * Math.Pow(p, options.MatchingFocalGamma) * Softplus(logit);
            var box = Slice(Boxes, query);
            double l1 = box.Zip(Gold, (left, right) => Math.Abs(left - right)).Sum();
            costs[query] = options.ClassCostWeight * classCost + options.L1CostWeight * l1
                - options.GIoUCostWeight * (1 - PlainGIoULoss(CenterToCorners(box), CenterToCorners(Gold)));
        }
        return costs[0] <= costs[1] ? 0 : 1;
    }

    /// <summary>Lin et al. 2017: FL = -alpha_t (1 - p_t)^gamma log(p_t), summed and divided by one target.</summary>
    private static double FocalClassification(double[] logits, int query, DetrSetLossOptions options)
    {
        double total = 0;
        for (int index = 0; index < logits.Length; index++)
        {
            bool positive = index == query * Classes + GoldClass;
            double p = Sigmoid(logits[index]);
            double pt = positive ? p : 1 - p;
            double alphaT = positive ? options.FocalAlpha : 1 - options.FocalAlpha;
            double negativeLogPt = positive ? Softplus(-logits[index]) : Softplus(logits[index]);
            total += alphaT * Math.Pow(1 - pt, options.FocalGamma) * negativeLogPt;
        }
        return total * options.ClassLossWeight;
    }

    private static double BoxObjective(double[] boxes, int query, DetrSetLossOptions options)
    {
        var box = Slice(boxes, query);
        double l1 = box.Zip(Gold, (left, right) => Math.Abs(left - right)).Sum();
        return options.L1LossWeight * l1 + options.GIoULossWeight * EngineGIoULoss(CenterToCorners(box), CenterToCorners(Gold));
    }

    private static double[] Slice(double[] boxes, int query) => boxes.Skip(query * 4).Take(4).ToArray();

    private static double[] CenterToCorners(double[] box) =>
        new[] { box[0] - box[2] / 2, box[1] - box[3] / 2, box[0] + box[2] / 2, box[1] + box[3] / 2 };

    private static double IoU(double[] predicted, double[] target)
    {
        var p = CenterToCorners(predicted);
        var t = CenterToCorners(target);
        double intersection = Math.Max(0, Math.Min(p[2], t[2]) - Math.Max(p[0], t[0]))
            * Math.Max(0, Math.Min(p[3], t[3]) - Math.Max(p[1], t[1]));
        return intersection / (predicted[2] * predicted[3] + target[2] * target[3] - intersection);
    }

    private static double PlainGIoULoss(double[] p, double[] t)
    {
        double intersection = Math.Max(0, Math.Min(p[2], t[2]) - Math.Max(p[0], t[0]))
            * Math.Max(0, Math.Min(p[3], t[3]) - Math.Max(p[1], t[1]));
        double union = (p[2] - p[0]) * (p[3] - p[1]) + (t[2] - t[0]) * (t[3] - t[1]) - intersection;
        double enclosure = (Math.Max(p[2], t[2]) - Math.Min(p[0], t[0])) * (Math.Max(p[3], t[3]) - Math.Min(p[1], t[1]));
        return 1 - intersection / union + (enclosure - union) / enclosure;
    }

    private static double EngineGIoULoss(double[] p, double[] t)
    {
        double intersection = Math.Max(0, Math.Min(p[2], t[2]) - Math.Max(p[0], t[0]))
            * Math.Max(0, Math.Min(p[3], t[3]) - Math.Max(p[1], t[1]));
        double union = (p[2] - p[0]) * (p[3] - p[1]) + (t[2] - t[0]) * (t[3] - t[1]) - intersection;
        double enclosure = (Math.Max(p[2], t[2]) - Math.Min(p[0], t[0])) * (Math.Max(p[3], t[3]) - Math.Min(p[1], t[1]));
        const double epsilon = 1e-7; // The engine's documented GIoU stabilizer, as in DetrSemanticTrainingLossTests.
        return 1 - intersection / (union + epsilon) + (enclosure - union) / (enclosure + epsilon);
    }

    private static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));

    private static double Softplus(double x) => Math.Log(1 + Math.Exp(x));

    private static double FiniteDifference(double[] point, int index, Func<double[], double> evaluate)
    {
        const double epsilon = 1e-6;
        var plus = (double[])point.Clone();
        var minus = (double[])point.Clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        return (evaluate(plus) - evaluate(minus)) / (2 * epsilon);
    }

    private static void Near(double expected, double actual, double tolerance) =>
        Assert.True(!double.IsNaN(actual) && !double.IsInfinity(actual) && Math.Abs(expected - actual) <= tolerance,
            $"Expected {expected:R}; actual {actual:R}; tolerance {tolerance:R}.");
}
