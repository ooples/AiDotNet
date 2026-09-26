using AiDotNet.ComputerVision.Detection;
using AiDotNet.ComputerVision.Detection.Losses;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;
using Oracle = AiDotNet.Tests.ModelFamilyTests.Base.TaskAlignedDetectionOracle;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>Task-aligned YOLO training against an independent oracle of the published assignment and losses.</summary>
public sealed class TaskAlignedDetectionLossTests
{
    private const int Classes = 2;
    private const int RegMax = 4;
    private const int ImageSize = 32;
    private static readonly int[] Strides = { 8, 16 };
    private static readonly int[] Sides = { 4, 2 }; // 32 / stride.

    public TaskAlignedDetectionLossTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void PublishedRecipe_IsTheDefault()
    {
        var options = new TaskAlignedLossOptions();
        // YOLOv10 Sec. 3.1: alpha 0.5, beta 6 for both heads, top-one for one-to-one; YOLOv9 Table 1 gains 7.5/0.5/1.5.
        Assert.Equal((10, 1, 0.5, 6.0, 7.5, 0.5, 1.5),
            (options.TopK, options.OneToOneTopK, options.Alpha, options.Beta, options.BoxGain, options.ClassGain, options.DflGain));
    }

    [Theory]
    [InlineData(10)]
    [InlineData(1)]
    public void ValueAndDerivatives_MatchTheIndependentOracle(int topK)
    {
        var (classLevels, distributionLevels, oracleLevels) = Heads(seed: 0.37);
        var gold = new[]
        {
            // A large object and a smaller overlapping one of the other class, so the two objects compete for anchors.
            new Oracle.Gold(0, 0.5, 0.5, 0.9, 0.85),
            new Oracle.Gold(1, 0.3, 0.35, 0.45, 0.5)
        };
        var positives = Oracle.Assign(oracleLevels, 1, Classes, RegMax, new[] { gold }, ImageSize, ImageSize, topK);
        Assert.NotEmpty(positives);
        if (topK == 1) Assert.Equal(2, positives.Count); // Top-one: exactly one anchor per object.
        else
        {
            Assert.Contains(positives, positive => positive.ClassId == 0);
            Assert.Contains(positives, positive => positive.ClassId == 1);
            Assert.True(positives.Count > 2);
        }

        var loss = new TaskAlignedDetectionLoss<double>(Classes, RegMax, new TaskAlignedLossOptions());
        var batch = Batch(gold);
        using var tape = new GradientTape<double>();
        var objective = loss.ComputeTapeLoss(classLevels, distributionLevels, Strides, ImageSize, ImageSize, batch, topK);
        var gradients = tape.ComputeGradients(objective, classLevels.Concat(distributionLevels).ToArray());
        double expected = Oracle.Loss(oracleLevels, 1, Classes, RegMax, positives);
        Near(expected, objective[0], 1e-9);
        Near(expected, loss.CalculateLoss(classLevels, distributionLevels, Strides, ImageSize, ImageSize, batch, topK), 1e-9);

        // Targets are detached: d/dx of the class term is gain * (sigmoid(x) - t) / normalizer.
        double normalizer = Math.Max(1, positives.Sum(positive => positive.Target));
        for (int level = 0; level < Strides.Length; level++)
        {
            Assert.True(gradients.TryGetValue(classLevels[level], out var classGradient));
            Assert.NotNull(classGradient);
            var targets = new double[classLevels[level].Length];
            foreach (var positive in positives.Where(p => p.Level == level))
                targets[positive.ClassId * Sides[level] * Sides[level] + positive.Cell] = positive.Target;
            for (int index = 0; index < targets.Length; index++)
                Near(0.5 * (Oracle.Sigmoid(classLevels[level][index]) - targets[index]) / normalizer, classGradient[index], 1e-9);

            Assert.True(gradients.TryGetValue(distributionLevels[level], out var distributionGradient));
            Assert.NotNull(distributionGradient);
            for (int index = 0; index < distributionLevels[level].Length; index++)
            {
                double derivative = FiniteDifference(oracleLevels[level].Distribution, index,
                    () => Oracle.Loss(oracleLevels, 1, Classes, RegMax, positives));
                Near(derivative, distributionGradient[index], 2e-6);
            }
        }
    }

    [Fact]
    public void EmptyImages_TrainBackgroundOnlyAndLeaveDistributionsUntouched()
    {
        var (classLevels, distributionLevels, oracleLevels) = Heads(seed: 1.1);
        var loss = new TaskAlignedDetectionLoss<double>(Classes, RegMax, new TaskAlignedLossOptions());
        var empty = new DetectionTrainingBatch<double>(new[] { Array.Empty<DetectionTrainingTarget<double>>() });
        using var tape = new GradientTape<double>();
        var objective = loss.ComputeTapeLoss(classLevels, distributionLevels, Strides, ImageSize, ImageSize, empty, 10);
        var gradients = tape.ComputeGradients(objective, distributionLevels.ToArray());
        Near(Oracle.Loss(oracleLevels, 1, Classes, RegMax, Array.Empty<Oracle.Positive>()), objective[0], 1e-9);
        foreach (var distribution in distributionLevels)
        {
            Assert.True(gradients.TryGetValue(distribution, out var gradient));
            Assert.NotNull(gradient);
            Assert.All(gradient.ToArray(), value => Assert.Equal(0.0, value));
        }
    }

    [Fact]
    public void AnchorsOutsideEveryObject_AreNeverPositive()
    {
        var (_, _, oracleLevels) = Heads(seed: 0.9);
        // A small box that contains exactly one stride-8 anchor center, (12, 12), and no stride-16 center.
        var gold = new[] { new Oracle.Gold(1, 12.0 / 32, 12.0 / 32, 3.0 / 32, 3.0 / 32) };
        var positives = Oracle.Assign(oracleLevels, 1, Classes, RegMax, new[] { gold }, ImageSize, ImageSize, 10);
        var positive = Assert.Single(positives);
        Assert.Equal((0, 5), (positive.Level, positive.Cell));
    }

    [Theory]
    [InlineData(nameof(TaskAlignedLossOptions.TopK), 0)]
    [InlineData(nameof(TaskAlignedLossOptions.OneToOneTopK), 0)]
    [InlineData(nameof(TaskAlignedLossOptions.Beta), -1)]
    [InlineData(nameof(TaskAlignedLossOptions.DflGain), double.NaN)]
    public void InvalidOptions_AreRejected(string property, double value)
    {
        var options = new TaskAlignedLossOptions();
        var info = typeof(TaskAlignedLossOptions).GetProperty(property);
        Assert.NotNull(info);
        info.SetValue(options, info.PropertyType == typeof(int) ? (object)(int)value : value);
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => new TaskAlignedDetectionLoss<double>(Classes, RegMax, options));
        Assert.Equal(property, error.ParamName);
    }

    [Fact]
    public void TargetsOfUnknownClasses_AreRejected()
    {
        var (classLevels, distributionLevels, _) = Heads(seed: 0.2);
        var loss = new TaskAlignedDetectionLoss<double>(Classes, RegMax, new TaskAlignedLossOptions());
        var batch = new DetectionTrainingBatch<double>(new[] { new[] { new DetectionTrainingTarget<double>(Classes, 0.5, 0.5, 0.2, 0.2) } });
        Assert.Throws<ArgumentException>(() => loss.CalculateLoss(classLevels, distributionLevels, Strides, ImageSize, ImageSize, batch, 10));
    }

    private static (Tensor<double>[] Classes, Tensor<double>[] Distributions, Oracle.Level[] Oracle) Heads(double seed)
    {
        var classLevels = new Tensor<double>[Strides.Length];
        var distributionLevels = new Tensor<double>[Strides.Length];
        var oracle = new Oracle.Level[Strides.Length];
        for (int level = 0; level < Strides.Length; level++)
        {
            int side = Sides[level];
            var classValues = Enumerable.Range(0, Classes * side * side).Select(i => Math.Sin(seed + i * 0.61) * 1.3).ToArray();
            var distributionValues = Enumerable.Range(0, 4 * RegMax * side * side).Select(i => Math.Cos(seed * 3 + i * 0.47) * 1.1).ToArray();
            classLevels[level] = new Tensor<double>((double[])classValues.Clone(), new[] { 1, Classes, side, side });
            distributionLevels[level] = new Tensor<double>((double[])distributionValues.Clone(), new[] { 1, 4 * RegMax, side, side });
            oracle[level] = new Oracle.Level(classValues, distributionValues, side, side, Strides[level]);
        }
        return (classLevels, distributionLevels, oracle);
    }

    private static DetectionTrainingBatch<double> Batch(IEnumerable<Oracle.Gold> gold) =>
        new(new[] { gold.Select(g => new DetectionTrainingTarget<double>(g.ClassId, g.CenterX, g.CenterY, g.Width, g.Height)).ToArray() });

    private static double FiniteDifference(double[] values, int index, Func<double> evaluate)
    {
        const double epsilon = 1e-6;
        double original = values[index];
        values[index] = original + epsilon;
        double plus = evaluate();
        values[index] = original - epsilon;
        double minus = evaluate();
        values[index] = original;
        return (plus - minus) / (2 * epsilon);
    }

    private static void Near(double expected, double actual, double tolerance) =>
        Assert.True(!double.IsNaN(actual) && !double.IsInfinity(actual) && Math.Abs(expected - actual) <= tolerance,
            $"Expected {expected:R}; actual {actual:R}; tolerance {tolerance:R}.");
}
