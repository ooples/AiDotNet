using AiDotNet.ComputerVision.Detection.Losses;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>Independent values and derivatives for the actual shared DETR objective.</summary>
public sealed class DetrSemanticTrainingLossTests
{
    public DetrSemanticTrainingLossTests() => TestModuleInitializer.EnsureInitialized();

    public enum LossRoute { Scalar, Tape }

    [Theory]
    [InlineData(LossRoute.Scalar)]
    [InlineData(LossRoute.Tape)]
    public void ExtremeFiniteUnselectedLogits_DoNotPoisonCorrectClassification(LossRoute route)
    {
        var predicted = Predictions(batch: 1, queries: 1);
        predicted[0, 0, 0] = double.MaxValue;
        predicted[0, 0, 1] = -double.MaxValue;
        predicted[0, 0, 2] = -double.MaxValue;
        var target = Target(0, 0.5, 0.5, 0.4, 0.4);
        var loss = new DETRSetLoss<double>(numClasses: 3, boxL1Weight: 0, boxGIoUWeight: 0);

        Near(0, Evaluate(loss, predicted, target, route));
    }

    [Theory]
    [InlineData(LossRoute.Scalar)]
    [InlineData(LossRoute.Tape)]
    public void EntireBatchWithoutObjects_StillLearnsNoObject(LossRoute route)
    {
        var predicted = Predictions(batch: 2, queries: 3);
        var target = new Tensor<double>(new[] { 2, 1, 5 });
        target[0, 0, 0] = -1;
        target[1, 0, 0] = -1;
        var loss = new DETRSetLoss<double>(numClasses: 3);

        Near(Math.Log(3), Evaluate(loss, predicted, target, route));
    }

    [Fact]
    public void EmptyBatch_BackgroundGradientIsNonZeroAndBoxGradientIsZero()
    {
        var predicted = Predictions(batch: 1, queries: 2);
        var target = EmptyTarget();
        var loss = new DETRSetLoss<double>(numClasses: 3);
        using var tape = new GradientTape<double>();
        var objective = loss.ComputeTapeLoss(predicted, target);
        var gradients = tape.ComputeGradients(objective, new[] { predicted });
        Assert.True(gradients.TryGetValue(predicted, out var gradient));
        Assert.NotNull(gradient);

        for (int query = 0; query < 2; query++)
        {
            Near(1.0 / 6, gradient[0, query, 0]);
            Near(1.0 / 6, gradient[0, query, 1]);
            Near(-1.0 / 3, gradient[0, query, 2]);
            for (int coordinate = 3; coordinate < 7; coordinate++)
                Near(0, gradient[0, query, coordinate]);
        }
    }

    [Theory]
    [InlineData(LossRoute.Scalar)]
    [InlineData(LossRoute.Tape)]
    public void ForegroundAndBackground_UseWeightedMeanNotQueryCount(LossRoute route)
    {
        var predicted = Predictions(batch: 1, queries: 2);
        var target = Target(0, 0.5, 0.5, 0.4, 0.4);
        Near(Math.Log(3), Evaluate(new DETRSetLoss<double>(numClasses: 3, boxL1Weight: 0, boxGIoUWeight: 0), predicted, target, route));
    }

    [Fact]
    public void UnmatchedQueries_ReceiveWeightedNoObjectGradients()
    {
        var predicted = Predictions(batch: 1, queries: 2);
        var target = Target(0, 0.5, 0.5, 0.4, 0.4);
        var loss = new DETRSetLoss<double>(numClasses: 3);
        using var tape = new GradientTape<double>();
        var objective = loss.ComputeTapeLoss(predicted, target);
        var gradients = tape.ComputeGradients(objective, new[] { predicted });
        Assert.True(gradients.TryGetValue(predicted, out var gradient));
        Assert.NotNull(gradient);

        Near((-2.0 / 3) / 1.1, gradient[0, 0, 0]);
        Near((1.0 / 3) / 1.1, gradient[0, 0, 1]);
        Near((1.0 / 3) / 1.1, gradient[0, 0, 2]);
        Near((0.1 / 3) / 1.1, gradient[0, 1, 0]);
        Near((0.1 / 3) / 1.1, gradient[0, 1, 1]);
        Near((-0.2 / 3) / 1.1, gradient[0, 1, 2]);
    }

    [Theory]
    [InlineData(LossRoute.Scalar)]
    [InlineData(LossRoute.Tape)]
    public void BoxL1_UsesCenterCoordinatesAndSumsFourCoordinates(LossRoute route)
    {
        var predicted = Predictions(batch: 1, queries: 1);
        var target = Target(0, 0.6, 0.55, 0.2, 0.3);
        var loss = new DETRSetLoss<double>(numClasses: 3, classWeight: 0, boxL1Weight: 1, boxGIoUWeight: 0);
        Near(0.45, Evaluate(loss, predicted, target, route));
    }

    [Theory]
    [InlineData(LossRoute.Scalar)]
    [InlineData(LossRoute.Tape)]
    public void GIoU_IsIncludedWithConfiguredWeight(LossRoute route)
    {
        var predicted = Predictions(batch: 1, queries: 1);
        var target = Target(0, 0.6, 0.55, 0.2, 0.3);
        var loss = new DETRSetLoss<double>(numClasses: 3, classWeight: 0, boxL1Weight: 0, boxGIoUWeight: 2);
        double expected = 2 * GIoULoss(new[] { 0.3, 0.3, 0.7, 0.7 }, new[] { 0.5, 0.4, 0.7, 0.7 });
        Near(expected, Evaluate(loss, predicted, target, route));
    }

    [Theory]
    [InlineData(LossRoute.Scalar)]
    [InlineData(LossRoute.Tape)]
    public void Assignment_MaximizesClassProbabilityNotProductOfProbabilities(LossRoute route)
    {
        var predicted = Predictions(batch: 1, queries: 2);
        double[,] probabilities = { { 0.55, 0.44, 0.01 }, { 0.3, 0.2, 0.5 } };
        for (int query = 0; query < 2; query++)
            for (int label = 0; label < 3; label++)
                predicted[0, query, label] = Math.Log(probabilities[query, label]);
        var targets = new Tensor<double>(new[] { 0.0, 0.5, 0.5, 0.4, 0.4, 1.0, 0.5, 0.5, 0.4, 0.4 }, new[] { 1, 2, 5 });
        // The diagonal wins SUM probability (.75 > .74), whereas -log incorrectly chooses
        // the off-diagonal PRODUCT (.132 > .11). No box term can mask this counterexample.
        double expected = (-Math.Log(0.55) - Math.Log(0.2)) / 2;
        var loss = new DETRSetLoss<double>(numClasses: 3, boxL1Weight: 0, boxGIoUWeight: 0);
        Near(expected, Evaluate(loss, predicted, targets, route));
    }

    [Theory]
    [InlineData(LossRoute.Scalar)]
    [InlineData(LossRoute.Tape)]
    public void BoxNormalization_UsesAllTargetsNotMeanOfImageMeans(LossRoute route)
    {
        var predicted = Predictions(batch: 2, queries: 2);
        var targets = new Tensor<double>(new[]
        {
            0.0, 0.5, 0.5, 0.2, 0.4, -1.0, 0, 0, 0, 0,
            0.0, 0.5, 0.5, 0.4, 0.4, 1.0, 0.5, 0.5, 0.4, 0.4
        }, new[] { 2, 2, 5 });
        var loss = new DETRSetLoss<double>(numClasses: 3, classWeight: 0, boxL1Weight: 1, boxGIoUWeight: 0);
        Near(0.2 / 3, Evaluate(loss, predicted, targets, route));
    }

    [Theory]
    [InlineData(LossRoute.Scalar)]
    [InlineData(LossRoute.Tape)]
    public void MoreTargetsThanQueries_IsRejectedRatherThanDropped(LossRoute route)
    {
        var predicted = Predictions(batch: 1, queries: 1);
        var targets = new Tensor<double>(new[] { 0.0, 0.5, 0.5, 0.4, 0.4, 1.0, 0.5, 0.5, 0.4, 0.4 }, new[] { 1, 2, 5 });
        var loss = new DETRSetLoss<double>(numClasses: 3);
        var error = Assert.Throws<ArgumentException>(() => Evaluate(loss, predicted, targets, route));
        Assert.Equal("targets", error.ParamName);
    }

    [Fact]
    public void StructuredShapeWithInvalidLabels_CannotBeInterpretedAsElementwiseMae()
    {
        var predicted = Predictions(batch: 1, queries: 1);
        var target = Predictions(batch: 1, queries: 1);
        target[0, 0, 0] = -2;
        Assert.Throws<ArgumentException>(() => new DETRSetLoss<double>(numClasses: 3).ComputeTapeLoss(predicted, target));
    }

    [Fact]
    public void IdenticalNonstructuredShapes_KeepVectorMaeCompatibility()
    {
        var loss = new DETRSetLoss<double>(numClasses: 3);
        var predicted = new Tensor<double>(new[] { 1.0, 2.0, 5.0, 9.0 }, new[] { 2, 2 });
        var target = new Tensor<double>(new[] { 0.0, 4.0, 2.0, 5.0 }, new[] { 2, 2 });
        Near(2.5, loss.CalculateLoss(new Vector<double>(predicted.ToArray()), new Vector<double>(target.ToArray())));
        using var tape = new GradientTape<double>();
        var objective = loss.ComputeTapeLoss(predicted, target);
        Near(2.5, objective[0]);
        var gradients = tape.ComputeGradients(objective, new[] { predicted });
        Assert.True(gradients.TryGetValue(predicted, out var gradient));
        Assert.NotNull(gradient);
        Assert.Equal(new[] { 0.25, -0.25, 0.25, 0.25 }, gradient.ToArray());
    }

    [Fact]
    public void ScalarLoss_DoesNotDisposeOrMutateBorrowedPredictionsAndTargets()
    {
        var predicted = Predictions(batch: 1, queries: 1);
        var target = Target(0, 0.6, 0.55, 0.2, 0.3);
        var beforePredicted = predicted.ToArray();
        var beforeTarget = target.ToArray();
        var loss = new DETRSetLoss<double>(numClasses: 3);
        double first = loss.CalculateLoss(predicted, target);
        Near(first, loss.CalculateLoss(predicted, target));
        Assert.Equal(beforePredicted, predicted.ToArray());
        Assert.Equal(beforeTarget, target.ToArray());
        using var doubled = AiDotNetEngine.Current.TensorMultiplyScalar(predicted, 2.0);
        Assert.Equal(beforePredicted.Select(value => value * 2), doubled.ToArray());
    }

    [Fact]
    public void GIoU_ActualBoxDerivativeMatchesIndependentFiniteDifference()
    {
        var predicted = Predictions(batch: 1, queries: 1);
        var target = Target(0, 0.65, 0.57, 0.25, 0.31);
        var loss = new DETRSetLoss<double>(numClasses: 3, classWeight: 0, boxL1Weight: 0, boxGIoUWeight: 2);
        using var tape = new GradientTape<double>();
        var objective = loss.ComputeTapeLoss(predicted, target);
        var gradients = tape.ComputeGradients(objective, new[] { predicted });
        Assert.True(gradients.TryGetValue(predicted, out var gradient));
        Assert.NotNull(gradient);

        var coordinates = new[] { 0.5, 0.5, 0.4, 0.4 };
        var gold = CenterToCorners(new[] { 0.65, 0.57, 0.25, 0.31 });
        for (int coordinate = 0; coordinate < 4; coordinate++)
        {
            double expected = FiniteDifference(coordinates, coordinate,
                box => 2 * GIoULoss(CenterToCorners(box), gold));
            Assert.True(Math.Abs(expected) > 0.01, "The independent gradient must be nonvacuous.");
            Near(expected, gradient[0, 0, coordinate + 3], 2e-5);
        }
    }

    [Fact]
    public void EngineLogSoftmax_PrimitiveGradientCoversEveryClass()
    {
        var logits = new Tensor<double>(new[] { 1, 3 });
        var target = new Tensor<double>(new[] { 1, 3 });
        target[0, 0] = 1;
        var engine = AiDotNetEngine.Current;
        using var tape = new GradientTape<double>();
        var logProbabilities = engine.TensorLogSoftmax(logits, axis: 1);
        var objective = engine.TensorNegate(engine.ReduceSum(engine.TensorMultiply(logProbabilities, target), null));
        var gradients = tape.ComputeGradients(objective, new[] { logits });
        Assert.True(gradients.TryGetValue(logits, out var gradient));
        Assert.NotNull(gradient);
        Near(-2.0 / 3, gradient[0, 0]);
        Near(1.0 / 3, gradient[0, 1]);
        Near(1.0 / 3, gradient[0, 2]);
    }

    [Fact]
    public void EngineGIoU_PrimitiveGradientMatchesIndependentFiniteDifference()
    {
        var coordinates = new[] { 0.15, 0.1, 0.65, 0.62 };
        var gold = new[] { 0.45, 0.22, 0.88, 0.76 };
        var predicted = new Tensor<double>(coordinates, new[] { 1, 4 });
        var target = new Tensor<double>(gold, new[] { 1, 4 });
        var engine = AiDotNetEngine.Current;
        using var tape = new GradientTape<double>();
        var objective = engine.ReduceSum(engine.TensorGIoULoss(predicted, target), null);
        var gradients = tape.ComputeGradients(objective, new[] { predicted });
        Assert.True(gradients.TryGetValue(predicted, out var gradient));
        Assert.NotNull(gradient);
        Near(GIoULoss(coordinates, gold), objective[0], 1e-6);
        for (int coordinate = 0; coordinate < 4; coordinate++)
        {
            double expected = FiniteDifference(coordinates, coordinate, box => GIoULoss(box, gold));
            Near(expected, gradient[0, coordinate], 2e-5);
        }
    }

    private static Tensor<double> Predictions(int batch, int queries)
    {
        var tensor = new Tensor<double>(new[] { batch, queries, 7 });
        for (int image = 0; image < batch; image++)
            for (int query = 0; query < queries; query++)
            {
                tensor[image, query, 3] = 0.5;
                tensor[image, query, 4] = 0.5;
                tensor[image, query, 5] = 0.4;
                tensor[image, query, 6] = 0.4;
            }
        return tensor;
    }

    private static Tensor<double> Target(int label, double cx, double cy, double width, double height) =>
        new(new[] { (double)label, cx, cy, width, height }, new[] { 1, 1, 5 });

    private static Tensor<double> EmptyTarget() => Target(-1, 0, 0, 0, 0);

    private static double Evaluate(DETRSetLoss<double> loss, Tensor<double> predicted, Tensor<double> target, LossRoute route) =>
        route switch
        {
            LossRoute.Scalar => loss.CalculateLoss(predicted, target),
            LossRoute.Tape => loss.ComputeTapeLoss(predicted, target)[0],
            _ => throw new ArgumentOutOfRangeException(nameof(route))
        };

    private static double[] CenterToCorners(double[] box) =>
        new[] { box[0] - box[2] / 2, box[1] - box[3] / 2, box[0] + box[2] / 2, box[1] + box[3] / 2 };

    private static double GIoULoss(double[] predicted, double[] target)
    {
        double intersection = Math.Max(0, Math.Min(predicted[2], target[2]) - Math.Max(predicted[0], target[0]))
            * Math.Max(0, Math.Min(predicted[3], target[3]) - Math.Max(predicted[1], target[1]));
        double union = (predicted[2] - predicted[0]) * (predicted[3] - predicted[1])
            + (target[2] - target[0]) * (target[3] - target[1]) - intersection;
        double enclosure = (Math.Max(predicted[2], target[2]) - Math.Min(predicted[0], target[0]))
            * (Math.Max(predicted[3], target[3]) - Math.Min(predicted[1], target[1]));
        // The existing engine's GIoU contract adds 1e-7 to union and enclosure, including their
        // occurrence in the enclosure-union numerator. Model that rule explicitly, not by widening
        // tolerances. It matters at identical boxes, where the stabilized loss is slightly nonzero.
        const double epsilon = 1e-7;
        return 1 - intersection / (union + epsilon) + (enclosure - union) / (enclosure + epsilon);
    }

    private static double FiniteDifference(double[] point, int coordinate, Func<double[], double> evaluate)
    {
        const double epsilon = 1e-6;
        var plus = (double[])point.Clone();
        var minus = (double[])point.Clone();
        plus[coordinate] += epsilon;
        minus[coordinate] -= epsilon;
        return (evaluate(plus) - evaluate(minus)) / (2 * epsilon);
    }

    private static void Near(double expected, double actual, double tolerance = 1e-8) =>
        Assert.True(!double.IsNaN(actual) && !double.IsInfinity(actual) && Math.Abs(expected - actual) <= tolerance,
            $"Expected {expected:R}; actual {actual:R}; tolerance {tolerance:R}.");
}
