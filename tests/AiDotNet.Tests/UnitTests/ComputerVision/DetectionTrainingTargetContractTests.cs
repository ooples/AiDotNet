using AiDotNet.ComputerVision.Detection;
using AiDotNet.ComputerVision.Detection.Losses;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

public sealed class DetectionTrainingTargetContractTests
{
    public DetectionTrainingTargetContractTests() => TestModuleInitializer.EnsureInitialized();

    public enum Coordinate { CenterX, CenterY, Width, Height }

    public static IEnumerable<object[]> InvalidCoordinates()
    {
        foreach (Coordinate coordinate in Enum.GetValues(typeof(Coordinate)))
        {
            foreach (double value in new[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, -0.01, 1.01 })
                yield return new object[] { coordinate, value };
            if (coordinate is Coordinate.Width or Coordinate.Height)
                yield return new object[] { coordinate, 0.0 };
        }
    }

    [Theory]
    [MemberData(nameof(InvalidCoordinates))]
    public void InvalidOrDegenerateTarget_IsRejected(Coordinate coordinate, double value)
    {
        double cx = coordinate == Coordinate.CenterX ? value : 0.5;
        double cy = coordinate == Coordinate.CenterY ? value : 0.5;
        double width = coordinate == Coordinate.Width ? value : 0.2;
        double height = coordinate == Coordinate.Height ? value : 0.2;
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => new DetectionTrainingTarget<double>(0, cx, cy, width, height));
        string expected = coordinate switch
        {
            Coordinate.CenterX => "centerX", Coordinate.CenterY => "centerY",
            Coordinate.Width => "width", Coordinate.Height => "height",
            _ => throw new ArgumentOutOfRangeException(nameof(coordinate))
        };
        Assert.Equal(expected, error.ParamName);
    }

    [Fact]
    public void PixelXywhConversion_UsesIndependentWidthAndHeight()
    {
        var target = DetectionTrainingTarget<double>.FromPixelXywh(7, 40, 30, 80, 60, imageWidth: 400, imageHeight: 200);
        Assert.Equal(7, target.ClassId);
        Assert.Equal(0.2, target.CenterX, 12);
        Assert.Equal(0.3, target.CenterY, 12);
        Assert.Equal(0.2, target.Width, 12);
        Assert.Equal(0.3, target.Height, 12);
    }

    [Fact]
    public void BatchOwnsListsAndTargetsExposeNoMutableProperties()
    {
        var original = new DetectionTrainingTarget<double>(1, 0.5, 0.5, 0.2, 0.3);
        var image = new List<DetectionTrainingTarget<double>> { original };
        var images = new List<List<DetectionTrainingTarget<double>>> { image, new() };
        var batch = new DetectionTrainingBatch<double>(images);
        image.Clear();
        images.Clear();
        Assert.Equal(2, batch.ImageCount);
        Assert.Equal(1, batch.TargetCount);
        Assert.Same(original, Assert.Single(batch[0]));
        Assert.Empty(batch[1]);
        Assert.All(typeof(DetectionTrainingTarget<double>).GetProperties(), property => Assert.False(property.CanWrite));
        var mutableInterface = Assert.IsAssignableFrom<IList<DetectionTrainingTarget<double>>>(batch[0]);
        Assert.Throws<NotSupportedException>(() => mutableInterface.Clear());
    }

    [Fact]
    public void CocoAdapter_PreservesContiguousClassIdsAndDoesNotConfuseXywhWithCenters()
    {
        // CocoDetectionDataLoader maps raw category IDs to contiguous indices starting at zero.
        // This is its emitted format, not raw annotation JSON with sparse category IDs.
        var labels = new Tensor<double>(new[]
        {
            0.0, 0.1, 0.15, 0.2, 0.3,
            79.0, 0.2, 0.3, 0.4, 0.2,
            0.0, 0, 0, 0, 0
        }, new[] { 1, 3, 5 });
        var batch = DetectionTrainingBatch<double>.FromPaddedCoco(labels);
        labels.Fill(0);
        Assert.Equal(2, batch.TargetCount);
        Assert.Equal(new[] { 0, 79 }, batch[0].Select(target => target.ClassId));
        Assert.Equal(0.2, batch[0][0].CenterX, 12);
        Assert.Equal(0.3, batch[0][0].CenterY, 12);
        Assert.Equal(0.4, batch[0][1].CenterX, 12);
        Assert.Equal(0.4, batch[0][1].CenterY, 12);
    }

    [Theory]
    [InlineData(-1.0)]
    [InlineData(0.5)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(2147483648.0)]
    public void CocoAdapter_RejectsInvalidClassRatherThanCastingIt(double label)
    {
        var labels = new Tensor<double>(new[] { label, 0.1, 0.1, 0.2, 0.2 }, new[] { 1, 1, 5 });
        Assert.Throws<ArgumentException>(() => DetectionTrainingBatch<double>.FromPaddedCoco(labels));
    }

    [Fact]
    public void CocoAdapter_RejectsPartialPaddingButAllowsWholeEmptyImages()
    {
        var empty = DetectionTrainingBatch<double>.FromPaddedCoco(new Tensor<double>(new[] { 2, 2, 5 }));
        Assert.Equal(2, empty.ImageCount);
        Assert.Equal(0, empty.TargetCount);
        var invalid = new Tensor<double>(new[] { 0.0, 0.1, 0, 0, 0 }, new[] { 1, 1, 5 });
        Assert.Throws<ArgumentOutOfRangeException>(() => DetectionTrainingBatch<double>.FromPaddedCoco(invalid));
    }

    [Fact]
    public void TypedLoss_RejectsBackgroundLabelAndExcessTargets()
    {
        var loss = new DETRSetLoss<double>(numClasses: 3);
        var logits = new Tensor<double>(new[] { 1, 1, 3 });
        var boxes = new Tensor<double>(new[] { 0.5, 0.5, 0.4, 0.4 }, new[] { 1, 1, 4 });
        var background = new DetectionTrainingBatch<double>(new[] { new[] { new DetectionTrainingTarget<double>(2, 0.5, 0.5, 0.2, 0.2) } });
        Assert.Throws<ArgumentException>(() => loss.ComputeTapeLoss(logits, boxes, background));
        var tooMany = new DetectionTrainingBatch<double>(new[] { new[]
        {
            new DetectionTrainingTarget<double>(0, 0.5, 0.5, 0.2, 0.2),
            new DetectionTrainingTarget<double>(1, 0.4, 0.4, 0.2, 0.2)
        } });
        Assert.Throws<ArgumentException>(() => loss.ComputeTapeLoss(logits, boxes, tooMany));
    }

    [Fact]
    public void ScalarTypedLoss_PreservesBorrowedHeads()
    {
        var loss = new DETRSetLoss<double>(numClasses: 3);
        var logits = new Tensor<double>(new[] { 1, 1, 3 });
        var boxes = new Tensor<double>(new[] { 0.5, 0.5, 0.4, 0.4 }, new[] { 1, 1, 4 });
        var targets = new DetectionTrainingBatch<double>(new[] { Array.Empty<DetectionTrainingTarget<double>>() });
        double first = loss.CalculateLoss(logits, boxes, targets);
        Assert.Equal(first, loss.CalculateLoss(logits, boxes, targets));
        Assert.Equal(new[] { 0.0, 0.0, 0.0 }, logits.ToArray());
        Assert.Equal(new[] { 0.5, 0.5, 0.4, 0.4 }, boxes.ToArray());
    }
}
