using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.TextDetection;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

public sealed class TextDetectionGeometryBoundaryReviewTests
{
    public enum Coordinate { Left, Top, Right, Bottom }
    public TextDetectionGeometryBoundaryReviewTests() => TestModuleInitializer.EnsureInitialized();

    public static IEnumerable<object[]> NonFiniteCases()
    {
        foreach (Coordinate coordinate in Enum.GetValues(typeof(Coordinate)))
            foreach (double value in new[] { double.NaN, double.NegativeInfinity, double.PositiveInfinity })
                yield return new object[] { coordinate, value };
    }

    [Theory]
    [MemberData(nameof(NonFiniteCases))]
    public void SharedRandomGeometryInvariant_RejectsEveryNonFiniteCoordinate(Coordinate coordinate, double value)
    {
        var (left, top, right, bottom) = coordinate switch
        {
            Coordinate.Left => (value, 0.0, 10.0, 10.0), Coordinate.Top => (0.0, value, 10.0, 10.0),
            Coordinate.Right => (0.0, 0.0, value, 10.0), Coordinate.Bottom => (0.0, 0.0, 10.0, value),
            _ => throw new ArgumentOutOfRangeException(nameof(coordinate))
        };
        var region = new TextRegion<double>(new BoundingBox<double>(left, top, right, bottom), 0.5);
        Assert.ThrowsAny<Xunit.Sdk.XunitException>(() => TextDetectionTestBase<double>.AssertBoxGeometricallyValid(region));
    }

    [Theory]
    [InlineData(0, 0, 10, 10)]
    [InlineData(-20, -12, 28, 20)]
    public void SharedRandomGeometryInvariant_AcceptsFinitePositiveAreaIncludingUnclippedText(double left, double top, double right, double bottom)
        => TextDetectionTestBase<double>.AssertBoxGeometricallyValid(
            new TextRegion<double>(new BoundingBox<double>(left, top, right, bottom), 0.5));
}
