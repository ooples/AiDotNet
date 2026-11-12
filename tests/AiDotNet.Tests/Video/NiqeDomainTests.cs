using AiDotNet.Video.Enhancement;
using Xunit;

namespace AiDotNet.Tests.Video;

public sealed class NiqeDomainTests
{
    [Fact]
    public void SingleUsablePatchIsUnscorableInsteadOfCrashingThePseudoinverse()
    {
        var image = new double[96, 96];
        var random = new Random(2294);
        for (int row = 0; row < 96; row++)
            for (int column = 0; column < 96; column++) image[row, column] = 255 * random.NextDouble();
        Assert.True(double.IsNaN(Niqe.Score(image)));
    }

    [Fact]
    public void NoUsablePatchIsUnscorableInsteadOfCrashingThePseudoinverse()
    {
        // A black image has no left/right AGGD samples, so its feature rows are all invalid.
        Assert.True(double.IsNaN(Niqe.Score(new double[192, 192])));
    }
}
