using AiDotNet.Tensors.Groups;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Groups;

public class LieGroupTests
{
    [Fact]
    public void So3_ExpLog_IdentityRoundTrip()
    {
        var group = new So3Group<double>();
        var zero = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

        var identity = group.Exp(zero);
        var log = group.Log(identity);

        Assert.Equal(1.0, identity.Matrix[0, 0], precision: 12);
        Assert.Equal(1.0, identity.Matrix[1, 1], precision: 12);
        Assert.Equal(1.0, identity.Matrix[2, 2], precision: 12);
        Assert.Equal(0.0, log[0], precision: 12);
        Assert.Equal(0.0, log[1], precision: 12);
        Assert.Equal(0.0, log[2], precision: 12);
    }

    [Fact]
    public void Su2_ExpLog_IdentityRoundTrip()
    {
        var group = new Su2Group<double>();
        var zero = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

        var identity = group.Exp(zero);
        var log = group.Log(identity);

        Assert.Equal(1.0, identity.W, precision: 12);
        Assert.Equal(0.0, identity.X, precision: 12);
        Assert.Equal(0.0, identity.Y, precision: 12);
        Assert.Equal(0.0, identity.Z, precision: 12);
        Assert.Equal(0.0, log[0], precision: 12);
        Assert.Equal(0.0, log[1], precision: 12);
        Assert.Equal(0.0, log[2], precision: 12);
    }

    [Fact]
    public void Se3_ExpLog_IdentityRoundTrip()
    {
        var group = new Se3Group<double>();
        var zero = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 });

        var identity = group.Exp(zero);
        var log = group.Log(identity);

        Assert.Equal(1.0, identity.Rotation.Matrix[0, 0], precision: 12);
        Assert.Equal(1.0, identity.Rotation.Matrix[1, 1], precision: 12);
        Assert.Equal(1.0, identity.Rotation.Matrix[2, 2], precision: 12);
        Assert.Equal(0.0, identity.Translation[0], precision: 12);
        Assert.Equal(0.0, identity.Translation[1], precision: 12);
        Assert.Equal(0.0, identity.Translation[2], precision: 12);
        Assert.Equal(0.0, log[0], precision: 12);
        Assert.Equal(0.0, log[1], precision: 12);
        Assert.Equal(0.0, log[2], precision: 12);
        Assert.Equal(0.0, log[3], precision: 12);
        Assert.Equal(0.0, log[4], precision: 12);
        Assert.Equal(0.0, log[5], precision: 12);
    }
}
