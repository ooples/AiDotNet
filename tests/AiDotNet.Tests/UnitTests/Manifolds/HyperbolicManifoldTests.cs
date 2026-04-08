using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Manifolds;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Manifolds;

public class HyperbolicManifoldTests
{
    [Fact]
    public void PoincareMobiusAdd_IdentityPreservesPoint()
    {
        var manifold = new PoincareBallManifold<double>();
        var x = new Vector<double>(new[] { 0.1, -0.2 });
        var zero = new Vector<double>(new[] { 0.0, 0.0 });

        var sum = manifold.MobiusAdd(x, zero);

        Assert.Equal(x[0], sum[0], precision: 10);
        Assert.Equal(x[1], sum[1], precision: 10);
    }

    [Fact]
    public void PoincareExpLog_RoundTripAtOrigin()
    {
        var manifold = new PoincareBallManifold<double>();
        var basePoint = new Vector<double>(new[] { 0.0, 0.0 });
        var tangent = new Vector<double>(new[] { 0.05, -0.02 });

        var point = manifold.ExpMap(tangent, basePoint);
        var recovered = manifold.LogMap(point, basePoint);

        Assert.Equal(tangent[0], recovered[0], precision: 8);
        Assert.Equal(tangent[1], recovered[1], precision: 8);
    }

    [Fact]
    public void HyperboloidDistance_SamePointIsZero()
    {
        var manifold = new HyperboloidManifold<double>();
        var x = new Vector<double>(new[] { 1.0, 0.0, 0.0 });

        var distance = manifold.Distance(x, x);

        Assert.Equal(0.0, distance, precision: 10);
    }

    [Fact]
    public void HyperboloidExpLog_RoundTrip()
    {
        var manifold = new HyperboloidManifold<double>();
        var basePoint = new Vector<double>(new[] { 1.0, 0.0, 0.0 });
        var tangent = new Vector<double>(new[] { 0.0, 0.1, 0.0 });

        var point = manifold.ExpMap(tangent, basePoint);
        var recovered = manifold.LogMap(point, basePoint);

        Assert.Equal(tangent[0], recovered[0], precision: 8);
        Assert.Equal(tangent[1], recovered[1], precision: 8);
        Assert.Equal(tangent[2], recovered[2], precision: 8);
    }
}
