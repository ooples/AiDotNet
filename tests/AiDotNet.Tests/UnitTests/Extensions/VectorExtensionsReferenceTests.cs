using System;
using AiDotNet.Extensions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Extensions;

/// <summary>
/// Reference values for the <see cref="VectorExtensions"/> members whose engine-op conversion (#2490) also
/// fixed a wrong result.
/// </summary>
public class VectorExtensionsReferenceTests
{
    [Fact]
    public void PointwiseSqrt_Half_ReturnsSquareRoots()
    {
        // Half is not IConvertible, so the former Convert.ToDouble route threw InvalidCastException.
        var vector = new Vector<Half>(new[] { (Half)4, (Half)9, (Half)16 });

        var roots = vector.PointwiseSqrt();

        Assert.Equal(3, roots.Length);
        Assert.Equal((Half)2, roots[0]);
        Assert.Equal((Half)3, roots[1]);
        Assert.Equal((Half)4, roots[2]);
    }
}
