using System;
using System.Threading.Tasks;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models;

/// <summary>
/// Reference behaviour for <see cref="VectorModel{T}"/> and the <see cref="ModelBase{T, TInput, TOutput}"/>
/// serialization it inherits, pinned where the engine-op conversion (#2490) fixed a wrong result.
/// </summary>
public class VectorModelReferenceTests
{
    [Fact]
    public void SerializeDeserialize_Half_RoundTripsCoefficients()
    {
        // Half is not IConvertible, so the former Convert.ToDouble serialization threw InvalidCastException.
        var coefficients = new Vector<Half>(new[] { (Half)1.5, (Half)(-2.25), (Half)0.125 });
        using var source = new VectorModel<Half>(coefficients);

        byte[] payload = source.Serialize();
        using var restored = new VectorModel<Half>(new Vector<Half>(3));
        restored.Deserialize(payload);

        Assert.Equal(3, restored.Coefficients.Length);
        Assert.Equal((Half)1.5, restored.Coefficients[0]);
        Assert.Equal((Half)(-2.25), restored.Coefficients[1]);
        Assert.Equal((Half)0.125, restored.Coefficients[2]);
    }

    [Fact]
    public async Task GetPartialDependenceAsync_OnePointGrid_IsTheGridStart()
    {
        // The grid spans [-1, 1]; a one-point grid is its start, -1 (the former i / (n - 1) step gave 0 / 0 = NaN).
        using var model = new VectorModel<double>(new Vector<double>(new[] { 2.0, 3.0 }));

        var pd = await model.GetPartialDependenceAsync(new Vector<int>(new[] { 1 }), gridResolution: 1);

        Assert.Equal(-1.0, pd.GridValues[1][0]);
        Assert.Equal(1, pd.PartialDependenceValues.Columns);
        Assert.Equal(-3.0, pd.PartialDependenceValues[0, 0], 1e-12);
    }
}
