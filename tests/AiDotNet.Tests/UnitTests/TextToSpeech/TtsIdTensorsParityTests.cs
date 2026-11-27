using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.TextToSpeech;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TextToSpeech;

/// <summary>
/// Pins <see cref="TtsIdTensors.From{T}"/> to the per-element loop the text-to-speech modules each carried:
/// <c>tensor[i] = NumOps.FromDouble(ids[i])</c>.
/// </summary>
public class TtsIdTensorsParityTests
{
    private static Tensor<T> Reference<T>(IReadOnlyList<int> ids)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = ops.FromDouble(ids[i]);
        return tensor;
    }

    public static TheoryData<int[]> IdLists() => new()
    {
        new[] { 0, 1, 2, 3 },
        new[] { 7 },
        new[] { 1023, 0, 512, 65535, 4 },
        // Past 2^24 a float cannot hold every integer: both conversions must round the same way
        new[] { 16777217, 16777219, 33554433, int.MaxValue, 123456789 },
        new[] { -1, -16777217, int.MinValue },
        Array.Empty<int>(),
        // A longer list with the same edge values
        Enumerable.Range(0, 40).Select(i => i % 5 == 0 ? 16777217 + i : i % 7 == 0 ? int.MinValue + i : i * 1009 - 7).ToArray(),
    };

    // Half takes the generic Cast path rather than the typed double and float paths
    [Theory]
    [MemberData(nameof(IdLists))]
    public void From_MatchesTheElementLoop_Half(int[] ids)
        => Assert.Equal(Reference<Half>(ids).ToArray(), TtsIdTensors.From<Half>(ids).ToArray());

    [Theory]
    [MemberData(nameof(IdLists))]
    public void From_MatchesTheElementLoop_Double(int[] ids)
    {
        var actual = TtsIdTensors.From<double>(ids);

        Assert.Equal(new[] { ids.Length }, actual.Shape.ToArray());
        Assert.Equal(Reference<double>(ids).ToArray(), actual.ToArray());
    }

    [Theory]
    [MemberData(nameof(IdLists))]
    public void From_MatchesTheElementLoop_Float(int[] ids)
    {
        var actual = TtsIdTensors.From<float>(ids);

        Assert.Equal(new[] { ids.Length }, actual.Shape.ToArray());
        Assert.Equal(Reference<float>(ids).ToArray(), actual.ToArray());
    }

    [Fact]
    public void From_AcceptsAnyReadOnlyList_AndDoesNotAliasAnArrayArgument()
    {
        var list = new List<int> { 5, 9, 2 };
        Assert.Equal(new[] { 5.0, 9.0, 2.0 }, TtsIdTensors.From<double>(list).ToArray());

        var array = new[] { 5, 9, 2 };
        var tensor = TtsIdTensors.From<double>(array);
        array[0] = 77;
        Assert.Equal(new[] { 5.0, 9.0, 2.0 }, tensor.ToArray());
    }

    [Fact]
    public void From_WithShape_FillsRowMajor_AndRejectsAMismatchedShape()
    {
        var ids = new[] { 4, 0, 16777217, -3, 9, 1 };

        var row = TtsIdTensors.From<float>(ids, new[] { 1, 6 });
        Assert.Equal(new[] { 1, 6 }, row.Shape.ToArray());
        Assert.Equal(Reference<float>(ids).ToArray(), row.ToArray());

        var grid = TtsIdTensors.From<double>(ids, new[] { 2, 3, 1 });
        Assert.Equal(new[] { 2, 3, 1 }, grid.Shape.ToArray());
        Assert.Equal(-3.0, grid[1, 0, 0]);

        Assert.Equal(new[] { 1, 0 }, TtsIdTensors.From<double>(Array.Empty<int>(), new[] { 1, 0 }).Shape.ToArray());
        Assert.Throws<ArgumentException>(() => TtsIdTensors.From<double>(ids, new[] { 4, 2 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => TtsIdTensors.From<double>(ids, new[] { -6, -1 }));
    }

    [Fact]
    public void FromGrid_KeepsOrTransposesTheLayout_LikeTheElementLoops()
    {
        var codes = new[,] { { 1, 2, 3 }, { 40, 50, 16777219 } };   // [codebooks 2, frames 3]

        var same = TtsIdTensors.FromGrid<float>(codes);
        Assert.Equal(new[] { 2, 3 }, same.Shape.ToArray());
        var transposed = TtsIdTensors.FromGrid<float>(codes, transpose: true);
        Assert.Equal(new[] { 3, 2 }, transposed.Shape.ToArray());

        var ops = MathHelper.GetNumericOperations<float>();
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < 3; c++)
            {
                Assert.Equal(ops.FromDouble(codes[r, c]), same[r, c]);
                Assert.Equal(ops.FromDouble(codes[r, c]), transposed[c, r]);
            }

        var row = TtsIdTensors.FromGridRow<double>(codes, 1);
        Assert.Equal(new[] { 3 }, row.Shape.ToArray());
        Assert.Equal(new[] { 40.0, 50.0, 16777219.0 }, row.ToArray());
        Assert.Throws<ArgumentOutOfRangeException>(() => TtsIdTensors.FromGridRow<double>(codes, 2));
    }
}
