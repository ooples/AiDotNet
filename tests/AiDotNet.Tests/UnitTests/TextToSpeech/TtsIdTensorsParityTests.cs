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
}
