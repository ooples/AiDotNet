using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.Serialization;
using Xunit;

namespace AiDotNetTests.UnitTests.Serialization;

public sealed class LayerStateBagTests
{
    [Fact]
    public void Nullable_state_distinguishes_null_empty_and_legacy_values()
    {
        var bag = new LayerStateBag(new Dictionary<string, object>
        {
            ["nullText"] = LayerStateBag.FormatNullable((string?)null),
            ["emptyText"] = LayerStateBag.FormatNullable(string.Empty),
            ["legacyText"] = "before-tags",
            ["nullArray"] = LayerStateBag.FormatNullable((int[]?)null),
            ["emptyArray"] = LayerStateBag.FormatNullable(Array.Empty<int>()),
            ["legacyArray"] = "2,3,5",
        }, "ProbeLayer");

        Assert.Null(bag.NullableString("nullText"));
        Assert.Equal(string.Empty, bag.NullableString("emptyText"));
        Assert.Equal("before-tags", bag.NullableString("legacyText"));
        Assert.Null(bag.NullableInt32Array("nullArray"));
        Assert.Empty(bag.NullableInt32Array("emptyArray")!);
        Assert.Equal(new[] { 2, 3, 5 }, bag.NullableInt32Array("legacyArray"));
    }

    [Fact]
    public void Enum_arrays_round_trip_by_name()
    {
        var expected = new[] { PNAAggregator.Max, PNAAggregator.StdDev };
        var bag = new LayerStateBag(new Dictionary<string, object>
        {
            ["aggregators"] = LayerStateBag.FormatEnumArray(expected),
        }, "ProbeLayer");

        Assert.Equal(expected, bag.EnumArray<PNAAggregator>("aggregators"));
    }

    [Fact]
    public void Live_json_configuration_is_deep_copied_through_the_durable_representation()
    {
        var config = new FlashAttentionConfig
        {
            BlockSizeQ = 17,
            UseCausalMask = true,
            Precision = FlashAttentionPrecision.Mixed,
        };
        var bag = new LayerStateBag(new Dictionary<string, object>
        {
            ["config"] = config,
        }, "ProbeLayer");

        var copy = bag.JsonObject<FlashAttentionConfig>("config");

        Assert.NotSame(config, copy);
        Assert.Equal(17, copy.BlockSizeQ);
        Assert.True(copy.UseCausalMask);
        Assert.Equal(FlashAttentionPrecision.Mixed, copy.Precision);
    }

    [Fact]
    public void Live_rectangular_reference_array_is_deep_copied_without_flattening()
    {
        var source = new[,]
        {
            { new List<int> { 1, 2 }, new List<int> { 3 } },
            { new List<int> { 4 }, new List<int> { 5, 6 } },
        };
        var bag = new LayerStateBag(new Dictionary<string, object>
        {
            ["grid"] = source,
        }, "RectangularArrayProbe");

        var copy = bag.CloneObject<List<int>[,]>("grid");

        Assert.Equal(source.GetLength(0), copy.GetLength(0));
        Assert.Equal(source.GetLength(1), copy.GetLength(1));
        Assert.NotSame(source, copy);
        Assert.NotSame(source[0, 0], copy[0, 0]);
        Assert.Equal(source[0, 0], copy[0, 0]);

        copy[0, 0].Add(99);
        Assert.Equal(new[] { 1, 2 }, source[0, 0]);
    }

    [Fact]
    public void Live_parametric_activation_preserves_configuration_in_an_independent_copy()
    {
        var source = new LeakyReLUActivation<float>(alpha: 0.2);
        var bag = new LayerStateBag(new Dictionary<string, object>
        {
            ["activation"] = source,
        }, "ActivationProbe");

        var copy = bag.Component<IActivationFunction<float>>("activation");

        var typed = Assert.IsType<LeakyReLUActivation<float>>(copy);
        Assert.NotSame(source, typed);
        Assert.Equal(source.Alpha, typed.Alpha);
    }
}
