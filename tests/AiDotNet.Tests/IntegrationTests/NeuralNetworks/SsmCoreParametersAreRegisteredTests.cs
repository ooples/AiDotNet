using System;
using System.IO;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Pins that the state-space layers expose their FULL parameter surface, not just their
/// input/output projections.
/// </summary>
/// <remarks>
/// <para>
/// Each of these layers' <c>UpdateParameters</c> applies gradients to every weight tensor it owns,
/// but only the projections carried <c>[TrainableParameter]</c>. The unregistered tensors -- the
/// state-space core itself: A, B, C, D and the discretization timescale -- were therefore invisible
/// to <c>GetParameters</c>/<c>SetParameters</c>, and so to serialization: saving a trained S4D
/// silently dropped the model it had learned and kept only the projections around it.
/// </para>
/// <para>
/// Measured on S4DLayer at the shape below: 40 parameters before, 144 after. The round trip
/// restored every parameter bit-exactly in BOTH cases -- the assertion the invariant suite already
/// made -- yet output still drifted by 4.147e-03, because the tensors that actually differed were
/// not in the parameter vector at all. LinearRecurrentUnitLayer made the point sharper: its
/// restored output sat 1.178 away from the original while a completely fresh random layer sat
/// 0.593 away, so restoring was worse than not restoring.
/// </para>
/// </remarks>
public class SsmCoreParametersAreRegisteredTests
{
    private static Tensor<double> Ramp(int[] shape)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = 0.01 * (i + 1);
        return t;
    }

    /// <summary>
    /// The count is a floor, not an equality: it is the projection-only surface these layers used to
    /// report, so the test fails if the core is ever silently dropped again, while remaining valid
    /// if a layer legitimately gains parameters.
    /// </summary>
    [Theory]
    [InlineData("S4D", 40)]
    [InlineData("S5", 40)]
    [InlineData("LRU", 40)]
    [InlineData("Hyena", 16)]
    public void StateSpaceCore_IsPartOfTheParameterSurface(string which, int projectionOnlyCount)
    {
        var layer = Make(which);
        layer.SetTrainingMode(false);
        layer.Forward(Ramp([1, 4, 4]));

        int count = layer.GetParameters().Length;
        Assert.True(count > projectionOnlyCount,
            $"{which} reported {count} parameters, which is not more than the {projectionOnlyCount} " +
            "the projections alone account for -- the state-space core is missing from the surface");
    }

    [Theory]
    [InlineData("S4D")]
    [InlineData("S5")]
    [InlineData("LRU")]
    [InlineData("Hyena")]
    public void Output_IsIdenticalAfterARoundTrip(string which)
    {
        var input = Ramp([1, 4, 4]);

        var original = Make(which);
        original.SetTrainingMode(false);
        original.ResetState();
        var expected = original.Forward(input).Clone();

        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            original.Serialize(writer);

        var restored = Make(which);
        ms.Position = 0;
        using (var reader = new BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            restored.Deserialize(reader);
        restored.SetTrainingMode(false);
        restored.ResetState();
        var actual = restored.Forward(input).Clone();

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(expected[i] == actual[i],
                $"{which} output[{i}] differs after a round trip: " +
                $"{expected[i]:G17} vs {actual[i]:G17}");
        }
    }

    /// <summary>
    /// The control that gives the test above its meaning: two independently constructed layers are
    /// randomly initialized and must NOT agree. Without this, a layer that emitted a constant would
    /// satisfy the round-trip check while proving nothing.
    /// </summary>
    [Theory]
    [InlineData("S4D")]
    [InlineData("S5")]
    [InlineData("LRU")]
    public void TwoFreshLayers_Disagree(string which)
    {
        var input = Ramp([1, 4, 4]);

        var a = Make(which);
        a.SetTrainingMode(false);
        a.ResetState();
        var outA = a.Forward(input).Clone();

        var b = Make(which);
        b.SetTrainingMode(false);
        b.ResetState();
        var outB = b.Forward(input).Clone();

        double maxDifference = 0;
        for (int i = 0; i < outA.Length; i++)
            maxDifference = Math.Max(maxDifference, Math.Abs(outA[i] - outB[i]));

        Assert.True(maxDifference > 1e-9,
            $"{which}: two independently initialized layers produced the same output " +
            $"(max difference {maxDifference:E3}), so the round-trip test above is vacuous");
    }

    private static LayerBase<double> Make(string which) => which switch
    {
        "S4D" => new S4DLayer<double>(4, 4, 4),
        "S5" => new S5Layer<double>(4, 4, 4),
        "LRU" => new LinearRecurrentUnitLayer<double>(4, 4, 4),
        "Hyena" => new HyenaLayer<double>(4, 4, 2, 4),
        _ => throw new ArgumentOutOfRangeException(nameof(which), which, "unknown layer"),
    };
}
