using System;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Regression tests for two defects in how <c>LayerTestBase.Serialize_Deserialize_ShouldPreserveBehavior</c>
/// compared a layer against itself. Both made correct layers report as broken.
/// </summary>
/// <remarks>
/// <para>
/// <b>Defect 1 — a difference-based tolerance cannot compare non-finite values.</b> The replay
/// comparison asserted <c>Math.Abs(a - b) &lt; 1e-12</c>. For two IDENTICAL infinities that is
/// <c>-inf - -inf = NaN</c>, <c>Math.Abs(NaN) = NaN</c>, and <c>NaN &lt; 1e-12</c> is false, so the
/// assertion fired on bit-identical values. <see cref="ALiBiPositionalBiasLayer{T}"/> hit this because
/// it masks with true <c>-Infinity</c> BY DESIGN (so <c>exp(-inf) = 0</c> exactly, where a large finite
/// sentinel can leak attention weight) and is annotated <c>ProducesNonFiniteOutput = true</c>. Matching
/// infinities by equality rather than by subtraction is the standard numeric convention — NumPy's
/// <c>allclose</c> does the same, and gates NaN behind <c>equal_nan</c>.
/// </para>
/// <para>
/// <b>Defect 2 — replaying a stateful layer without resetting it.</b> The same test ran a second
/// <c>Forward</c> to check that serializing had not perturbed the layer, but did not call
/// <c>ResetState()</c> first, while the sibling comparison ten lines below did. For a stateful layer
/// the second forward legitimately differs because the recurrence advanced, so the assertion blamed
/// serialization for ordinary statefulness.
/// </para>
/// </remarks>
public class NonFiniteAndStatefulComparisonRegressionTests
{
    /// <summary>
    /// The exact arithmetic that produced the false failure, pinned directly: a difference-based
    /// tolerance rejects equal infinities, an equality-first comparison accepts them.
    /// </summary>
    [Theory]
    [InlineData(double.NegativeInfinity)]
    [InlineData(double.PositiveInfinity)]
    public void EqualNonFiniteValues_AreNotReportedAsDiffering(double value)
    {
        double a = value, b = value;

        // What the old assertion computed. Documented here so the regression is unambiguous.
        Assert.True(double.IsNaN(Math.Abs(a - b)),
            "subtracting two equal infinities must yield NaN — this is why the old check failed");
        Assert.False(Math.Abs(a - b) < 1e-12,
            "the difference-based tolerance rejects values that are bit-identical");

        // What the comparison does now.
        Assert.True(a.Equals(b), "equality must accept two identical non-finite values");
    }

    [Fact]
    public void FiniteDrift_IsStillRejected()
    {
        // The equality escape must not become a blanket pass: genuinely different finite values still
        // have to fail the tolerance.
        double a = 1.0, b = 1.0 + 1e-6;

        Assert.False(a.Equals(b));
        Assert.False(Math.Abs(a - b) < 1e-12, "real drift between finite values must still be caught");
    }

    /// <summary>
    /// ALiBi masks with true -Infinity by design, so its bias tensor is expected to contain
    /// non-finite entries and to be REPRODUCIBLE across calls. Both properties together are what the
    /// serialization comparison needs.
    /// </summary>
    [Fact]
    public void ALiBiBias_ContainsMaskedInfinities_AndIsReproducible()
    {
        var layer = new ALiBiPositionalBiasLayer<double>(numHeads: 4);

        var first = layer.ComputeBias(queryLen: 6, keyLen: 6);
        var second = layer.ComputeBias(queryLen: 6, keyLen: 6);

        int negativeInfinities = 0;
        for (int i = 0; i < first.Length; i++)
        {
            if (double.IsNegativeInfinity(first[i])) negativeInfinities++;

            // Equality — not a difference — because the tensor deliberately holds -Infinity.
            Assert.True(first[i].Equals(second[i]),
                $"ALiBi bias element [{i}] is not reproducible: {first[i]:G17} vs {second[i]:G17}");
        }

        Assert.True(negativeInfinities > 0,
            "causal ALiBi must mask future positions with -Infinity; without any the test proves nothing");
    }

    /// <summary>
    /// A stateful layer advances on every forward, so two consecutive forwards over the same input
    /// are NOT required to agree — but a forward after ResetState must reproduce the first one. That
    /// is the property the serialization comparison depends on.
    /// </summary>
    [Fact]
    public void StatefulLayer_ReproducesItsFirstForward_OnlyAfterResetState()
    {
        var layer = new ReservoirLayer<double>(inputSize: 4, reservoirSize: 8);
        layer.SetTrainingMode(false);

        var input = new Tensor<double>(new[] { 1, 4 });
        for (int i = 0; i < input.Length; i++) input[i] = 0.25 * (i + 1);

        var first = layer.Forward(input).Clone();

        layer.ResetState();
        var afterReset = layer.Forward(input).Clone();

        Assert.Equal(first.Length, afterReset.Length);
        for (int i = 0; i < first.Length; i++)
        {
            Assert.True(Math.Abs(first[i] - afterReset[i]) < 1e-12,
                $"a reset reservoir must reproduce its first forward at [{i}]: " +
                $"{first[i]:G17} vs {afterReset[i]:G17}");
        }
    }
}
