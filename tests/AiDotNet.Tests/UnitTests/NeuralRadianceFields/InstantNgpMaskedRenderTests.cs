using System;
using System.Reflection;
using AiDotNet.Models.Options;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields;

/// <summary>
/// Instant-NGP's training step queries every sample and masks the density of samples in empty occupancy cells, so the
/// step has one shape per batch size. It must render the color <see cref="InstantNGP{T}.RenderRays"/> renders by
/// skipping those samples, or training would fit a different image than inference shows.
/// </summary>
public class InstantNgpMaskedRenderTests
{
    [Fact]
    public void MaskedRender_MatchesTheOccupancySkippingRender()
    {
        var model = new InstantNGP<double>(new InstantNGPOptions<double>
        {
            HashTableSize = 64, NumLevels = 2, FeaturesPerLevel = 2, FinestResolution = 8, CoarsestResolution = 2,
            MlpHiddenDim = 8, MlpNumLayers = 1, FeatureDim = 4, ColorHiddenDim = 4, ColorNumLayers = 1,
            OccupancyGridResolution = 4, RenderSamples = 6,
            SceneMin = new Vector<double>(new[] { -2.0, -2.0, -2.0 }),
            SceneMax = new Vector<double>(new[] { 2.0, 2.0, 2.0 }),
            Seed = 3,
        });

        // Empty every other occupancy cell, so some samples are skipped by RenderRays and masked by the step.
        var bitfieldField = typeof(InstantNGP<double>).GetField("_occupancyBitfield", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(bitfieldField);
        if (bitfieldField?.GetValue(model) is not uint[] bitfield) throw new InvalidOperationException("no occupancy bitfield");
        for (int i = 0; i < bitfield.Length; i++) bitfield[i] = 0x55555555u;

        const int rays = 5, samples = 6;
        var origins = new Tensor<double>(new[] { rays, 3 });
        var directions = new Tensor<double>(new[] { rays, 3 });
        for (int r = 0; r < rays; r++)
        {
            origins[r * 3] = -1.5 + 0.6 * r;
            origins[r * 3 + 1] = 0.3 * Math.Sin(r);
            origins[r * 3 + 2] = -1.8;
            directions[r * 3] = 0.1 * r;
            directions[r * 3 + 2] = 1.0;
        }

        var expected = model.RenderRays(origins, directions, samples, 0.1, 3.5);

        var sample = typeof(InstantNGP<double>).GetMethod("SampleRaysWithOccupancy", BindingFlags.NonPublic | BindingFlags.Instance);
        var render = typeof(InstantNGP<double>).GetMethod("RenderSamples", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(sample);
        Assert.NotNull(render);
        var sampled = sample?.Invoke(model, new object[] { origins, directions, samples, 0.1, 3.5 });
        if (sampled is not ValueTuple<Tensor<double>, Tensor<double>, bool[], double[], double[], double[]> s)
            throw new InvalidOperationException("unexpected sampling result");
        int total = rays * samples;
        var occupancy = new Tensor<double>(new[] { total, 1 });
        int empty = 0;
        for (int i = 0; i < total; i++)
        {
            occupancy[i] = s.Item3[i] ? 1.0 : 0.0;
            if (!s.Item3[i]) empty++;
        }
        Assert.True(empty > 0 && empty < total, $"{empty} of {total} samples are empty; the check needs both kinds.");
        var spacings = AiDotNet.NeuralRadianceFields.Helpers.VolumeRenderer<double>.Spacings(rays, samples, s.Item5, s.Item6, s.Item4);
        var actual = render?.Invoke(model, new object[] { s.Item1, s.Item2, occupancy, spacings }) as Tensor<double>;
        Assert.NotNull(actual);

        Assert.Equal(expected.Length, actual?.Length ?? -1);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - (actual?[i] ?? double.NaN)) <= 1e-12,
                $"pixel component {i}: masked {actual?[i]:R} vs skipped {expected[i]:R}");
    }
}