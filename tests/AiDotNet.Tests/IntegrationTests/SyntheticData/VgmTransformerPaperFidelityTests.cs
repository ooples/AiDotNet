using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.SyntheticData;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

/// <summary>
/// Paper-fidelity guards for <see cref="TabularDataTransformer{T}"/>'s variational
/// Gaussian-mixture mode-specific normalization (Xu et al. 2019 §4.2 — the "VGM" the
/// whole tabular-GAN family shares). Pins the two behaviours the previous plain-EM +
/// argmax implementation got wrong:
///   1. the variational Dirichlet prior must DISCOVER the number of modes (a clean
///      2-component mixture must recover ~2 active modes near the true means, not the
///      configured maximum);
///   2. forward transform must SAMPLE the mode from the responsibilities, and the
///      forward→inverse round-trip must reconstruct continuous values closely.
/// </summary>
public class VgmTransformerPaperFidelityTests
{
    private const int Seed = 1234;

    /// <summary>
    /// A column drawn from a well-separated 2-component mixture
    /// (N(-5, 0.4) and N(+5, 0.4)) must be fit with ~2 active modes whose means sit
    /// near ±5, even though up to 10 modes are allowed. This is the variational
    /// auto-pruning the "V" in VGM provides; plain EM kept all 10 alive.
    /// </summary>
    [Fact]
    public void Vgm_RecoversTwoModeMixture_StructureAndMeans()
    {
        const int perMode = 300;
        var rng = new Random(Seed);
        var data = new Matrix<double>(perMode * 2, 1);
        for (int i = 0; i < perMode; i++)
            data[i, 0] = -5.0 + 0.4 * SampleNormal(rng);
        for (int i = 0; i < perMode; i++)
            data[perMode + i, 0] = 5.0 + 0.4 * SampleNormal(rng);

        var columns = new List<ColumnMetadata> { new("X", ColumnDataType.Continuous, columnIndex: 0) };

        var t = new TabularDataTransformer<double>(vgmModes: 10, random: new Random(Seed));
        t.Fit(data, columns);

        // transformedWidth = 1 (normalized) + numActiveModes. Recover the mode count.
        var info = t.GetTransformInfo(0);
        int activeModes = info.Width - 1;

        Assert.True(activeModes >= 2 && activeModes <= 3,
            $"VGM should recover ~2 modes from a clean 2-component mixture, got {activeModes} " +
            "(variational pruning regressed — plain EM would keep all 10).");

        // Forward then inverse should reconstruct each value within a few tenths
        // (mode-relative normalization is near-lossless for in-distribution data).
        var transformed = t.Transform(data);
        var recon = t.InverseTransform(transformed);
        double maxErr = 0;
        for (int i = 0; i < data.Rows; i++)
            maxErr = Math.Max(maxErr, Math.Abs(recon[i, 0] - data[i, 0]));
        Assert.True(maxErr < 1.0,
            $"VGM forward→inverse round-trip error {maxErr:F3} too large (mode normalization broken).");

        // Reconstructed values must cluster near ±5 (the two real modes), never in
        // the empty gap around 0 — proof the mode structure survived the round-trip.
        int nearLow = 0, nearHigh = 0, inGap = 0;
        for (int i = 0; i < recon.Rows; i++)
        {
            double v = recon[i, 0];
            if (Math.Abs(v - (-5.0)) < 2.0) nearLow++;
            else if (Math.Abs(v - 5.0) < 2.0) nearHigh++;
            else if (Math.Abs(v) < 2.0) inGap++;
        }
        Assert.True(nearLow > perMode / 2 && nearHigh > perMode / 2,
            $"Reconstructed values did not cluster at both modes (low={nearLow}, high={nearHigh}).");
        Assert.True(inGap < data.Rows / 20,
            $"Too many reconstructions ({inGap}) landed in the empty gap around 0 — mode collapse.");
    }

    private static double SampleNormal(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
