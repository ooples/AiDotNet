using System.Linq;
using AiDotNet.Safety.Watermarking;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// The SynthID-style watermarker against what Gowal et al. (2025, Sec. 8) state for SynthID-O: a 136-bit payload
/// without stated error correction, and decisions calibrated to a 0.1% false positive rate.
/// </summary>
public class SynthIdImageWatermarkerPaperTests
{
    private static SynthIdImageWatermarker<float> Small() =>
        new(new SynthIdImageOptions { ImageSize = 32, SecretBits = 136, Seed = 4 }, "paper-key");

    [Fact]
    public void Payload_IsAll136Bits_WithNoErrorCorrection()
    {
        var model = Small();
        Assert.False(new SynthIdImageOptions().UseErrorCorrection);
        var secret = model.KeySecret();
        Assert.Equal(136, secret.Length);
        // No parity and no padding: the key stream fills every bit, so both values occur throughout.
        Assert.Contains(true, secret.Skip(91));
        Assert.Contains(false, secret.Skip(91));
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task AtThePapersRate_ACalibrationTooSmallToReachIt_NeverClaimsAWatermark()
    {
        await System.Threading.Tasks.Task.Yield();
        // A conformal clean p-value is at least 1 / (n + 1), so a 0.1% rate needs n >= 999 clean calibration images
        // before any image can be called watermarked. With 24 the decision can only abstain or clear an image.
        Assert.Equal(0.001, new SynthIdImageOptions().FalsePositiveRate);
        var model = Small();
        const int n = 24;
        model.Calibrate(StegaStampWatermarkerTests.Images(RandomHelper.CreateSeededRandom(1), n),
            StegaStampWatermarkerTests.Images(RandomHelper.CreateSeededRandom(2), n));
        var probes = StegaStampWatermarkerTests.Images(RandomHelper.CreateSeededRandom(3), 8);
        for (int i = 0; i < probes.Shape[0]; i++)
        {
            var decision = model.Decide(probes.Slice(i));
            Assert.True(decision.CleanPValue >= 1.0 / (n + 1) - 1e-12);
            Assert.NotEqual(WatermarkVerdict.Watermarked, decision.Verdict);
        }
    }
}
