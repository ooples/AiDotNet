using System.Linq;
using AiDotNet.Safety.Watermarking;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// StegaStamp (Tancik et al., 2020) checked against reference values: the JPEG tables of its compression noise and
/// the layout of its error-corrected message.
/// </summary>
public class StegaStampWatermarkerPaperTests
{
    private static StegaStampWatermarker<float> Small(bool errorCorrection = true) =>
        new(new StegaStampOptions { ImageSize = 16, SecretBits = 100, UseErrorCorrection = errorCorrection, Seed = 2 }, "paper-key");

    [Fact]
    public void JpegNoise_UsesTheStandardTables_ScaledByTheIjgQualityRule()
    {
        // Sec. 4.2 trains through a differentiable JPEG. At quality 50 the steps are the ITU-T T.81 Annex K tables;
        // at quality 75 the IJG rule gives the familiar 8 6 5 8 12 20 26 31 first luminance row.
        var model = Small();
        var q50 = model.QuantizationTable(1, 16, 50);
        var q75 = model.QuantizationTable(1, 16, 75);
        int blocks = 4;
        double[] luminance50 = { 16, 11, 10, 16, 24, 40, 51, 61 };
        double[] chrominance50 = { 17, 18, 24, 47, 99, 99, 99, 99 };
        double[] luminance75 = { 8, 6, 5, 8, 12, 20, 26, 31 };
        for (int k = 0; k < 8; k++)
        {
            Assert.Equal(luminance50[k], q50[0, k]);
            Assert.Equal(chrominance50[k], q50[blocks, k]);
            Assert.Equal(chrominance50[k], q50[2 * blocks, k]);
            Assert.Equal(luminance75[k], q75[0, k]);
        }

        // Every block of a channel shares the channel's table.
        for (int blk = 1; blk < blocks; blk++)
            for (int k = 0; k < 64; k++)
                Assert.Equal(q50[0, k], q50[blk, k]);
    }

    [Fact]
    public void KeyedMessage_Is56DataBits_ProtectedByBch_InThe100BitSecret()
    {
        // Sec. 4.1 and 5.4: a 100-bit secret carrying 56 data bits under a BCH code that corrects 5 errors.
        var model = Small();
        var secret = model.KeySecret();
        Assert.Equal(100, secret.Length);
        var code = BchCode.StegaStamp();
        Assert.Equal(56, code.DataBits);
        Assert.Equal(5, code.Capacity);

        var codeword = secret.Take(code.CodewordBits).ToArray();
        var decoded = code.Decode(codeword);
        Assert.True(decoded.HasValue);
        Assert.Equal(0, decoded!.Value.Corrected);
        Assert.Equal(codeword.Take(56).ToArray(), decoded.Value.Data);
        Assert.All(secret.Skip(code.CodewordBits), bit => Assert.False(bit));

        // Without error correction all 100 bits are keyed data, and the data bits are the same key stream.
        var plain = Small(errorCorrection: false).KeySecret();
        Assert.Equal(secret.Take(56), plain.Take(56));
    }
}
