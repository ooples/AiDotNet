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

    /// <summary>
    /// The residual loss's border map is the reference's two passes over an s x s array (rows i and -i, then columns,
    /// for i below s / 4, each multiplied by (cos(4 pi i / s + pi) + 1) / 2), computed here as one factor per row times
    /// one per column.
    /// </summary>
    [Theory]
    [InlineData(16)]
    [InlineData(400)]
    public void EdgeFalloff_IsTheReferencesRowThenColumnPasses(int s)
    {
        var falloff = new double[s, s];
        for (int y = 0; y < s; y++)
            for (int x = 0; x < s; x++) falloff[y, x] = 1.0;
        for (int i = 0; i < s / 4; i++)
        {
            double factor = (System.Math.Cos(4 * System.Math.PI * i / s + System.Math.PI) + 1) / 2;
            int minus = i == 0 ? 0 : s - i;   // numpy's row -i
            for (int x = 0; x < s; x++) { falloff[minus, x] *= factor; falloff[i, x] *= factor; }
        }

        for (int j = 0; j < s / 4; j++)
        {
            double factor = (System.Math.Cos(4 * System.Math.PI * j / s + System.Math.PI) + 1) / 2;
            int minus = j == 0 ? 0 : s - j;
            for (int y = 0; y < s; y++) { falloff[y, minus] *= factor; falloff[y, j] *= factor; }
        }

        var edge = StegaStampWatermarker<double>.EdgeFactors(s);
        for (int y = 0; y < s; y++)
            for (int x = 0; x < s; x++)
                Assert.True(System.Math.Abs(edge[y] * edge[x] - falloff[y, x]) <= 1e-15, $"({y}, {x}): {edge[y] * edge[x]} against {falloff[y, x]}");
    }

    /// <summary>
    /// An untrained network hides nothing readable, so it is not ready; one training step makes it ready, and the step
    /// count is model state, so a deserialized copy stays ready.
    /// </summary>
    [Fact]
    public void IsReady_FollowsTraining_AndSurvivesSerialization()
    {
        var model = Small(errorCorrection: false);
        Assert.False(model.IsReady);

        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(3);
        var images = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 2, 3, 16, 16 });
        for (int i = 0; i < images.Length; i++) images.SetFlat(i, (float)rng.NextDouble());
        var secrets = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 2, 100 });
        for (int i = 0; i < secrets.Length; i++) secrets.SetFlat(i, rng.Next(2));
        model.Train(images, secrets);
        Assert.True(model.IsReady);
        Assert.Equal(1, model.TrainingStep);

        var restored = Small(errorCorrection: false);
        restored.Deserialize(model.Serialize());
        Assert.Equal(1, restored.TrainingStep);
        Assert.True(restored.IsReady);
    }

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
