using AiDotNet.Document.PixelToSequence;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Document;

/// <summary>
/// Pix2Struct's page rescale against the reference processor, which resizes with
/// torch.nn.functional.interpolate(mode="bilinear", align_corners=False, antialias=True).
/// </summary>
public class Pix2StructResizeTests
{
    /// <summary>
    /// 8x8 to 3x5 with x[i, j] = (3i + 5j) mod 11. The expected values are torch 2.12's output for that call. Plain
    /// bilinear (the engine interpolation the resize used before) differs from them by up to 2.43, because it reads
    /// only the two source pixels nearest each output instead of the triangle filter widened by the shrink factor.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void Downscale_matches_torch_antialiased_bilinear()
    {
        var data = new double[64];
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                data[i * 8 + j] = (3 * i + 5 * j) % 11;
        var image = new Tensor<double>(data, new[] { 1, 1, 8, 8 });

        var resized = Pix2Struct<double>.ResizeBilinearAntialiased(image, 3, 5);

        double[] expected =
        {
            3.9401913875598082, 5.504873294346979, 5.473684210526315, 4.7670565302144245, 4.059808612440191,
            4.795454545454546, 4.3765432098765435, 5.738095238095238, 5.140211640211641, 4.3354978354978355,
            5.650717703349282, 4.8206627680311875, 4.499999999999999, 5.819688109161794, 5.1124401913875595,
        };
        Assert.Equal(4, resized.Rank);
        Assert.Equal(3, resized.Shape[2]);
        Assert.Equal(5, resized.Shape[3]);
        var actual = resized.ToArray();
        for (int k = 0; k < expected.Length; k++)
            Assert.Equal(expected[k], actual[k], 12);
    }
}
