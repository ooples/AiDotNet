using AiDotNet.Preprocessing.Document;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Regression tests for integer overflow in Otsu's between-class variance
/// (<c>wB * wF</c> was evaluated in int32 before being widened to double).
/// </summary>
public class BinarizationOtsuPrecisionTests
{
    [Fact(Timeout = 120000)]
    public async Task Otsu_OnHundredThousandPixelBimodalImage_SeparatesTheTwoLevels()
    {
        // 250 x 400 = 100,000 pixels: top half 0.2 (histogram bin 51), bottom half 1.0 (bin 255).
        // For every candidate threshold 51..254, wB = wF = 50,000 and wB * wF = 2.5e9, which
        // overflowed int32 to a negative value, so no variance ever beat 0 and the threshold
        // stayed at bin 0 (everything became foreground).
        const int height = 250;
        const int width = 400;
        var data = new Vector<double>(height * width);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                data[y * width + x] = y < height / 2 ? 0.2 : 1.0;
            }
        }

        var image = new Tensor<double>(new[] { 1, height, width }, data);
        var binarizer = new Binarization<double>();

        var result = binarizer.OtsuBinarization(image);

        // Correct Otsu threshold is bin 51 (0.2): dark pixels map to 0, bright pixels to 1.
        Assert.Equal(0.0, result[0, 0, 0]);
        Assert.Equal(0.0, result[0, height / 2 - 1, width - 1]);
        Assert.Equal(1.0, result[0, height / 2, 0]);
        Assert.Equal(1.0, result[0, height - 1, width - 1]);
    }

    [Fact(Timeout = 120000)]
    public async Task Otsu_HalfPixels_ThousandsPerLevel_SplitsAtTheBestThreshold()
    {
        await Task.Yield();

        // Half pixels: 100 at 0.1 (bin 25), 3000 at 0.5 (bin 127), 3000 at 0.9 (bin 229). The between-class variance
        // peaks at t = 127 (377 at t = 25 against 2771 at t = 127 with class probabilities), so the 0.1 pixels fall
        // below 127 / 255 and the rest above. Counting the histogram in Half stops at 2048, and the count-scale
        // product wB * wF (about 3e5 .. 9e6) overflows Half's 65504 to infinity for every split, so the first
        // split, t = 25, would win and every pixel would end up above 25 / 255.
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();
        int width = 6100;
        var pixels = new Half[width];
        for (int i = 0; i < width; i++)
        {
            pixels[i] = ops.FromDouble(i < 100 ? 0.1 : i < 3100 ? 0.5 : 0.9);
        }

        using var binarization = new Binarization<Half>();
        var result = binarization.OtsuBinarization(
            new AiDotNet.Tensors.LinearAlgebra.Tensor<Half>(pixels, new[] { 1, 1, width })).ToArray();

        Assert.Equal(0.0, ops.ToDouble(result[0]));
        Assert.Equal(0.0, ops.ToDouble(result[99]));
        Assert.Equal(1.0, ops.ToDouble(result[100]));
        Assert.Equal(1.0, ops.ToDouble(result[3099]));
        Assert.Equal(1.0, ops.ToDouble(result[3100]));
    }
}
