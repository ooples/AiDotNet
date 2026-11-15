using System.Threading.Tasks;
using AiDotNet.Preprocessing.Document;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Half-precision document images wider than 2048 pixels: pixel coordinates must not be held in Half, whose
/// integers past 2048 are spaced 2 apart (2049 rounds to 2048), or the resampling reads the neighbouring pixel.
/// </summary>
public class DocumentHalfCoordinateTests
{
    /// <summary>A [1, height, width] Half image of zeros with a single 1 at column <paramref name="column"/> of every row.</summary>
    private static Tensor<Half> MarkedColumn(int height, int width, int column)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();
        var pixels = new Half[height * width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                pixels[y * width + x] = x == column ? ops.One : ops.Zero;
            }
        }

        return new Tensor<Half>(pixels, new[] { 1, height, width });
    }

    [Fact(Timeout = 120000)]
    public async Task DocumentPreprocessor_Resize_Half_SameSize_KeepsColumnsPast2048()
    {
        await Task.Yield();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();

        // Resizing to the same size samples source column x at output column x. A Half coordinate 2049 rounds to
        // 2048, which would move the marked column 2049 one to the left.
        using var preprocessor = new DocumentPreprocessor<Half>();
        var image = MarkedColumn(1, 2100, 2049);

        var resized = preprocessor.Resize(image, 2100, 1);

        Assert.Equal(1.0, ops.ToDouble(resized[0, 0, 2049]));
        Assert.Equal(0.0, ops.ToDouble(resized[0, 0, 2048]));
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutNormalization_ResizeWithPadding_Half_SameSize_KeepsColumnsPast2048()
    {
        await Task.Yield();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();

        using var layout = new LayoutNormalization<Half>();
        var image = MarkedColumn(1, 2100, 2049);

        var resized = layout.ResizeWithPadding(image, 2100, 1, ops.Zero);

        Assert.Equal(1.0, ops.ToDouble(resized[0, 0, 2049]));
        Assert.Equal(0.0, ops.ToDouble(resized[0, 0, 2048]));
    }

    [Fact(Timeout = 120000)]
    public async Task Deskew_RotateImage_Half_ZeroAngle_KeepsColumnsPast2048()
    {
        await Task.Yield();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();

        // A zero rotation maps row 0 onto itself (the last row falls outside the y < height - 1 test and turns
        // white). Source x = (2049 - 1050) + 1050 in Half is 2048, which would read the neighbouring column.
        using var deskew = new Deskew<Half>();
        var image = MarkedColumn(2, 2100, 2049);

        var rotated = deskew.RotateImage(image, 0.0);

        Assert.Equal(1.0, ops.ToDouble(rotated[0, 0, 2049]));
        Assert.Equal(0.0, ops.ToDouble(rotated[0, 0, 2048]));
    }
}
