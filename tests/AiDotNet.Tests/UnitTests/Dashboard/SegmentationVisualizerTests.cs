using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Segmentation.Common;
using AiDotNet.Models.Results;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Dashboard;

/// <summary>
/// Verifies the Dashboard segmentation overlay, and in particular the hand-written BMP encoder behind
/// its HTML output.
/// </summary>
/// <remarks>
/// <para>
/// A hand-rolled binary format is exactly the kind of code that silently produces a file which is
/// subtly wrong — mirrored vertically, colour-swapped, or misaligned by a padding byte — and still
/// "looks like an image" to a casual glance. These tests decode the emitted BMP back out of the HTML
/// data URI and assert the actual bytes, rather than checking that a string was produced.
/// </para>
/// <para>
/// Everything is asserted through the public <see cref="AiModelResult{T,TInput,TOutput}"/> facade,
/// so the tests exercise the path a caller actually uses without exposing the presentation helper.
/// </para>
/// </remarks>
public class SegmentationVisualizerTests
{
    private const int Width = 6;
    private const int Height = 4;

    /// <summary>
    /// BMP's two classic traps: rows are stored BOTTOM-UP, and each row is padded to a 4-byte boundary.
    /// A red pixel placed at the image's top-left must therefore appear in the LAST row of the encoded
    /// pixel data, and the row stride must include the padding.
    /// </summary>
    [Fact]
    public async Task EncodedBmp_HasCorrectHeaderDimensionsAndBottomUpRowOrder()
    {
        await Task.Yield();
        // Distinct top-left pixel so orientation is detectable; everything else black.
        var image = new Tensor<float>([3, Height, Width]);
        image[0, 0, 0] = 1.0f;   // pure red at (row 0, col 0)

        var output = MakeSingleInstanceOutput();

        var result = CreateResult(
            new SegmentationVisualizationConfig
            {
                Alpha = 0.0,          // leave the source pixels untouched so orientation is unambiguous
                DrawContours = false,
                ShowLabels = false,
                ShowScores = false,
                ShowBoundingBoxes = false,
            });
        var html = result.GenerateSegmentationHtmlOverlay(image, output);

        byte[] bmp = ExtractBmp(html);

        // BITMAPFILEHEADER
        Assert.Equal((byte)'B', bmp[0]);
        Assert.Equal((byte)'M', bmp[1]);
        Assert.Equal(54, ReadInt32(bmp, 10));            // pixel data offset
        Assert.Equal(bmp.Length, ReadInt32(bmp, 2));     // file size matches actual length

        // BITMAPINFOHEADER
        Assert.Equal(40, ReadInt32(bmp, 14));
        Assert.Equal(Width, ReadInt32(bmp, 18));
        Assert.Equal(Height, ReadInt32(bmp, 22));
        Assert.Equal(24, bmp[28]);                       // bits per pixel

        int rowBytes = Width * 3;
        int padding = (4 - (rowBytes % 4)) % 4;
        int stride = rowBytes + padding;
        Assert.Equal(54 + stride * Height, bmp.Length);

        // Row 0 of the image is the LAST row of BMP pixel data.
        int lastRowStart = 54 + stride * (Height - 1);
        Assert.Equal(0, bmp[lastRowStart + 0]);      // B
        Assert.Equal(0, bmp[lastRowStart + 1]);      // G
        Assert.Equal(255, bmp[lastRowStart + 2]);    // R  -> the red pixel, bottom-up

        // ...and the FIRST encoded row (image row Height-1) is still black.
        Assert.Equal(0, bmp[54 + 0]);
        Assert.Equal(0, bmp[54 + 1]);
        Assert.Equal(0, bmp[54 + 2]);
    }

    /// <summary>
    /// BMP stores channels as B, G, R. Getting that order wrong produces an image whose colours look
    /// plausible but are swapped, which no shape assertion would catch.
    /// </summary>
    [Fact]
    public async Task EncodedBmp_WritesChannelsInBgrOrder()
    {
        await Task.Yield();
        var image = new Tensor<float>([3, Height, Width]);
        image[2, 0, 0] = 1.0f;   // pure BLUE at the top-left

        var html = CreateResult().GenerateSegmentationHtmlOverlay(
            image, MakeSingleInstanceOutput(),
            new SegmentationVisualizationConfig
            {
                Alpha = 0.0, DrawContours = false, ShowLabels = false, ShowScores = false, ShowBoundingBoxes = false,
            });

        byte[] bmp = ExtractBmp(html);
        int stride = Width * 3 + (4 - ((Width * 3) % 4)) % 4;
        int lastRowStart = 54 + stride * (Height - 1);

        Assert.Equal(255, bmp[lastRowStart + 0]);   // B holds the blue
        Assert.Equal(0, bmp[lastRowStart + 1]);
        Assert.Equal(0, bmp[lastRowStart + 2]);
    }

    /// <summary>
    /// A [0,1] image must be scaled to 8-bit, not truncated to 0/1 — otherwise every overlay renders
    /// as an almost-black image.
    /// </summary>
    [Fact]
    public async Task EncodedBmp_ScalesNormalizedImagesToFullByteRange()
    {
        await Task.Yield();
        var image = new Tensor<float>([3, Height, Width]);
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < Height; y++)
                for (int x = 0; x < Width; x++)
                    image[c, y, x] = 1.0f;   // white, in [0,1] terms

        var html = CreateResult().GenerateSegmentationHtmlOverlay(
            image, MakeSingleInstanceOutput(),
            new SegmentationVisualizationConfig
            {
                Alpha = 0.0, DrawContours = false, ShowLabels = false, ShowScores = false, ShowBoundingBoxes = false,
            });

        byte[] bmp = ExtractBmp(html);
        Assert.Equal(255, bmp[54 + 0]);
        Assert.Equal(255, bmp[54 + 1]);
        Assert.Equal(255, bmp[54 + 2]);
    }

    /// <summary>
    /// A [0,255] image must be written through unscaled. Multiplying by 255 again would clamp a
    /// mid-grey source to solid white.
    /// </summary>
    [Fact]
    public async Task EncodedBmp_PassesEightBitImagesThroughUnscaled()
    {
        await Task.Yield();
        var image = new Tensor<float>([3, Height, Width]);
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < Height; y++)
                for (int x = 0; x < Width; x++)
                    image[c, y, x] = 128.0f;

        var html = CreateResult().GenerateSegmentationHtmlOverlay(
            image, MakeSingleInstanceOutput(),
            new SegmentationVisualizationConfig
            {
                Alpha = 0.0,
                DrawContours = false,
                ShowLabels = false,
                ShowScores = false,
                ShowBoundingBoxes = false,
            });
        byte[] bmp = ExtractBmp(html);
        Assert.Equal(128, bmp[54 + 0]);
        Assert.Equal(128, bmp[54 + 1]);
        Assert.Equal(128, bmp[54 + 2]);
    }

    [Fact]
    public async Task RenderAsciiOverlay_WritesToTheProvidedWriterAndUnbatchesClassMaps()
    {
        await Task.Yield();
        var classMap = new Tensor<float>([1, Height, Width]);
        for (int y = 0; y < Height; y++)
            for (int x = 0; x < Width; x++)
                classMap[0, y, x] = 1.0f;
        var output = new SegmentationOutput<float>
        {
            ClassMap = classMap,
            NumClasses = 2,
            ImageHeight = Height,
            ImageWidth = Width,
        };
        using var writer = new StringWriter();

        CreateResult().RenderSegmentationAscii(output, maxWidth: Width, writer: writer);

        string[] mapRows = writer.ToString()
            .Split(new[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries)
            .Where(line => line == new string('1', Width))
            .ToArray();
        Assert.Equal(2, mapRows.Length); // vertical step 2 preserves terminal character aspect ratio
        Assert.Contains("2 class(es)", writer.ToString(), StringComparison.Ordinal);
    }

    /// <summary>The HTML is self-contained and carries the legend, so it can be attached to a report.</summary>
    [Fact]
    public async Task GenerateHtmlOverlay_ProducesSelfContainedPageWithLegend()
    {
        await Task.Yield();
        var image = new Tensor<float>([3, Height, Width]);
        var html = CreateResult().GenerateSegmentationHtmlOverlay(
            image, MakeSingleInstanceOutput(),
            new SegmentationVisualizationConfig { ShowLabels = false, ShowScores = false },
            title: "Overlay <check>");

        Assert.Contains("<!DOCTYPE html>", html);
        Assert.Contains("data:image/bmp;base64,", html);
        Assert.Contains("Overlay &lt;check&gt;", html);   // title is HTML-escaped
        Assert.Contains("<table>", html);                 // legend present
    }

    private static AiModelResult<float, Tensor<float>, Tensor<float>> CreateResult(
        SegmentationVisualizationConfig? config = null) =>
        new() { SegmentationVisualization = config };

    private static SegmentationOutput<float> MakeSingleInstanceOutput()
    {
        var masks = new Tensor<float>([1, Height, Width]);
        masks[0, Height - 1, Width - 1] = 1.0f;   // a mask well away from the probed pixel
        return new SegmentationOutput<float>
        {
            InstanceMasks = masks,
            InstanceClasses = new[] { 3 },
            InstanceScores = new[] { 0.5f },
            NumClasses = 4,
            ImageHeight = Height,
            ImageWidth = Width,
        };
    }

    private static byte[] ExtractBmp(string html)
    {
        var match = Regex.Match(html, "data:image/bmp;base64,([A-Za-z0-9+/=]+)");
        Assert.True(match.Success, "No BMP data URI found in the generated HTML.");
        return Convert.FromBase64String(match.Groups[1].Value);
    }

    private static int ReadInt32(byte[] buffer, int offset) =>
        buffer[offset] | (buffer[offset + 1] << 8) | (buffer[offset + 2] << 16) | (buffer[offset + 3] << 24);
}
