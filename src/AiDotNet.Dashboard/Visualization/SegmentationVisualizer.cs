using System;
using System.Globalization;
using System.Text;
using AiDotNet.ComputerVision.Segmentation.Common;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using SysConsole = System.Console;

namespace AiDotNet.Dashboard.Visualization;

/// <summary>
/// Presents a segmentation overlay in the two forms this dashboard can actually deliver: a terminal
/// rendering and a self-contained HTML page.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A segmentation model labels every pixel of an image. This turns that result
/// into something you can look at — either printed straight into your terminal as characters, or as
/// an HTML page with the coloured overlay embedded in it, which you can open in a browser or attach
/// to a report.
/// </para>
/// <para>
/// The pixel compositing is NOT reimplemented here. It is delegated to
/// <see cref="SegmentationRenderer"/> in the main library, which follows torchvision's
/// <c>draw_segmentation_masks</c> shape. This type only handles presentation, which keeps one
/// definition of what an overlay looks like.
/// </para>
/// <para>
/// The HTML embeds the image as an uncompressed 24-bit BMP data URI. BMP rather than PNG because PNG
/// requires zlib framing and CRCs, and this is a debugging view — a dependency-free encoder that every
/// browser understands is the better trade here.
/// </para>
/// </remarks>
public class SegmentationVisualizer
{
    /// <summary>Characters used for the terminal overlay, from background to densest coverage.</summary>
    private static readonly char[] DensityChars = { ' ', '.', ':', '-', '=', '+', '*', '#', '@' };

    /// <summary>
    /// Prints an ASCII overlay to the console: one character per cell, chosen by which instance covers
    /// it, downsampled to fit the requested width.
    /// </summary>
    /// <remarks>
    /// A segmentation map is far less legible as text than a heatmap is, so this shows WHICH instance
    /// owns each cell (by index digit) rather than pretending to render colour. It exists for CI logs
    /// and headless runs, where no browser is available.
    /// </remarks>
    /// <param name="output">The segmentation result to display.</param>
    /// <param name="title">Optional heading.</param>
    /// <param name="maxWidth">Maximum console width; the map is downsampled to fit.</param>
    public void RenderAsciiOverlay<T>(SegmentationOutput<T> output, string? title = null, int maxWidth = 80)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));

        var masks = output.InstanceMasks;
        var classMap = output.ClassMap;
        if (masks is null && classMap is null)
        {
            SysConsole.WriteLine("(segmentation output carries neither instance masks nor a class map)");
            return;
        }

        if (title is { Length: > 0 })
        {
            SysConsole.WriteLine();
            SysConsole.WriteLine(title);
            SysConsole.WriteLine(new string('=', title.Length));
        }

        int height = masks?.Shape[1] ?? classMap!.Shape[0];
        int width = masks?.Shape[2] ?? classMap!.Shape[1];
        int step = Math.Max(1, (int)Math.Ceiling(width / (double)Math.Max(1, maxWidth)));

        var numOps = MathHelper.GetNumericOperations<T>();
        var builder = new StringBuilder();

        for (int y = 0; y < height; y += step)
        {
            builder.Clear();
            for (int x = 0; x < width; x += step)
            {
                int owner = OwnerAt(masks, classMap, numOps, y, x);
                builder.Append(owner < 0 ? '.' : (char)('0' + (owner % 10)));
            }
            SysConsole.WriteLine(builder.ToString());
        }

        SysConsole.WriteLine();
        SysConsole.WriteLine($"{output.NumInstances} instance(s), {output.NumClasses} class(es); '.' = unlabelled.");
    }

    private static int OwnerAt<T>(
        Tensor<T>? masks, Tensor<T>? classMap, INumericOperations<T> numOps, int y, int x)
    {
        if (masks is not null)
        {
            var half = numOps.FromDouble(0.5);
            for (int m = 0; m < masks.Shape[0]; m++)
                if (numOps.GreaterThan(masks[m, y, x], half)) return m;
            return -1;
        }

        int cls = (int)Math.Round(numOps.ToDouble(classMap![y, x]));
        return cls <= 0 ? -1 : cls;
    }

    /// <summary>
    /// Produces a self-contained HTML page showing the overlay as an embedded image, with a legend.
    /// </summary>
    /// <param name="image">Source image [3, H, W] (or [H, W] greyscale).</param>
    /// <param name="output">Segmentation result to draw.</param>
    /// <param name="config">Overlay settings; when null the library defaults are used.</param>
    /// <param name="title">Page heading.</param>
    public string GenerateHtmlOverlay<T>(
        Tensor<T> image,
        SegmentationOutput<T> output,
        SegmentationVisualizationConfig? config = null,
        string title = "Segmentation Overlay")
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (output is null) throw new ArgumentNullException(nameof(output));

        var rendered = SegmentationRenderer.Render(image, output, config);
        string dataUri = "data:image/bmp;base64," + Convert.ToBase64String(EncodeBmp(rendered));

        var html = new StringBuilder();
        html.AppendLine("<!DOCTYPE html>");
        html.AppendLine("<html><head><meta charset=\"utf-8\">");
        html.AppendLine($"<title>{Escape(title)}</title>");
        html.AppendLine("<style>");
        html.AppendLine("body{font-family:system-ui,sans-serif;margin:2rem;background:#111;color:#eee}");
        html.AppendLine("img{image-rendering:pixelated;max-width:100%;border:1px solid #444}");
        html.AppendLine("table{border-collapse:collapse;margin-top:1rem}");
        html.AppendLine("td,th{padding:.25rem .75rem;border-bottom:1px solid #333;text-align:left}");
        html.AppendLine("</style></head><body>");
        html.AppendLine($"<h1>{Escape(title)}</h1>");
        html.AppendLine($"<img src=\"{dataUri}\" alt=\"Segmentation overlay\">");
        html.AppendLine(BuildLegend(output));
        html.AppendLine("</body></html>");
        return html.ToString();
    }

    private static string BuildLegend<T>(SegmentationOutput<T> output)
    {
        if (output.InstanceClasses is null || output.InstanceClasses.Length == 0)
            return $"<p>{output.NumClasses} class(es).</p>";

        var numOps = MathHelper.GetNumericOperations<T>();
        var table = new StringBuilder();
        table.AppendLine("<table><tr><th>#</th><th>Class</th><th>Score</th></tr>");
        for (int i = 0; i < output.InstanceClasses.Length; i++)
        {
            string score = output.InstanceScores is not null && i < output.InstanceScores.Length
                ? numOps.ToDouble(output.InstanceScores[i]).ToString("0.00", CultureInfo.InvariantCulture)
                : "-";
            table.AppendLine($"<tr><td>{i}</td><td>{output.InstanceClasses[i]}</td><td>{score}</td></tr>");
        }
        table.AppendLine("</table>");
        return table.ToString();
    }

    private static string Escape(string value) =>
        value.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&quot;");

    /// <summary>
    /// Encodes an RGB image tensor [3, H, W] as an uncompressed 24-bit BMP.
    /// </summary>
    /// <remarks>
    /// BMP stores rows bottom-up and pads each row to a 4-byte boundary, both of which are handled
    /// here. Values are read in the tensor's own range: a [0,1] image is scaled by 255, a [0,255] image
    /// is used as-is, matching how <see cref="SegmentationRenderer"/> preserves the input range.
    /// </remarks>
    internal static byte[] EncodeBmp<T>(Tensor<T> rgb)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int height = rgb.Shape[1];
        int width = rgb.Shape[2];

        bool normalized = true;
        for (int c = 0; c < 3 && normalized; c++)
            for (int y = 0; y < height && normalized; y++)
                for (int x = 0; x < width; x++)
                    if (numOps.ToDouble(rgb[c, y, x]) > 1.0) { normalized = false; break; }
        double scale = normalized ? 255.0 : 1.0;

        int rowBytes = width * 3;
        int padding = (4 - (rowBytes % 4)) % 4;
        int pixelBytes = (rowBytes + padding) * height;
        const int headerSize = 54;
        var bmp = new byte[headerSize + pixelBytes];

        // BITMAPFILEHEADER
        bmp[0] = (byte)'B'; bmp[1] = (byte)'M';
        WriteInt32(bmp, 2, headerSize + pixelBytes);
        WriteInt32(bmp, 10, headerSize);
        // BITMAPINFOHEADER
        WriteInt32(bmp, 14, 40);
        WriteInt32(bmp, 18, width);
        WriteInt32(bmp, 22, height);
        bmp[26] = 1;                    // planes
        bmp[28] = 24;                   // bits per pixel
        WriteInt32(bmp, 34, pixelBytes);

        int offset = headerSize;
        for (int y = height - 1; y >= 0; y--)   // BMP rows run bottom-up
        {
            for (int x = 0; x < width; x++)
            {
                bmp[offset++] = Clamp(numOps.ToDouble(rgb[2, y, x]) * scale);   // B
                bmp[offset++] = Clamp(numOps.ToDouble(rgb[1, y, x]) * scale);   // G
                bmp[offset++] = Clamp(numOps.ToDouble(rgb[0, y, x]) * scale);   // R
            }
            offset += padding;
        }
        return bmp;
    }

    private static byte Clamp(double value) =>
        value <= 0 ? (byte)0 : value >= 255 ? (byte)255 : (byte)Math.Round(value);

    private static void WriteInt32(byte[] buffer, int offset, int value)
    {
        buffer[offset] = (byte)(value & 0xFF);
        buffer[offset + 1] = (byte)((value >> 8) & 0xFF);
        buffer[offset + 2] = (byte)((value >> 16) & 0xFF);
        buffer[offset + 3] = (byte)((value >> 24) & 0xFF);
    }
}
