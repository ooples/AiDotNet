using System;
using System.Globalization;
using System.IO;
using System.Text;
using AiDotNet.ComputerVision.Segmentation.Common;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using SysConsole = System.Console;

namespace AiDotNet.Models.Results;

/// <summary>
/// Implements the two segmentation presentation formats exposed by <see cref="AiModelResult{T,TInput,TOutput}"/>:
/// a terminal rendering and a self-contained HTML page.
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
internal sealed class SegmentationVisualizer
{
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
    /// <param name="writer">Destination for the overlay. Defaults to <see cref="Console.Out"/>.</param>
    public void RenderAsciiOverlay<T>(
        SegmentationOutput<T> output,
        string? title = null,
        int maxWidth = 80,
        TextWriter? writer = null)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        writer ??= SysConsole.Out;

        var masks = output.InstanceMasks is null
            ? null
            : SegmentationTensorOps.EnsureUnbatched(output.InstanceMasks);
        var classMap = output.ClassMap is null
            ? null
            : SegmentationTensorOps.EnsureUnbatchedClassMap(output.ClassMap);
        if (masks is null && classMap is null)
        {
            writer.WriteLine("(segmentation output carries neither instance masks nor a class map)");
            return;
        }

        if (masks is not null && masks.Rank != 3)
            throw new ArgumentException($"InstanceMasks must be [N,H,W] or [B,N,H,W]; got rank {masks.Rank}.", nameof(output));

        if (title is { Length: > 0 })
        {
            writer.WriteLine();
            writer.WriteLine(title);
            writer.WriteLine(new string('=', title.Length));
        }

        int height = masks?.Shape[1] ?? (classMap ?? throw new InvalidOperationException("classMap is null where it is required.")).Shape[0];
        int width = masks?.Shape[2] ?? (classMap ?? throw new InvalidOperationException("classMap is null where it is required.")).Shape[1];
        int horizontalStep = Math.Max(1, (int)Math.Ceiling(width / (double)Math.Max(1, maxWidth)));
        int verticalStep = horizontalStep * 2;

        var rows = SampledIndices(height, verticalStep);
        var columns = SampledIndices(width, horizontalStep);
        var owners = OwnerMap(masks, classMap, rows, columns);
        var builder = new StringBuilder();

        for (int r = 0; r < rows.Length; r++)
        {
            builder.Clear();
            for (int c = 0; c < columns.Length; c++)
            {
                int owner = owners[r * columns.Length + c];
                builder.Append(owner < 0 ? '.' : (char)('0' + (owner % 10)));
            }
            writer.WriteLine(builder.ToString());
        }

        writer.WriteLine();
        writer.WriteLine($"{output.NumInstances} instance(s), {output.NumClasses} class(es); '.' = unlabelled.");
    }

    /// <summary>The indices 0, step, 2 step, ... below <paramref name="extent"/>.</summary>
    private static int[] SampledIndices(int extent, int step)
    {
        var indices = new int[(extent + step - 1) / step];
        for (int i = 0; i < indices.Length; i++) indices[i] = i * step;
        return indices;
    }

    /// <summary>
    /// The owner of every displayed cell (the given rows and columns), row-major: the first instance whose
    /// mask exceeds 0.5, or the rounded class id (both -1 where unlabelled).
    /// </summary>
    /// <remarks>
    /// Only the displayed rows and columns are selected before thresholding and reducing, so a large mask
    /// stack downsampled to a console width does not produce full-resolution intermediate tensors.
    /// </remarks>
    private static int[] OwnerMap<T>(Tensor<T>? masks, Tensor<T>? classMap, int[] rows, int[] columns)
    {
        var engine = AiDotNetEngine.Current;
        var owners = new int[rows.Length * columns.Length];
        var rowIndices = new Tensor<int>(rows, new[] { rows.Length });
        var columnIndices = new Tensor<int>(columns, new[] { columns.Length });
        if (masks is not null)
        {
            if (masks.Shape[0] == 0)
            {
                for (int i = 0; i < owners.Length; i++) owners[i] = -1;
                return owners;
            }

            var numOps = MathHelper.GetNumericOperations<T>();
            var displayed = engine.TensorIndexSelect(engine.TensorIndexSelect(masks, rowIndices, 1), columnIndices, 2);
            var inside = engine.TensorGreaterThan(displayed, numOps.FromDouble(0.5));
            var first = engine.TensorArgMax(inside, 0).ToArray();
            var any = engine.ReduceMax(inside, new[] { 0 }, false).Cast<double>().ToArray();
            for (int i = 0; i < owners.Length; i++)
            {
                owners[i] = any[i] > 0 ? first[i] : -1;
            }

            return owners;
        }

        var map = classMap ?? throw new InvalidOperationException("classMap is null where it is required.");
        var displayedClasses = engine.TensorIndexSelect(engine.TensorIndexSelect(map, rowIndices, 0), columnIndices, 1);
        var classes = engine.TensorRound(displayedClasses).Cast<double>().ToArray();
        for (int i = 0; i < owners.Length; i++)
        {
            int cls = (int)classes[i];
            owners[i] = cls <= 0 ? -1 : cls;
        }

        return owners;
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

        var scores = output.InstanceScores is { Length: > 0 } instanceScores
            ? new Tensor<T>(instanceScores, new[] { instanceScores.Length }).Cast<double>().ToArray()
            : Array.Empty<double>();
        var table = new StringBuilder();
        table.AppendLine("<table><tr><th>#</th><th>Class</th><th>Score</th></tr>");
        for (int i = 0; i < output.InstanceClasses.Length; i++)
        {
            string score = i < scores.Length
                ? scores[i].ToString("0.00", CultureInfo.InvariantCulture)
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

        double scale = 255.0 / SegmentationRenderer.DetectImageScale(rgb);

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

        // BMP stores rows bottom-up and pixels as B, G, R: flip the channel and row axes of the scaled
        // [3, H, W] image and interleave to [H, W, 3].
        var engine = AiDotNetEngine.Current;
        var channels = engine.TensorSlice(rgb, new[] { 0, 0, 0 }, new[] { 3, height, width });
        var scaled = engine.TensorMultiplyScalar(channels, numOps.FromDouble(scale));
        var bgrBottomUp = engine.TensorPermute(engine.TensorFlip(scaled, new[] { 0, 1 }), new[] { 1, 2, 0 });
        // The permute is a strided view; Cast walks the backing storage in its own order, so the view is
        // materialized in logical (interleaved B, G, R) order first.
        var samples = bgrBottomUp.Contiguous().Cast<double>().ToArray();

        int offset = headerSize;
        int sample = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < rowBytes; x++)
            {
                bmp[offset++] = Clamp(samples[sample++]);
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
