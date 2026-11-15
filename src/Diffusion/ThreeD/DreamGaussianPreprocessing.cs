namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// DreamGaussian's input pre-processing after background removal (<c>process.py</c>): recentring the object, with
/// OpenCV's <c>cv2.resize(..., interpolation=cv2.INTER_AREA)</c>.
/// </summary>
internal static class DreamGaussianPreprocessing
{
    /// <summary>
    /// <c>process.py</c>'s recenter: the mask (<c>alpha &gt; 0</c>) bounding box <c>x_min:x_max, y_min:y_max</c> (the maxima
    /// exclusive, as the reference slices) is resized with <c>INTER_AREA</c> so its longer side is
    /// <c>int(size * (1 - border_ratio))</c> (each side <c>int(side * scale)</c>) and pasted centred
    /// (<c>(size - side) // 2</c>) into a transparent <paramref name="size"/> x <paramref name="size"/> image.
    /// </summary>
    /// <param name="rgba">Planar RGBA [4, H, W].</param>
    /// <param name="height">H.</param>
    /// <param name="width">W.</param>
    /// <param name="size">The output size (<c>--size</c>).</param>
    /// <param name="borderRatio">The empty border (<c>--border_ratio</c>, 0.2).</param>
    /// <returns>Planar RGBA [4, size, size], or null when the mask's box is empty or collapses (the reference fails
    /// there).</returns>
    internal static double[]? Recenter(double[] rgba, int height, int width, int size, double borderRatio)
    {
        if (rgba is null) throw new ArgumentNullException(nameof(rgba));
        int pixels = height * width;
        int xMin = int.MaxValue, xMax = -1, yMin = int.MaxValue, yMax = -1;
        for (int x = 0; x < height; x++)
        {
            for (int y = 0; y < width; y++)
            {
                if (!(rgba[(3 * pixels) + (x * width) + y] > 0.0)) continue;
                xMin = Math.Min(xMin, x);
                xMax = Math.Max(xMax, x);
                yMin = Math.Min(yMin, y);
                yMax = Math.Max(yMax, y);
            }
        }

        if (xMax < 0) return null;
        int h = xMax - xMin, w = yMax - yMin;
        int desired = (int)(size * (1.0 - borderRatio));
        if (Math.Max(h, w) == 0) return null;
        double scale = (double)desired / Math.Max(h, w);
        int h2 = (int)(h * scale), w2 = (int)(w * scale);
        if (h2 < 1 || w2 < 1) return null;
        int x2Min = (size - h2) / 2, y2Min = (size - w2) / 2;

        var crop = new double[4 * h * w];
        for (int c = 0; c < 4; c++)
        {
            for (int x = 0; x < h; x++)
            {
                for (int y = 0; y < w; y++)
                    crop[(((c * h) + x) * w) + y] = rgba[(c * pixels) + ((xMin + x) * width) + yMin + y];
            }
        }

        var resized = ResizeArea(crop, 4, h, w, h2, w2);
        var output = new double[4 * size * size];
        for (int c = 0; c < 4; c++)
        {
            for (int x = 0; x < h2; x++)
            {
                for (int y = 0; y < w2; y++)
                    output[(((c * size) + x2Min + x) * size) + y2Min + y] = resized[(((c * h2) + x) * w2) + y];
            }
        }

        return output;
    }

    /// <summary>
    /// <c>cv2.resize(..., interpolation=cv2.INTER_AREA)</c> of a planar image: when neither side grows, OpenCV's
    /// pixel-area weights (<c>computeResizeAreaTab</c>); otherwise its INTER_AREA zoom coefficients
    /// (<c>sx = floor(dx * scale)</c>, <c>fx = (dx + 1) - (sx + 1) / scale</c>, fractional part), close to nearest
    /// neighbour.
    /// </summary>
    internal static double[] ResizeArea(double[] source, int channels, int height, int width, int outHeight, int outWidth)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (height == outHeight && width == outWidth) return (double[])source.Clone();
        bool area = height >= outHeight && width >= outWidth;
        var rows = area ? AreaTaps(height, outHeight) : ZoomTaps(height, outHeight);
        var columns = area ? AreaTaps(width, outWidth) : ZoomTaps(width, outWidth);
        var output = new double[channels * outHeight * outWidth];
        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < outHeight; y++)
            {
                for (int x = 0; x < outWidth; x++)
                {
                    double sum = 0.0;
                    foreach (var (sy, wy) in rows[y])
                    {
                        foreach (var (sx, wx) in columns[x])
                            sum += wy * wx * source[(((c * height) + sy) * width) + sx];
                    }

                    output[(((c * outHeight) + y) * outWidth) + x] = sum;
                }
            }
        }

        return output;
    }

    /// <summary>OpenCV's <c>computeResizeAreaTab</c> for one axis.</summary>
    internal static List<(int Index, double Weight)>[] AreaTaps(int sourceSize, int size)
    {
        double scale = (double)sourceSize / size;
        var taps = new List<(int Index, double Weight)>[size];
        for (int dx = 0; dx < size; dx++)
        {
            var list = new List<(int Index, double Weight)>();
            double fsx1 = dx * scale, fsx2 = fsx1 + scale;
            double cellWidth = Math.Min(scale, sourceSize - fsx1);
            int sx1 = (int)Math.Ceiling(fsx1), sx2 = (int)Math.Floor(fsx2);
            sx2 = Math.Min(sx2, sourceSize - 1);
            sx1 = Math.Min(sx1, sx2);
            if (sx1 - fsx1 > 1e-3) list.Add((sx1 - 1, (sx1 - fsx1) / cellWidth));
            for (int sx = sx1; sx < sx2; sx++) list.Add((sx, 1.0 / cellWidth));
            if (fsx2 - sx2 > 1e-3) list.Add((sx2, Math.Min(Math.Min(fsx2 - sx2, 1.0), cellWidth) / cellWidth));
            taps[dx] = list;
        }

        return taps;
    }

    /// <summary>OpenCV's INTER_AREA coefficients of the generic (zoom) resize path for one axis.</summary>
    internal static List<(int Index, double Weight)>[] ZoomTaps(int sourceSize, int size)
    {
        double scale = (double)sourceSize / size, inverse = (double)size / sourceSize;
        var taps = new List<(int Index, double Weight)>[size];
        for (int dx = 0; dx < size; dx++)
        {
            int sx = (int)Math.Floor(dx * scale);
            double fx = (dx + 1) - ((sx + 1) * inverse);
            fx = fx <= 0.0 ? 0.0 : fx - Math.Floor(fx);
            if (sx < 0)
            {
                sx = 0;
                fx = 0.0;
            }

            if (sx >= sourceSize - 1)
            {
                sx = sourceSize - 1;
                fx = 0.0;
            }

            taps[dx] = fx == 0.0
                ? new List<(int Index, double Weight)> { (sx, 1.0) }
                : new List<(int Index, double Weight)> { (sx, 1.0 - fx), (sx + 1, fx) };
        }

        return taps;
    }
}
