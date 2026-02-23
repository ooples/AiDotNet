namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
/// </summary>
/// <remarks>
/// <para>
/// CLAHE improves upon standard histogram equalization by dividing the image into tiles
/// and equalizing each tile independently, with a clip limit to prevent noise amplification.
/// The tile boundaries are blended using bilinear interpolation for smooth results.
/// Based on Zuiderveld (1994).
/// </para>
/// <para><b>For Beginners:</b> Regular histogram equalization can make noise very visible.
/// CLAHE works on small regions and limits how much contrast it adds, so you get better
/// detail without amplifying noise. It's widely used in medical imaging.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CLAHE<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the clip limit for contrast limiting. Higher values allow more contrast.
    /// </summary>
    public double ClipLimit { get; }

    /// <summary>
    /// Gets the number of tiles in each dimension.
    /// </summary>
    public int TileGridSize { get; }

    /// <summary>
    /// Creates a new CLAHE augmentation.
    /// </summary>
    /// <param name="clipLimit">Contrast clip limit. Default is 4.0.</param>
    /// <param name="tileGridSize">Number of tiles per dimension. Default is 8.</param>
    /// <param name="probability">Probability of applying. Default is 0.5.</param>
    public CLAHE(double clipLimit = 4.0, int tileGridSize = 8, double probability = 0.5) : base(probability)
    {
        if (clipLimit <= 0) throw new ArgumentOutOfRangeException(nameof(clipLimit));
        if (tileGridSize < 1) throw new ArgumentOutOfRangeException(nameof(tileGridSize));
        ClipLimit = clipLimit;
        TileGridSize = tileGridSize;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        int h = data.Height;
        int w = data.Width;
        int numBins = 256;
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        // Apply CLAHE to each channel independently
        for (int c = 0; c < data.Channels; c++)
        {
            ApplyClaheChannel(data, result, h, w, c, numBins, maxVal);
        }

        return result;
    }

    private void ApplyClaheChannel(ImageTensor<T> src, ImageTensor<T> dst,
        int h, int w, int channel, int numBins, double maxVal)
    {
        int tileH = (h + TileGridSize - 1) / TileGridSize;
        int tileW = (w + TileGridSize - 1) / TileGridSize;

        // Compute CDF for each tile
        var tileCdfs = new double[TileGridSize, TileGridSize, numBins];

        for (int ty = 0; ty < TileGridSize; ty++)
        {
            for (int tx = 0; tx < TileGridSize; tx++)
            {
                int startY = ty * tileH;
                int startX = tx * tileW;
                int endY = Math.Min(startY + tileH, h);
                int endX = Math.Min(startX + tileW, w);

                var histogram = new double[numBins];
                int pixelCount = 0;

                for (int y = startY; y < endY; y++)
                {
                    for (int x = startX; x < endX; x++)
                    {
                        double val = NumOps.ToDouble(src.GetPixel(y, x, channel));
                        int bin = Math.Max(0, Math.Min(numBins - 1, (int)(val / maxVal * (numBins - 1))));
                        histogram[bin]++;
                        pixelCount++;
                    }
                }

                // Clip histogram
                double clipThreshold = ClipLimit * pixelCount / numBins;
                double excess = 0;
                for (int i = 0; i < numBins; i++)
                {
                    if (histogram[i] > clipThreshold)
                    {
                        excess += histogram[i] - clipThreshold;
                        histogram[i] = clipThreshold;
                    }
                }

                // Redistribute excess
                double redistrib = excess / numBins;
                for (int i = 0; i < numBins; i++)
                    histogram[i] += redistrib;

                // Build CDF
                tileCdfs[ty, tx, 0] = histogram[0];
                for (int i = 1; i < numBins; i++)
                    tileCdfs[ty, tx, i] = tileCdfs[ty, tx, i - 1] + histogram[i];

                // Normalize CDF
                double totalCdf = tileCdfs[ty, tx, numBins - 1];
                if (totalCdf > 0)
                {
                    for (int i = 0; i < numBins; i++)
                        tileCdfs[ty, tx, i] = tileCdfs[ty, tx, i] / totalCdf * maxVal;
                }
            }
        }

        // Apply with bilinear interpolation between tiles
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                double val = NumOps.ToDouble(src.GetPixel(y, x, channel));
                int bin = Math.Max(0, Math.Min(numBins - 1, (int)(val / maxVal * (numBins - 1))));

                // Find surrounding tiles
                double tyf = ((double)y / tileH) - 0.5;
                double txf = ((double)x / tileW) - 0.5;

                int ty0 = Math.Max(0, Math.Min(TileGridSize - 1, (int)Math.Floor(tyf)));
                int tx0 = Math.Max(0, Math.Min(TileGridSize - 1, (int)Math.Floor(txf)));
                int ty1 = Math.Min(TileGridSize - 1, ty0 + 1);
                int tx1 = Math.Min(TileGridSize - 1, tx0 + 1);

                double fy = Math.Max(0, Math.Min(1, tyf - ty0));
                double fx = Math.Max(0, Math.Min(1, txf - tx0));

                // Bilinear interpolation of mapped values
                double mapped = tileCdfs[ty0, tx0, bin] * (1 - fx) * (1 - fy) +
                                tileCdfs[ty0, tx1, bin] * fx * (1 - fy) +
                                tileCdfs[ty1, tx0, bin] * (1 - fx) * fy +
                                tileCdfs[ty1, tx1, bin] * fx * fy;

                dst.SetPixel(y, x, channel, NumOps.FromDouble(mapped));
            }
        }
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["clip_limit"] = ClipLimit;
        p["tile_grid_size"] = TileGridSize;
        return p;
    }
}
