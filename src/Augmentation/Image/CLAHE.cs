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

        var engine = AiDotNetEngine.Current;
        int grid = TileGridSize;
        int tiles = grid * grid;
        var bins = ImageTensor<T>.HistogramBins(src.GetChannelPlane(channel), maxVal, numBins);

        // Tile of every pixel: (y / tileH, x / tileW)
        var tileRow = engine.Reshape(engine.TensorDivideScalar(engine.TensorArange<int>(0, h, 1), tileH), [h, 1]);
        var tileCol = engine.Reshape(engine.TensorDivideScalar(engine.TensorArange<int>(0, w, 1), tileW), [1, w]);
        var tileId = engine.TensorAdd(engine.TensorMultiplyScalar(tileRow, grid), tileCol);

        // Histogram of every tile in one bincount over (tile, bin) keys
        var keys = engine.TensorAdd(engine.TensorMultiplyScalar(tileId, numBins), bins);
        var histogram = engine.Reshape(
            engine.TensorBinCount(engine.Reshape(keys, [h * w]), tiles * numBins).Cast<T>(),
            [tiles, numBins]);
        var pixelCount = engine.ReduceSum(histogram, [1], true);

        // Clip histogram and redistribute the excess evenly
        T binCount = NumOps.FromDouble(numBins);
        var clipThreshold = engine.TensorBroadcastTo(
            engine.TensorDivideScalar(engine.TensorMultiplyScalar(pixelCount, NumOps.FromDouble(ClipLimit)), binCount), [tiles, numBins]);
        var excess = engine.ReduceSum(engine.TensorMax(engine.TensorSubtract(histogram, clipThreshold), NumOps.Zero), [1], true);
        var clipped = engine.TensorAdd(engine.TensorMin(histogram, clipThreshold), engine.TensorDivideScalar(excess, binCount));

        // Build and normalize the CDF of every tile
        var cdf = engine.TensorCumSum(clipped, 1);
        var totalCdf = engine.TensorSlice(cdf, [0, numBins - 1], [tiles, 1]);
        var normalized = engine.TensorMultiplyScalar(engine.TensorDivide(cdf, totalCdf), NumOps.FromDouble(maxVal));
        var hasMass = engine.TensorBroadcastTo(engine.TensorGreaterThan(totalCdf, NumOps.Zero), [tiles, numBins]);
        var tileCdfs = engine.Reshape(engine.TensorWhere(hasMass, normalized, cdf), [tiles * numBins]);

        // Apply with bilinear interpolation between tiles
        var (ty0, ty1, fy) = TileNeighbours(engine, h, tileH, grid);
        var (tx0, tx1, fx) = TileNeighbours(engine, w, tileW, grid);
        ty0 = engine.Reshape(ty0, [h, 1]); ty1 = engine.Reshape(ty1, [h, 1]); fy = engine.Reshape(fy, [h, 1]);
        tx0 = engine.Reshape(tx0, [1, w]); tx1 = engine.Reshape(tx1, [1, w]); fx = engine.Reshape(fx, [1, w]);

        Tensor<T> Lookup(Tensor<int> rowTile, Tensor<int> colTile)
        {
            var tile = engine.TensorAdd(engine.TensorMultiplyScalar(rowTile, grid), colTile);
            var index = engine.TensorAdd(engine.TensorMultiplyScalar(tile, numBins), bins);
            return engine.TensorTake(tileCdfs, index);
        }

        var oneMinusFx = engine.ScalarMinusTensor(NumOps.One, fx);
        var oneMinusFy = engine.ScalarMinusTensor(NumOps.One, fy);
        var mapped = engine.TensorMultiply(engine.TensorMultiply(Lookup(ty0, tx0), oneMinusFx), oneMinusFy);
        mapped = engine.TensorAdd(mapped, engine.TensorMultiply(engine.TensorMultiply(Lookup(ty0, tx1), fx), oneMinusFy));
        mapped = engine.TensorAdd(mapped, engine.TensorMultiply(engine.TensorMultiply(Lookup(ty1, tx0), oneMinusFx), fy));
        mapped = engine.TensorAdd(mapped, engine.TensorMultiply(engine.TensorMultiply(Lookup(ty1, tx1), fx), fy));

        dst.SetChannelPlane(channel, mapped);
    }

    /// <summary>
    /// For positions 0..size-1: the lower and upper neighbouring tile (clamped to the grid) and the
    /// interpolation weight towards the upper tile, using tile centres at (t + 0.5) * tileSize.
    /// </summary>
    private static (Tensor<int> Lower, Tensor<int> Upper, Tensor<T> Weight) TileNeighbours(
        IEngine engine, int size, int tileSize, int grid)
    {
        var position = engine.TensorSubtractScalar(
            engine.TensorDivideScalar(engine.TensorArange<T>(0, size, 1), NumOps.FromDouble(tileSize)), NumOps.FromDouble(0.5));
        var lower = engine.TensorClamp(engine.TensorFloor(position), NumOps.Zero, NumOps.FromDouble(grid - 1));
        var upper = engine.TensorClampMax(engine.TensorAddScalar(lower, NumOps.One), NumOps.FromDouble(grid - 1));
        var weight = engine.TensorClamp(engine.TensorSubtract(position, lower), NumOps.Zero, NumOps.One);
        return (lower.Cast<int>(), upper.Cast<int>(), weight);
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["clip_limit"] = ClipLimit;
        p["tile_grid_size"] = TileGridSize;
        return p;
    }
}
