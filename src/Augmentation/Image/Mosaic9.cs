namespace AiDotNet.Augmentation.Image;

/// <summary>
/// 9-image mosaic variant that arranges images in a 3x3 grid.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Mosaic9<T> : ImageAugmenterBase<T>
{
    public int OutputHeight { get; }
    public int OutputWidth { get; }

    public Mosaic9(int outputHeight = 640, int outputWidth = 640,
        double probability = 0.5) : base(probability)
    {
        OutputHeight = outputHeight; OutputWidth = outputWidth;
    }

    /// <summary>
    /// Applies 9-image mosaic. Caller provides 8 additional images.
    /// </summary>
    public ImageTensor<T> ApplyMosaic9(ImageTensor<T>[] images, AugmentationContext<T> context)
    {
        if (images.Length < 9)
            throw new ArgumentException("Mosaic9 requires exactly 9 images", nameof(images));

        var result = new ImageTensor<T>(height: OutputHeight, width: OutputWidth, channels: images[0].Channels);
        int cellH = OutputHeight / 3;
        int cellW = OutputWidth / 3;

        for (int i = 0; i < 9; i++)
        {
            int row = i / 3;
            int col = i % 3;
            int startY = row * cellH;
            int startX = col * cellW;
            int h = (row == 2) ? OutputHeight - startY : cellH;
            int w = (col == 2) ? OutputWidth - startX : cellW;

            var resized = new Resize<T>(h, w).Apply(images[i], context);

            for (int y = 0; y < h && (startY + y) < result.Height; y++)
                for (int x = 0; x < w && (startX + x) < result.Width; x++)
                    for (int c = 0; c < Math.Min(result.Channels, resized.Channels); c++)
                        result.SetPixel(startY + y, startX + x, c, resized.GetPixel(y, x, c));
        }

        return result;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        // Single-image fallback
        var images = new ImageTensor<T>[9];
        for (int i = 0; i < 9; i++) images[i] = data;
        return ApplyMosaic9(images, context);
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["output_height"] = OutputHeight; p["output_width"] = OutputWidth;
        return p;
    }
}
