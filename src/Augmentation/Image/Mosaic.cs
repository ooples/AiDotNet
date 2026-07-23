namespace AiDotNet.Augmentation.Image;

/// <summary>
/// YOLO-style 4-image mosaic augmentation (Bochkovskiy et al., 2020).
/// Combines 4 images into a single image with a random center point.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Mosaic<T> : ImageAugmenterBase<T>
{
    public int OutputHeight { get; }
    public int OutputWidth { get; }
    public double MinCenterRatio { get; }
    public double MaxCenterRatio { get; }

    public Mosaic(int outputHeight = 640, int outputWidth = 640,
        double minCenterRatio = 0.25, double maxCenterRatio = 0.75,
        double probability = 0.5) : base(probability)
    {
        OutputHeight = outputHeight; OutputWidth = outputWidth;
        MinCenterRatio = minCenterRatio; MaxCenterRatio = maxCenterRatio;
    }

    /// <summary>
    /// Applies 4-image mosaic. Caller provides 3 additional images.
    /// </summary>
    public ImageTensor<T> ApplyMosaic(ImageTensor<T> image1, ImageTensor<T> image2,
        ImageTensor<T> image3, ImageTensor<T> image4, AugmentationContext<T> context)
    {
        int centerY = (int)(OutputHeight * context.GetRandomDouble(MinCenterRatio, MaxCenterRatio));
        int centerX = (int)(OutputWidth * context.GetRandomDouble(MinCenterRatio, MaxCenterRatio));

        var result = new ImageTensor<T>(height: OutputHeight, width: OutputWidth, channels: image1.Channels);

        // Top-left: image1
        PlaceImage(result, image1, 0, 0, centerY, centerX);
        // Top-right: image2
        PlaceImage(result, image2, 0, centerX, centerY, OutputWidth - centerX);
        // Bottom-left: image3
        PlaceImage(result, image3, centerY, 0, OutputHeight - centerY, centerX);
        // Bottom-right: image4
        PlaceImage(result, image4, centerY, centerX, OutputHeight - centerY, OutputWidth - centerX);

        return result;
    }

    private void PlaceImage(ImageTensor<T> canvas, ImageTensor<T> source,
        int startY, int startX, int regionH, int regionW)
    {
        // Resize source to fit region
        var resized = new Resize<T>(regionH, regionW).Apply(source, new AugmentationContext<T>());

        for (int y = 0; y < regionH && (startY + y) < canvas.Height; y++)
            for (int x = 0; x < regionW && (startX + x) < canvas.Width; x++)
                for (int c = 0; c < Math.Min(canvas.Channels, resized.Channels); c++)
                    canvas.SetPixel(startY + y, startX + x, c, resized.GetPixel(y, x, c));
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        // Single-image fallback: use the same image in all 4 quadrants
        return ApplyMosaic(data, data, data, data, context);
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["output_height"] = OutputHeight; p["output_width"] = OutputWidth;
        p["min_center_ratio"] = MinCenterRatio; p["max_center_ratio"] = MaxCenterRatio;
        return p;
    }
}
