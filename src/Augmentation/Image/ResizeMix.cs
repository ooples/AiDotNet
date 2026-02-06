namespace AiDotNet.Augmentation.Image;

/// <summary>
/// ResizeMix augmentation - resizes one image and pastes it onto another.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ResizeMix<T> : ImageAugmenterBase<T>
{
    public double MinScale { get; }
    public double MaxScale { get; }

    public ResizeMix(double minScale = 0.1, double maxScale = 0.8,
        double probability = 0.5) : base(probability)
    {
        MinScale = minScale; MaxScale = maxScale;
    }

    /// <summary>
    /// Resizes image2 and pastes it onto a random location in image1.
    /// </summary>
    public ImageTensor<T> ApplyResizeMix(ImageTensor<T> image1, ImageTensor<T> image2,
        AugmentationContext<T> context)
    {
        var result = image1.Clone();
        double scale = context.GetRandomDouble(MinScale, MaxScale);
        int pasteH = Math.Max(1, (int)(image1.Height * scale));
        int pasteW = Math.Max(1, (int)(image1.Width * scale));

        var resized = new Resize<T>(pasteH, pasteW).Apply(image2, context);

        int startY = context.GetRandomInt(0, Math.Max(1, image1.Height - pasteH));
        int startX = context.GetRandomInt(0, Math.Max(1, image1.Width - pasteW));

        for (int y = 0; y < pasteH && (startY + y) < image1.Height; y++)
            for (int x = 0; x < pasteW && (startX + x) < image1.Width; x++)
                for (int c = 0; c < Math.Min(image1.Channels, resized.Channels); c++)
                    result.SetPixel(startY + y, startX + x, c, resized.GetPixel(y, x, c));

        return result;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        return data.Clone();
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_scale"] = MinScale; p["max_scale"] = MaxScale;
        return p;
    }
}
