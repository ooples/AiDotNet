namespace AiDotNet.Augmentation.Image;

/// <summary>
/// General compression artifact simulation that randomly selects between JPEG and WebP styles.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ImageCompression<T> : ImageAugmenterBase<T>
{
    public int MinQuality { get; }
    public int MaxQuality { get; }

    public ImageCompression(int minQuality = 40, int maxQuality = 100, double probability = 0.5)
        : base(probability)
    {
        MinQuality = minQuality; MaxQuality = maxQuality;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        // Randomly choose between JPEG and WebP style compression
        if (context.GetRandomBool())
        {
            var jpeg = new JpegCompression<T>(MinQuality, MaxQuality, probability: 1.0);
            return jpeg.Apply(data, context);
        }
        else
        {
            var webp = new WebPCompression<T>(MinQuality, MaxQuality, probability: 1.0);
            return webp.Apply(data, context);
        }
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["min_quality"] = MinQuality; p["max_quality"] = MaxQuality; return p;
    }
}
