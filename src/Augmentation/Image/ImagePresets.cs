namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Preset preprocessing configurations for common models.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public static class ImagePresets<T>
{
    /// <summary>
    /// Standard ImageNet preprocessing: resize 256, center crop 224, normalize.
    /// </summary>
    public static ImagePreprocessor<T> ImageNet(bool training = false)
    {
        var pipeline = new ImagePreprocessor<T>();

        if (training)
        {
            pipeline.Add(new RandomResizedCrop<T>(224, 224))
                    .RandomHorizontalFlip();
        }
        else
        {
            pipeline.Resize(256, 256)
                    .CenterCrop(224, 224);
        }

        pipeline.ToTensor()
                .Normalize(
                    new[] { 0.485, 0.456, 0.406 },
                    new[] { 0.229, 0.224, 0.225 });

        return pipeline;
    }

    /// <summary>
    /// COCO detection preprocessing.
    /// </summary>
    public static ImagePreprocessor<T> COCO(int targetSize = 640)
    {
        return new ImagePreprocessor<T>()
            .Add(new ResizeWithAspectRatio<T>(targetSize, targetSize))
            .ToTensor()
            .Normalize(
                new[] { 0.485, 0.456, 0.406 },
                new[] { 0.229, 0.224, 0.225 });
    }

    /// <summary>
    /// Pascal VOC preprocessing.
    /// </summary>
    public static ImagePreprocessor<T> VOC(int targetSize = 512)
    {
        return new ImagePreprocessor<T>()
            .Resize(targetSize, targetSize)
            .ToTensor()
            .Normalize(
                new[] { 0.485, 0.456, 0.406 },
                new[] { 0.229, 0.224, 0.225 });
    }

    /// <summary>
    /// CLIP model preprocessing: resize 224, center crop 224, normalize with CLIP stats.
    /// </summary>
    public static ImagePreprocessor<T> CLIP()
    {
        return new ImagePreprocessor<T>()
            .Resize(224, 224)
            .CenterCrop(224, 224)
            .ToTensor()
            .Normalize(
                new[] { 0.48145466, 0.4578275, 0.40821073 },
                new[] { 0.26862954, 0.26130258, 0.27577711 });
    }

    /// <summary>
    /// DINO/DINOv2 preprocessing.
    /// </summary>
    public static ImagePreprocessor<T> DINO(int imageSize = 224)
    {
        return new ImagePreprocessor<T>()
            .Resize(imageSize, imageSize)
            .CenterCrop(imageSize, imageSize)
            .ToTensor()
            .Normalize(
                new[] { 0.485, 0.456, 0.406 },
                new[] { 0.229, 0.224, 0.225 });
    }

    /// <summary>
    /// SAM (Segment Anything Model) preprocessing.
    /// </summary>
    public static ImagePreprocessor<T> SAM(int imageSize = 1024)
    {
        return new ImagePreprocessor<T>()
            .Add(new LongestMaxSize<T>(imageSize))
            .Add(new PadToSquare<T>())
            .ToTensor()
            .Normalize(
                new[] { 0.485, 0.456, 0.406 },
                new[] { 0.229, 0.224, 0.225 });
    }
}
