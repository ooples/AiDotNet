namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Unified preprocessing pipeline builder for chaining image transformations.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ImagePreprocessor<T>
{
    private readonly List<IAugmentation<T, ImageTensor<T>>> _transforms = new();

    /// <summary>
    /// Adds a transform to the pipeline.
    /// </summary>
    public ImagePreprocessor<T> Add(IAugmentation<T, ImageTensor<T>> transform)
    {
        _transforms.Add(transform);
        return this;
    }

    /// <summary>
    /// Adds a resize transform.
    /// </summary>
    public ImagePreprocessor<T> Resize(int height, int width,
        InterpolationMode mode = InterpolationMode.Bilinear)
    {
        _transforms.Add(new Resize<T>(height, width, mode));
        return this;
    }

    /// <summary>
    /// Adds center crop.
    /// </summary>
    public ImagePreprocessor<T> CenterCrop(int height, int width)
    {
        _transforms.Add(new CenterCrop<T>(height, width));
        return this;
    }

    /// <summary>
    /// Adds normalization.
    /// </summary>
    public ImagePreprocessor<T> Normalize(double[] mean, double[] std)
    {
        _transforms.Add(new Normalize<T>(mean, std));
        return this;
    }

    /// <summary>
    /// Adds ToTensor conversion.
    /// </summary>
    public ImagePreprocessor<T> ToTensor(double scaleFactor = 255.0)
    {
        _transforms.Add(new ToTensor<T>(scaleFactor));
        return this;
    }

    /// <summary>
    /// Adds horizontal flip augmentation.
    /// </summary>
    public ImagePreprocessor<T> RandomHorizontalFlip(double probability = 0.5)
    {
        _transforms.Add(new HorizontalFlip<T>(probability));
        return this;
    }

    /// <summary>
    /// Adds random crop augmentation.
    /// </summary>
    public ImagePreprocessor<T> RandomCrop(int height, int width)
    {
        _transforms.Add(new RandomCrop<T>(height, width));
        return this;
    }

    /// <summary>
    /// Applies the entire pipeline to an image.
    /// </summary>
    public ImageTensor<T> Process(ImageTensor<T> image, AugmentationContext<T>? context = null)
    {
        context ??= new AugmentationContext<T>();
        var result = image;
        foreach (var transform in _transforms)
            result = transform.Apply(result, context);
        return result;
    }

    /// <summary>
    /// Gets the list of transforms in this pipeline.
    /// </summary>
    public IReadOnlyList<IAugmentation<T, ImageTensor<T>>> Transforms => _transforms.AsReadOnly();
}
