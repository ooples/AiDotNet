using AiDotNet.Augmentation.Base;
using AiDotNet.Augmentation.Data;

namespace AiDotNet.Augmentation.Image.Geometric;

/// <summary>
/// Randomly crops a region from the image.
/// </summary>
/// <remarks>
/// <para>
/// Random cropping extracts a random rectangular region from the image. This is one of the
/// most effective augmentations for teaching models to recognize objects even when partially
/// visible or at different positions within the frame.
/// </para>
/// <para><b>For Beginners:</b> Think of this like taking a photo with a camera that randomly
/// zooms in on different parts of the scene. Even if only part of an object is visible,
/// it's still that object. This teaches your model to recognize objects even when they're
/// partially out of frame.
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Image classification where objects may be partially visible</item>
/// <item>Training for translation invariance</item>
/// <item>When input images are larger than needed for the model</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Object detection (may crop out target objects entirely)</item>
/// <item>When the full context of the image is important</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RandomCrop<T> : SpatialAugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Gets the output height after cropping.
    /// </summary>
    public int CropHeight { get; }

    /// <summary>
    /// Gets the output width after cropping.
    /// </summary>
    public int CropWidth { get; }

    /// <summary>
    /// Gets the minimum scale factor for the crop region relative to image size.
    /// </summary>
    public double MinScale { get; }

    /// <summary>
    /// Gets the maximum scale factor for the crop region relative to image size.
    /// </summary>
    public double MaxScale { get; }

    /// <summary>
    /// Gets the minimum aspect ratio (width/height) for the crop.
    /// </summary>
    public double MinAspectRatio { get; }

    /// <summary>
    /// Gets the maximum aspect ratio (width/height) for the crop.
    /// </summary>
    public double MaxAspectRatio { get; }

    /// <summary>
    /// Gets whether to use scale-based random cropping (like RandomResizedCrop).
    /// </summary>
    public bool UseScaleCropping { get; }

    /// <summary>
    /// Creates a new random crop augmentation with fixed output size.
    /// </summary>
    /// <param name="cropHeight">
    /// The output height after cropping.
    /// </param>
    /// <param name="cropWidth">
    /// The output width after cropping.
    /// </param>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Industry standard default is 1.0 (always crop).
    /// </param>
    public RandomCrop(int cropHeight, int cropWidth, double probability = 1.0)
        : base(probability)
    {
        if (cropHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(cropHeight), "Crop height must be positive");
        if (cropWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(cropWidth), "Crop width must be positive");

        CropHeight = cropHeight;
        CropWidth = cropWidth;
        UseScaleCropping = false;
        MinScale = 1.0;
        MaxScale = 1.0;
        MinAspectRatio = 1.0;
        MaxAspectRatio = 1.0;
    }

    /// <summary>
    /// Creates a new random resized crop augmentation (scale-based cropping).
    /// </summary>
    /// <param name="outputHeight">
    /// The output height after cropping and resizing.
    /// </param>
    /// <param name="outputWidth">
    /// The output width after cropping and resizing.
    /// </param>
    /// <param name="minScale">
    /// The minimum scale of the crop relative to image area.
    /// Industry standard default is 0.08 (8% of image area).
    /// </param>
    /// <param name="maxScale">
    /// The maximum scale of the crop relative to image area.
    /// Industry standard default is 1.0 (full image).
    /// </param>
    /// <param name="minAspectRatio">
    /// The minimum aspect ratio (width/height) for the crop.
    /// Industry standard default is 0.75 (3:4 ratio).
    /// </param>
    /// <param name="maxAspectRatio">
    /// The maximum aspect ratio (width/height) for the crop.
    /// Industry standard default is 1.333 (4:3 ratio).
    /// </param>
    /// <param name="probability">
    /// The probability of applying this augmentation.
    /// </param>
    public RandomCrop(
        int outputHeight,
        int outputWidth,
        double minScale,
        double maxScale,
        double minAspectRatio = 0.75,
        double maxAspectRatio = 1.333,
        double probability = 1.0)
        : base(probability)
    {
        if (outputHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputHeight), "Output height must be positive");
        if (outputWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputWidth), "Output width must be positive");
        if (minScale <= 0 || minScale > 1)
            throw new ArgumentOutOfRangeException(nameof(minScale), "Scale must be between 0 and 1");
        if (maxScale <= 0 || maxScale > 1)
            throw new ArgumentOutOfRangeException(nameof(maxScale), "Scale must be between 0 and 1");
        if (minScale > maxScale)
            throw new ArgumentException("minScale must be <= maxScale");
        if (minAspectRatio <= 0 || maxAspectRatio <= 0)
            throw new ArgumentException("Aspect ratios must be positive");
        if (minAspectRatio > maxAspectRatio)
            throw new ArgumentException("minAspectRatio must be <= maxAspectRatio");

        CropHeight = outputHeight;
        CropWidth = outputWidth;
        MinScale = minScale;
        MaxScale = maxScale;
        MinAspectRatio = minAspectRatio;
        MaxAspectRatio = maxAspectRatio;
        UseScaleCropping = true;
    }

    /// <summary>
    /// Applies the random crop transformation and returns transform parameters.
    /// </summary>
    protected override (ImageTensor<T> data, IDictionary<string, object> parameters) ApplyWithTransformParams(
        ImageTensor<T> data,
        AugmentationContext<T> context)
    {
        int height = data.Height;
        int width = data.Width;
        int channels = data.Channels;

        int cropX, cropY, cropW, cropH;

        if (UseScaleCropping)
        {
            // RandomResizedCrop style: sample scale and aspect ratio
            (cropX, cropY, cropW, cropH) = GetRandomResizedCropParams(
                width, height, context);
        }
        else
        {
            // Simple random crop: just pick a random position
            cropW = Math.Min(CropWidth, width);
            cropH = Math.Min(CropHeight, height);
            cropX = context.GetRandomInt(0, Math.Max(1, width - cropW + 1));
            cropY = context.GetRandomInt(0, Math.Max(1, height - cropH + 1));
        }

        // Create output tensor (pass explicit ChannelOrder to avoid ambiguity)
        var result = new ImageTensor<T>(CropHeight, CropWidth, channels, data.ChannelOrder, data.ColorSpace);

        // Copy and resize if needed
        if (UseScaleCropping)
        {
            // Need to resize the crop to output dimensions
            ResizeCropToOutput(data, result, cropX, cropY, cropW, cropH);
        }
        else
        {
            // Direct copy
            for (int y = 0; y < CropHeight; y++)
            {
                for (int x = 0; x < CropWidth; x++)
                {
                    int srcY = cropY + y;
                    int srcX = cropX + x;

                    if (srcY < height && srcX < width)
                    {
                        for (int c = 0; c < channels; c++)
                        {
                            result.SetPixel(y, x, c, data.GetPixel(srcY, srcX, c));
                        }
                    }
                }
            }
        }

        var parameters = new Dictionary<string, object>
        {
            ["crop_x"] = cropX,
            ["crop_y"] = cropY,
            ["crop_width"] = cropW,
            ["crop_height"] = cropH,
            ["output_width"] = CropWidth,
            ["output_height"] = CropHeight,
            ["original_width"] = width,
            ["original_height"] = height,
            ["scale_x"] = (double)CropWidth / cropW,
            ["scale_y"] = (double)CropHeight / cropH
        };

        return (result, parameters);
    }

    private (int x, int y, int w, int h) GetRandomResizedCropParams(
        int width, int height, AugmentationContext<T> context)
    {
        int area = width * height;

        // Try to find a valid crop with the given constraints
        for (int attempt = 0; attempt < 10; attempt++)
        {
            double scale = context.GetRandomDouble(MinScale, MaxScale);
            double aspectRatio = Math.Exp(context.GetRandomDouble(
                Math.Log(MinAspectRatio), Math.Log(MaxAspectRatio)));

            int cropW = (int)Math.Sqrt(area * scale * aspectRatio);
            int cropH = (int)Math.Sqrt(area * scale / aspectRatio);

            if (cropW <= width && cropH <= height)
            {
                int x = context.GetRandomInt(0, width - cropW + 1);
                int y = context.GetRandomInt(0, height - cropH + 1);
                return (x, y, cropW, cropH);
            }
        }

        // Fallback: center crop with target aspect ratio
        double targetRatio = (double)CropWidth / CropHeight;
        int fallbackW, fallbackH;

        if ((double)width / height > targetRatio)
        {
            fallbackH = height;
            fallbackW = (int)(height * targetRatio);
        }
        else
        {
            fallbackW = width;
            fallbackH = (int)(width / targetRatio);
        }

        int fallbackX = (width - fallbackW) / 2;
        int fallbackY = (height - fallbackH) / 2;

        return (fallbackX, fallbackY, fallbackW, fallbackH);
    }

    private void ResizeCropToOutput(
        ImageTensor<T> source,
        ImageTensor<T> dest,
        int cropX, int cropY, int cropW, int cropH)
    {
        int outH = dest.Height;
        int outW = dest.Width;
        int channels = source.Channels;

        double scaleX = (double)cropW / outW;
        double scaleY = (double)cropH / outH;

        for (int y = 0; y < outH; y++)
        {
            for (int x = 0; x < outW; x++)
            {
                double srcX = cropX + x * scaleX;
                double srcY = cropY + y * scaleY;

                for (int c = 0; c < channels; c++)
                {
                    T value = BilinearSample(source, srcX, srcY, c);
                    dest.SetPixel(y, x, c, value);
                }
            }
        }
    }

    private static T BilinearSample(ImageTensor<T> image, double x, double y, int channel)
    {
        int x0 = (int)Math.Floor(x);
        int x1 = x0 + 1;
        int y0 = (int)Math.Floor(y);
        int y1 = y0 + 1;

        x0 = Math.Max(0, Math.Min(image.Width - 1, x0));
        x1 = Math.Max(0, Math.Min(image.Width - 1, x1));
        y0 = Math.Max(0, Math.Min(image.Height - 1, y0));
        y1 = Math.Max(0, Math.Min(image.Height - 1, y1));

        double xFrac = x - Math.Floor(x);
        double yFrac = y - Math.Floor(y);

        double v00 = Convert.ToDouble(image.GetPixel(y0, x0, channel));
        double v01 = Convert.ToDouble(image.GetPixel(y0, x1, channel));
        double v10 = Convert.ToDouble(image.GetPixel(y1, x0, channel));
        double v11 = Convert.ToDouble(image.GetPixel(y1, x1, channel));

        double v0 = v00 * (1 - xFrac) + v01 * xFrac;
        double v1 = v10 * (1 - xFrac) + v11 * xFrac;
        double result = v0 * (1 - yFrac) + v1 * yFrac;

        return (T)Convert.ChangeType(result, typeof(T));
    }

    /// <summary>
    /// Transforms a bounding box after random crop.
    /// </summary>
    protected override BoundingBox<T> TransformBoundingBox(
        BoundingBox<T> box,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        int cropX = (int)transformParams["crop_x"];
        int cropY = (int)transformParams["crop_y"];
        int cropW = (int)transformParams["crop_width"];
        int cropH = (int)transformParams["crop_height"];
        double scaleX = (double)transformParams["scale_x"];
        double scaleY = (double)transformParams["scale_y"];

        var (x, y, w, h) = box.ToXYWH();

        // Transform coordinates relative to crop region
        double newX = (x - cropX) * scaleX;
        double newY = (y - cropY) * scaleY;
        double newW = w * scaleX;
        double newH = h * scaleY;

        // Clip to output bounds
        newX = Math.Max(0, newX);
        newY = Math.Max(0, newY);
        double newX2 = Math.Min(CropWidth, newX + newW);
        double newY2 = Math.Min(CropHeight, newY + newH);

        var result = box.Clone();
        result.X1 = (T)Convert.ChangeType(newX, typeof(T));
        result.Y1 = (T)Convert.ChangeType(newY, typeof(T));
        result.X2 = (T)Convert.ChangeType(newX2, typeof(T));
        result.Y2 = (T)Convert.ChangeType(newY2, typeof(T));
        result.Format = BoundingBoxFormat.XYXY;
        return result;
    }

    /// <summary>
    /// Transforms a keypoint after random crop.
    /// </summary>
    protected override Keypoint<T> TransformKeypoint(
        Keypoint<T> keypoint,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        int cropX = (int)transformParams["crop_x"];
        int cropY = (int)transformParams["crop_y"];
        double scaleX = (double)transformParams["scale_x"];
        double scaleY = (double)transformParams["scale_y"];

        double x = Convert.ToDouble(keypoint.X);
        double y = Convert.ToDouble(keypoint.Y);

        double newX = (x - cropX) * scaleX;
        double newY = (y - cropY) * scaleY;

        var result = keypoint.Clone();
        result.X = (T)Convert.ChangeType(newX, typeof(T));
        result.Y = (T)Convert.ChangeType(newY, typeof(T));

        // Mark as not visible if outside crop bounds (0 = not labeled/not visible)
        if (newX < 0 || newX >= CropWidth || newY < 0 || newY >= CropHeight)
        {
            result.Visibility = 0;
        }

        return result;
    }

    /// <summary>
    /// Transforms a segmentation mask after random crop.
    /// </summary>
    protected override SegmentationMask<T> TransformMask(
        SegmentationMask<T> mask,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        int cropX = (int)transformParams["crop_x"];
        int cropY = (int)transformParams["crop_y"];
        int cropW = (int)transformParams["crop_width"];
        int cropH = (int)transformParams["crop_height"];

        var data = mask.ToDense();
        var cropped = new T[CropHeight, CropWidth];

        double scaleX = (double)cropW / CropWidth;
        double scaleY = (double)cropH / CropHeight;

        for (int y = 0; y < CropHeight; y++)
        {
            for (int x = 0; x < CropWidth; x++)
            {
                int srcX = cropX + (int)(x * scaleX);
                int srcY = cropY + (int)(y * scaleY);

                if (srcX >= 0 && srcX < mask.Width && srcY >= 0 && srcY < mask.Height)
                {
                    cropped[y, x] = data[srcY, srcX];
                }
            }
        }

        var result = new SegmentationMask<T>(cropped, mask.Type, mask.ClassIndex)
        {
            ClassName = mask.ClassName,
            InstanceId = mask.InstanceId
        };

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["crop_height"] = CropHeight;
        parameters["crop_width"] = CropWidth;
        parameters["use_scale_cropping"] = UseScaleCropping;
        if (UseScaleCropping)
        {
            parameters["min_scale"] = MinScale;
            parameters["max_scale"] = MaxScale;
            parameters["min_aspect_ratio"] = MinAspectRatio;
            parameters["max_aspect_ratio"] = MaxAspectRatio;
        }
        return parameters;
    }
}
