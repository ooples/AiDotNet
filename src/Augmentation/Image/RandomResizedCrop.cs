namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Randomly crops and resizes a region of the image (PyTorch-style RandomResizedCrop).
/// </summary>
/// <remarks>
/// <para>
/// RandomResizedCrop first extracts a random crop with an area between <see cref="MinScale"/>
/// and <see cref="MaxScale"/> of the original, with an aspect ratio between <see cref="MinRatio"/>
/// and <see cref="MaxRatio"/>, then resizes to the target output size. This is the standard
/// training augmentation for ImageNet and many other tasks.
/// </para>
/// <para><b>For Beginners:</b> This picks a random-sized chunk of the image (from 8% to 100%
/// of the area) with a random shape (from slightly tall to slightly wide), then resizes it to
/// a fixed size. This teaches your model to recognize objects at different scales and crops.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Standard ImageNet training pipeline (the single most important augmentation)</item>
/// <item>Any classification task as default training augmentation</item>
/// <item>Self-supervised learning (SimCLR, BYOL, etc.)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RandomResizedCrop<T> : SpatialImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the target output height.
    /// </summary>
    public int OutputHeight { get; }

    /// <summary>
    /// Gets the target output width.
    /// </summary>
    public int OutputWidth { get; }

    /// <summary>
    /// Gets the minimum scale (fraction of image area).
    /// </summary>
    public double MinScale { get; }

    /// <summary>
    /// Gets the maximum scale (fraction of image area).
    /// </summary>
    public double MaxScale { get; }

    /// <summary>
    /// Gets the minimum aspect ratio (width/height).
    /// </summary>
    public double MinRatio { get; }

    /// <summary>
    /// Gets the maximum aspect ratio (width/height).
    /// </summary>
    public double MaxRatio { get; }

    /// <summary>
    /// Gets the interpolation mode.
    /// </summary>
    public InterpolationMode Interpolation { get; }

    /// <summary>
    /// Creates a new RandomResizedCrop.
    /// </summary>
    /// <param name="outputHeight">Target height. Must be positive.</param>
    /// <param name="outputWidth">Target width. Must be positive.</param>
    /// <param name="minScale">Minimum crop area as fraction. Default is 0.08 (8%).</param>
    /// <param name="maxScale">Maximum crop area as fraction. Default is 1.0 (100%).</param>
    /// <param name="minRatio">Minimum aspect ratio. Default is 0.75 (3:4).</param>
    /// <param name="maxRatio">Maximum aspect ratio. Default is 1.333 (4:3).</param>
    /// <param name="interpolation">Interpolation mode. Default is Bilinear.</param>
    /// <param name="probability">Probability of applying. Default is 1.0.</param>
    public RandomResizedCrop(
        int outputHeight,
        int outputWidth,
        double minScale = 0.08,
        double maxScale = 1.0,
        double minRatio = 0.75,
        double maxRatio = 1.333,
        InterpolationMode interpolation = InterpolationMode.Bilinear,
        double probability = 1.0) : base(probability)
    {
        if (outputHeight <= 0) throw new ArgumentOutOfRangeException(nameof(outputHeight));
        if (outputWidth <= 0) throw new ArgumentOutOfRangeException(nameof(outputWidth));
        if (minScale <= 0 || maxScale > 1 || minScale > maxScale)
            throw new ArgumentException("Scale must satisfy 0 < minScale <= maxScale <= 1.");
        if (minRatio <= 0 || maxRatio <= 0 || minRatio > maxRatio)
            throw new ArgumentException("Ratio must be positive with minRatio <= maxRatio.");

        OutputHeight = outputHeight;
        OutputWidth = outputWidth;
        MinScale = minScale;
        MaxScale = maxScale;
        MinRatio = minRatio;
        MaxRatio = maxRatio;
        Interpolation = interpolation;
    }

    /// <summary>
    /// Creates a square RandomResizedCrop.
    /// </summary>
    public RandomResizedCrop(int outputSize, double minScale = 0.08, double maxScale = 1.0,
        double minRatio = 0.75, double maxRatio = 1.333,
        InterpolationMode interpolation = InterpolationMode.Bilinear, double probability = 1.0)
        : this(outputSize, outputSize, minScale, maxScale, minRatio, maxRatio, interpolation, probability)
    {
    }

    /// <inheritdoc />
    protected override (ImageTensor<T> data, IDictionary<string, object> parameters) ApplyWithTransformParams(
        ImageTensor<T> data, AugmentationContext<T> context)
    {
        int h = data.Height;
        int w = data.Width;
        int area = h * w;

        int cropX, cropY, cropW, cropH;

        // Try to find valid crop parameters
        bool found = false;
        cropX = cropY = cropW = cropH = 0;

        for (int attempt = 0; attempt < 10; attempt++)
        {
            double scale = context.GetRandomDouble(MinScale, MaxScale);
            double ratio = Math.Exp(context.GetRandomDouble(Math.Log(MinRatio), Math.Log(MaxRatio)));

            cropW = (int)Math.Round(Math.Sqrt(area * scale * ratio));
            cropH = (int)Math.Round(Math.Sqrt(area * scale / ratio));

            if (cropW > 0 && cropH > 0 && cropW <= w && cropH <= h)
            {
                cropX = context.GetRandomInt(0, w - cropW + 1);
                cropY = context.GetRandomInt(0, h - cropH + 1);
                found = true;
                break;
            }
        }

        if (!found)
        {
            // Fallback: center crop with target aspect ratio
            double targetRatio = (double)OutputWidth / OutputHeight;
            if ((double)w / h > targetRatio)
            {
                cropH = h;
                cropW = (int)(h * targetRatio);
            }
            else
            {
                cropW = w;
                cropH = (int)(w / targetRatio);
            }
            cropX = (w - cropW) / 2;
            cropY = (h - cropH) / 2;
        }

        // Extract and resize
        var result = new ImageTensor<T>(OutputHeight, OutputWidth, data.Channels, data.ChannelOrder, data.ColorSpace)
        {
            IsNormalized = data.IsNormalized,
            NormalizationMean = data.NormalizationMean,
            NormalizationStd = data.NormalizationStd,
            OriginalRange = data.OriginalRange
        };

        double scaleX = (double)cropW / OutputWidth;
        double scaleY = (double)cropH / OutputHeight;

        for (int y = 0; y < OutputHeight; y++)
        {
            for (int x = 0; x < OutputWidth; x++)
            {
                double srcX = cropX + (x + 0.5) * scaleX - 0.5;
                double srcY = cropY + (y + 0.5) * scaleY - 0.5;
                srcX = Math.Max(0, Math.Min(w - 1, srcX));
                srcY = Math.Max(0, Math.Min(h - 1, srcY));

                for (int c = 0; c < data.Channels; c++)
                {
                    result.SetPixel(y, x, c, BilinearSample(data, srcX, srcY, c));
                }
            }
        }

        var parameters = new Dictionary<string, object>
        {
            ["crop_x"] = cropX,
            ["crop_y"] = cropY,
            ["crop_width"] = cropW,
            ["crop_height"] = cropH,
            ["output_width"] = OutputWidth,
            ["output_height"] = OutputHeight,
            ["original_width"] = w,
            ["original_height"] = h,
            ["scale_x"] = (double)OutputWidth / cropW,
            ["scale_y"] = (double)OutputHeight / cropH
        };

        return (result, parameters);
    }

    private static T BilinearSample(ImageTensor<T> image, double x, double y, int channel)
    {
        int x0 = (int)Math.Floor(x), x1 = x0 + 1;
        int y0 = (int)Math.Floor(y), y1 = y0 + 1;
        x0 = Math.Max(0, Math.Min(image.Width - 1, x0));
        x1 = Math.Max(0, Math.Min(image.Width - 1, x1));
        y0 = Math.Max(0, Math.Min(image.Height - 1, y0));
        y1 = Math.Max(0, Math.Min(image.Height - 1, y1));
        double fx = x - Math.Floor(x), fy = y - Math.Floor(y);
        double v = NumOps.ToDouble(image.GetPixel(y0, x0, channel)) * (1 - fx) * (1 - fy) +
                   NumOps.ToDouble(image.GetPixel(y0, x1, channel)) * fx * (1 - fy) +
                   NumOps.ToDouble(image.GetPixel(y1, x0, channel)) * (1 - fx) * fy +
                   NumOps.ToDouble(image.GetPixel(y1, x1, channel)) * fx * fy;
        return NumOps.FromDouble(v);
    }

    /// <inheritdoc />
    protected override BoundingBox<T> TransformBoundingBox(BoundingBox<T> box,
        IDictionary<string, object> transformParams, AugmentationContext<T> context)
    {
        int cropX = (int)transformParams["crop_x"];
        int cropY = (int)transformParams["crop_y"];
        double scaleX = (double)transformParams["scale_x"];
        double scaleY = (double)transformParams["scale_y"];

        var (bx, by, bw, bh) = box.ToXYWH();
        double x1 = Math.Max(0, Math.Min(OutputWidth, (bx - cropX) * scaleX));
        double y1 = Math.Max(0, Math.Min(OutputHeight, (by - cropY) * scaleY));
        double x2 = Math.Max(0, Math.Min(OutputWidth, (bx + bw - cropX) * scaleX));
        double y2 = Math.Max(0, Math.Min(OutputHeight, (by + bh - cropY) * scaleY));

        var result = box.Clone();
        result.X1 = NumOps.FromDouble(x1);
        result.Y1 = NumOps.FromDouble(y1);
        result.X2 = NumOps.FromDouble(x2);
        result.Y2 = NumOps.FromDouble(y2);
        result.Format = BoundingBoxFormat.XYXY;
        return result;
    }

    /// <inheritdoc />
    protected override Keypoint<T> TransformKeypoint(Keypoint<T> keypoint,
        IDictionary<string, object> transformParams, AugmentationContext<T> context)
    {
        int cropX = (int)transformParams["crop_x"];
        int cropY = (int)transformParams["crop_y"];
        double scaleX = (double)transformParams["scale_x"];
        double scaleY = (double)transformParams["scale_y"];

        double x = (NumOps.ToDouble(keypoint.X) - cropX) * scaleX;
        double y = (NumOps.ToDouble(keypoint.Y) - cropY) * scaleY;

        var result = keypoint.Clone();
        result.X = NumOps.FromDouble(x);
        result.Y = NumOps.FromDouble(y);
        if (x < 0 || x >= OutputWidth || y < 0 || y >= OutputHeight)
            result.Visibility = 0;
        return result;
    }

    /// <inheritdoc />
    protected override SegmentationMask<T> TransformMask(SegmentationMask<T> mask,
        IDictionary<string, object> transformParams, AugmentationContext<T> context)
    {
        int cropX = (int)transformParams["crop_x"];
        int cropY = (int)transformParams["crop_y"];
        int cropW = (int)transformParams["crop_width"];
        int cropH = (int)transformParams["crop_height"];

        var dense = mask.ToDense();
        var cropped = new T[OutputHeight, OutputWidth];
        double scaleX = (double)cropW / OutputWidth;
        double scaleY = (double)cropH / OutputHeight;

        for (int y = 0; y < OutputHeight; y++)
        {
            for (int x = 0; x < OutputWidth; x++)
            {
                int srcX = cropX + (int)(x * scaleX);
                int srcY = cropY + (int)(y * scaleY);
                if (srcX >= 0 && srcX < mask.Width && srcY >= 0 && srcY < mask.Height)
                    cropped[y, x] = dense[srcY, srcX];
            }
        }

        return new SegmentationMask<T>(cropped, mask.Type, mask.ClassIndex)
        {
            ClassName = mask.ClassName,
            InstanceId = mask.InstanceId
        };
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["output_height"] = OutputHeight;
        parameters["output_width"] = OutputWidth;
        parameters["min_scale"] = MinScale;
        parameters["max_scale"] = MaxScale;
        parameters["min_ratio"] = MinRatio;
        parameters["max_ratio"] = MaxRatio;
        parameters["interpolation"] = Interpolation.ToString();
        return parameters;
    }
}
