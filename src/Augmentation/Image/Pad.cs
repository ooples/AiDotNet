namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Specifies the padding mode for filling new pixels.
/// </summary>
public enum PaddingMode
{
    /// <summary>
    /// Fill with a constant value (default 0).
    /// </summary>
    Constant,

    /// <summary>
    /// Replicate edge pixels outward.
    /// </summary>
    Edge,

    /// <summary>
    /// Reflect pixels at the boundary (not duplicating the edge pixel).
    /// </summary>
    Reflect,

    /// <summary>
    /// Reflect pixels at the boundary (duplicating the edge pixel).
    /// </summary>
    Symmetric
}

/// <summary>
/// Pads an image with configurable padding amounts and fill modes.
/// </summary>
/// <remarks>
/// <para>
/// Padding adds pixels around the border of an image. Multiple fill modes control
/// how the new pixels are filled:
/// <list type="bullet">
/// <item><b>Constant</b>: Fills with a fixed value (e.g., 0 for black, 255 for white)</item>
/// <item><b>Edge</b>: Extends the edge pixel values outward</item>
/// <item><b>Reflect</b>: Mirrors the image at the boundary</item>
/// <item><b>Symmetric</b>: Like reflect but includes the boundary pixel</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Padding makes an image larger by adding extra pixels around
/// the edges. This is useful when you need a specific size but don't want to crop or stretch
/// the image. Black padding (constant=0) is most common.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Making images a specific size without distortion</item>
/// <item>Preparing for convolutions that reduce spatial size</item>
/// <item>Adding borders for visual presentation</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Pad<T> : SpatialImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the padding on the left side.
    /// </summary>
    public int PadLeft { get; }

    /// <summary>
    /// Gets the padding on the right side.
    /// </summary>
    public int PadRight { get; }

    /// <summary>
    /// Gets the padding on the top side.
    /// </summary>
    public int PadTop { get; }

    /// <summary>
    /// Gets the padding on the bottom side.
    /// </summary>
    public int PadBottom { get; }

    /// <summary>
    /// Gets the padding mode.
    /// </summary>
    public PaddingMode Mode { get; }

    /// <summary>
    /// Gets the constant fill value (used when Mode is Constant).
    /// </summary>
    public double FillValue { get; }

    /// <summary>
    /// Creates a new pad augmentation with uniform padding.
    /// </summary>
    /// <param name="padding">The number of pixels to add on all sides.</param>
    /// <param name="mode">The padding mode. Default is Constant.</param>
    /// <param name="fillValue">The fill value for Constant mode. Default is 0 (black).</param>
    /// <param name="probability">The probability of applying this augmentation.</param>
    public Pad(int padding, PaddingMode mode = PaddingMode.Constant, double fillValue = 0,
        double probability = 1.0)
        : this(padding, padding, padding, padding, mode, fillValue, probability)
    {
    }

    /// <summary>
    /// Creates a new pad augmentation with separate horizontal and vertical padding.
    /// </summary>
    /// <param name="padHorizontal">Padding for left and right sides.</param>
    /// <param name="padVertical">Padding for top and bottom sides.</param>
    /// <param name="mode">The padding mode.</param>
    /// <param name="fillValue">The fill value for Constant mode.</param>
    /// <param name="probability">The probability of applying this augmentation.</param>
    public Pad(int padHorizontal, int padVertical, PaddingMode mode = PaddingMode.Constant,
        double fillValue = 0, double probability = 1.0)
        : this(padHorizontal, padHorizontal, padVertical, padVertical, mode, fillValue, probability)
    {
    }

    /// <summary>
    /// Creates a new pad augmentation with individual padding per side.
    /// </summary>
    /// <param name="padLeft">Padding for the left side.</param>
    /// <param name="padRight">Padding for the right side.</param>
    /// <param name="padTop">Padding for the top side.</param>
    /// <param name="padBottom">Padding for the bottom side.</param>
    /// <param name="mode">The padding mode.</param>
    /// <param name="fillValue">The fill value for Constant mode.</param>
    /// <param name="probability">The probability of applying this augmentation.</param>
    public Pad(int padLeft, int padRight, int padTop, int padBottom,
        PaddingMode mode = PaddingMode.Constant, double fillValue = 0, double probability = 1.0)
        : base(probability)
    {
        if (padLeft < 0 || padRight < 0 || padTop < 0 || padBottom < 0)
            throw new ArgumentOutOfRangeException("Padding values must be non-negative.");

        PadLeft = padLeft;
        PadRight = padRight;
        PadTop = padTop;
        PadBottom = padBottom;
        Mode = mode;
        FillValue = fillValue;
    }

    /// <summary>
    /// Applies the padding operation and returns transformation parameters.
    /// </summary>
    protected override (ImageTensor<T> data, IDictionary<string, object> parameters) ApplyWithTransformParams(
        ImageTensor<T> data, AugmentationContext<T> context)
    {
        int srcH = data.Height;
        int srcW = data.Width;
        int newH = srcH + PadTop + PadBottom;
        int newW = srcW + PadLeft + PadRight;

        var result = new ImageTensor<T>(newH, newW, data.Channels, data.ChannelOrder, data.ColorSpace)
        {
            IsNormalized = data.IsNormalized,
            NormalizationMean = data.NormalizationMean,
            NormalizationStd = data.NormalizationStd,
            OriginalRange = data.OriginalRange
        };

        // Fill with constant value if in Constant mode
        if (Mode == PaddingMode.Constant)
        {
            T fill = NumOps.FromDouble(FillValue);
            for (int y = 0; y < newH; y++)
            {
                for (int x = 0; x < newW; x++)
                {
                    for (int c = 0; c < data.Channels; c++)
                    {
                        result.SetPixel(y, x, c, fill);
                    }
                }
            }
        }

        // Fill all pixels using the appropriate mode
        for (int y = 0; y < newH; y++)
        {
            for (int x = 0; x < newW; x++)
            {
                int srcY = y - PadTop;
                int srcX = x - PadLeft;

                if (srcY >= 0 && srcY < srcH && srcX >= 0 && srcX < srcW)
                {
                    // Inside original image
                    for (int c = 0; c < data.Channels; c++)
                    {
                        result.SetPixel(y, x, c, data.GetPixel(srcY, srcX, c));
                    }
                }
                else if (Mode != PaddingMode.Constant)
                {
                    // Outside original image, use fill mode
                    int mappedY = MapCoordinate(srcY, srcH, Mode);
                    int mappedX = MapCoordinate(srcX, srcW, Mode);

                    for (int c = 0; c < data.Channels; c++)
                    {
                        result.SetPixel(y, x, c, data.GetPixel(mappedY, mappedX, c));
                    }
                }
            }
        }

        var parameters = new Dictionary<string, object>
        {
            ["pad_left"] = PadLeft,
            ["pad_right"] = PadRight,
            ["pad_top"] = PadTop,
            ["pad_bottom"] = PadBottom,
            ["original_width"] = srcW,
            ["original_height"] = srcH,
            ["mode"] = Mode.ToString()
        };

        return (result, parameters);
    }

    private static int MapCoordinate(int coord, int size, PaddingMode mode)
    {
        if (coord >= 0 && coord < size)
            return coord;

        switch (mode)
        {
            case PaddingMode.Edge:
                return Math.Max(0, Math.Min(coord, size - 1));

            case PaddingMode.Reflect:
                if (coord < 0)
                    coord = -coord - 1;
                if (coord >= size)
                    coord = 2 * size - coord - 1;
                return Math.Max(0, Math.Min(coord, size - 1));

            case PaddingMode.Symmetric:
                if (coord < 0)
                    coord = -coord;
                if (coord >= size)
                    coord = 2 * size - coord - 2;
                return Math.Max(0, Math.Min(coord, size - 1));

            default:
                return Math.Max(0, Math.Min(coord, size - 1));
        }
    }

    /// <summary>
    /// Transforms a bounding box after padding.
    /// </summary>
    protected override BoundingBox<T> TransformBoundingBox(
        BoundingBox<T> box,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        var (x, y, w, h) = box.ToXYWH();

        var result = box.Clone();
        result.X1 = NumOps.FromDouble(x + PadLeft);
        result.Y1 = NumOps.FromDouble(y + PadTop);
        result.X2 = NumOps.FromDouble(x + w + PadLeft);
        result.Y2 = NumOps.FromDouble(y + h + PadTop);
        result.Format = BoundingBoxFormat.XYXY;
        return result;
    }

    /// <summary>
    /// Transforms a keypoint after padding.
    /// </summary>
    protected override Keypoint<T> TransformKeypoint(
        Keypoint<T> keypoint,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        double x = NumOps.ToDouble(keypoint.X) + PadLeft;
        double y = NumOps.ToDouble(keypoint.Y) + PadTop;

        var result = keypoint.Clone();
        result.X = NumOps.FromDouble(x);
        result.Y = NumOps.FromDouble(y);
        return result;
    }

    /// <summary>
    /// Transforms a segmentation mask after padding.
    /// </summary>
    protected override SegmentationMask<T> TransformMask(
        SegmentationMask<T> mask,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        int newH = mask.Height + PadTop + PadBottom;
        int newW = mask.Width + PadLeft + PadRight;
        var dense = mask.ToDense();
        var padded = new T[newH, newW];

        for (int y = 0; y < mask.Height; y++)
        {
            for (int x = 0; x < mask.Width; x++)
            {
                padded[y + PadTop, x + PadLeft] = dense[y, x];
            }
        }

        return new SegmentationMask<T>(padded, mask.Type, mask.ClassIndex)
        {
            ClassName = mask.ClassName,
            InstanceId = mask.InstanceId
        };
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["pad_left"] = PadLeft;
        parameters["pad_right"] = PadRight;
        parameters["pad_top"] = PadTop;
        parameters["pad_bottom"] = PadBottom;
        parameters["mode"] = Mode.ToString();
        parameters["fill_value"] = FillValue;
        return parameters;
    }
}
