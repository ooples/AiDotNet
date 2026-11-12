using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Specifies the channel ordering of an image tensor.
/// </summary>
public enum ChannelOrder
{
    /// <summary>
    /// Channels, Height, Width (PyTorch/Caffe convention).
    /// </summary>
    CHW,

    /// <summary>
    /// Height, Width, Channels (TensorFlow/NumPy convention).
    /// </summary>
    HWC,

    /// <summary>
    /// Batch, Channels, Height, Width (batched CHW).
    /// </summary>
    BCHW,

    /// <summary>
    /// Batch, Height, Width, Channels (batched HWC).
    /// </summary>
    BHWC
}

/// <summary>
/// Specifies the color space of an image.
/// </summary>
public enum ColorSpace
{
    /// <summary>
    /// Red, Green, Blue color space.
    /// </summary>
    RGB,

    /// <summary>
    /// Blue, Green, Red color space (OpenCV default).
    /// </summary>
    BGR,

    /// <summary>
    /// Single channel grayscale.
    /// </summary>
    Grayscale,

    /// <summary>
    /// Red, Green, Blue, Alpha (with transparency).
    /// </summary>
    RGBA,

    /// <summary>
    /// Hue, Saturation, Value color space.
    /// </summary>
    HSV,

    /// <summary>
    /// Hue, Saturation, Lightness color space.
    /// </summary>
    HSL,

    /// <summary>
    /// CIE L*a*b* color space.
    /// </summary>
    LAB,

    /// <summary>
    /// YCbCr color space (JPEG encoding).
    /// </summary>
    YCbCr
}

/// <summary>
/// Specifies the interpolation method for image resizing.
/// </summary>
public enum InterpolationMode
{
    /// <summary>
    /// Nearest neighbor interpolation (fastest, blocky).
    /// </summary>
    Nearest,

    /// <summary>
    /// Bilinear interpolation (good balance).
    /// </summary>
    Bilinear,

    /// <summary>
    /// Bicubic interpolation (smoother, slower).
    /// </summary>
    Bicubic,

    /// <summary>
    /// Lanczos resampling (high quality, slowest).
    /// </summary>
    Lanczos,

    /// <summary>
    /// Area-based resampling (good for downscaling).
    /// </summary>
    Area
}

/// <summary>
/// Represents an image as a tensor with image-specific metadata and operations.
/// </summary>
/// <remarks>
/// <para>
/// ImageTensor wraps a Tensor&lt;T&gt; to provide image-specific functionality:
/// - Channel ordering (CHW vs HWC)
/// - Color space awareness (RGB, BGR, HSV, etc.)
/// - Normalization state tracking
/// - Image-specific operations (crop, resize, color conversion)
/// </para>
/// <para><b>For Beginners:</b> An image on a computer is stored as numbers representing
/// pixel colors. This class represents those numbers in a way that's optimized for
/// machine learning, while keeping track of important details like whether the image
/// is in RGB or BGR format.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for pixel values.</typeparam>
public class ImageTensor<T>
{
    private readonly Tensor<T> _data;

    /// <summary>
    /// Gets the underlying tensor data.
    /// </summary>
    public Tensor<T> Data => _data;

    /// <summary>
    /// Gets the image height in pixels.
    /// </summary>
    public int Height { get; private set; }

    /// <summary>
    /// Gets the image width in pixels.
    /// </summary>
    public int Width { get; private set; }

    /// <summary>
    /// Gets the number of channels.
    /// </summary>
    public int Channels { get; private set; }

    /// <summary>
    /// Gets the batch size (1 for single images).
    /// </summary>
    public int BatchSize { get; private set; } = 1;

    /// <summary>
    /// Gets or sets the channel ordering.
    /// </summary>
    public ChannelOrder ChannelOrder { get; set; }

    /// <summary>
    /// Gets or sets the color space.
    /// </summary>
    public ColorSpace ColorSpace { get; set; }

    /// <summary>
    /// Gets or sets whether the image is normalized to [0, 1].
    /// </summary>
    public bool IsNormalized { get; set; }

    /// <summary>
    /// Gets or sets the normalization mean (per channel).
    /// </summary>
    public T[]? NormalizationMean { get; set; }

    /// <summary>
    /// Gets or sets the normalization std (per channel).
    /// </summary>
    public T[]? NormalizationStd { get; set; }

    /// <summary>
    /// Gets or sets the original value range before normalization.
    /// </summary>
    public (T min, T max)? OriginalRange { get; set; }

    /// <summary>
    /// Gets or sets additional metadata.
    /// </summary>
    public IDictionary<string, object>? Metadata { get; set; }

    /// <summary>
    /// Creates an ImageTensor from an existing tensor.
    /// </summary>
    /// <param name="data">The tensor data.</param>
    /// <param name="channelOrder">The channel ordering.</param>
    /// <param name="colorSpace">The color space.</param>
    public ImageTensor(Tensor<T> data, ChannelOrder channelOrder = ChannelOrder.CHW, ColorSpace colorSpace = ColorSpace.RGB)
    {
        Guard.NotNull(data);
        _data = data;
        ChannelOrder = channelOrder;
        ColorSpace = colorSpace;

        ParseDimensions();
    }

    /// <summary>
    /// Creates an ImageTensor with specified dimensions.
    /// </summary>
    /// <param name="height">The image height.</param>
    /// <param name="width">The image width.</param>
    /// <param name="channels">The number of channels.</param>
    /// <param name="channelOrder">The channel ordering.</param>
    /// <param name="colorSpace">The color space.</param>
    public ImageTensor(int height, int width, int channels = 3, ChannelOrder channelOrder = ChannelOrder.CHW, ColorSpace colorSpace = ColorSpace.RGB)
    {
        Height = height;
        Width = width;
        Channels = channels;
        ChannelOrder = channelOrder;
        ColorSpace = colorSpace;

        int[] dimensions = channelOrder switch
        {
            ChannelOrder.CHW => [channels, height, width],
            ChannelOrder.HWC => [height, width, channels],
            ChannelOrder.BCHW => [1, channels, height, width],
            ChannelOrder.BHWC => [1, height, width, channels],
            _ => throw new ArgumentException($"Unknown channel order: {channelOrder}")
        };

        _data = new Tensor<T>(dimensions);
    }

    /// <summary>
    /// Creates a batched ImageTensor with specified dimensions.
    /// </summary>
    /// <param name="batchSize">The batch size.</param>
    /// <param name="height">The image height.</param>
    /// <param name="width">The image width.</param>
    /// <param name="channels">The number of channels.</param>
    /// <param name="channelOrder">The channel ordering (must be BCHW or BHWC).</param>
    /// <param name="colorSpace">The color space.</param>
    public ImageTensor(int batchSize, int height, int width, int channels = 3, ChannelOrder channelOrder = ChannelOrder.BCHW, ColorSpace colorSpace = ColorSpace.RGB)
    {
        BatchSize = batchSize;
        Height = height;
        Width = width;
        Channels = channels;
        ChannelOrder = channelOrder;
        ColorSpace = colorSpace;

        int[] dimensions = channelOrder switch
        {
            ChannelOrder.BCHW => [batchSize, channels, height, width],
            ChannelOrder.BHWC => [batchSize, height, width, channels],
            _ => throw new ArgumentException($"Batched images require BCHW or BHWC channel order, got: {channelOrder}")
        };

        _data = new Tensor<T>(dimensions);
    }

    /// <summary>
    /// Creates a deep copy of this image tensor.
    /// </summary>
    /// <returns>A new ImageTensor with copied data.</returns>
    public ImageTensor<T> Clone()
    {
        // Create a copy of the tensor data — defensive-copy the shape array
        // so the clone never aliases the original's mutable shape metadata.
        var clonedData = new Tensor<T>((int[])_data._shape.Clone());
        _data.Contiguous().CopyTo(clonedData.AsWritableSpan());

        return new ImageTensor<T>(clonedData, ChannelOrder, ColorSpace)
        {
            IsNormalized = IsNormalized,
            NormalizationMean = NormalizationMean is not null ? (T[])NormalizationMean.Clone() : null,
            NormalizationStd = NormalizationStd is not null ? (T[])NormalizationStd.Clone() : null,
            OriginalRange = OriginalRange,
            Metadata = Metadata is not null ? new Dictionary<string, object>(Metadata) : null
        };
    }

    /// <summary>
    /// Converts this image to a different channel order.
    /// </summary>
    /// <param name="targetOrder">The target channel order.</param>
    /// <returns>A new ImageTensor with the specified channel order.</returns>
    public ImageTensor<T> ToChannelOrder(ChannelOrder targetOrder)
    {
        if (ChannelOrder == targetOrder)
        {
            return Clone();
        }

        // Determine target dimensions
        int[] targetDimensions = targetOrder switch
        {
            ChannelOrder.CHW => [Channels, Height, Width],
            ChannelOrder.HWC => [Height, Width, Channels],
            ChannelOrder.BCHW => [BatchSize, Channels, Height, Width],
            ChannelOrder.BHWC => [BatchSize, Height, Width, Channels],
            _ => throw new ArgumentException($"Unknown target order: {targetOrder}")
        };

        var result = new Tensor<T>(targetDimensions);

        // Transpose the data
        TransposeData(_data, result, ChannelOrder, targetOrder);

        return new ImageTensor<T>(result, targetOrder, ColorSpace)
        {
            IsNormalized = IsNormalized,
            NormalizationMean = NormalizationMean,
            NormalizationStd = NormalizationStd,
            OriginalRange = OriginalRange,
            Metadata = Metadata
        };
    }

    /// <summary>
    /// Gets a pixel value at the specified coordinates.
    /// </summary>
    /// <param name="y">The y coordinate (row).</param>
    /// <param name="x">The x coordinate (column).</param>
    /// <param name="channel">The channel index.</param>
    /// <returns>The pixel value.</returns>
    public T GetPixel(int y, int x, int channel = 0)
    {
        int index = CalculateIndex(0, y, x, channel);
        return _data[index];
    }

    /// <summary>
    /// Sets a pixel value at the specified coordinates.
    /// </summary>
    /// <param name="y">The y coordinate (row).</param>
    /// <param name="x">The x coordinate (column).</param>
    /// <param name="channel">The channel index.</param>
    /// <param name="value">The value to set.</param>
    public void SetPixel(int y, int x, int channel, T value)
    {
        int index = CalculateIndex(0, y, x, channel);
        _data[index] = value;
    }

    /// <summary>
    /// Gets all channel values at a pixel location.
    /// </summary>
    /// <param name="y">The y coordinate.</param>
    /// <param name="x">The x coordinate.</param>
    /// <returns>Array of channel values.</returns>
    public T[] GetPixelChannels(int y, int x)
    {
        var result = new T[Channels];
        for (int c = 0; c < Channels; c++)
        {
            result[c] = GetPixel(y, x, c);
        }
        return result;
    }

    /// <summary>
    /// Sets all channel values at a pixel location.
    /// </summary>
    /// <param name="y">The y coordinate.</param>
    /// <param name="x">The x coordinate.</param>
    /// <param name="values">The channel values.</param>
    public void SetPixelChannels(int y, int x, T[] values)
    {
        for (int c = 0; c < Channels && c < values.Length; c++)
        {
            SetPixel(y, x, c, values[c]);
        }
    }

    /// <summary>
    /// Extracts a rectangular region from the image.
    /// </summary>
    /// <param name="x">The left edge x coordinate.</param>
    /// <param name="y">The top edge y coordinate.</param>
    /// <param name="width">The region width.</param>
    /// <param name="height">The region height.</param>
    /// <returns>A new ImageTensor containing the extracted region.</returns>
    public ImageTensor<T> Crop(int x, int y, int width, int height)
    {
        if (x < 0 || y < 0 || x + width > Width || y + height > Height)
        {
            throw new ArgumentOutOfRangeException("Crop region exceeds image bounds.");
        }

        var result = new ImageTensor<T>(height, width, Channels, ChannelOrder, ColorSpace)
        {
            IsNormalized = IsNormalized,
            NormalizationMean = NormalizationMean,
            NormalizationStd = NormalizationStd,
            OriginalRange = OriginalRange
        };

        for (int dy = 0; dy < height; dy++)
        {
            for (int dx = 0; dx < width; dx++)
            {
                for (int c = 0; c < Channels; c++)
                {
                    result.SetPixel(dy, dx, c, GetPixel(y + dy, x + dx, c));
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Gets the tensor dimensions as an array.
    /// </summary>
    /// <returns>The dimensions array.</returns>
    public int[] GetDimensions()
    {
        return (int[])_data._shape.Clone();
    }

    /// <summary>
    /// Parses dimensions from the tensor based on channel order.
    /// </summary>
    private void ParseDimensions()
    {
        var dims = _data._shape;

        switch (ChannelOrder)
        {
            case ChannelOrder.CHW when dims.Length == 3:
                Channels = dims[0];
                Height = dims[1];
                Width = dims[2];
                BatchSize = 1;
                break;

            case ChannelOrder.HWC when dims.Length == 3:
                Height = dims[0];
                Width = dims[1];
                Channels = dims[2];
                BatchSize = 1;
                break;

            case ChannelOrder.BCHW when dims.Length == 4:
                BatchSize = dims[0];
                Channels = dims[1];
                Height = dims[2];
                Width = dims[3];
                break;

            case ChannelOrder.BHWC when dims.Length == 4:
                BatchSize = dims[0];
                Height = dims[1];
                Width = dims[2];
                Channels = dims[3];
                break;

            default:
                throw new ArgumentException($"Tensor dimensions {string.Join("x", dims)} incompatible with channel order {ChannelOrder}");
        }
    }

    /// <summary>
    /// Calculates the flat index for a pixel location.
    /// </summary>
    private int CalculateIndex(int batch, int y, int x, int channel)
    {
        return ChannelOrder switch
        {
            ChannelOrder.CHW => channel * Height * Width + y * Width + x,
            ChannelOrder.HWC => y * Width * Channels + x * Channels + channel,
            ChannelOrder.BCHW => batch * Channels * Height * Width + channel * Height * Width + y * Width + x,
            ChannelOrder.BHWC => batch * Height * Width * Channels + y * Width * Channels + x * Channels + channel,
            _ => throw new InvalidOperationException($"Unknown channel order: {ChannelOrder}")
        };
    }

    /// <summary>
    /// Transposes data between channel orderings.
    /// </summary>
    private static void TransposeData(Tensor<T> source, Tensor<T> target, ChannelOrder sourceOrder, ChannelOrder targetOrder)
    {
        var sourceDims = source._shape;
        int height, width, channels, batchSize;

        // Parse source dimensions
        switch (sourceOrder)
        {
            case ChannelOrder.CHW:
                channels = sourceDims[0];
                height = sourceDims[1];
                width = sourceDims[2];
                batchSize = 1;
                break;
            case ChannelOrder.HWC:
                height = sourceDims[0];
                width = sourceDims[1];
                channels = sourceDims[2];
                batchSize = 1;
                break;
            case ChannelOrder.BCHW:
                batchSize = sourceDims[0];
                channels = sourceDims[1];
                height = sourceDims[2];
                width = sourceDims[3];
                break;
            case ChannelOrder.BHWC:
                batchSize = sourceDims[0];
                height = sourceDims[1];
                width = sourceDims[2];
                channels = sourceDims[3];
                break;
            default:
                throw new ArgumentException($"Unknown source order: {sourceOrder}");
        }

        // Bring the source into canonical [batch, channel, height, width] layout.
        var engine = AiDotNetEngine.Current;
        Tensor<T> canonical = sourceOrder switch
        {
            ChannelOrder.CHW => engine.Reshape(source, [1, channels, height, width]),
            ChannelOrder.HWC => engine.TensorPermute(engine.Reshape(source, [1, height, width, channels]), [0, 3, 1, 2]),
            ChannelOrder.BCHW => source,
            _ => engine.TensorPermute(source, [0, 3, 1, 2])
        };

        // Single-image targets have no batch axis, so later batches overwrite earlier ones:
        // the last batch is the one that remains.
        Tensor<T> arranged = targetOrder switch
        {
            ChannelOrder.CHW => engine.TensorSliceAxis(canonical, 0, batchSize - 1),
            ChannelOrder.HWC => engine.TensorPermute(engine.TensorSliceAxis(canonical, 0, batchSize - 1), [1, 2, 0]),
            ChannelOrder.BCHW => canonical,
            ChannelOrder.BHWC => engine.TensorPermute(canonical, [0, 2, 3, 1]),
            _ => throw new InvalidOperationException($"Unknown channel order: {targetOrder}")
        };

        arranged.Contiguous().CopyTo(target.AsWritableSpan());
    }

    /// <summary>
    /// Returns the first image's pixels as a new contiguous [channels, height, width] tensor,
    /// whatever the stored channel order is.
    /// </summary>
    internal Tensor<T> ToPlanarTensor()
    {
        var engine = AiDotNetEngine.Current;
        Tensor<T> image = _data.Rank == 4 ? engine.TensorSliceAxis(_data, 0, 0) : _data;
        bool channelsLast = ChannelOrder == ChannelOrder.HWC || ChannelOrder == ChannelOrder.BHWC;
        Tensor<T> planar = channelsLast ? engine.TensorPermute(image, [2, 0, 1]) : image;
        var copy = new Tensor<T>([Channels, Height, Width]);
        planar.Contiguous().CopyTo(copy.AsWritableSpan());
        return copy;
    }

    /// <summary>
    /// Returns w0 * channel0 + w1 * channel1 + w2 * channel2 of a [channels, height, width]
    /// tensor as a [height, width] plane.
    /// </summary>
    internal static Tensor<T> MixChannels(Tensor<T> planar, double w0, double w1, double w2)
    {
        var engine = AiDotNetEngine.Current;
        return MixPlanes(
            engine.TensorSliceAxis(planar, 0, 0), engine.TensorSliceAxis(planar, 0, 1), engine.TensorSliceAxis(planar, 0, 2),
            w0, w1, w2);
    }

    /// <summary>
    /// Returns w0 * p0 + w1 * p1 + w2 * p2 for three equally shaped planes.
    /// </summary>
    internal static Tensor<T> MixPlanes(Tensor<T> p0, Tensor<T> p1, Tensor<T> p2, double w0, double w1, double w2)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var a = engine.TensorMultiplyScalar(p0, numOps.FromDouble(w0));
        var b = engine.TensorMultiplyScalar(p1, numOps.FromDouble(w1));
        var c = engine.TensorMultiplyScalar(p2, numOps.FromDouble(w2));
        return engine.TensorAdd(engine.TensorAdd(a, b), c);
    }

    /// <summary>
    /// Removes the sRGB transfer curve: v / 12.92 for v &lt;= 0.04045, else ((v + 0.055) / 1.055)^2.4.
    /// </summary>
    internal static Tensor<T> SrgbToLinear(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var linear = engine.TensorDivideScalar(values, numOps.FromDouble(12.92));
        var curved = engine.TensorPower(
            engine.TensorDivideScalar(engine.TensorAddScalar(values, numOps.FromDouble(0.055)), numOps.FromDouble(1.055)),
            numOps.FromDouble(2.4));
        return engine.TensorWhere(engine.TensorGreaterThan(values, numOps.FromDouble(0.04045)), curved, linear);
    }

    /// <summary>
    /// Splits the first three channels of a [channels, height, width] tensor into [height, width]
    /// planes, dividing by 255 when the image is not normalized so the planes are in [0, 1].
    /// </summary>
    internal static (Tensor<T> R, Tensor<T> G, Tensor<T> B) UnitRgbPlanes(Tensor<T> planar, bool isNormalized)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var r = engine.TensorSliceAxis(planar, 0, 0);
        var g = engine.TensorSliceAxis(planar, 0, 1);
        var b = engine.TensorSliceAxis(planar, 0, 2);
        if (!isNormalized)
        {
            T scale = numOps.FromDouble(255.0);
            r = engine.TensorDivideScalar(r, scale);
            g = engine.TensorDivideScalar(g, scale);
            b = engine.TensorDivideScalar(b, scale);
        }

        return (r, g, b);
    }

    /// <summary>
    /// Computes hue normalized to [0, 1) from RGB planes and their per-pixel max and max - min,
    /// using the hexcone model shared by HSV and HLS (hue 0 where max - min is below 1e-10).
    /// </summary>
    internal static Tensor<T> HexconeHue(Tensor<T> r, Tensor<T> g, Tensor<T> b, Tensor<T> max, Tensor<T> delta)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        T eps = numOps.FromDouble(1e-10);
        T sixty = numOps.FromDouble(60.0);

        var six = new Tensor<T>(delta._shape);
        engine.TensorFill(six, numOps.FromDouble(6.0));

        // max == r: 60 * (((g - b) / delta) % 6)
        var hueR = engine.TensorMultiplyScalar(
            engine.TensorFmod(engine.TensorDivide(engine.TensorSubtract(g, b), delta), six), sixty);
        // max == g: 60 * (((b - r) / delta) + 2)
        var hueG = engine.TensorMultiplyScalar(
            engine.TensorAddScalar(engine.TensorDivide(engine.TensorSubtract(b, r), delta), numOps.FromDouble(2.0)), sixty);
        // otherwise: 60 * (((r - g) / delta) + 4)
        var hueB = engine.TensorMultiplyScalar(
            engine.TensorAddScalar(engine.TensorDivide(engine.TensorSubtract(r, g), delta), numOps.FromDouble(4.0)), sixty);

        var isR = engine.TensorLessThan(engine.TensorAbs(engine.TensorSubtract(max, r)), eps);
        var isG = engine.TensorLessThan(engine.TensorAbs(engine.TensorSubtract(max, g)), eps);
        var flat = engine.TensorLessThan(delta, eps);
        var zero = new Tensor<T>(delta._shape);

        var hue = engine.TensorWhere(flat, zero, engine.TensorWhere(isR, hueR, engine.TensorWhere(isG, hueG, hueB)));
        var negative = engine.TensorLessThan(hue, numOps.Zero);
        hue = engine.TensorWhere(negative, engine.TensorAddScalar(hue, numOps.FromDouble(360.0)), hue);
        return engine.TensorDivideScalar(hue, numOps.FromDouble(360.0));
    }

    /// <summary>
    /// Returns a tensor shaped like <paramref name="like"/> with every element set to <paramref name="value"/>.
    /// </summary>
    internal static Tensor<T> Constant(Tensor<T> like, double value)
    {
        var constant = new Tensor<T>((int[])like._shape.Clone());
        AiDotNetEngine.Current.TensorFill(constant, MathHelper.GetNumericOperations<T>().FromDouble(value));
        return constant;
    }

    /// <summary>
    /// Wraps hue values into [0, 1): (h % 1), plus 1 where the remainder is negative.
    /// </summary>
    internal static Tensor<T> WrapUnitInterval(Tensor<T> h)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var wrapped = engine.TensorFmod(h, Constant(h, 1.0));
        var negative = engine.TensorLessThan(wrapped, numOps.Zero);
        return engine.TensorWhere(negative, engine.TensorAddScalar(wrapped, numOps.One), wrapped);
    }

    /// <summary>
    /// Converts hue (in [0, 1)), saturation and value planes back to RGB planes using the
    /// hexcone sectors of h * 6.
    /// </summary>
    internal static (Tensor<T> R, Tensor<T> G, Tensor<T> B) HsvToRgbPlanes(Tensor<T> h, Tensor<T> s, Tensor<T> v)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var hPrime = engine.TensorMultiplyScalar(h, numOps.FromDouble(6.0));
        var c = engine.TensorMultiply(v, s);
        var x = engine.TensorMultiply(c, engine.ScalarMinusTensor(numOps.One,
            engine.TensorAbs(engine.TensorSubtractScalar(engine.TensorFmod(hPrime, Constant(hPrime, 2.0)), numOps.One))));
        var m = engine.TensorSubtract(v, c);
        var zero = Constant(hPrime, 0.0);

        var s1 = engine.TensorLessThan(hPrime, numOps.FromDouble(1.0));
        var s2 = engine.TensorLessThan(hPrime, numOps.FromDouble(2.0));
        var s3 = engine.TensorLessThan(hPrime, numOps.FromDouble(3.0));
        var s4 = engine.TensorLessThan(hPrime, numOps.FromDouble(4.0));
        var s5 = engine.TensorLessThan(hPrime, numOps.FromDouble(5.0));

        Tensor<T> Pick(Tensor<T> a0, Tensor<T> a1, Tensor<T> a2, Tensor<T> a3, Tensor<T> a4, Tensor<T> a5) =>
            engine.TensorWhere(s1, a0, engine.TensorWhere(s2, a1, engine.TensorWhere(s3, a2,
                engine.TensorWhere(s4, a3, engine.TensorWhere(s5, a4, a5)))));

        var r = engine.TensorAdd(Pick(c, x, zero, zero, x, c), m);
        var g = engine.TensorAdd(Pick(x, c, c, x, zero, zero), m);
        var b = engine.TensorAdd(Pick(zero, zero, x, c, c, x), m);
        return (r, g, b);
    }

    /// <summary>
    /// Pads the spatial axes of a [channels, height, width] tensor. Out-of-range rows/columns
    /// either repeat the edge (<paramref name="mirror"/> false) or mirror including the edge
    /// pixel (index -1 maps to 0, index size maps to size - 1).
    /// </summary>
    internal static Tensor<T> PadSpatial(Tensor<T> planar, int top, int bottom, int left, int right, bool mirror)
    {
        var engine = AiDotNetEngine.Current;
        var rows = BorderIndices(planar.Shape[1], top, bottom, mirror);
        var cols = BorderIndices(planar.Shape[2], left, right, mirror);
        return engine.TensorIndexSelect(engine.TensorIndexSelect(planar, rows, 1), cols, 2);
    }

    /// <summary>
    /// Source indices for positions -before .. size + after - 1 of an axis of length size.
    /// </summary>
    internal static Tensor<int> BorderIndices(int size, int before, int after, bool mirror)
    {
        var engine = AiDotNetEngine.Current;
        var idx = engine.TensorArange<int>(-before, size + after, 1);
        if (mirror)
        {
            // i < 0 -> -i - 1; i >= size -> 2 * size - i - 1
            var low = engine.ScalarMinusTensor(-1, idx);
            var high = engine.ScalarMinusTensor(2 * size - 1, idx);
            idx = engine.TensorWhere(engine.TensorLessThan(idx, 0), low,
                engine.TensorWhere(engine.TensorGreaterThan(idx, size - 1), high, idx));
        }

        return engine.TensorClamp(idx, 0, size - 1);
    }

    /// <summary>
    /// Returns the edge-replicated k x k neighbourhood of every pixel of a
    /// [channels, height, width] tensor as a [channels, height, width, k * k] tensor.
    /// </summary>
    internal static Tensor<T> Neighbourhoods(Tensor<T> planar, int kernelSize)
    {
        var engine = AiDotNetEngine.Current;
        int channels = planar.Shape[0], height = planar.Shape[1], width = planar.Shape[2];
        int half = kernelSize / 2;
        var padded = PadSpatial(planar, half, kernelSize - 1 - half, half, kernelSize - 1 - half, mirror: false);
        var windows = engine.TensorUnfold(engine.TensorUnfold(padded, 1, kernelSize, 1), 2, kernelSize, 1);
        return engine.Reshape(windows.Contiguous(), [channels, height, width, kernelSize * kernelSize]);
    }

    /// <summary>
    /// Converts values laid out row-major as [height, width, channels] (the order a y, x, c loop
    /// visits them) to a [channels, height, width] tensor.
    /// </summary>
    internal static Tensor<T> PlanarFromInterleaved(Tensor<T> values, int height, int width, int channels)
    {
        var engine = AiDotNetEngine.Current;
        var interleaved = engine.Reshape(values, [height, width, channels]);
        return engine.TensorPermute(interleaved, [2, 0, 1]).Contiguous();
    }

    /// <summary>
    /// Returns a copy of a [channels, height, width] tensor with the rectangle starting at
    /// (<paramref name="y"/>, <paramref name="x"/>) of size <paramref name="h"/> x <paramref name="w"/>
    /// set to <paramref name="value"/> in every channel.
    /// </summary>
    internal static Tensor<T> FillRect(Tensor<T> planar, int y, int x, int h, int w, T value)
    {
        var block = new Tensor<T>([planar.Shape[0], h, w]);
        AiDotNetEngine.Current.TensorFill(block, value);
        return AiDotNetEngine.Current.TensorSetSlice(planar, block, [0, y, x]);
    }

    /// <summary>
    /// Fills a rectangle of a [channels, height, width] tensor either with a constant or with
    /// uniform noise in [0, maxValue) drawn per pixel in y, x, c order (the order of a per-pixel loop).
    /// </summary>
    internal static Tensor<T> EraseRect(Tensor<T> planar, int y, int x, int h, int w, bool withNoise,
        double fillValue, double maxValue, AugmentationContext<T> context)
    {
        if (!withNoise)
        {
            return FillRect(planar, y, x, h, w, MathHelper.GetNumericOperations<T>().FromDouble(fillValue));
        }

        int channels = planar.Shape[0];
        var noise = AugmentationRandom<T>.Uniform(context, [h, w, channels], 0, maxValue);

        return AiDotNetEngine.Current.TensorSetSlice(planar, PlanarFromInterleaved(noise, h, w, channels), [0, y, x]);
    }

    /// <summary>
    /// Blends every channel towards <paramref name="maxValue"/> with a per-pixel weight:
    /// clamp(v * (1 - a) + maxValue * a, 0, maxValue) for a [height, width] weight map a.
    /// </summary>
    internal static Tensor<T> BlendTowardsMax(Tensor<T> planar, Tensor<T> weights, double maxValue)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var a = engine.Reshape(weights, [1, planar.Shape[1], planar.Shape[2]]);
        var blended = engine.TensorAdd(
            engine.TensorMultiply(planar, engine.ScalarMinusTensor(numOps.One, a)),
            engine.TensorMultiplyScalar(a, numOps.FromDouble(maxValue)));
        return engine.TensorClamp(blended, numOps.Zero, numOps.FromDouble(maxValue));
    }

    /// <summary>
    /// Per-channel mean and sqrt(population variance + 1e-8) of a [channels, height, width] tensor,
    /// each shaped [channels, 1, 1].
    /// </summary>
    internal static (Tensor<T> Mean, Tensor<T> Std) ChannelMeanStd(Tensor<T> planar)
    {
        var engine = AiDotNetEngine.Current;
        var mean = engine.ReduceMean(planar, [1, 2], true);
        var centered = engine.TensorSubtract(planar, mean);
        var variance = engine.ReduceMean(engine.TensorMultiply(centered, centered), [1, 2], true);
        var std = engine.TensorSqrt(engine.TensorAddScalar(variance, MathHelper.GetNumericOperations<T>().FromDouble(1e-8)));
        return (mean, std);
    }

    /// <summary>
    /// Returns the first <paramref name="count"/> channels of a [channels, height, width] tensor.
    /// </summary>
    internal static Tensor<T> LeadingChannels(Tensor<T> planar, int count)
    {
        return AiDotNetEngine.Current.TensorSlice(planar, [0, 0, 0], [count, planar.Shape[1], planar.Shape[2]]);
    }

    /// <summary>
    /// Replaces the first block.Shape[0] channels of a [channels, height, width] tensor with
    /// <paramref name="block"/>, keeping the remaining channels.
    /// </summary>
    internal static Tensor<T> ReplaceLeadingBlock(Tensor<T> planar, Tensor<T> block)
    {
        int channels = planar.Shape[0];
        int count = block.Shape[0];
        if (count >= channels)
        {
            return block;
        }

        var rest = AiDotNetEngine.Current.TensorSlice(planar, [count, 0, 0], [channels - count, planar.Shape[1], planar.Shape[2]]);
        return AiDotNetEngine.Current.TensorConcatenate([block, rest], 0);
    }

    /// <summary>
    /// Per-pixel gradient magnitude summed over channels, sum_c sqrt(dx^2 + dy^2) with central
    /// differences, for interior pixels of a [channels, height, width] tensor; border pixels are 0.
    /// Returns a [height, width] tensor.
    /// </summary>
    internal static Tensor<T> GradientMagnitude(Tensor<T> planar)
    {
        var engine = AiDotNetEngine.Current;
        int channels = planar.Shape[0], height = planar.Shape[1], width = planar.Shape[2];
        var result = new Tensor<T>([height, width]);
        if (height < 3 || width < 3)
        {
            return result;
        }

        int ih = height - 2, iw = width - 2;
        var dx = engine.TensorSubtract(
            engine.TensorSlice(planar, [0, 1, 2], [channels, ih, iw]),
            engine.TensorSlice(planar, [0, 1, 0], [channels, ih, iw]));
        var dy = engine.TensorSubtract(
            engine.TensorSlice(planar, [0, 2, 1], [channels, ih, iw]),
            engine.TensorSlice(planar, [0, 0, 1], [channels, ih, iw]));
        var magnitude = engine.TensorSqrt(engine.TensorAdd(engine.TensorMultiply(dx, dx), engine.TensorMultiply(dy, dy)));
        var summed = engine.ReduceSum(magnitude, [0], false);
        return engine.TensorSetSlice(result, summed, [1, 1]);
    }

    /// <summary>
    /// Returns one channel of the first image as a [height, width] tensor.
    /// </summary>
    internal Tensor<T> GetChannelPlane(int channel)
    {
        return AiDotNetEngine.Current.TensorSliceAxis(ToPlanarTensor(), 0, channel);
    }

    /// <summary>
    /// Overwrites one channel of the first image with a [height, width] tensor.
    /// </summary>
    internal void SetChannelPlane(int channel, Tensor<T> plane)
    {
        var planar = ToPlanarTensor();
        AiDotNetEngine.Current.TensorSetSliceAxis(planar, plane.Contiguous(), 0, channel);
        SetFromPlanarTensor(planar);
    }

    /// <summary>
    /// Histogram bin of every value: clamp((int)(v / maxValue * (numBins - 1)), 0, numBins - 1).
    /// </summary>
    internal static Tensor<int> HistogramBins(Tensor<T> values, double maxValue, int numBins)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var scaled = engine.TensorMultiplyScalar(
            engine.TensorDivideScalar(values, numOps.FromDouble(maxValue)), numOps.FromDouble(numBins - 1.0));
        return engine.TensorClamp(engine.TensorTrunc(scaled), numOps.Zero, numOps.FromDouble(numBins - 1.0)).Cast<int>();
    }

    /// <summary>
    /// Histogram-equalization levels per bin: (cdf[bin] - cdfMin) / max(1, total - cdfMin) * maxValue,
    /// where cdfMin is the first non-zero CDF value.
    /// </summary>
    internal static Tensor<T> EqualizationLevels(Tensor<int> bins, int numBins, double maxValue)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var flatBins = engine.Reshape(bins, [bins.Length]);

        // Integer CDF; cdfMin is its smallest positive entry (0 when there is none)
        var cdfCounts = engine.TensorCumSum(engine.TensorBinCount(flatBins, numBins), 0);
        var positive = engine.TensorGreaterThan(cdfCounts, 0);
        int cdfMin = engine.TensorMaxValue(positive) > 0
            ? engine.TensorMinValue(engine.TensorWhere(positive, cdfCounts, ImageTensor<int>.Constant(cdfCounts, int.MaxValue)))
            : 0;
        int denom = Math.Max(1, bins.Length - cdfMin);
        var cdf = cdfCounts.Cast<T>();
        return engine.TensorMultiplyScalar(
            engine.TensorDivideScalar(engine.TensorSubtractScalar(cdf, numOps.FromDouble(cdfMin)), numOps.FromDouble(denom)),
            numOps.FromDouble(maxValue));
    }

    /// <summary>
    /// Applies separable resampling to a [channels, height, width] tensor:
    /// out[c] = rowWeights * in[c] * colWeights^T with rowWeights [newHeight, height] and
    /// colWeights [newWidth, width]. Returns [channels, newHeight, newWidth].
    /// </summary>
    internal static Tensor<T> ResampleSeparable(Tensor<T> planar, Tensor<T> rowWeights, Tensor<T> colWeights)
    {
        var engine = AiDotNetEngine.Current;
        int channels = planar.Shape[0], height = planar.Shape[1], width = planar.Shape[2];
        int newHeight = rowWeights.Shape[0], newWidth = colWeights.Shape[0];

        var byRow = engine.Reshape(engine.TensorPermute(planar, [1, 0, 2]).Contiguous(), [height, channels * width]);
        var rowsResampled = engine.Reshape(engine.TensorMatMul(rowWeights, byRow), [newHeight, channels, width]);
        var byChannel = engine.Reshape(engine.TensorPermute(rowsResampled, [1, 0, 2]).Contiguous(), [channels * newHeight, width]);
        var resampled = engine.TensorMatMul(byChannel, engine.TensorTranspose(colWeights));
        return engine.Reshape(resampled, [channels, newHeight, newWidth]);
    }

    /// <summary>
    /// Value noise: a [height / cellSize + 2, width / cellSize + 2] grid of U(min, max) draws (row-major
    /// draw order) bilinearly interpolated to [height, width], pixel (y, x) sampling grid position
    /// (y / cellSize, x / cellSize). With <paramref name="smoothstep"/> the interpolation fractions
    /// t become t * t * (3 - 2 * t).
    /// </summary>
    internal static Tensor<T> GridNoise(AugmentationContext<T> context, int height, int width, int cellSize,
        double min, double max, bool smoothstep)
    {
        var engine = AiDotNetEngine.Current;
        int gridHeight = height / cellSize + 2;
        int gridWidth = width / cellSize + 2;
        var grid = AugmentationRandom<T>.Uniform(context, [gridHeight, gridWidth], min, max);
        var numOps = MathHelper.GetNumericOperations<T>();
        T cell = numOps.FromDouble(cellSize);
        var rowPositions = engine.TensorDivideScalar(engine.TensorArange<T>(0, height, 1), cell);
        var colPositions = engine.TensorDivideScalar(engine.TensorArange<T>(0, width, 1), cell);
        return InterpolateGrid(grid, rowPositions, colPositions, smoothstep);
    }

    /// <summary>
    /// Bilinear interpolation of a [gridHeight, gridWidth] grid at fractional row positions [height]
    /// and column positions [width] (non-negative): out[y, x] mixes the grid points floor(f) and
    /// min(floor(f) + 1, last) along each axis with fraction f - floor(f) (smoothstepped when asked).
    /// </summary>
    internal static Tensor<T> InterpolateGrid(Tensor<T> grid, Tensor<T> rowPositions, Tensor<T> colPositions, bool smoothstep)
    {
        var engine = AiDotNetEngine.Current;
        var rows = InterpolationWeights(rowPositions, grid.Shape[0], smoothstep);
        var cols = InterpolationWeights(colPositions, grid.Shape[1], smoothstep);
        return engine.TensorMatMul(engine.TensorMatMul(rows, grid), engine.TensorTranspose(cols));
    }

    /// <summary>
    /// [size, gridLength] linear interpolation weights: position f samples between grid points floor(f)
    /// and min(floor(f) + 1, gridLength - 1) with fraction f - floor(f).
    /// </summary>
    private static Tensor<T> InterpolationWeights(Tensor<T> positionsAlongAxis, int gridLength, bool smoothstep)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int size = positionsAlongAxis.Length;
        int[] shape = [size, gridLength];
        var f = engine.Reshape(positionsAlongAxis, [size, 1]);
        var lower = engine.TensorFloor(f);
        var upper = engine.TensorClampMax(engine.TensorAddScalar(lower, numOps.One), numOps.FromDouble(gridLength - 1));
        var t = engine.TensorSubtract(f, lower);
        if (smoothstep)
        {
            var curve = engine.ScalarMinusTensor(numOps.FromDouble(3.0), engine.TensorMultiplyScalar(t, numOps.FromDouble(2.0)));
            t = engine.TensorMultiply(engine.TensorMultiply(t, t), curve);
        }

        var positions = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, gridLength, 1), [1, gridLength]), shape);
        var atLower = engine.TensorEquals(positions, engine.TensorBroadcastTo(lower, shape));
        var atUpper = engine.TensorEquals(positions, engine.TensorBroadcastTo(upper, shape));
        return engine.TensorAdd(
            engine.TensorMultiply(atLower, engine.TensorBroadcastTo(engine.ScalarMinusTensor(numOps.One, t), shape)),
            engine.TensorMultiply(atUpper, engine.TensorBroadcastTo(t, shape)));
    }

    /// <summary>
    /// Squared distance (dy^2 + dx^2) of every tap of a size x size kernel from its centre tap
    /// size / 2, as a [size, size] integer tensor.
    /// </summary>
    internal static Tensor<int> KernelRadiusSquared(int size)
    {
        var engine = AiDotNetEngine.Current;
        int center = size / 2;
        var offsets = engine.TensorArange<int>(-center, size - center, 1);
        var squared = engine.TensorMultiply(offsets, offsets);
        int[] shape = [size, size];
        return engine.TensorAdd(
            engine.TensorBroadcastTo(engine.Reshape(squared, [size, 1]), shape),
            engine.TensorBroadcastTo(engine.Reshape(squared, [1, size]), shape));
    }

    /// <summary>
    /// Normalized Gaussian kernel: exp(-(dy^2 + dx^2) / (2 sigma^2)) divided by its sum.
    /// </summary>
    internal static Tensor<T> GaussianKernel(int size, double sigma)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var exponent = engine.TensorDivideScalar(KernelRadiusSquared(size).Cast<T>(), numOps.FromDouble(-2 * sigma * sigma));
        var weights = engine.TensorExp(exponent);
        return engine.TensorDivideScalar(weights, engine.TensorSum(weights));
    }

    /// <summary>
    /// Divides a kernel by its sum when the sum is positive.
    /// </summary>
    internal static Tensor<T> NormalizeKernel(Tensor<T> kernel)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = AiDotNetEngine.Current.TensorSum(kernel);
        return numOps.GreaterThan(sum, numOps.Zero) ? AiDotNetEngine.Current.TensorDivideScalar(kernel, sum) : kernel;
    }

    /// <summary>
    /// Octave weights 1 / 2^k for k = 0 .. count - 1, as exact powers of two in <typeparamref name="T"/>.
    /// </summary>
    internal static T[] OctaveWeights(int count)
    {
        // 2 * cumprod(0.5, ..., 0.5) = 1, 1/2, 1/4, ...: every step halves, so each weight is exact
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        var halves = new Tensor<T>([count]);
        engine.TensorFill(halves, numOps.FromDouble(0.5));
        return engine.TensorMultiplyScalar(engine.TensorCumProd(halves, 0), numOps.FromDouble(2.0)).ToArray();
    }

    /// <summary>
    /// Converts a fixed table of literal kernel coefficients to a [rows, columns] tensor of
    /// <typeparamref name="T"/> in one bulk conversion. Only for literal tables: anything computed
    /// from input data or random draws is combined in <typeparamref name="T"/> by the caller.
    /// </summary>
    internal static Tensor<T> KernelTensor(double[,] kernel)
    {
        int rows = kernel.GetLength(0);
        int cols = kernel.GetLength(1);
        var flat = new double[rows * cols];
        Buffer.BlockCopy(kernel, 0, flat, 0, flat.Length * sizeof(double));
        return new Tensor<double>(flat, [rows, cols]).Cast<T>();
    }

    /// <summary>
    /// Cross-correlates every channel of a [channels, height, width] tensor with a centred
    /// [kh, kw] kernel: out[c, y, x] = sum k[i, j] * in[c, y + i - kh / 2, x + j - kw / 2], with
    /// out-of-range taps resolved by <see cref="PadSpatial"/>.
    /// </summary>
    internal static Tensor<T> Correlate(Tensor<T> planar, Tensor<T> kernel, bool mirror)
    {
        var engine = AiDotNetEngine.Current;
        int channels = planar.Shape[0], height = planar.Shape[1], width = planar.Shape[2];
        int kh = kernel.Shape[0], kw = kernel.Shape[1];
        int ry = kh / 2, rx = kw / 2;
        var padded = PadSpatial(planar, ry, kh - 1 - ry, rx, kw - 1 - rx, mirror);
        var input = engine.Reshape(padded, [channels, 1, height + kh - 1, width + kw - 1]);
        var output = engine.Conv2D(input, engine.Reshape(kernel, [1, 1, kh, kw]), 1, 0, 1);
        return engine.Reshape(output, [channels, height, width]);
    }

    /// <summary>
    /// Builds a [channels, height, width] tensor whose leading channels are the given
    /// [height, width] planes and whose remaining channels are copied from <paramref name="planar"/>.
    /// </summary>
    internal static Tensor<T> ReplaceLeadingChannels(Tensor<T> planar, params Tensor<T>[] planes)
    {
        var engine = AiDotNetEngine.Current;
        var leading = engine.TensorStack(planes, 0);
        int channels = planar.Shape[0];
        if (channels <= planes.Length)
        {
            return leading;
        }

        var rest = engine.TensorSlice(planar, [planes.Length, 0, 0],
            [channels - planes.Length, planar.Shape[1], planar.Shape[2]]);
        return engine.TensorConcatenate([leading, rest], 0);
    }

    /// <summary>
    /// Writes a [channels, height, width] tensor into the first image, converting to the
    /// stored channel order. Other images in a batch are left unchanged.
    /// </summary>
    internal void SetFromPlanarTensor(Tensor<T> planar)
    {
        var engine = AiDotNetEngine.Current;
        bool channelsLast = ChannelOrder == ChannelOrder.HWC || ChannelOrder == ChannelOrder.BHWC;
        Tensor<T> image = channelsLast ? engine.TensorPermute(planar, [1, 2, 0]) : planar;
        image = image.Contiguous();
        if (_data.Rank == 4)
        {
            engine.TensorSetSliceAxis(_data, image, 0, 0);
        }
        else
        {
            image.CopyTo(_data.AsWritableSpan());
        }
    }

    /// <summary>
    /// Static version of index calculation for transposition.
    /// </summary>
    private static int CalculateIndexStatic(int batch, int y, int x, int channel, int height, int width, int channels, ChannelOrder order)
    {
        return order switch
        {
            ChannelOrder.CHW => channel * height * width + y * width + x,
            ChannelOrder.HWC => y * width * channels + x * channels + channel,
            ChannelOrder.BCHW => batch * channels * height * width + channel * height * width + y * width + x,
            ChannelOrder.BHWC => batch * height * width * channels + y * width * channels + x * channels + channel,
            _ => throw new InvalidOperationException($"Unknown channel order: {order}")
        };
    }
}
