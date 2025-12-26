using AiDotNet.Tensors.LinearAlgebra;

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
        _data = data ?? throw new ArgumentNullException(nameof(data));
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
        // Create a copy of the tensor data
        var clonedData = new Tensor<T>((int[])_data.Shape.Clone());
        for (int i = 0; i < _data.Length; i++)
        {
            clonedData[i] = _data[i];
        }

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
        return (int[])_data.Shape.Clone();
    }

    /// <summary>
    /// Parses dimensions from the tensor based on channel order.
    /// </summary>
    private void ParseDimensions()
    {
        var dims = _data.Shape;

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
        var sourceDims = source.Shape;
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

        // Copy with transposition
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int sourceIdx = CalculateIndexStatic(b, y, x, c, height, width, channels, sourceOrder);
                        int targetIdx = CalculateIndexStatic(b, y, x, c, height, width, channels, targetOrder);
                        target[targetIdx] = source[sourceIdx];
                    }
                }
            }
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
