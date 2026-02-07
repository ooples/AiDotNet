using System;
using System.IO;
using System.Text;
using AiDotNet.LinearAlgebra;
#if NET6_0_OR_GREATER
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
#endif

namespace AiDotNet.Helpers;

/// <summary>
/// Helper class for loading and saving images as tensors.
/// </summary>
/// <remarks>
/// <para>
/// Supports common image formats without external dependencies:
/// - BMP: Windows Bitmap format (uncompressed)
/// - PPM/PGM: Portable Pixmap/Graymap (simple text or binary)
/// - RAW: Raw pixel data with specified dimensions
/// </para>
/// <para>
/// <b>For Beginners:</b> This class converts image files into tensors for neural networks.
/// Images are loaded as [channels, height, width] or [batch, channels, height, width] tensors.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor values.</typeparam>
public static class ImageHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Loads an image from a file path and returns it as a tensor.
    /// </summary>
    /// <param name="filePath">Path to the image file.</param>
    /// <param name="normalize">Whether to normalize pixel values to [0, 1] range.</param>
    /// <returns>Tensor with shape [1, channels, height, width].</returns>
    /// <exception cref="FileNotFoundException">If the file does not exist.</exception>
    /// <exception cref="NotSupportedException">If the image format is not supported.</exception>
    public static Tensor<T> LoadImage(string filePath, bool normalize = true)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Image file not found: {filePath}", filePath);
        }

        var extension = Path.GetExtension(filePath).ToLowerInvariant();
        return extension switch
        {
            ".bmp" => LoadBmp(filePath, normalize),
            ".ppm" => LoadPpm(filePath, normalize),
            ".pgm" => LoadPgm(filePath, normalize),
            ".raw" => throw new NotSupportedException("RAW format requires explicit dimensions. Use LoadRaw method."),
#if NET6_0_OR_GREATER
            _ => LoadImageWithImageSharp(filePath, normalize)
#else
            _ => throw new NotSupportedException($"Unsupported image format: {extension}. On .NET Framework only .bmp, .ppm, .pgm are supported. Use .NET 6+ for PNG/JPEG/GIF/TIFF support.")
#endif
        };
    }

    /// <summary>
    /// Loads a BMP (Windows Bitmap) image file.
    /// </summary>
    /// <param name="filePath">Path to the BMP file.</param>
    /// <param name="normalize">Whether to normalize to [0, 1].</param>
    /// <returns>Tensor with shape [1, 3, height, width] for RGB images.</returns>
    public static Tensor<T> LoadBmp(string filePath, bool normalize = true)
    {
        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);

        // BMP Header (14 bytes)
        var signature = reader.ReadUInt16();
        if (signature != 0x4D42) // "BM" in little-endian
        {
            throw new InvalidDataException("Invalid BMP file signature.");
        }

        reader.ReadUInt32(); // File size
        reader.ReadUInt32(); // Reserved
        var dataOffset = reader.ReadUInt32();

        // DIB Header
        var headerSize = reader.ReadUInt32();
        int width, height;
        ushort bitsPerPixel;
        uint compression = 0;

        if (headerSize >= 40) // BITMAPINFOHEADER or later
        {
            width = reader.ReadInt32();
            height = reader.ReadInt32();
            reader.ReadUInt16(); // Color planes
            bitsPerPixel = reader.ReadUInt16();
            compression = reader.ReadUInt32();
            // Skip rest of header
            stream.Seek(dataOffset, SeekOrigin.Begin);
        }
        else
        {
            throw new NotSupportedException($"Unsupported BMP header size: {headerSize}");
        }

        if (compression != 0)
        {
            throw new NotSupportedException("Compressed BMP files are not supported.");
        }

        // Handle bottom-up vs top-down
        bool bottomUp = height > 0;
        height = Math.Abs(height);

        int channels = bitsPerPixel / 8;
        if (channels < 3)
        {
            channels = 3; // Treat as RGB
        }

        // Row padding (rows must be multiple of 4 bytes)
        int rowSize = (width * (bitsPerPixel / 8) + 3) & ~3;

        var tensor = new Tensor<T>(new[] { 1, 3, height, width });
        var span = tensor.AsWritableSpan();
        var normFactor = normalize ? 255.0 : 1.0;

        var rowBuffer = new byte[rowSize];

        for (int y = 0; y < height; y++)
        {
            int actualY = bottomUp ? (height - 1 - y) : y;
            reader.Read(rowBuffer, 0, rowSize);

            for (int x = 0; x < width; x++)
            {
                int pixelOffset = x * (bitsPerPixel / 8);

                // BMP stores as BGR
                byte b = rowBuffer[pixelOffset];
                byte g = rowBuffer[pixelOffset + 1];
                byte r = rowBuffer[pixelOffset + 2];

                // Store as RGB in [C, H, W] format
                span[0 * height * width + actualY * width + x] = NumOps.FromDouble(r / normFactor);
                span[1 * height * width + actualY * width + x] = NumOps.FromDouble(g / normFactor);
                span[2 * height * width + actualY * width + x] = NumOps.FromDouble(b / normFactor);
            }
        }

        return tensor;
    }

    /// <summary>
    /// Loads a PPM (Portable Pixmap) image file.
    /// </summary>
    /// <param name="filePath">Path to the PPM file.</param>
    /// <param name="normalize">Whether to normalize to [0, 1].</param>
    /// <returns>Tensor with shape [1, 3, height, width].</returns>
    public static Tensor<T> LoadPpm(string filePath, bool normalize = true)
    {
        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);

        // Read magic number
        var magic = ReadPnmToken(reader);
        bool isBinary = magic == "P6";
        if (magic != "P3" && magic != "P6")
        {
            throw new InvalidDataException($"Invalid PPM magic number: {magic}. Expected P3 or P6.");
        }

        // Read dimensions
        int width = int.Parse(ReadPnmToken(reader));
        int height = int.Parse(ReadPnmToken(reader));
        int maxVal = int.Parse(ReadPnmToken(reader));

        var tensor = new Tensor<T>(new[] { 1, 3, height, width });
        var span = tensor.AsWritableSpan();
        var normFactor = normalize ? (double)maxVal : 1.0;

        if (isBinary)
        {
            // Binary PPM
            var bytesPerChannel = maxVal > 255 ? 2 : 1;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double r, g, b;
                    if (bytesPerChannel == 1)
                    {
                        r = reader.ReadByte();
                        g = reader.ReadByte();
                        b = reader.ReadByte();
                    }
                    else
                    {
                        r = (reader.ReadByte() << 8) | reader.ReadByte();
                        g = (reader.ReadByte() << 8) | reader.ReadByte();
                        b = (reader.ReadByte() << 8) | reader.ReadByte();
                    }

                    span[0 * height * width + y * width + x] = NumOps.FromDouble(r / normFactor);
                    span[1 * height * width + y * width + x] = NumOps.FromDouble(g / normFactor);
                    span[2 * height * width + y * width + x] = NumOps.FromDouble(b / normFactor);
                }
            }
        }
        else
        {
            // ASCII PPM
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double r = int.Parse(ReadPnmToken(reader));
                    double g = int.Parse(ReadPnmToken(reader));
                    double b = int.Parse(ReadPnmToken(reader));

                    span[0 * height * width + y * width + x] = NumOps.FromDouble(r / normFactor);
                    span[1 * height * width + y * width + x] = NumOps.FromDouble(g / normFactor);
                    span[2 * height * width + y * width + x] = NumOps.FromDouble(b / normFactor);
                }
            }
        }

        return tensor;
    }

    /// <summary>
    /// Loads a PGM (Portable Graymap) image file.
    /// </summary>
    /// <param name="filePath">Path to the PGM file.</param>
    /// <param name="normalize">Whether to normalize to [0, 1].</param>
    /// <returns>Tensor with shape [1, 1, height, width].</returns>
    public static Tensor<T> LoadPgm(string filePath, bool normalize = true)
    {
        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);

        var magic = ReadPnmToken(reader);
        bool isBinary = magic == "P5";
        if (magic != "P2" && magic != "P5")
        {
            throw new InvalidDataException($"Invalid PGM magic number: {magic}. Expected P2 or P5.");
        }

        int width = int.Parse(ReadPnmToken(reader));
        int height = int.Parse(ReadPnmToken(reader));
        int maxVal = int.Parse(ReadPnmToken(reader));

        var tensor = new Tensor<T>(new[] { 1, 1, height, width });
        var span = tensor.AsWritableSpan();
        var normFactor = normalize ? (double)maxVal : 1.0;

        if (isBinary)
        {
            var bytesPerPixel = maxVal > 255 ? 2 : 1;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double val;
                    if (bytesPerPixel == 1)
                    {
                        val = reader.ReadByte();
                    }
                    else
                    {
                        val = (reader.ReadByte() << 8) | reader.ReadByte();
                    }
                    span[y * width + x] = NumOps.FromDouble(val / normFactor);
                }
            }
        }
        else
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double val = int.Parse(ReadPnmToken(reader));
                    span[y * width + x] = NumOps.FromDouble(val / normFactor);
                }
            }
        }

        return tensor;
    }

    /// <summary>
    /// Loads raw pixel data from a file.
    /// </summary>
    /// <param name="filePath">Path to the raw data file.</param>
    /// <param name="width">Image width.</param>
    /// <param name="height">Image height.</param>
    /// <param name="channels">Number of channels (1 for grayscale, 3 for RGB).</param>
    /// <param name="bytesPerChannel">Bytes per channel (1 for 8-bit, 2 for 16-bit).</param>
    /// <param name="normalize">Whether to normalize values.</param>
    /// <returns>Tensor with shape [1, channels, height, width].</returns>
    public static Tensor<T> LoadRaw(string filePath, int width, int height, int channels = 3,
        int bytesPerChannel = 1, bool normalize = true)
    {
        var data = File.ReadAllBytes(filePath);
        var expectedSize = width * height * channels * bytesPerChannel;
        if (data.Length < expectedSize)
        {
            throw new InvalidDataException($"Raw file too small. Expected {expectedSize} bytes, got {data.Length}.");
        }

        var tensor = new Tensor<T>(new[] { 1, channels, height, width });
        var span = tensor.AsWritableSpan();
        var maxVal = bytesPerChannel == 1 ? 255.0 : 65535.0;
        var normFactor = normalize ? maxVal : 1.0;

        int idx = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    double val;
                    if (bytesPerChannel == 1)
                    {
                        val = data[idx++];
                    }
                    else
                    {
                        val = data[idx++] | (data[idx++] << 8);
                    }
                    span[c * height * width + y * width + x] = NumOps.FromDouble(val / normFactor);
                }
            }
        }

        return tensor;
    }

    /// <summary>
    /// Saves a tensor as a BMP image file.
    /// </summary>
    /// <param name="tensor">Tensor with shape [1, channels, height, width] or [channels, height, width].</param>
    /// <param name="filePath">Output file path.</param>
    /// <param name="denormalize">Whether to denormalize from [0, 1] to [0, 255].</param>
    public static void SaveBmp(Tensor<T> tensor, string filePath, bool denormalize = true)
    {
        var shape = tensor.Shape;
        int channels, height, width;

        if (shape.Length == 4)
        {
            channels = shape[1];
            height = shape[2];
            width = shape[3];
        }
        else if (shape.Length == 3)
        {
            channels = shape[0];
            height = shape[1];
            width = shape[2];
        }
        else
        {
            throw new ArgumentException("Tensor must have 3 or 4 dimensions.");
        }

        var span = tensor.AsSpan();
        var scale = denormalize ? 255.0 : 1.0;

        // BMP row padding
        int rowSize = (width * 3 + 3) & ~3;
        int dataSize = rowSize * height;
        int fileSize = 54 + dataSize;

        using var stream = File.Create(filePath);
        using var writer = new BinaryWriter(stream);

        // BMP Header
        writer.Write((ushort)0x4D42); // "BM"
        writer.Write(fileSize);
        writer.Write(0); // Reserved
        writer.Write(54); // Data offset

        // DIB Header (BITMAPINFOHEADER)
        writer.Write(40); // Header size
        writer.Write(width);
        writer.Write(-height); // Negative for top-down
        writer.Write((ushort)1); // Color planes
        writer.Write((ushort)24); // Bits per pixel
        writer.Write(0); // Compression
        writer.Write(dataSize);
        writer.Write(2835); // Horizontal resolution (72 DPI)
        writer.Write(2835); // Vertical resolution
        writer.Write(0); // Colors in palette
        writer.Write(0); // Important colors

        // Pixel data
        var rowBuffer = new byte[rowSize];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double r, g, b;
                if (channels == 1)
                {
                    r = g = b = NumOps.ToDouble(span[y * width + x]);
                }
                else
                {
                    r = NumOps.ToDouble(span[0 * height * width + y * width + x]);
                    g = NumOps.ToDouble(span[1 * height * width + y * width + x]);
                    b = NumOps.ToDouble(span[2 * height * width + y * width + x]);
                }

                // BMP is BGR
                rowBuffer[x * 3 + 0] = (byte)MathPolyfill.Clamp(b * scale, 0, 255);
                rowBuffer[x * 3 + 1] = (byte)MathPolyfill.Clamp(g * scale, 0, 255);
                rowBuffer[x * 3 + 2] = (byte)MathPolyfill.Clamp(r * scale, 0, 255);
            }
            writer.Write(rowBuffer, 0, rowSize);
        }
    }

    /// <summary>
    /// Saves a tensor as a PPM image file.
    /// </summary>
    /// <param name="tensor">Tensor with shape [1, 3, height, width] or [3, height, width].</param>
    /// <param name="filePath">Output file path.</param>
    /// <param name="denormalize">Whether to denormalize from [0, 1] to [0, 255].</param>
    /// <param name="binary">Whether to save as binary (P6) or ASCII (P3).</param>
    public static void SavePpm(Tensor<T> tensor, string filePath, bool denormalize = true, bool binary = true)
    {
        var shape = tensor.Shape;
        int height, width;

        if (shape.Length == 4)
        {
            height = shape[2];
            width = shape[3];
        }
        else if (shape.Length == 3)
        {
            height = shape[1];
            width = shape[2];
        }
        else
        {
            throw new ArgumentException("Tensor must have 3 or 4 dimensions.");
        }

        var span = tensor.AsSpan();
        var scale = denormalize ? 255.0 : 1.0;

        using var stream = File.Create(filePath);
        using var writer = new StreamWriter(stream, Encoding.ASCII);

        writer.WriteLine(binary ? "P6" : "P3");
        writer.WriteLine($"{width} {height}");
        writer.WriteLine("255");
        writer.Flush();

        if (binary)
        {
            using var bw = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double r = NumOps.ToDouble(span[0 * height * width + y * width + x]);
                    double g = NumOps.ToDouble(span[1 * height * width + y * width + x]);
                    double b = NumOps.ToDouble(span[2 * height * width + y * width + x]);

                    bw.Write((byte)MathPolyfill.Clamp(r * scale, 0, 255));
                    bw.Write((byte)MathPolyfill.Clamp(g * scale, 0, 255));
                    bw.Write((byte)MathPolyfill.Clamp(b * scale, 0, 255));
                }
            }
        }
        else
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double r = NumOps.ToDouble(span[0 * height * width + y * width + x]);
                    double g = NumOps.ToDouble(span[1 * height * width + y * width + x]);
                    double b = NumOps.ToDouble(span[2 * height * width + y * width + x]);

                    writer.Write($"{(int)MathPolyfill.Clamp(r * scale, 0, 255)} ");
                    writer.Write($"{(int)MathPolyfill.Clamp(g * scale, 0, 255)} ");
                    writer.Write($"{(int)MathPolyfill.Clamp(b * scale, 0, 255)} ");
                }
                writer.WriteLine();
            }
        }
    }

#if NET6_0_OR_GREATER
    /// <summary>
    /// Loads an image using SixLabors.ImageSharp for formats not natively supported (PNG, JPEG, GIF, TIFF, WebP, etc.).
    /// </summary>
    /// <param name="filePath">Path to the image file.</param>
    /// <param name="normalize">Whether to normalize pixel values to [0, 1] range.</param>
    /// <returns>Tensor with shape [1, channels, height, width].</returns>
    private static Tensor<T> LoadImageWithImageSharp(string filePath, bool normalize)
    {
        using var image = Image.Load<Rgba32>(filePath);
        int width = image.Width;
        int height = image.Height;

        var tensor = new Tensor<T>(new[] { 1, 3, height, width });
        var span = tensor.AsWritableSpan();
        var normFactor = normalize ? 255.0 : 1.0;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var pixel = image[x, y];
                span[0 * height * width + y * width + x] = NumOps.FromDouble(pixel.R / normFactor);
                span[1 * height * width + y * width + x] = NumOps.FromDouble(pixel.G / normFactor);
                span[2 * height * width + y * width + x] = NumOps.FromDouble(pixel.B / normFactor);
            }
        }

        return tensor;
    }
#endif

    /// <summary>
    /// Reads a token from a PNM file (skipping comments and whitespace).
    /// </summary>
    private static string ReadPnmToken(BinaryReader reader)
    {
        var sb = new StringBuilder();
        bool inComment = false;

        while (true)
        {
            int b = reader.Read();
            if (b < 0) break;

            char c = (char)b;

            if (c == '#')
            {
                inComment = true;
                continue;
            }

            if (inComment)
            {
                if (c == '\n' || c == '\r')
                {
                    inComment = false;
                }
                continue;
            }

            if (char.IsWhiteSpace(c))
            {
                if (sb.Length > 0)
                {
                    break;
                }
                continue;
            }

            sb.Append(c);
        }

        return sb.ToString();
    }
}
