namespace AiDotNet.ComputerVision.Segmentation.Salient;

/// <summary>
/// rembg's <c>u2net</c> session (<c>rembg/sessions/u2net.py</c>): Lanczos resize to 320 x 320, scale by the image
/// maximum, ImageNet normalisation, U²-Net, min-max normalisation of the fused map <c>d0</c>, 8-bit quantisation and a
/// Lanczos resize back to the image size, which is the alpha of rembg's (naive) cut-out.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// The network starts at PyTorch's default initialisation; load the released <c>u2net.pth</c> with
/// <see cref="U2Net{T}.LoadTorchWeights"/> for rembg's masks. Images are quantised to 8 bits after each resize, as
/// Pillow does for the 8-bit images rembg resizes.
/// </remarks>
public sealed class U2NetForegroundSegmenter<T> : IForegroundSegmenter<T>
{
    private static readonly double[] Mean = { 0.485, 0.456, 0.406 };
    private static readonly double[] Std = { 0.229, 0.224, 0.225 };

    /// <summary>Creates the segmenter.</summary>
    /// <param name="network">The network; null creates a full U²-Net.</param>
    /// <param name="inputSize">The network resolution. rembg: 320.</param>
    /// <param name="seed">Initialisation seed of a created network.</param>
    public U2NetForegroundSegmenter(U2Net<T>? network = null, int inputSize = 320, int? seed = null)
    {
        if (inputSize < 1) throw new ArgumentOutOfRangeException(nameof(inputSize), "inputSize must be positive.");
        Network = network ?? new U2Net<T>(U2NetConfiguration.Full(), seed);
        if (Network.Configuration.InputChannels != 3 || Network.Configuration.OutputChannels != 1)
            throw new ArgumentException("Foreground segmentation needs a 3-channel input and a 1-channel output.", nameof(network));
        InputSize = inputSize;
    }

    /// <summary>The U²-Net.</summary>
    public U2Net<T> Network { get; }

    /// <summary>The network resolution.</summary>
    public int InputSize { get; }

    /// <inheritdoc />
    public Tensor<T> Segment(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (image.Rank != 4 || image.Shape[0] != 1 || image.Shape[1] != 3)
            throw new ArgumentException("The image must be [1, 3, H, W].", nameof(image));
        int height = image.Shape[2], width = image.Shape[3], size = InputSize;

        var rgb = image.Cast<double>().ToArray();
        var resized = Quantize(LanczosResampler.Resize(rgb, 3, height, width, size, size, 0.0, 1.0));
        double max = 1e-6;
        foreach (double v in resized) max = Math.Max(max, v);
        int pixels = size * size;
        var input = new double[3 * pixels];
        for (int c = 0; c < 3; c++)
        {
            for (int p = 0; p < pixels; p++)
                input[(c * pixels) + p] = ((resized[(c * pixels) + p] / max) - Mean[c]) / Std[c];
        }

        var outputs = Network.Forward(new Tensor<double>(new[] { 1, 3, size, size }, new Vector<double>(input)).Cast<T>());
        var prediction = outputs[0].Cast<double>().ToArray();
        double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
        foreach (double v in prediction)
        {
            lo = Math.Min(lo, v);
            hi = Math.Max(hi, v);
        }

        // (pred - mi) / (ma - mi), then Image.fromarray((pred * 255).astype("uint8")).
        var mask = new double[pixels];
        double range = hi - lo;
        for (int p = 0; p < pixels; p++)
            mask[p] = range > 0.0 ? Math.Floor(((prediction[p] - lo) / range) * 255.0) / 255.0 : 0.0;

        var alpha = Quantize(LanczosResampler.Resize(mask, 1, size, size, height, width, 0.0, 1.0));
        return new Tensor<double>(new[] { height, width }, new Vector<double>(alpha)).Cast<T>();
    }

    /// <summary>Rounds to the 8-bit grid, as Pillow stores a resampled 8-bit image.</summary>
    private static double[] Quantize(double[] values)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = Math.Round(values[i] * 255.0, MidpointRounding.AwayFromZero) / 255.0;
        return values;
    }
}
