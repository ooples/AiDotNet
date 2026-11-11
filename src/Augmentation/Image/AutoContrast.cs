namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Maximizes image contrast by stretching the intensity range to fill [0, max].
/// </summary>
/// <remarks>
/// <para>AutoContrast finds the minimum and maximum pixel values per channel and linearly
/// scales them to span the full range. This is equivalent to a per-channel min-max normalization.</para>
/// <para><b>For Beginners:</b> If your darkest pixel is 50 and brightest is 200, this stretches
/// them to 0 and 255, making the image use the full brightness range.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AutoContrast<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the percentage of lightest/darkest pixels to ignore when finding min/max.
    /// </summary>
    public double Cutoff { get; }

    public AutoContrast(double cutoff = 0, double probability = 0.5) : base(probability)
    {
        if (cutoff < 0 || cutoff >= 50)
            throw new ArgumentOutOfRangeException(nameof(cutoff), "Must be [0, 50).");
        Cutoff = cutoff;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        int h = data.Height, w = data.Width;
        int totalPixels = h * w;
        int cutoffPixels = (int)(totalPixels * Cutoff / 100.0);

        var engine = Engine;
        int channels = data.Channels;
        var image = data.ToPlanarTensor();

        // Per channel: sort the pixel values and take the cutoff percentiles
        var sorted = engine.TensorSort(engine.Reshape(image, [channels, totalPixels]), 1, false).Values;
        var lo = engine.Reshape(engine.TensorSlice(sorted, [0, cutoffPixels], [channels, 1]), [channels, 1, 1]);
        var hi = engine.Reshape(engine.TensorSlice(sorted, [0, totalPixels - 1 - cutoffPixels], [channels, 1]), [channels, 1, 1]);
        var spread = engine.TensorSubtract(hi, lo);

        // Map [lo, hi] to [0, maxVal]
        var maxTensor = new Tensor<T>([channels, 1, 1]);
        engine.TensorFill(maxTensor, NumOps.FromDouble(maxVal));
        var scale = engine.TensorDivide(maxTensor, spread);
        var mapped = engine.TensorClamp(
            engine.TensorMultiply(engine.TensorSubtract(image, lo), scale),
            NumOps.Zero, NumOps.FromDouble(maxVal));

        // Channels whose percentiles coincide are left unchanged
        var flat = engine.TensorBroadcastTo(
            engine.TensorLessThan(engine.TensorAbs(spread), NumOps.FromDouble(1e-10)), [channels, h, w]);
        result.SetFromPlanarTensor(engine.TensorWhere(flat, image, mapped));

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["cutoff"] = Cutoff; return p;
    }
}
