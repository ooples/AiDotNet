namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Independently shifts each RGB channel by a random amount.
/// </summary>
/// <remarks>
/// <para>RgbShift adds different random offsets to the R, G, and B channels independently,
/// creating subtle color cast effects. This simulates white balance variations and color
/// calibration differences between cameras.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RgbShift<T> : ImageAugmenterBase<T>
{
    public double RedShiftLimit { get; }
    public double GreenShiftLimit { get; }
    public double BlueShiftLimit { get; }

    public RgbShift(double redShiftLimit = 0.05, double greenShiftLimit = 0.05,
        double blueShiftLimit = 0.05, double probability = 0.5) : base(probability)
    {
        RedShiftLimit = redShiftLimit;
        GreenShiftLimit = greenShiftLimit;
        BlueShiftLimit = blueShiftLimit;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels < 3) return data.Clone();

        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double rShift = context.GetRandomDouble(-RedShiftLimit, RedShiftLimit) * maxVal;
        double gShift = context.GetRandomDouble(-GreenShiftLimit, GreenShiftLimit) * maxVal;
        double bShift = context.GetRandomDouble(-BlueShiftLimit, BlueShiftLimit) * maxVal;
        var shifts = new Tensor<T>([NumOps.FromDouble(rShift), NumOps.FromDouble(gShift), NumOps.FromDouble(bShift)], [3, 1, 1]);

        var result = data.Clone();

        var engine = Engine;
        var image = data.ToPlanarTensor();
        var rgb = engine.TensorSlice(image, [0, 0, 0], [3, data.Height, data.Width]);
        var shifted = engine.TensorClamp(
            engine.TensorAdd(rgb, shifts),
            NumOps.Zero, NumOps.FromDouble(maxVal));
        result.SetFromPlanarTensor(ImageTensor<T>.ReplaceLeadingChannels(image,
            engine.TensorSliceAxis(shifted, 0, 0), engine.TensorSliceAxis(shifted, 0, 1), engine.TensorSliceAxis(shifted, 0, 2)));

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["r_shift_limit"] = RedShiftLimit;
        p["g_shift_limit"] = GreenShiftLimit;
        p["b_shift_limit"] = BlueShiftLimit;
        return p;
    }
}
