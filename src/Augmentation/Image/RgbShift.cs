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

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels < 3) return data.Clone();

        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double rShift = context.GetRandomDouble(-RedShiftLimit, RedShiftLimit) * maxVal;
        double gShift = context.GetRandomDouble(-GreenShiftLimit, GreenShiftLimit) * maxVal;
        double bShift = context.GetRandomDouble(-BlueShiftLimit, BlueShiftLimit) * maxVal;
        double[] shifts = [rShift, gShift, bShift];

        var result = data.Clone();

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < 3; c++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c)) + shifts[c];
                    result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, val))));
                }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["r_shift_limit"] = RedShiftLimit;
        p["g_shift_limit"] = GreenShiftLimit;
        p["b_shift_limit"] = BlueShiftLimit;
        return p;
    }
}
