namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Randomly adjusts hue, saturation, and value in HSV space.
/// </summary>
/// <remarks>
/// <para>Converts the image to HSV, applies random shifts to each component, then converts
/// back to RGB. This provides intuitive color augmentation by directly manipulating color
/// properties.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class HueSaturationValue<T> : ImageAugmenterBase<T>
{
    public double HueShiftLimit { get; }
    public double SatShiftLimit { get; }
    public double ValShiftLimit { get; }

    public HueSaturationValue(double hueShiftLimit = 0.05, double satShiftLimit = 0.1,
        double valShiftLimit = 0.1, double probability = 0.5) : base(probability)
    {
        HueShiftLimit = hueShiftLimit;
        SatShiftLimit = satShiftLimit;
        ValShiftLimit = valShiftLimit;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels < 3) return data.Clone();

        double hShift = context.GetRandomDouble(-HueShiftLimit, HueShiftLimit);
        double sShift = context.GetRandomDouble(-SatShiftLimit, SatShiftLimit);
        double vShift = context.GetRandomDouble(-ValShiftLimit, ValShiftLimit);

        var result = data.Clone();

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
            {
                double r = NumOps.ToDouble(data.GetPixel(y, x, 0));
                double g = NumOps.ToDouble(data.GetPixel(y, x, 1));
                double b = NumOps.ToDouble(data.GetPixel(y, x, 2));
                double maxVal = data.IsNormalized ? 1.0 : 255.0;

                r /= maxVal; g /= maxVal; b /= maxVal;

                // RGB to HSV
                double max = Math.Max(r, Math.Max(g, b));
                double min = Math.Min(r, Math.Min(g, b));
                double delta = max - min;

                double h = 0;
                if (delta > 1e-10)
                {
                    if (Math.Abs(max - r) < 1e-10) h = ((g - b) / delta) % 6.0;
                    else if (Math.Abs(max - g) < 1e-10) h = (b - r) / delta + 2.0;
                    else h = (r - g) / delta + 4.0;
                    h /= 6.0;
                    if (h < 0) h += 1.0;
                }
                double s = max < 1e-10 ? 0 : delta / max;
                double v = max;

                // Apply shifts
                h = (h + hShift) % 1.0;
                if (h < 0) h += 1.0;
                s = Math.Max(0, Math.Min(1, s + sShift));
                v = Math.Max(0, Math.Min(1, v + vShift));

                // HSV to RGB
                var (rr, gg, bb) = RgbToHsv<T>.HsvToRgb(h, s, v);
                result.SetPixel(y, x, 0, NumOps.FromDouble(rr * maxVal));
                result.SetPixel(y, x, 1, NumOps.FromDouble(gg * maxVal));
                result.SetPixel(y, x, 2, NumOps.FromDouble(bb * maxVal));
            }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["hue_shift_limit"] = HueShiftLimit;
        p["sat_shift_limit"] = SatShiftLimit;
        p["val_shift_limit"] = ValShiftLimit;
        return p;
    }
}
