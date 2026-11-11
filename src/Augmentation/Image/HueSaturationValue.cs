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

        var engine = Engine;
        var image = data.ToPlanarTensor();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        T maxT = NumOps.FromDouble(maxVal);
        var r = engine.TensorDivideScalar(engine.TensorSliceAxis(image, 0, 0), maxT);
        var g = engine.TensorDivideScalar(engine.TensorSliceAxis(image, 0, 1), maxT);
        var b = engine.TensorDivideScalar(engine.TensorSliceAxis(image, 0, 2), maxT);

        // RGB to HSV
        var max = engine.TensorMax(r, engine.TensorMax(g, b));
        var min = engine.TensorMin(r, engine.TensorMin(g, b));
        var delta = engine.TensorSubtract(max, min);
        var h = ImageTensor<T>.HexconeHue(r, g, b, max, delta);
        var black = engine.TensorLessThan(max, NumOps.FromDouble(1e-10));
        var s = engine.TensorWhere(black, ImageTensor<T>.Constant(max, 0.0), engine.TensorDivide(delta, max));

        // Apply shifts
        h = ImageTensor<T>.WrapUnitInterval(engine.TensorAddScalar(h, NumOps.FromDouble(hShift)));
        s = engine.TensorClamp(engine.TensorAddScalar(s, NumOps.FromDouble(sShift)), NumOps.Zero, NumOps.One);
        var v = engine.TensorClamp(engine.TensorAddScalar(max, NumOps.FromDouble(vShift)), NumOps.Zero, NumOps.One);

        // HSV to RGB
        var (rr, gg, bb) = ImageTensor<T>.HsvToRgbPlanes(h, s, v);
        result.SetFromPlanarTensor(ImageTensor<T>.ReplaceLeadingChannels(image,
            engine.TensorMultiplyScalar(rr, maxT), engine.TensorMultiplyScalar(gg, maxT), engine.TensorMultiplyScalar(bb, maxT)));

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
