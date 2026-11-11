namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates rain streaks on the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Rain<T> : ImageAugmenterBase<T>
{
    public double MinSlant { get; }
    public double MaxSlant { get; }
    public double DropLength { get; }
    public double DropWidth { get; }
    public double Intensity { get; }
    public double Brightness { get; }

    public Rain(double minSlant = -10, double maxSlant = 10,
        double dropLength = 0.05, double dropWidth = 1,
        double intensity = 0.3, double brightness = 0.7,
        double probability = 0.5) : base(probability)
    {
        MinSlant = minSlant; MaxSlant = maxSlant;
        DropLength = dropLength; DropWidth = dropWidth;
        Intensity = intensity; Brightness = brightness;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double slant = context.GetRandomDouble(MinSlant, MaxSlant);
        double slantRad = slant * Math.PI / 180.0;
        int numDrops = (int)(Intensity * data.Height * data.Width / 100.0);
        int dropLen = Math.Max(3, (int)(DropLength * data.Height));

        // Mark the drop pixels (RNG draws in the original order); every drop pixel brightens the
        // original value, so overlapping drops are idempotent
        var dropMask = new int[data.Height * data.Width];
        for (int d = 0; d < numDrops; d++)
        {
            int startY = context.GetRandomInt(0, data.Height);
            int startX = context.GetRandomInt(0, data.Width);

            int halfWidth = Math.Max(0, (int)((DropWidth - 1) / 2));

            for (int i = 0; i < dropLen; i++)
            {
                int y = startY + i;
                int xCenter = startX + (int)(i * Math.Tan(slantRad));

                for (int dx = -halfWidth; dx <= halfWidth; dx++)
                {
                    int x = xCenter + dx;
                    if (y < 0 || y >= data.Height || x < 0 || x >= data.Width) continue;
                    dropMask[y * data.Width + x] = 1;
                }
            }
        }

        var engine = Engine;
        var image = data.ToPlanarTensor();
        var drops = engine.TensorClamp(
            engine.TensorAddScalar(image, NumOps.FromDouble(Brightness * maxVal * 0.3)),
            NumOps.Zero, NumOps.FromDouble(maxVal));
        var mask = engine.TensorBroadcastTo(
            new Tensor<int>(dropMask, [1, data.Height, data.Width]).Cast<T>(), image._shape);
        var rained = engine.TensorWhere(mask, drops, image);

        // Slight overall brightness reduction for overcast
        var overcast = engine.TensorMax(engine.TensorMultiplyScalar(rained, NumOps.FromDouble(0.95)), NumOps.Zero);
        result.SetFromPlanarTensor(overcast);

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_slant"] = MinSlant; p["max_slant"] = MaxSlant;
        p["drop_length"] = DropLength; p["drop_width"] = DropWidth;
        p["intensity"] = Intensity; p["brightness"] = Brightness;
        return p;
    }
}
