namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Plasma fractal transformation - applies a procedural plasma-like color perturbation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PlasmaTransform<T> : ImageAugmenterBase<T>
{
    public double MinIntensity { get; }
    public double MaxIntensity { get; }

    public PlasmaTransform(double minIntensity = 0.05, double maxIntensity = 0.2,
        double probability = 0.5) : base(probability)
    {
        MinIntensity = minIntensity; MaxIntensity = maxIntensity;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double intensity = context.GetRandomDouble(MinIntensity, MaxIntensity);

        var engine = Engine;
        var image = data.ToPlanarTensor();
        var planes = new Tensor<T>[data.Channels];
        T zero = NumOps.Zero;
        T max = NumOps.FromDouble(maxVal);
        T scale = NumOps.FromDouble(intensity * maxVal);

        // Generate plasma pattern using diamond-square algorithm (multi-octave noise per channel)
        T[] octaveWeights = ImageTensor<T>.OctaveWeights(4);
        for (int c = 0; c < data.Channels; c++)
        {
            // Octave k: cells of max(2, 4 * 2^k) with smoothstep interpolation, weight 1 / 2^k
            Tensor<T>? plasma = null;
            for (int octave = 0; octave < 4; octave++)
            {
                int freq = Math.Max(2, 4 << octave);
                var layer = engine.TensorMultiplyScalar(
                    ImageTensor<T>.GridNoise(context, data.Height, data.Width, freq, -1, 1, smoothstep: true),
                    octaveWeights[octave]);
                plasma = plasma is null ? layer : engine.TensorAdd(plasma, layer);
            }

            // Apply plasma perturbation
            var perturbation = engine.TensorMultiplyScalar(plasma ?? new Tensor<T>([data.Height, data.Width]), scale);
            planes[c] = engine.TensorClamp(engine.TensorAdd(engine.TensorSliceAxis(image, 0, c), perturbation), zero, max);
        }

        result.SetFromPlanarTensor(engine.TensorStack(planes, 0));

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_intensity"] = MinIntensity; p["max_intensity"] = MaxIntensity;
        return p;
    }
}
