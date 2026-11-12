namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates MRI spike (Herringbone) artifacts caused by electrical interference in k-space.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SpikeArtifact<T> : ImageAugmenterBase<T>
{
    public int MinSpikes { get; }
    public int MaxSpikes { get; }
    public double MinIntensity { get; }
    public double MaxIntensity { get; }

    public SpikeArtifact(int minSpikes = 1, int maxSpikes = 3,
        double minIntensity = 0.1, double maxIntensity = 0.5,
        double probability = 0.5) : base(probability)
    {
        MinSpikes = minSpikes; MaxSpikes = maxSpikes;
        MinIntensity = minIntensity; MaxIntensity = maxIntensity;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        int numSpikes = context.GetRandomInt(MinSpikes, MaxSpikes + 1);

        var engine = Engine;
        var image = data.ToPlanarTensor();
        int height = data.Height, width = data.Width;
        if (numSpikes > 0)
        {
            // Per spike, in draw order: intensity, x/y frequencies (cycles * 2 pi) and phase
            var draws = AugmentationRandom<T>.Unit(context, [numSpikes, 4]);
            T twoPi = NumOps.FromDouble(Math.PI * 2);
            var intensity = Parameter(engine, draws, 0, MinIntensity, MaxIntensity, numSpikes);
            var freqX = engine.TensorMultiplyScalar(Parameter(engine, draws, 1, 0.1, 0.5, numSpikes), twoPi);
            var freqY = engine.TensorMultiplyScalar(Parameter(engine, draws, 2, 0.1, 0.5, numSpikes), twoPi);
            var phase = Parameter(engine, draws, 3, 0, Math.PI * 2, numSpikes);
            int[] rowShape = [numSpikes, height, 1];
            int[] colShape = [numSpikes, 1, width];
            int[] shape = [numSpikes, height, width];

            // Herringbone pattern: sin(x * freqX + y * freqY + phase) * intensity * maxVal
            var xs = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, width, 1), [1, 1, width]), colShape);
            var ys = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, height, 1), [1, height, 1]), rowShape);
            var alongX = engine.TensorMultiply(xs, engine.TensorBroadcastTo(freqX, colShape));
            var alongY = engine.TensorMultiply(ys, engine.TensorBroadcastTo(freqY, rowShape));
            var phaseMap = engine.TensorAdd(
                engine.TensorAdd(engine.TensorBroadcastTo(alongX, shape), engine.TensorBroadcastTo(alongY, shape)),
                engine.TensorBroadcastTo(phase, shape));
            var patterns = engine.TensorMultiplyScalar(
                engine.TensorMultiply(engine.TensorSin(phaseMap), engine.TensorBroadcastTo(intensity, shape)), NumOps.FromDouble(maxVal));

            // Spikes add one after another (clamped after each)
            T zero = NumOps.Zero;
            T max = NumOps.FromDouble(maxVal);
            for (int s = 0; s < numSpikes; s++)
            {
                var pattern3 = engine.Reshape(engine.TensorSliceAxis(patterns, 0, s), [1, height, width]);
                image = engine.TensorClamp(engine.TensorAdd(image, pattern3), zero, max);
            }
        }

        result.SetFromPlanarTensor(image);

        return result;
    }

    /// <summary>
    /// Column <paramref name="column"/> of the [count, k] unit draws mapped to [min, max), shaped [count, 1, 1].
    /// </summary>
    private static Tensor<T> Parameter(IEngine engine, Tensor<T> draws, int column, double min, double max, int count)
    {
        var unit = engine.Reshape(engine.TensorSliceAxis(draws, 1, column).Contiguous(), [count, 1, 1]);
        return engine.TensorAddScalar(engine.TensorMultiplyScalar(unit, NumOps.FromDouble(max - min)), NumOps.FromDouble(min));
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_spikes"] = MinSpikes; p["max_spikes"] = MaxSpikes;
        p["min_intensity"] = MinIntensity; p["max_intensity"] = MaxIntensity;
        return p;
    }
}
