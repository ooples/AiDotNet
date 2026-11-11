namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Adds random shadow regions to the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Shadow<T> : ImageAugmenterBase<T>
{
    public int MinShadows { get; }
    public int MaxShadows { get; }
    public double MinDarkness { get; }
    public double MaxDarkness { get; }

    public Shadow(int minShadows = 1, int maxShadows = 3,
        double minDarkness = 0.3, double maxDarkness = 0.7,
        double probability = 0.5) : base(probability)
    {
        if (minShadows < 0) throw new ArgumentOutOfRangeException(nameof(minShadows));
        if (maxShadows < minShadows) throw new ArgumentOutOfRangeException(nameof(maxShadows), "maxShadows must be >= minShadows.");
        if (minDarkness < 0 || minDarkness > 1) throw new ArgumentOutOfRangeException(nameof(minDarkness));
        if (maxDarkness < minDarkness || maxDarkness > 1) throw new ArgumentOutOfRangeException(nameof(maxDarkness), "maxDarkness must be >= minDarkness and <= 1.");
        MinShadows = minShadows; MaxShadows = maxShadows;
        MinDarkness = minDarkness; MaxDarkness = maxDarkness;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        int numShadows = context.GetRandomInt(MinShadows, MaxShadows + 1);

        var engine = Engine;
        var image = data.ToPlanarTensor();
        int height = data.Height, width = data.Width;
        if (numShadows > 0)
        {
            // Per shadow, in draw order: darkness, edge point (x1, y1) and edge-normal angle
            var draws = AugmentationRandom<T>.Unit(context, [numShadows, 4]);
            var darkness = Parameter(engine, draws, 0, MinDarkness, MaxDarkness, numShadows);
            var x1 = Parameter(engine, draws, 1, 0, width, numShadows);
            var y1 = Parameter(engine, draws, 2, 0, height, numShadows);
            var angle = Parameter(engine, draws, 3, 0, Math.PI * 2, numShadows);
            int[] rowShape = [numShadows, height, 1];
            int[] colShape = [numShadows, 1, width];
            int[] shape = [numShadows, height, width];

            // Signed distance from each shadow edge: 0 below -20, 1 above 20, linear in between
            var xs = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, width, 1), [1, 1, width]), colShape);
            var ys = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, height, 1), [1, height, 1]), rowShape);
            var alongX = engine.TensorMultiply(
                engine.TensorSubtract(xs, engine.TensorBroadcastTo(x1, colShape)),
                engine.TensorBroadcastTo(engine.TensorCos(angle), colShape));
            var alongY = engine.TensorMultiply(
                engine.TensorSubtract(ys, engine.TensorBroadcastTo(y1, rowShape)),
                engine.TensorBroadcastTo(engine.TensorSin(angle), rowShape));
            var dist = engine.TensorAdd(engine.TensorBroadcastTo(alongX, shape), engine.TensorBroadcastTo(alongY, shape));
            var shadowFactor = engine.TensorClamp(
                engine.TensorDivideScalar(engine.TensorAddScalar(dist, NumOps.FromDouble(20.0)), NumOps.FromDouble(40.0)),
                NumOps.Zero, NumOps.One);
            var multipliers = engine.ScalarMinusTensor(NumOps.One,
                engine.TensorMultiply(shadowFactor, engine.TensorBroadcastTo(darkness, shape)));

            // Shadows darken the image one after another (clamped after each)
            T zero = NumOps.Zero;
            T max = NumOps.FromDouble(maxVal);
            for (int s = 0; s < numShadows; s++)
            {
                var scale = engine.Reshape(engine.TensorSliceAxis(multipliers, 0, s), [1, height, width]);
                image = engine.TensorClamp(engine.TensorMultiply(image, scale), zero, max);
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

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_shadows"] = MinShadows; p["max_shadows"] = MaxShadows;
        p["min_darkness"] = MinDarkness; p["max_darkness"] = MaxDarkness;
        return p;
    }
}
