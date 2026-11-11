namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates MRI ghosting artifacts caused by motion during acquisition.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GhostingArtifact<T> : ImageAugmenterBase<T>
{
    public int NumGhosts { get; }
    public double MinIntensity { get; }
    public double MaxIntensity { get; }

    public GhostingArtifact(int numGhosts = 3, double minIntensity = 0.05,
        double maxIntensity = 0.3, double probability = 0.5) : base(probability)
    {
        NumGhosts = numGhosts; MinIntensity = minIntensity; MaxIntensity = maxIntensity;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double ghostIntensity = context.GetRandomDouble(MinIntensity, MaxIntensity);

        // Ghosts appear as shifted copies along the phase-encode direction
        bool horizontal = context.GetRandomBool();

        var engine = Engine;
        var original = data.ToPlanarTensor();
        var accumulated = original;
        T min = NumOps.Zero;
        T max = NumOps.FromDouble(maxVal);
        // Ghost g has weight ghostIntensity / g
        var ghostNumbers = engine.Reshape(engine.TensorArange<T>(1, NumGhosts + 1, 1), [NumGhosts, 1, 1, 1]);
        var weights = engine.TensorDivide(ImageTensor<T>.Constant(ghostNumbers, ghostIntensity), ghostNumbers);
        for (int g = 1; g <= NumGhosts; g++)
        {
            int shift = (int)(g * (horizontal ? data.Width : data.Height) / (NumGhosts + 1.0) * 0.3);

            // ghost[y, x] = data[(y + shift) % H, x] (or along x): a roll by -shift
            int axis = horizontal ? 2 : 1;
            int size = horizontal ? data.Width : data.Height;
            int roll = ((-shift % size) + size) % size;
            var ghost = engine.TensorRoll(original, [roll], [axis]);
            accumulated = engine.TensorClamp(
                engine.TensorAdd(accumulated, engine.TensorMultiply(ghost, engine.TensorSliceAxis(weights, 0, g - 1))), min, max);
        }

        result.SetFromPlanarTensor(accumulated);

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["num_ghosts"] = NumGhosts;
        p["min_intensity"] = MinIntensity; p["max_intensity"] = MaxIntensity;
        return p;
    }
}
