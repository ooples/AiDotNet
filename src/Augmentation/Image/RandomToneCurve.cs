namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies a random tone curve transformation to adjust image tonality.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RandomToneCurve<T> : ImageAugmenterBase<T>
{
    public double Scale { get; }
    public int NumControlPoints { get; }

    public RandomToneCurve(double scale = 0.1, int numControlPoints = 4,
        double probability = 0.5) : base(probability)
    {
        if (numControlPoints < 0) throw new ArgumentOutOfRangeException(nameof(numControlPoints), "Must be non-negative.");
        Scale = scale; NumControlPoints = numControlPoints;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        // Random control points for the tone curve: x_i = i / (n + 1) for i = 0 .. n + 1, and
        // y_i = clamp(x_i + U(-Scale, Scale), 0, 1) for the inner points (y = 0 and 1 at the ends)
        var engine = Engine;
        int points = NumControlPoints + 2;
        var knotsX = engine.TensorDivideScalar(
            engine.TensorArange<T>(0, points, 1), NumOps.FromDouble(NumControlPoints + 1));
        var innerY = engine.TensorClamp(
            engine.TensorAdd(
                engine.TensorSlice(knotsX, [1], [NumControlPoints]),
                AugmentationRandom<T>.Uniform(context, [NumControlPoints], -Scale, Scale)),
            NumOps.Zero, NumOps.One);
        var knotsY = engine.TensorConcatenate(
            [new Tensor<T>([1]), innerY, ImageTensor<T>.Constant(new Tensor<T>([1]), 1.0)], 0);

        // Piecewise-linear lookup between control points: segment = first control point at or above
        // the normalized value (the last segment when none is)
        var upperX = engine.TensorSlice(knotsX, [1], [points - 1]);

        var image = data.ToPlanarTensor();
        var normalized = engine.TensorDivideScalar(image, NumOps.FromDouble(maxVal));
        var seg = engine.TensorClampMax(engine.TensorSearchSorted(upperX, normalized, false), points - 2);
        var next = engine.TensorAddScalar(seg, 1);

        var x0 = engine.TensorTake(knotsX, seg);
        var x1 = engine.TensorTake(knotsX, next);
        var y0 = engine.TensorTake(knotsY, seg);
        var y1 = engine.TensorTake(knotsY, next);

        var width = engine.TensorSubtract(x1, x0);
        var t = engine.TensorWhere(
            engine.TensorGreaterThan(width, NumOps.FromDouble(1e-10)),
            engine.TensorDivide(engine.TensorSubtract(normalized, x0), width),
            ImageTensor<T>.Constant(width, 0.0));
        var mapped = engine.TensorAdd(y0, engine.TensorMultiply(t, engine.TensorSubtract(y1, y0)));

        result.SetFromPlanarTensor(engine.TensorClamp(
            engine.TensorMultiplyScalar(mapped, NumOps.FromDouble(maxVal)), NumOps.Zero, NumOps.FromDouble(maxVal)));

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["scale"] = Scale; p["num_control_points"] = NumControlPoints;
        return p;
    }
}
