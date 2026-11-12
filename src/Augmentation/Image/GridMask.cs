namespace AiDotNet.Augmentation.Image;

/// <summary>
/// GridMask augmentation (Chen et al. 2020) that masks grid-shaped regions.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GridMask<T> : ImageAugmenterBase<T>
{
    public double MinRatio { get; }
    public double MaxRatio { get; }
    public int GridDMin { get; }
    public int GridDMax { get; }
    public bool Rotate { get; }
    public double FillValue { get; }

    public GridMask(double minRatio = 0.3, double maxRatio = 0.7,
        int gridDMin = 64, int gridDMax = 128, bool rotate = true,
        double fillValue = 0.0, double probability = 0.5) : base(probability)
    {
        MinRatio = minRatio; MaxRatio = maxRatio;
        GridDMin = gridDMin; GridDMax = gridDMax;
        Rotate = rotate; FillValue = fillValue;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        int d = context.GetRandomInt(GridDMin, GridDMax + 1);
        double ratio = context.GetRandomDouble(MinRatio, MaxRatio);
        int maskLen = (int)(d * ratio);

        double angle = Rotate ? context.GetRandomDouble(0, Math.PI * 2) : 0;
        double cosA = Math.Cos(angle);
        double sinA = Math.Sin(angle);

        double centerY = data.Height / 2.0;
        double centerX = data.Width / 2.0;

        int offsetY = context.GetRandomInt(0, d);
        int offsetX = context.GetRandomInt(0, d);

        // Rotate every pixel coordinate around the center and keep the grid cell offsets
        var engine = Engine;
        T cos = NumOps.FromDouble(cosA);
        T sin = NumOps.FromDouble(sinA);
        T cy = NumOps.FromDouble(centerY);
        T cx = NumOps.FromDouble(centerX);
        var dy = engine.Reshape(engine.TensorSubtractScalar(engine.TensorArange<T>(0, data.Height, 1), cy), [data.Height, 1]);
        var dx = engine.Reshape(engine.TensorSubtractScalar(engine.TensorArange<T>(0, data.Width, 1), cx), [1, data.Width]);
        var ry = engine.TensorAddScalar(engine.TensorSubtract(engine.TensorMultiplyScalar(dy, cos), engine.TensorMultiplyScalar(dx, sin)), cy);
        var rx = engine.TensorAddScalar(engine.TensorAdd(engine.TensorMultiplyScalar(dy, sin), engine.TensorMultiplyScalar(dx, cos)), cx);

        // ((v + offset) % d + d) % d, compared against maskLen (truncation keeps v < maskLen equivalent)
        var gridY = PositiveRemainder(engine, engine.TensorAddScalar(ry, NumOps.FromDouble(offsetY)), d);
        var gridX = PositiveRemainder(engine, engine.TensorAddScalar(rx, NumOps.FromDouble(offsetX)), d);
        T limit = NumOps.FromDouble(maskLen);
        var masked = engine.TensorMultiply(engine.TensorLessThan(gridY, limit), engine.TensorLessThan(gridX, limit));

        var image = data.ToPlanarTensor();
        var mask = engine.TensorBroadcastTo(engine.Reshape(masked, [1, data.Height, data.Width]), image._shape);
        var filled = ImageTensor<T>.Constant(image, FillValue);
        result.SetFromPlanarTensor(engine.TensorWhere(mask, filled, image));

        return result;
    }

    private static Tensor<T> PositiveRemainder(IEngine engine, Tensor<T> values, int d)
    {
        var divisor = ImageTensor<T>.Constant(values, d);
        var inner = engine.TensorAddScalar(engine.TensorFmod(values, divisor), NumOps.FromDouble(d));
        return engine.TensorFmod(inner, divisor);
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_ratio"] = MinRatio; p["max_ratio"] = MaxRatio;
        p["grid_d_min"] = GridDMin; p["grid_d_max"] = GridDMax;
        p["rotate"] = Rotate; p["fill_value"] = FillValue;
        return p;
    }
}
