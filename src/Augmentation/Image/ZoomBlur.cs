namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies radial/zoom blur emanating from the image center.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ZoomBlur<T> : ImageAugmenterBase<T>
{
    public double MinZoomFactor { get; }
    public double MaxZoomFactor { get; }

    public ZoomBlur(double minZoomFactor = 0.01, double maxZoomFactor = 0.05, double probability = 0.5)
        : base(probability)
    {
        MinZoomFactor = minZoomFactor; MaxZoomFactor = maxZoomFactor;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        double factor = context.GetRandomDouble(MinZoomFactor, MaxZoomFactor);
        int numSteps = Math.Max(3, (int)(factor * 100));
        double cx = data.Width / 2.0, cy = data.Height / 2.0;
        var result = data.Clone();

        // Every step samples the image at points pulled towards the centre by t; the row index
        // depends only on y and the column index only on x, so each step is a row/column gather.
        var engine = Engine;
        var image = data.ToPlanarTensor();
        // Step s pulls by t = s * factor / numSteps: sample indices of every step at once, [steps, size]
        var steps = engine.Reshape(engine.TensorDivideScalar(
            engine.TensorMultiplyScalar(engine.TensorArange<T>(0, numSteps, 1), NumOps.FromDouble(factor)), NumOps.FromDouble(numSteps)),
            [numSteps, 1]);
        var columnIndices = SampleIndices(engine, steps, data.Width, cx);
        var rowIndices = SampleIndices(engine, steps, data.Height, cy);
        var sum = new Tensor<T>(image._shape);
        for (int s = 0; s < numSteps; s++)
        {
            var ix = engine.TensorSliceAxis(columnIndices, 0, s).Contiguous();
            var iy = engine.TensorSliceAxis(rowIndices, 0, s).Contiguous();
            sum = engine.TensorAdd(sum, engine.TensorIndexSelect(engine.TensorIndexSelect(image, iy, 1), ix, 2));
        }

        result.SetFromPlanarTensor(engine.TensorDivideScalar(sum, NumOps.FromDouble(numSteps)));

        return result;
    }

    /// <summary>
    /// round(p + (center - p) * t) clamped to [0, size - 1] for every step t ([steps, 1]) and position p.
    /// </summary>
    private static Tensor<int> SampleIndices(IEngine engine, Tensor<T> steps, int size, double center)
    {
        int count = steps.Shape[0];
        int[] shape = [count, size];
        var positions = engine.Reshape(engine.TensorArange<T>(0, size, 1), [1, size]);
        var toCenter = engine.ScalarMinusTensor(NumOps.FromDouble(center), positions);
        var pulled = engine.TensorAdd(
            engine.TensorBroadcastTo(positions, shape),
            engine.TensorMultiply(engine.TensorBroadcastTo(toCenter, shape), engine.TensorBroadcastTo(steps, shape)));
        return engine.TensorClamp(engine.TensorRound(pulled), NumOps.Zero, NumOps.FromDouble(size - 1)).Cast<int>();
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_zoom_factor"] = MinZoomFactor; p["max_zoom_factor"] = MaxZoomFactor;
        return p;
    }
}
