namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Drops random object masks from the segmentation mask.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MaskDropout<T> : ImageAugmenterBase<T>
{
    public double DropRate { get; }
    public double FillValue { get; }

    public MaskDropout(double dropRate = 0.3, double fillValue = 0.0,
        double probability = 0.5) : base(probability)
    {
        DropRate = dropRate; FillValue = fillValue;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();

        // Object IDs are the truncated values of the first channel, in order of first appearance
        var engine = Engine;
        var image = data.ToPlanarTensor();
        var ids = engine.TensorTrunc(engine.TensorSliceAxis(image, 0, 0));
        var objectIds = engine.TensorUnique(ids, false);
        int[] isObject = engine.TensorNotEquals(objectIds, NumOps.Zero).Cast<int>().ToArray();

        // Randomly select non-zero objects to drop (one draw per object, in first-appearance order)
        var dropPositions = new List<int>();
        for (int i = 0; i < isObject.Length; i++)
        {
            if (isObject[i] != 0 && context.GetRandomDouble(0, 1) < DropRate)
            {
                dropPositions.Add(i);
            }
        }

        // Fill dropped object regions
        if (dropPositions.Count > 0)
        {
            var dropIds = engine.TensorIndexSelect(objectIds, new Tensor<int>(dropPositions.ToArray(), [dropPositions.Count]), 0);
            var dropped = engine.TensorIsIn(ids, dropIds);
            var droppedMask = engine.TensorWhere(dropped, ImageTensor<T>.Constant(ids, 1.0), ImageTensor<T>.Constant(ids, 0.0));
            var mask = engine.TensorBroadcastTo(engine.Reshape(droppedMask, [1, data.Height, data.Width]), image._shape);
            result.SetFromPlanarTensor(engine.TensorWhere(mask, ImageTensor<T>.Constant(image, FillValue), image));
        }

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["drop_rate"] = DropRate; p["fill_value"] = FillValue;
        return p;
    }
}
