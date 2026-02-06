namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Clips bounding boxes to image boundaries.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BBoxClipToImage<T> : SpatialImageAugmenterBase<T>
{
    public BBoxClipToImage(double probability = 1.0) : base(probability)
    {
    }

    protected override (ImageTensor<T> data, IDictionary<string, object> parameters) ApplyWithTransformParams(
        ImageTensor<T> data, AugmentationContext<T> context)
    {
        var parms = new Dictionary<string, object>
        {
            ["height"] = data.Height,
            ["width"] = data.Width
        };
        return (data.Clone(), parms);
    }

    protected override BoundingBox<T> TransformBoundingBox(BoundingBox<T> box,
        IDictionary<string, object> transformParams, AugmentationContext<T> context)
    {
        int height = (int)transformParams["height"];
        int width = (int)transformParams["width"];

        var (bx, by, bw, bh) = box.ToXYWH();
        double newX1 = Math.Max(0, Math.Min(width, bx));
        double newY1 = Math.Max(0, Math.Min(height, by));
        double newX2 = Math.Max(0, Math.Min(width, bx + bw));
        double newY2 = Math.Max(0, Math.Min(height, by + bh));

        if (newX2 <= newX1 || newY2 <= newY1)
        {
            var empty = box.Clone();
            empty.X1 = NumOps.FromDouble(0); empty.Y1 = NumOps.FromDouble(0);
            empty.X2 = NumOps.FromDouble(0); empty.Y2 = NumOps.FromDouble(0);
            empty.Format = BoundingBoxFormat.XYXY;
            return empty;
        }

        var result = box.Clone();
        result.X1 = NumOps.FromDouble(newX1); result.Y1 = NumOps.FromDouble(newY1);
        result.X2 = NumOps.FromDouble(newX2); result.Y2 = NumOps.FromDouble(newY2);
        result.Format = BoundingBoxFormat.XYXY;
        return result;
    }

    protected override Keypoint<T> TransformKeypoint(Keypoint<T> keypoint,
        IDictionary<string, object> transformParams, AugmentationContext<T> context)
    {
        int height = (int)transformParams["height"];
        int width = (int)transformParams["width"];

        double kx = NumOps.ToDouble(keypoint.X);
        double ky = NumOps.ToDouble(keypoint.Y);

        if (kx < 0 || kx >= width || ky < 0 || ky >= height)
        {
            var hidden = keypoint.Clone();
            hidden.Visibility = 0;
            return hidden;
        }

        return keypoint;
    }

    protected override SegmentationMask<T> TransformMask(SegmentationMask<T> mask,
        IDictionary<string, object> transformParams, AugmentationContext<T> context)
    {
        return mask;
    }

    public override IDictionary<string, object> GetParameters()
    {
        return base.GetParameters();
    }
}
