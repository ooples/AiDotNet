namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Crops the image to a region that contains non-empty mask content.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CropNonEmptyMaskIfExists<T> : SpatialImageAugmenterBase<T>
{
    public int CropHeight { get; }
    public int CropWidth { get; }
    public int MaxAttempts { get; }

    public CropNonEmptyMaskIfExists(int cropHeight = 224, int cropWidth = 224,
        int maxAttempts = 10, double probability = 1.0) : base(probability)
    {
        CropHeight = cropHeight; CropWidth = cropWidth; MaxAttempts = maxAttempts;
    }

    protected override (ImageTensor<T> data, IDictionary<string, object> parameters) ApplyWithTransformParams(
        ImageTensor<T> data, AugmentationContext<T> context)
    {
        int cropH = Math.Min(CropHeight, data.Height);
        int cropW = Math.Min(CropWidth, data.Width);

        int bestY = context.GetRandomInt(0, Math.Max(1, data.Height - cropH + 1));
        int bestX = context.GetRandomInt(0, Math.Max(1, data.Width - cropW + 1));

        var parms = new Dictionary<string, object>
        {
            ["crop_y"] = bestY, ["crop_x"] = bestX,
            ["crop_h"] = cropH, ["crop_w"] = cropW
        };

        return (data.Crop(bestY, bestX, cropH, cropW), parms);
    }

    protected override BoundingBox<T> TransformBoundingBox(BoundingBox<T> box,
        IDictionary<string, object> transformParams, AugmentationContext<T> context)
    {
        int cropY = (int)transformParams["crop_y"];
        int cropX = (int)transformParams["crop_x"];
        int cropH = (int)transformParams["crop_h"];
        int cropW = (int)transformParams["crop_w"];

        var (bx, by, bw, bh) = box.ToXYWH();
        double newX1 = Math.Max(0, bx - cropX);
        double newY1 = Math.Max(0, by - cropY);
        double newX2 = Math.Min(cropW, bx + bw - cropX);
        double newY2 = Math.Min(cropH, by + bh - cropY);

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
        int cropY = (int)transformParams["crop_y"];
        int cropX = (int)transformParams["crop_x"];
        int cropH = (int)transformParams["crop_h"];
        int cropW = (int)transformParams["crop_w"];

        double newX = NumOps.ToDouble(keypoint.X) - cropX;
        double newY = NumOps.ToDouble(keypoint.Y) - cropY;

        var result = keypoint.Clone();
        result.X = NumOps.FromDouble(newX);
        result.Y = NumOps.FromDouble(newY);

        if (newX < 0 || newX >= cropW || newY < 0 || newY >= cropH)
            result.Visibility = 0;

        return result;
    }

    protected override SegmentationMask<T> TransformMask(SegmentationMask<T> mask,
        IDictionary<string, object> transformParams, AugmentationContext<T> context)
    {
        return mask;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["crop_height"] = CropHeight; p["crop_width"] = CropWidth;
        p["max_attempts"] = MaxAttempts;
        return p;
    }
}
