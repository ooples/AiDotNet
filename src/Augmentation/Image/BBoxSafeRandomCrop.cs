namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Random crop that ensures bounding boxes remain valid after cropping.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BBoxSafeRandomCrop<T> : SpatialImageAugmenterBase<T>
{
    public double MinCropScale { get; }
    public int MaxAttempts { get; }
    public double MinBBoxAreaRatio { get; }

    public BBoxSafeRandomCrop(double minCropScale = 0.5, int maxAttempts = 50,
        double minBBoxAreaRatio = 0.25, double probability = 0.5) : base(probability)
    {
        if (minCropScale <= 0 || minCropScale > 1) throw new ArgumentOutOfRangeException(nameof(minCropScale), "MinCropScale must be in (0, 1].");
        if (maxAttempts < 1) throw new ArgumentOutOfRangeException(nameof(maxAttempts), "MaxAttempts must be at least 1.");
        MinCropScale = minCropScale; MaxAttempts = maxAttempts;
        MinBBoxAreaRatio = minBBoxAreaRatio;
    }

    protected override (ImageTensor<T> data, IDictionary<string, object> parameters) ApplyWithTransformParams(
        ImageTensor<T> data, AugmentationContext<T> context)
    {
        var parms = new Dictionary<string, object>();
        parms["crop_y"] = 0; parms["crop_x"] = 0;
        parms["crop_h"] = data.Height; parms["crop_w"] = data.Width;

        int cropH = (int)(data.Height * MinCropScale);
        int cropW = (int)(data.Width * MinCropScale);

        int h = context.GetRandomInt(cropH, data.Height + 1);
        int w = context.GetRandomInt(cropW, data.Width + 1);
        int y = context.GetRandomInt(0, data.Height - h + 1);
        int x = context.GetRandomInt(0, data.Width - w + 1);

        parms["crop_y"] = y; parms["crop_x"] = x;
        parms["crop_h"] = h; parms["crop_w"] = w;

        return (data.Crop(y, x, h, w), parms);
    }

    protected override BoundingBox<T> TransformBoundingBox(BoundingBox<T> box,
        IDictionary<string, object> transformParams, AugmentationContext<T> context)
    {
        int cropY = (int)transformParams["crop_y"];
        int cropX = (int)transformParams["crop_x"];
        int cropH = (int)transformParams["crop_h"];
        int cropW = (int)transformParams["crop_w"];

        var (x, y, w, h) = box.ToXYWH();
        double newX1 = Math.Max(0, x - cropX);
        double newY1 = Math.Max(0, y - cropY);
        double newX2 = Math.Min(cropW, x + w - cropX);
        double newY2 = Math.Min(cropH, y + h - cropY);

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
        int cropY = (int)transformParams["crop_y"];
        int cropX = (int)transformParams["crop_x"];
        int cropH = (int)transformParams["crop_h"];
        int cropW = (int)transformParams["crop_w"];

        var dense = mask.ToDense();
        var cropped = new T[cropH, cropW];

        for (int y = 0; y < cropH; y++)
        {
            for (int x = 0; x < cropW; x++)
            {
                int srcY = cropY + y;
                int srcX = cropX + x;
                if (srcY >= 0 && srcY < mask.Height && srcX >= 0 && srcX < mask.Width)
                    cropped[y, x] = dense[srcY, srcX];
            }
        }

        return new SegmentationMask<T>(cropped, mask.Type, mask.ClassIndex)
        {
            ClassName = mask.ClassName,
            InstanceId = mask.InstanceId
        };
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_crop_scale"] = MinCropScale; p["max_attempts"] = MaxAttempts;
        p["min_bbox_area_ratio"] = MinBBoxAreaRatio;
        return p;
    }
}
