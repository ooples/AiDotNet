namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Random-sized crop with bounding box safety, followed by resize to target size.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RandomSizedBBoxSafeCrop<T> : SpatialImageAugmenterBase<T>
{
    public int TargetHeight { get; }
    public int TargetWidth { get; }
    public double MinCropScale { get; }
    public double MaxCropScale { get; }

    public RandomSizedBBoxSafeCrop(int targetHeight = 224, int targetWidth = 224,
        double minCropScale = 0.5, double maxCropScale = 1.0,
        double probability = 0.5) : base(probability)
    {
        TargetHeight = targetHeight; TargetWidth = targetWidth;
        MinCropScale = minCropScale; MaxCropScale = maxCropScale;
    }

    protected override (ImageTensor<T> data, IDictionary<string, object> parameters) ApplyWithTransformParams(
        ImageTensor<T> data, AugmentationContext<T> context)
    {
        double scale = context.GetRandomDouble(MinCropScale, MaxCropScale);
        int cropH = Math.Max(1, (int)(data.Height * scale));
        int cropW = Math.Max(1, (int)(data.Width * scale));

        int y = context.GetRandomInt(0, Math.Max(1, data.Height - cropH + 1));
        int x = context.GetRandomInt(0, Math.Max(1, data.Width - cropW + 1));

        var cropped = data.Crop(y, x, cropH, cropW);
        var resized = new Resize<T>(TargetHeight, TargetWidth).Apply(cropped, context);

        var parms = new Dictionary<string, object>
        {
            ["crop_y"] = y, ["crop_x"] = x,
            ["crop_h"] = cropH, ["crop_w"] = cropW,
            ["target_h"] = TargetHeight, ["target_w"] = TargetWidth
        };

        return (resized, parms);
    }

    protected override BoundingBox<T> TransformBoundingBox(BoundingBox<T> box,
        IDictionary<string, object> transformParams, AugmentationContext<T> context)
    {
        int cropY = (int)transformParams["crop_y"];
        int cropX = (int)transformParams["crop_x"];
        int cropH = (int)transformParams["crop_h"];
        int cropW = (int)transformParams["crop_w"];
        int targetH = (int)transformParams["target_h"];
        int targetW = (int)transformParams["target_w"];

        var (bx, by, bw, bh) = box.ToXYWH();
        double newX1 = Math.Max(0, bx - cropX) / cropW * targetW;
        double newY1 = Math.Max(0, by - cropY) / cropH * targetH;
        double newX2 = Math.Min(cropW, bx + bw - cropX) / cropW * targetW;
        double newY2 = Math.Min(cropH, by + bh - cropY) / cropH * targetH;

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
        int targetH = (int)transformParams["target_h"];
        int targetW = (int)transformParams["target_w"];

        double newX = (NumOps.ToDouble(keypoint.X) - cropX) / cropW * targetW;
        double newY = (NumOps.ToDouble(keypoint.Y) - cropY) / cropH * targetH;

        var result = keypoint.Clone();
        result.X = NumOps.FromDouble(newX);
        result.Y = NumOps.FromDouble(newY);

        if (newX < 0 || newX >= targetW || newY < 0 || newY >= targetH)
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
        int targetH = (int)transformParams["target_h"];
        int targetW = (int)transformParams["target_w"];

        var dense = mask.ToDense();
        var resized = new T[targetH, targetW];

        for (int y = 0; y < targetH; y++)
        {
            for (int x = 0; x < targetW; x++)
            {
                int srcY = cropY + (int)((double)y / targetH * cropH);
                int srcX = cropX + (int)((double)x / targetW * cropW);
                if (srcY >= 0 && srcY < mask.Height && srcX >= 0 && srcX < mask.Width)
                    resized[y, x] = dense[srcY, srcX];
            }
        }

        return new SegmentationMask<T>(resized, mask.Type, mask.ClassIndex)
        {
            ClassName = mask.ClassName,
            InstanceId = mask.InstanceId
        };
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["target_height"] = TargetHeight; p["target_width"] = TargetWidth;
        p["min_crop_scale"] = MinCropScale; p["max_crop_scale"] = MaxCropScale;
        return p;
    }
}
