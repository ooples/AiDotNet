namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Random crop with minimum IoU guarantee for bounding boxes (SSD-style).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MinIoURandomCrop<T> : SpatialImageAugmenterBase<T>
{
    public double[] MinIoUThresholds { get; }
    public int MaxAttempts { get; }

    public MinIoURandomCrop(double[]? minIoUThresholds = null, int maxAttempts = 50,
        double probability = 0.5) : base(probability)
    {
        MinIoUThresholds = minIoUThresholds ?? new[] { 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 };
        MaxAttempts = maxAttempts;
    }

    protected override (ImageTensor<T> data, IDictionary<string, object> parameters) ApplyWithTransformParams(
        ImageTensor<T> data, AugmentationContext<T> context)
    {
        int threshIdx = context.GetRandomInt(0, MinIoUThresholds.Length);
        double minIoU = MinIoUThresholds[threshIdx];

        var parms = new Dictionary<string, object>
        {
            ["crop_y"] = 0, ["crop_x"] = 0,
            ["crop_h"] = data.Height, ["crop_w"] = data.Width,
            ["min_iou"] = minIoU
        };

        if (Math.Abs(minIoU - 1.0) < 1e-10) return (data.Clone(), parms);

        int bestY = 0, bestX = 0, bestCropH = data.Height, bestCropW = data.Width;

        for (int attempt = 0; attempt < MaxAttempts; attempt++)
        {
            double scale = context.GetRandomDouble(0.3, 1.0);
            double aspectRatio = context.GetRandomDouble(0.5, 2.0);

            int cropW = (int)(data.Width * scale * Math.Sqrt(aspectRatio));
            int cropH = (int)(data.Height * scale / Math.Sqrt(aspectRatio));
            cropW = Math.Max(1, Math.Min(cropW, data.Width));
            cropH = Math.Max(1, Math.Min(cropH, data.Height));

            int y = context.GetRandomInt(0, Math.Max(1, data.Height - cropH + 1));
            int x = context.GetRandomInt(0, Math.Max(1, data.Width - cropW + 1));

            bestY = y; bestX = x; bestCropH = cropH; bestCropW = cropW;

            // Compute IoU of crop region with full image
            double cropArea = (double)cropH * cropW;
            double imageArea = (double)data.Height * data.Width;
            double iou = cropArea / imageArea;

            if (iou >= minIoU) break;
        }

        parms["crop_y"] = bestY; parms["crop_x"] = bestX;
        parms["crop_h"] = bestCropH; parms["crop_w"] = bestCropW;

        return (data.Crop(bestY, bestX, bestCropH, bestCropW), parms);
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
        p["min_iou_thresholds"] = MinIoUThresholds; p["max_attempts"] = MaxAttempts;
        return p;
    }
}
