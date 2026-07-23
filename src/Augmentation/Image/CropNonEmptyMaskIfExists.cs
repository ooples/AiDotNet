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

        // Try multiple attempts to find a crop containing non-empty content
        // Use pixel variance as a proxy for non-empty regions when no mask is available
        double bestScore = -1;
        for (int attempt = 0; attempt < MaxAttempts; attempt++)
        {
            int candY = context.GetRandomInt(0, Math.Max(1, data.Height - cropH + 1));
            int candX = context.GetRandomInt(0, Math.Max(1, data.Width - cropW + 1));

            // Check if region has non-zero content
            double sum = 0;
            int sampleStep = Math.Max(1, cropH * cropW / 100);
            int count = 0;
            for (int i = 0; i < cropH * cropW; i += sampleStep)
            {
                int sy = candY + i / cropW;
                int sx = candX + i % cropW;
                if (sy < data.Height && sx < data.Width)
                {
                    // Check all channels for non-empty content
                    for (int c = 0; c < data.Channels; c++)
                        sum += Math.Abs(NumOps.ToDouble(data.GetPixel(sy, sx, c)));
                    count++;
                }
            }

            double score = count > 0 ? sum / count : 0;
            if (score > bestScore)
            {
                bestScore = score;
                bestY = candY;
                bestX = candX;
            }

            if (score > 0) break;
        }

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
        p["crop_height"] = CropHeight; p["crop_width"] = CropWidth;
        p["max_attempts"] = MaxAttempts;
        return p;
    }
}
