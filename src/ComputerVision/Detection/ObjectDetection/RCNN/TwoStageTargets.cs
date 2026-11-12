namespace AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;

/// <summary>Shared target conversion for training the two-stage detectors.</summary>
internal static class TwoStageTargets
{
    /// <summary>A normalized center-format target as corner coordinates in input pixels.</summary>
    internal static double[] PixelCorners<T>(DetectionTrainingTarget<T> target, int width, int height)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        double cx = ops.ToDouble(target.CenterX) * width;
        double cy = ops.ToDouble(target.CenterY) * height;
        double halfWidth = ops.ToDouble(target.Width) * width / 2;
        double halfHeight = ops.ToDouble(target.Height) * height / 2;
        return new[] { cx - halfWidth, cy - halfHeight, cx + halfWidth, cy + halfHeight };
    }

    /// <summary>The first image's rows of a [batch, rows, width] head output, as [rows, width].</summary>
    internal static Tensor<T> FirstImage<T>(Tensor<T> head)
    {
        if (head.Rank != 3 || head.Shape[0] != 1)
            throw new InvalidOperationException("Two-stage training expects the proposal outputs of exactly one image.");
        return AiDotNet.Tensors.Engines.AiDotNetEngine.Current.Reshape(head, new[] { head.Shape[1], head.Shape[2] });
    }

    /// <summary>Appends constant corner boxes to detached proposal boxes [proposals, 4].</summary>
    internal static Tensor<T> AppendBoxes<T>(Tensor<T> proposals, IReadOnlyList<double[]> boxes)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        int existing = proposals.Shape[0];
        var combined = new Tensor<T>(new[] { existing + boxes.Count, 4 });
        var source = proposals.ToArray();
        for (int i = 0; i < source.Length; i++) combined[i] = source[i];
        for (int b = 0; b < boxes.Count; b++)
            for (int k = 0; k < 4; k++)
                combined[(existing + b) * 4 + k] = ops.FromDouble(boxes[b][k]);
        return combined;
    }
}
