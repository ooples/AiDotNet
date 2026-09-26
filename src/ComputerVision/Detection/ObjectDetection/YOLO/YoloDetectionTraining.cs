using AiDotNet.ComputerVision.Detection.Losses;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;

/// <summary>Shared input checks and head splitting for task-aligned YOLO detection training.</summary>
internal static class YoloDetectionTraining
{
    /// <summary>Rejects inputs and targets the model cannot train on, before any forward pass or update.</summary>
    internal static void Validate<T>(Tensor<T> input, DetectionTrainingBatch<T> targets, int numClasses, string family)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (input.Rank != 4 || input.Shape[0] <= 0 || input.Shape[1] != 3 || input.Shape[2] <= 0 || input.Shape[3] <= 0)
            throw new ArgumentException($"{family} training requires a nonempty NCHW three-channel image batch.", nameof(input));
        targets.ValidateForModel(input.Shape[0], numClasses, int.MaxValue);
    }

    /// <summary>
    /// The loss of one YOLOv8-style head whose class levels start at <paramref name="offset"/> in
    /// <paramref name="heads"/>, followed by its distribution levels.
    /// </summary>
    internal static Tensor<T> HeadLoss<T>(TaskAlignedDetectionLoss<T> loss, List<Tensor<T>> heads, int offset, int levels,
        int[] strides, int imageHeight, int imageWidth, DetectionTrainingBatch<T> targets, int topK)
    {
        if (levels <= 0 || levels != strides.Length || heads.Count < offset + 2 * levels)
            throw new InvalidOperationException("YOLO training requires one class and one distribution output per pyramid level.");
        return loss.ComputeTapeLoss(heads.GetRange(offset, levels), heads.GetRange(offset + levels, levels),
            strides, imageHeight, imageWidth, targets, topK);
    }
}
