using AiDotNet.ComputerVision.Detection;

namespace AiDotNet;

/// <summary>Explicit semantic detection training through the fluent model-builder facade.</summary>
public static class DetectionBuilderExtensions
{
    /// <summary>Runs one detection-task update on the configured capable model.</summary>
    /// <remarks>
    /// Unlike raw tensor Train, this API deliberately selects the configured model's assignment and
    /// classification/box objective. Unsupported model families fail explicitly; there is no MSE fallback.
    /// Input images must already have the preprocessing expected by the model's Predict method.
    /// </remarks>
    public static IAiModelBuilder<T, Tensor<T>, Tensor<T>> TrainDetections<T>(
        this IAiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> input, DetectionTrainingBatch<T> targets)
    {
        if (builder is null) throw new ArgumentNullException(nameof(builder));
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (builder is not AiModelBuilder<T, Tensor<T>, Tensor<T>> facade)
            throw new NotSupportedException("This operation requires the AiModelBuilder facade.");
        if (facade.ConfiguredModel is null)
            throw new InvalidOperationException("Configure a detection model before training it.");
        if (facade.ConfiguredModel is not IDetectionTrainingModel<T> model)
            throw new NotSupportedException("The configured model does not implement semantic detection training.");
        model.TrainDetections(input, targets);
        return builder;
    }

    /// <summary>Explicitly adapts the COCO loader's normalized, zero-padded xywh labels before training.</summary>
    /// <remarks>This conversion is never selected by guessing the shape passed to raw Train.</remarks>
    public static IAiModelBuilder<T, Tensor<T>, Tensor<T>> TrainCocoDetections<T>(
        this IAiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> input, Tensor<T> paddedCocoLabels) =>
        builder.TrainDetections(input, DetectionTrainingBatch<T>.FromPaddedCoco(paddedCocoLabels));
}
