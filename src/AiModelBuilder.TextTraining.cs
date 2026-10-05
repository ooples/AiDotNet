using AiDotNet.ComputerVision.Detection.TextDetection;

namespace AiDotNet;

/// <summary>
/// Annotation-based training for the text families through the facade: text detection on polygons and
/// recognition on transcriptions. Each model applies its paper's target assignment and loss.
/// </summary>
public static class TextTrainingBuilderExtensions
{
    /// <summary>Trains the configured text detector one step on page images and their text polygons.</summary>
    public static IAiModelBuilder<T, Tensor<T>, Tensor<T>> TrainTextDetections<T>(
        this IAiModelBuilder<T, Tensor<T>, Tensor<T>> builder, Tensor<T> images, TextDetectionTrainingBatch targets)
    {
        if (builder is null) throw new ArgumentNullException(nameof(builder));
        if (images is null) throw new ArgumentNullException(nameof(images));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        var model = ConfiguredModel<ITextDetectionTrainingModel<T>, T>(builder, "a text detector", "text detection training");
        model.TrainTextDetections(images, targets);
        return builder;
    }

    /// <summary>Trains the configured text recogniser one step on text-line images and their transcriptions.</summary>
    public static IAiModelBuilder<T, Tensor<T>, Tensor<T>> TrainRecognition<T>(
        this IAiModelBuilder<T, Tensor<T>, Tensor<T>> builder, Tensor<T> images, IReadOnlyList<string> transcriptions)
    {
        if (builder is null) throw new ArgumentNullException(nameof(builder));
        if (images is null) throw new ArgumentNullException(nameof(images));
        if (transcriptions is null) throw new ArgumentNullException(nameof(transcriptions));
        var model = ConfiguredModel<IRecognitionTrainingModel<T>, T>(builder, "a text recogniser", "recognition training");
        model.TrainRecognition(images, transcriptions);
        return builder;
    }

    private static TModel ConfiguredModel<TModel, T>(IAiModelBuilder<T, Tensor<T>, Tensor<T>> builder, string kind, string capability)
        where TModel : class
    {
        if (builder is not AiModelBuilder<T, Tensor<T>, Tensor<T>> facade)
            throw new NotSupportedException("This operation requires the AiModelBuilder facade.");
        if (facade.ConfiguredModel is null)
            throw new InvalidOperationException($"Configure {kind} before training it.");
        return facade.ConfiguredModel as TModel
            ?? throw new NotSupportedException($"The configured model does not implement {capability}.");
    }
}
