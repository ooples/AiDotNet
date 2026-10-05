using AiDotNet.ComputerVision.Detection.TextDetection;

namespace AiDotNet.Interfaces;

/// <summary>
/// A text recogniser that trains on transcriptions, using its paper's objective (CTC for CRNN, teacher-forced
/// cross-entropy for TrOCR).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IRecognitionTrainingModel<T>
{
    /// <summary>One training step on cropped text-line images and the text each one shows.</summary>
    /// <param name="images">The text-line images, <c>[batch, channels, height, width]</c> or one <c>[channels, height, width]</c>.</param>
    /// <param name="transcriptions">One transcription per image, using only characters of the model's character set.</param>
    void TrainRecognition(Tensor<T> images, IReadOnlyList<string> transcriptions);
}

/// <summary>
/// A text detector that trains on text polygons, using its paper's target assignment and loss (DBNet's
/// shrunk-kernel and border maps, CRAFT's character and affinity heatmaps, EAST's score and geometry maps).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ITextDetectionTrainingModel<T>
{
    /// <summary>One training step on page images and the text regions annotated on each.</summary>
    /// <param name="images">The page images, <c>[batch, channels, height, width]</c> or one <c>[channels, height, width]</c>.</param>
    /// <param name="targets">The text polygons of each image, in that image's pixel coordinates.</param>
    void TrainTextDetections(Tensor<T> images, TextDetectionTrainingBatch targets);
}
