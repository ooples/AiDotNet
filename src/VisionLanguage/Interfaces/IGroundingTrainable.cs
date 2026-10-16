using AiDotNet.VisionLanguage.Grounding;

namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// A grounding model that trains on grounded phrases: an image, a caption, and the boxes the caption's phrases name.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Such a model's paper loss matches its predictions to the boxes and scores them against the caption's tokens. A tensor
/// target cannot carry the phrases, so these models refuse <c>Train(input, expected)</c> and train here instead.
/// </remarks>
public interface IGroundingTrainable<T>
{
    /// <summary>One training step with the model's paper loss.</summary>
    /// <param name="image">One RGB image <c>[3, H, W]</c> with values in 0..1.</param>
    /// <param name="caption">Period-separated phrases, such as "cat . remote control .".</param>
    /// <param name="boxes">The boxes, each naming one of the caption's phrases.</param>
    void TrainGrounding(Tensor<T> image, string caption, IReadOnlyList<GroundingBox> boxes);

    /// <summary>The paper loss on the same supervision, without training.</summary>
    T EvaluateGroundingLoss(Tensor<T> image, string caption, IReadOnlyList<GroundingBox> boxes);
}
