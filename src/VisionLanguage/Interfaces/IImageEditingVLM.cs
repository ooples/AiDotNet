namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for VLMs that edit images based on natural language instructions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Image editing VLMs take an input image and a text instruction and produce an edited version.
/// These models understand the semantic intent of the instruction and apply targeted modifications.
/// </para>
/// </remarks>
public interface IImageEditingVLM<T> : IVisualEncoder<T>
{
    /// <summary>
    /// Edits an image according to a natural language instruction.
    /// </summary>
    /// <param name="image">Input image tensor in [channels, height, width] format.</param>
    /// <param name="instruction">Natural language editing instruction (e.g., "make the sky blue").</param>
    /// <returns>Edited image tensor in [channels, height, width] format.</returns>
    Tensor<T> EditImage(Tensor<T> image, string instruction);

    /// <summary>
    /// Gets the output image resolution.
    /// </summary>
    int OutputImageSize { get; }
}
