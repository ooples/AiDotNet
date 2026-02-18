namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for 3D vision-language models that understand point clouds, 3D scenes, and spatial relationships.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// 3D vision-language models extend traditional 2D VLMs to process point clouds, voxel grids,
/// and 3D scene representations. They enable spatial reasoning, 3D object grounding,
/// and scene-level question answering.
/// </para>
/// </remarks>
public interface IThreeDVisionLanguageModel<T> : IGenerativeVisionLanguageModel<T>
{
    /// <summary>
    /// Processes a 3D point cloud and generates language output conditioned on a text prompt.
    /// </summary>
    /// <param name="pointCloud">Point cloud tensor in [numPoints, channels] format (XYZ + optional color/normals).</param>
    /// <param name="prompt">Text prompt or question about the 3D scene.</param>
    /// <returns>Output tensor of token logits for text generation.</returns>
    Tensor<T> GenerateFrom3D(Tensor<T> pointCloud, string prompt);

    /// <summary>
    /// Gets the maximum number of 3D points the model can process.
    /// </summary>
    int MaxPoints { get; }

    /// <summary>
    /// Gets the number of channels per point (e.g., 3 for XYZ, 6 for XYZ+RGB).
    /// </summary>
    int PointChannels { get; }
}
