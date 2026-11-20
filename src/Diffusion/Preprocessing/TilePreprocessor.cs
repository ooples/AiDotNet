using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Tile preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Produces a blurred/downsampled version of the input image for tile-based
/// ControlNet conditioning. This guides the model to preserve overall color
/// and composition while allowing fine detail regeneration.
/// </para>
/// <para>
/// <b>For Beginners:</b> This creates a blurry version of your image that
/// ControlNet uses to keep the same colors and general layout, while letting
/// the AI add sharp details. It's commonly used for upscaling and detail enhancement.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TilePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    private readonly int _blurRadius;

    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.Tile;
    /// <inheritdoc />
    public override int OutputChannels => 3;

    /// <summary>
    /// Initializes a new tile preprocessor.
    /// </summary>
    /// <param name="blurRadius">Radius of the Gaussian-like blur. Default: 4.</param>
    public TilePreprocessor(int blurRadius = 4)
    {
        _blurRadius = blurRadius;
    }

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        // Box blur approximation: per channel, the mean over the neighbourhood clipped to the image
        return PreprocessorFilters.ClippedBoxMean(PreprocessorFilters.ToRgb(data), _blurRadius);
    }
}
