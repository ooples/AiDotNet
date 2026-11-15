using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Inpainting mask preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Processes inpainting masks by binarizing and optionally feathering edges.
/// The output is a single-channel mask where 1.0 indicates regions to inpaint
/// and 0.0 indicates regions to preserve.
/// </para>
/// <para>
/// <b>For Beginners:</b> This prepares a mask that tells the AI which parts
/// of the image to regenerate (white) and which to keep (black). The feathering
/// option creates smooth transitions at mask edges for more natural blending.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class InpaintingMaskPreprocessor<T> : DiffusionPreprocessorBase<T>
{
    private readonly double _binarizeThreshold;
    private readonly int _featherRadius;

    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.InpaintMask;
    /// <inheritdoc />
    public override int OutputChannels => 1;

    /// <summary>
    /// Initializes a new inpainting mask preprocessor.
    /// </summary>
    /// <param name="binarizeThreshold">Threshold for binarizing the mask. Default: 0.5.</param>
    /// <param name="featherRadius">Radius for edge feathering. 0 disables feathering. Default: 0.</param>
    public InpaintingMaskPreprocessor(double binarizeThreshold = 0.5, int featherRadius = 0)
    {
        _binarizeThreshold = binarizeThreshold;
        _featherRadius = featherRadius;
    }

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        // Binarize the mask
        var result = PreprocessorFilters.Above(PreprocessorFilters.Channel(data, 0), _binarizeThreshold);

        // Apply feathering if requested: mean over the neighbourhood clipped to the image
        if (_featherRadius > 0)
            result = PreprocessorFilters.ClippedBoxMean(result, _featherRadius);

        return result;
    }
}
