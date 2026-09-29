using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the EAST document model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> EAST finds text in a photograph in one pass, without the usual chain of
/// candidate-generation and filtering steps — which is what the name stands for, Efficient and
/// Accurate Scene Text detector. The values here are the ones it ships with.
/// </para>
/// </remarks>
public class EASTOptions : DocumentNeuralNetworkOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    /// <remarks>
    /// <para>
    /// ImageSize and BackboneChannels are declared by DocumentNeuralNetworkOptions and were
    /// previously left unset, with the values living in constructor parameters instead. EAST
    /// (Zhou et al., CVPR 2017) detects on a 512-pixel image over a 512-channel backbone.
    /// </para>
    /// </remarks>
    public EASTOptions()
    {
        ImageSize = 512;
        BackboneChannels = 512;
        FeatureChannels = 128;
        GeometryType = EASTGeometryType.RBox;
    }

    /// <summary>
    /// Gets or sets the width of the feature-merging branch. Default: 128.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Distinct from <see cref="DocumentNeuralNetworkOptions.BackboneChannels"/>: that is the
    /// backbone's width, this is the width of the branch that merges the backbone's levels back
    /// together before the prediction heads.
    /// </para>
    /// </remarks>
    public int FeatureChannels { get; set; }

    /// <summary>
    /// Gets or sets the shape predicted for each detected piece of text.
    /// Default: <see cref="EASTGeometryType.RBox"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This changes the model, not just the output format: the geometry head emits five channels
    /// for a rotated box and eight for a quadrilateral. It was previously a <c>string</c>, where
    /// a misspelling compiled and silently produced the five-channel head.
    /// </para>
    /// </remarks>
    public EASTGeometryType GeometryType { get; set; }

    /// <summary>
    /// Gets or sets the learning rate used when the model creates its own optimizer.
    /// Default: 1e-3.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Both constructors previously built a BARE optimizer -- <c>new AdamOptimizer&lt;...&gt;(this)</c>
    /// -- so the model trained at the optimizer's own default and no configured rate could reach
    /// it. 1e-3 IS Adam's default, so behaviour is unchanged.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-3;
}
