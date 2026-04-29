using AiDotNet.Tensors;

namespace AiDotNet.Interfaces;

/// <summary>
/// Mixin interface for neural networks that produce multi-scale feature pyramids — typically
/// detection / segmentation backbones (ResNet, CSPDarknet, EfficientNet, SwinTransformer)
/// whose outputs feed FPN, PAN, anchor generators, or DETR-style transformer heads.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically <c>float</c> or <c>double</c>).</typeparam>
/// <remarks>
/// <para>
/// Replaces the legacy <c>BackboneBase&lt;T&gt;.ExtractFeatures()</c> contract. By implementing
/// this interface alongside <see cref="INeuralNetworkModel{T}"/>, a backbone gains the standard
/// neural-network surface (Layers, GetNamedLayerActivations, UpdateParameters, SetTrainingMode,
/// state-dict serialization) AND keeps the multi-scale extraction API that detection heads need.
/// </para>
/// <para>
/// The contract is paper-faithful: <see cref="GetFeatureMaps"/> returns one tensor per scale in
/// resolution-descending order (high-resolution first), with corresponding entries in
/// <see cref="OutputChannels"/> and <see cref="Strides"/>. For ResNet-50 detection use, that's
/// typically <c>{ C2, C3, C4, C5 }</c> with channels <c>{ 256, 512, 1024, 2048 }</c> and
/// strides <c>{ 4, 8, 16, 32 }</c>.
/// </para>
/// <para><b>For Beginners:</b> Detection models (YOLO, FasterRCNN, DETR) need to look at an image
/// at multiple zoom levels — small features for tiny objects, large features for big objects.
/// A "feature pyramid" provides exactly that: the same network outputs feature maps at several
/// resolutions. This interface is the contract any model implements when it wants to plug into
/// a detection head as a feature pyramid source.</para>
/// </remarks>
public interface IFeatureMapProvider<T>
{
    /// <summary>
    /// Runs the network on <paramref name="input"/> and returns the multi-scale feature maps,
    /// in resolution-descending order (the highest-resolution map first).
    /// </summary>
    /// <param name="input">The input image tensor of shape <c>[B, C, H, W]</c>.</param>
    /// <returns>A list of feature maps, one per scale.</returns>
    IReadOnlyList<Tensor<T>> GetFeatureMaps(Tensor<T> input);

    /// <summary>
    /// The number of channels in each scale's feature map. Indexed in the same order as
    /// <see cref="GetFeatureMaps"/>'s return value.
    /// </summary>
    int[] OutputChannels { get; }

    /// <summary>
    /// The downsampling stride at each scale (input pixel count divided by feature pixel count).
    /// Indexed in the same order as <see cref="GetFeatureMaps"/>'s return value. Standard
    /// detection backbones produce <c>{ 4, 8, 16, 32 }</c>.
    /// </summary>
    int[] Strides { get; }
}
