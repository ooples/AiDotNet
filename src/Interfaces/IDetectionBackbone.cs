namespace AiDotNet.Interfaces;

/// <summary>
/// Marker interface for detection backbones. Combines the two contracts a detection
/// model expects from its backbone:
/// <list type="bullet">
///   <item><see cref="INeuralNetworkModel{T}"/> — standard neural-network surface
///         (Layers, GetNamedLayerActivations, UpdateParameters, state-dict serialization).</item>
///   <item><see cref="IFeatureMapProvider{T}"/> — multi-scale feature pyramid output
///         (GetFeatureMaps, OutputChannels, Strides) consumed by FPN, anchor generators,
///         and DETR-style transformer heads.</item>
/// </list>
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Use this interface as the field type in detection consumers (ObjectDetectorBase,
/// TextDetectorBase) instead of the concrete <c>BackboneBase&lt;T&gt;</c> base class.
/// That keeps detection models decoupled from a specific base-class hierarchy and lets
/// any future backbone implementation that satisfies the two underlying contracts plug
/// in without inheriting from <c>BackboneBase</c>.
/// </para>
/// <para><b>For Beginners:</b> A detection model needs two things from its "backbone"
/// (the network that processes the image first): the standard ability to train and run
/// like any other neural network, and a special ability to output features at multiple
/// scales for spotting both small and large objects. This interface bundles those two
/// requirements so detection models can ask for "anything that can do both" rather than
/// "specifically a BackboneBase".</para>
/// </remarks>
public interface IDetectionBackbone<T> : INeuralNetworkModel<T>, IFeatureMapProvider<T>
{
}
