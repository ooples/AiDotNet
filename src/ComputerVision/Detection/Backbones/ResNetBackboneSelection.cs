using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// Picks the ResNet a detector builds from its options' <see cref="BackboneType"/>.
/// </summary>
/// <remarks>
/// <para>
/// DETR, DINO and DBNet build a ResNet backbone, and each one ignored the options' <c>Backbone</c>
/// and built a ResNet-50: the papers' default. <see cref="ObjectDetectionOptions{T}.Backbone"/> defaults
/// to CSPDarknet for the YOLO family, which none of these can use, so a backbone that is not a ResNet
/// still selects the paper's ResNet-50 and nothing changes for existing callers. A backbone that names a
/// ResNet is now honoured: DBNet's paper reports DB-ResNet-18 alongside DB-ResNet-50, DETR's reports
/// ResNet-101, and a smaller trunk is what bounds these models at test scale.
/// </para>
/// <para><b>For Beginners:</b> The backbone is the part of a detector that turns an image into
/// features. This lets you choose a smaller or larger ResNet; if you name something that is not a
/// ResNet, the model keeps the one its paper used.</para>
/// </remarks>
public static class ResNetBackboneSelection
{
    /// <summary>The ResNet variant <paramref name="backbone"/> names, or <paramref name="paperDefault"/>.</summary>
    /// <param name="backbone">The options' backbone choice.</param>
    /// <param name="paperDefault">The variant the model's paper uses when no ResNet is named.</param>
    public static ResNetVariant Resolve(BackboneType backbone, ResNetVariant paperDefault = ResNetVariant.ResNet50)
        => backbone switch
        {
            BackboneType.ResNet18 => ResNetVariant.ResNet18,
            BackboneType.ResNet34 => ResNetVariant.ResNet34,
            BackboneType.ResNet50 => ResNetVariant.ResNet50,
            BackboneType.ResNet101 => ResNetVariant.ResNet101,
            _ => paperDefault,
        };
}
