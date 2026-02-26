using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the OneFormer universal segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> OneFormer options inherit from NeuralNetworkOptions. OneFormer is trained
/// once on panoptic data and can perform semantic, instance, or panoptic segmentation by simply
/// providing a text prompt describing which task to perform. This "one model, all tasks" approach
/// simplifies deployment.
/// </para>
/// </remarks>
public class OneFormerOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public OneFormerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public OneFormerOptions(OneFormerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
