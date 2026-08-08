using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// Configuration options for the ViT-CoMer semantic segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ViT-CoMer options inherit from NeuralNetworkOptions, which provides
/// a Seed property for reproducibility. ViT-CoMer is a hybrid model that runs CNN and transformer
/// branches in parallel and fuses them to get excellent boundary quality in segmentation.
/// </para>
/// </remarks>
public class ViTCoMerOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ViTCoMerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ViTCoMerOptions(ViTCoMerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
    }

    /// <summary>
    /// Gets or sets the initial AdamW learning rate. The default follows the ViT-CoMer
    /// ADE20K training configuration from the original paper.
    /// </summary>
    public double LearningRate { get; set; } = 6e-5;
}
