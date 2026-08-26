using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Mamba;

/// <summary>
/// Configuration options for Vision Mamba (Vim) bidirectional SSM segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VisionMamba model. Default values follow the original paper settings.</para>
/// </remarks>
public class VisionMambaOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public VisionMambaOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VisionMambaOptions(VisionMambaOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    /// <summary>Gets or sets the AdamW learning rate used by the ADE20K recipe.</summary>
    /// <value>Defaults to <c>6e-5</c>, as specified for Vision Mamba ADE20K training.</value>
    /// <remarks><para><b>For Beginners:</b> This controls the size of each optimizer update.</para></remarks>
    public double LearningRate { get; set; } = 6e-5;

    /// <summary>Gets or sets the AdamW decoupled weight decay used by the ADE20K recipe.</summary>
    /// <value>Defaults to <c>0.01</c>, as specified for Vision Mamba ADE20K training.</value>
    /// <remarks><para><b>For Beginners:</b> Weight decay regularizes the learned weights.</para></remarks>
    public double WeightDecay { get; set; } = 0.01;
}
