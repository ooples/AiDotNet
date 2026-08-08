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
        ChannelDimensions = other.ChannelDimensions?.ToArray();
        StageDepths = other.StageDepths?.ToArray();
        AttentionHeads = other.AttentionHeads?.ToArray();
        DecoderDimension = other.DecoderDimension;
        WindowSize = other.WindowSize;
        PatchSize = other.PatchSize;
        MlpRatio = other.MlpRatio;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        MaxGradientNorm = other.MaxGradientNorm;
    }

    /// <summary>
    /// Gets or sets the four Swin stage dimensions. A null value uses the paper's
    /// Swin-L dimensions [192, 384, 768, 1536]. Each stage must be twice the preceding stage.
    /// </summary>
    public int[]? ChannelDimensions { get; set; }

    /// <summary>
    /// Gets or sets the four Swin stage depths. A null value uses the paper's [2, 2, 18, 2].
    /// </summary>
    public int[]? StageDepths { get; set; }

    /// <summary>
    /// Gets or sets the four Swin attention-head counts. A null value uses [6, 12, 24, 48].
    /// </summary>
    public int[]? AttentionHeads { get; set; }

    /// <summary>Gets or sets the decoder width. A null value uses the paper default of 256.</summary>
    public int? DecoderDimension { get; set; }

    /// <summary>Gets or sets the Swin attention window size. The paper default is 7.</summary>
    public int WindowSize { get; set; } = 7;

    /// <summary>Gets or sets the Swin patch size. The paper default is 4.</summary>
    public int PatchSize { get; set; } = 4;

    /// <summary>Gets or sets the Swin MLP expansion ratio. The paper default is 4.</summary>
    public int MlpRatio { get; set; } = 4;

    /// <summary>Gets or sets the AdamW learning rate. The paper default is 1e-4 for OneFormer.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the decoupled AdamW weight decay. The paper default is 0.05.</summary>
    public double WeightDecay { get; set; } = 0.05;

    /// <summary>Gets or sets the global gradient-norm clipping threshold. Set to 0 to disable clipping.</summary>
    public double MaxGradientNorm { get; set; } = 0.01;
}
