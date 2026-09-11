using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the CRNN document model.
/// </summary>
public class CRNNOptions : DocumentNeuralNetworkOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    /// <remarks>
    /// <para>
    /// ImageWidth and MaxSequenceLength are declared by DocumentNeuralNetworkOptions and were
    /// previously left unset, with the values living in constructor parameters instead. CRNN
    /// (Shi et al. 2017) reads a 128-pixel-wide text crop and emits up to 32 timesteps.
    /// </para>
    /// </remarks>
    public CRNNOptions()
    {
        ImageWidth = 128;
        MaxSequenceLength = 32;
        CnnChannels = 512;
        RnnHiddenSize = 256;
        RnnLayers = 2;
    }

    /// <summary>
    /// Gets or sets the width of the final convolutional feature map. Default: 512.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> CRNN looks at the picture with a convolutional network first,
    /// then reads the result left to right with a recurrent one. This is how many features the
    /// looking half produces for each slice of the image.</para>
    /// </remarks>
    public int CnnChannels { get; set; }

    /// <summary>
    /// Gets or sets the hidden width of the recurrent layers. Default: 256.
    /// </summary>
    public int RnnHiddenSize { get; set; }

    /// <summary>
    /// Gets or sets how many recurrent layers read the feature sequence. Default: 2.
    /// </summary>
    public int RnnLayers { get; set; }

    /// <summary>
    /// Gets or sets the recognisable character set. Null uses the model's default charset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The character count is the width of the CTC output layer, so this changes the model's
    /// shape and not merely how results are decoded.
    /// </para>
    /// </remarks>
    public string? Charset { get; set; }

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
