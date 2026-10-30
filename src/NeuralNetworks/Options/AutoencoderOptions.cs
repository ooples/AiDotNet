using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the Autoencoder neural network.
/// </summary>
public class AutoencoderOptions : ModelHyperparameterOptions
{

    /// <summary>
    /// Gets or sets batch size. Default: <c>32</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many examples the model looks at before it updates itself once.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets epochs. Default: <c>1</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many complete passes the model makes over your training data.</para>
    /// </remarks>
    public int Epochs { get; set; } = 1;

    /// <summary>
    /// Gets or sets the width of the bottleneck (the code) in the default stack; null derives it as
    /// max(input size / 4, 4).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Used only when the architecture supplies no layers of its own. The default stack is
    /// input → hidden → bottleneck → hidden → output, with the hidden width max(input / 2, 8).
    /// </para>
    /// <para><b>For Beginners:</b> How many numbers each input is compressed into. Smaller forces a
    /// more aggressive summary.</para>
    /// </remarks>
    public int? LatentSize { get; set; }

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(BatchSize, nameof(BatchSize));
        Require(Epochs, nameof(Epochs));
    }
}
