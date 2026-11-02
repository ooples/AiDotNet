using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the DeepBeliefNetwork.
/// </summary>
public class DeepBeliefNetworkOptions : ModelHyperparameterOptions
{

    /// <summary>
    /// Gets or sets batch size. Default: <c>32</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many examples the model looks at before it updates itself once.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets epochs. Default: <c>10</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many complete passes the model makes over your training data.</para>
    /// </remarks>
    public int Epochs { get; set; } = 10;

    /// <summary>
    /// Gets or sets the hidden units of each stacked RBM, input side first. Default: <c>{ 500, 500, 2000 }</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The default is the stack Hinton, Osindero &amp; Teh 2006 trained on MNIST (784-500-500-2000). Used
    /// only when the architecture supplies no layers of its own; a supervised head maps the last width
    /// to the output.
    /// </para>
    /// <para><b>For Beginners:</b> Each entry is one layer of feature detectors, learned one layer at a
    /// time before the whole network is fine-tuned.</para>
    /// </remarks>
    public int[] HiddenLayers { get; set; } = { 500, 500, 2000 };

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(BatchSize, nameof(BatchSize));
        Require(Epochs, nameof(Epochs));
    }
}
