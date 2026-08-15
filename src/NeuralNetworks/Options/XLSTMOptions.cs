using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the XLSTMLanguageModel.
/// </summary>
public class XLSTMOptions : NeuralNetworkOptions
{
    /// <summary>
    /// AdamW learning rate. Defaults to 1e-3, the rate used for the xLSTM language models in
    /// Beck et al., 2024 (S4.1).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How big a step the model takes each time it learns. The default is
    /// the value from the xLSTM paper. Lower it if training becomes unstable; raise it if the loss
    /// barely moves.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-3;
}
