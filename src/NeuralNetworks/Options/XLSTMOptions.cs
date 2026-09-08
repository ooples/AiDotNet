using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the XLSTMLanguageModel.
/// </summary>
public class XLSTMOptions : SequenceModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="XLSTMOptions"/> class carrying
    /// this model's shipped defaults.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You do not need to set any of these. They are the values the
    /// model has always used, moved here from its constructor so they can be seen and
    /// changed in one place.
    /// </para>
    /// <para>
    /// Carried over unchanged. Whether each matches the published paper is verified, and
    /// corrected where it does not, in a later phase of issue #2090 — kept separate so a
    /// change in behaviour is never buried in a mechanical move.
    /// </para>
    /// </remarks>
    public XLSTMOptions()
    {
        VocabSize = 50277;
        ModelDimension = 256;
        NumLayers = 4;
        NumHeads = 8;
        MaxSequenceLength = 512;
    }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore(requiresHeads: true, requiresState: false);
    }
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
