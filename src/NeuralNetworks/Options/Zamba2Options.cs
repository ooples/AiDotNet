using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the Zamba2LanguageModel.
/// </summary>
public class Zamba2Options : SequenceModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Zamba2Options"/> class carrying
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
    public Zamba2Options()
    {
        VocabSize = 32000;
        ModelDimension = 3584;
        NumLayers = 81;
        StateDimension = 64;
        NumHeads = 32;
        AttentionInterval = 6;
        MaxSequenceLength = 4096;
    }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore(requiresHeads: true, requiresState: true);
    }
}
