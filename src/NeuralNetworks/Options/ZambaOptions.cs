using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the ZambaLanguageModel.
/// </summary>
public class ZambaOptions : SequenceModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ZambaOptions"/> class carrying
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
    public ZambaOptions()
    {
        VocabSize = 32000;
        ModelDimension = 3712;
        NumLayers = 76;
        StateDimension = 16;
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
        ValidateCore(requiresHeads: false, requiresState: true);
    }
}
