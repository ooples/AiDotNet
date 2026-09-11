using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the RWKV7LanguageModel.
/// </summary>
public class RWKV7Options : SequenceModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RWKV7Options"/> class carrying
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
    public RWKV7Options()
    {
        VocabSize = 65536;
        ModelDimension = 256;
        NumLayers = 4;
        NumHeads = 4;
        FfnMultiplier = 3.5;
        MaxSequenceLength = 512;
    }

    /// <summary>Initializes an instance by copying every declared and inherited setting.</summary>
    /// <param name="other">The source options.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public RWKV7Options(RWKV7Options other) : base(other)
    {
    }

    /// <summary>
    /// Throws if a required model dimension or consumed training setting is invalid.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is non-positive, or a consumed numeric setting is
    /// non-finite or outside its supported range. The message identifies the invalid property.
    /// </exception>
    public void Validate()
    {
        ValidateCore(requiresHeads: true, requiresState: false);
        Require(FfnMultiplier, nameof(FfnMultiplier));
    }
}
