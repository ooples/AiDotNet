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

    /// <summary>Initializes an instance by copying every declared and inherited setting.</summary>
    /// <param name="other">The source options.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public Zamba2Options(Zamba2Options other) : base(other)
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
        ValidateCore(requiresHeads: true, requiresState: true);
        Require(AttentionInterval, nameof(AttentionInterval));
    }
}
