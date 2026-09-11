using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configures the dimensions and attention spacing of the Samba language model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Samba combines a running memory for older information with
/// attention to recent tokens. Its attention interval controls how often recurrent blocks
/// are interleaved with attention blocks.</para>
/// <para>Ren et al., <i>Samba: Simple Hybrid State Space Models for Efficient Unlimited
/// Context Language Modeling</i> (2024), combine Mamba with sliding-window attention and
/// evaluate models up to 3.8 billion parameters. The existing 256-wide, eight-layer defaults
/// are a small library configuration, not the published 3.8B checkpoint. These settings do
/// not independently specify the paper's entire training recipe or attention-window policy.</para>
/// </remarks>
/// <seealso href="https://arxiv.org/abs/2406.07522">Original Samba paper.</seealso>
public class SambaOptions : SequenceModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="SambaOptions"/> class carrying
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
    public SambaOptions()
    {
        VocabSize = 32000;
        ModelDimension = 256;
        NumLayers = 8;
        StateDimension = 16;
        AttentionInterval = 2;
        MaxSequenceLength = 512;
    }

    /// <summary>Initializes an instance by copying every declared and inherited setting.</summary>
    /// <param name="other">The source options.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public SambaOptions(SambaOptions other) : base(other)
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
        ValidateCore(requiresHeads: false, requiresState: true);
        Require(AttentionInterval, nameof(AttentionInterval));
    }
}
