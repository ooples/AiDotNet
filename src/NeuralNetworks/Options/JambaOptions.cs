using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configures the dimensions and attention spacing of the Jamba language model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Jamba combines a compact recurrent memory with attention,
/// which can look back at individual tokens. <see cref="SequenceModelOptions.AttentionInterval"/>
/// controls how frequently this model inserts an attention block.</para>
/// <para>Lieber et al., <i>Jamba: A Hybrid Transformer-Mamba Language Model</i> (2024),
/// describe interleaved Transformer/Mamba layers and mixture-of-experts feed-forward layers.
/// The original release has 52 billion total parameters and 12 billion active per token.
/// These options expose the library's sequence dimensions, not the paper's complete MoE
/// configuration. The existing 256-wide, eight-layer defaults remain small library defaults,
/// not a reproduction of that checkpoint.</para>
/// </remarks>
/// <seealso href="https://arxiv.org/abs/2403.19887">Original Jamba paper.</seealso>
public class JambaOptions : SequenceModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="JambaOptions"/> class carrying
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
    public JambaOptions()
    {
        VocabSize = 65536;
        ModelDimension = 256;
        NumLayers = 8;
        StateDimension = 16;
        AttentionInterval = 8;
        MaxSequenceLength = 512;
    }

    /// <summary>Initializes an instance by copying every declared and inherited setting.</summary>
    /// <param name="other">The source options.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public JambaOptions(JambaOptions other) : base(other)
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
