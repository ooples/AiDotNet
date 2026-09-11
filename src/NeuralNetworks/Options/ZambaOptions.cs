using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configures the dimensions and attention spacing of the Zamba language model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Zamba uses a recurrent backbone for compact memory and adds
/// attention periodically to recover token-level information. Width and depth determine
/// model capacity, while the attention interval controls how often the two mechanisms mix.</para>
/// <para>Glorioso et al., <i>Zamba: A Compact 7B SSM Hybrid Model</i> (2024), describe a
/// seven-billion-parameter Mamba backbone with a shared attention module. These options
/// retain the library's existing 3712-wide, 76-layer configuration. Matching those dimensions
/// alone does not establish checkpoint, weight-sharing, or training-recipe equivalence.</para>
/// </remarks>
/// <seealso href="https://arxiv.org/abs/2405.16712">Original Zamba paper.</seealso>
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

    /// <summary>Initializes an instance by copying every declared and inherited setting.</summary>
    /// <param name="other">The source options.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public ZambaOptions(ZambaOptions other) : base(other)
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
