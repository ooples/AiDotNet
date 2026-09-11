using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the RWKV4LanguageModel.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> RWKV-4 keeps a compact recurrent memory while training with
/// parallel token operations. Width and layer count trade memory and compute for capacity.</para>
/// <para>Peng et al., <i>RWKV: Reinventing RNNs for the Transformer Era</i> (2023), Table 2,
/// report 169M (width 768, 12 layers), 1.5B (2048, 24), 7B (4096, 32), and 14B (5120, 40)
/// configurations. Those are published sizing examples, not the defaults below: this library
/// retains its existing width 256 and four layers for compatibility. Matching size alone does
/// not reproduce pretrained weights or the full training recipe.</para>
/// </remarks>
/// <seealso href="https://arxiv.org/abs/2305.13048">Original RWKV paper.</seealso>
public class RWKV4Options : SequenceModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RWKV4Options"/> class carrying
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
    public RWKV4Options()
    {
        VocabSize = 50277;
        ModelDimension = 256;
        NumLayers = 4;
        MaxSequenceLength = 512;
    }

    /// <summary>Initializes an instance by copying every declared and inherited setting.</summary>
    /// <param name="other">The source options.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public RWKV4Options(RWKV4Options other) : base(other)
    {
    }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore(requiresHeads: false, requiresState: false);
    }
}
