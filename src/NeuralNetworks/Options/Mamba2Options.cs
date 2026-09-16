using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configures the dimensions, state width, and head count of the Mamba-2 language model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Mamba-2 keeps a running summary instead of retaining an
/// attention score for every earlier token. Its state width controls the size of that memory;
/// its head count divides the internal computation into groups.</para>
/// <para>Dao and Gu, <i>Transformers are SSMs: Generalized Models and Efficient Algorithms
/// Through Structured State Space Duality</i> (2024), introduce the structured state-space
/// duality formulation used by Mamba-2. Released configurations range from 130 million to
/// 2.7 billion parameters. The existing 256-wide, four-layer defaults are the library's small
/// configuration, not the dimensions of those pretrained checkpoints.</para>
/// </remarks>
/// <seealso href="https://arxiv.org/abs/2405.21060">Original Mamba-2 paper.</seealso>
/// <seealso href="https://github.com/state-spaces/mamba">Authors' implementation and checkpoint roster.</seealso>
public class Mamba2Options : SequenceModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Mamba2Options"/> class carrying
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
    public Mamba2Options()
    {
        VocabSize = 50277;
        ModelDimension = 256;
        NumLayers = 4;
        StateDimension = 64;
        NumHeads = 8;
        MaxSequenceLength = 512;
    }

    /// <summary>Initializes an instance by copying every declared and inherited setting.</summary>
    /// <param name="other">The source options.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public Mamba2Options(Mamba2Options other) : base(other)
    {
    }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    internal void Validate()
    {
        ValidateCore(requiresHeads: true, requiresState: true);
    }
}
