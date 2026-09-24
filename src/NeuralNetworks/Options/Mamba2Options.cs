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
/// 2.7 billion parameters; the defaults are the smallest of them, Mamba-2 130M.</para>
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
    /// <b>For Beginners:</b> You do not need to set any of these. They reproduce the smallest
    /// released Mamba-2 model (about 130 million parameters). Lower them for a faster, smaller
    /// model.
    /// </para>
    /// <para>
    /// Each value is the Mamba-2 130M checkpoint's. VocabSize, ModelDimension and NumLayers are its
    /// published config (state-spaces/mamba2-130m: vocab_size 50277, d_model 768, n_layer 24).
    /// StateDimension and NumHeads are the Mamba2 layer's own defaults in the authors'
    /// implementation, which that config does not override: d_state 128, and headdim 64 over an
    /// expand-2 inner width, so 2 x 768 / 64 = 24 heads. MaxSequenceLength is not a paper value -
    /// a state-space model has no positional limit and the checkpoint declares none - it is the
    /// library's bound on the sequence it prepares.
    /// </para>
    /// </remarks>
    public Mamba2Options()
    {
        VocabSize = 50277;
        ModelDimension = 768;
        NumLayers = 24;
        StateDimension = 128;
        NumHeads = 24;
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
