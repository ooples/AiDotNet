using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for <see cref="MambaLanguageModel{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Mamba is a language model that reads text left to right while
/// carrying a compact running summary of everything it has seen, instead of re-reading the
/// whole passage at every step the way a transformer does. That makes it fast on long text.
/// You do not need to set anything here — the defaults come with the model.
/// </para>
/// <para>
/// <b>Values are carried over unchanged from the constructor parameters they replace and are
/// NOT yet the paper's.</b> Mamba-130M is 768 wide with 24 layers; the values below are the
/// demo-sized ones this model has always shipped. Correcting them is a deliberate behavioural
/// change, made in its own phase of issue #2090 so it is reviewed apart from this mechanical
/// move.
/// </para>
/// </remarks>
/// <seealso href="https://arxiv.org/abs/2312.00752">
/// Mamba: Linear-Time Sequence Modeling with Selective State Spaces
/// </seealso>
public class MambaOptions : SequenceModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MambaOptions"/> class with this model's
    /// shipped defaults.
    /// </summary>
    public MambaOptions()
    {
        VocabSize = 50277;
        ModelDimension = 256;
        NumLayers = 4;
        StateDimension = 16;
        ExpandFactor = 2;
        MaxSequenceLength = 512;
    }

    /// <summary>Initializes an instance by copying every declared and inherited setting.</summary>
    /// <param name="other">The source options.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public MambaOptions(MambaOptions other) : base(other)
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
        ValidateCore(requiresHeads: false, requiresState: true);
        Require(ExpandFactor, nameof(ExpandFactor));
    }
}
