using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configures the dimensions and default optimizer learning rate of the xLSTM language model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> xLSTM extends the familiar recurrent memory cell with more
/// expressive gates and memory. Width, depth, and head count control the model's capacity;
/// the learning rate controls the size of each training step.</para>
/// <para>Beck et al., <i>xLSTM: Extended Long Short-Term Memory</i> (2024), introduce
/// exponential gating, scalar-memory sLSTM, and matrix-memory mLSTM blocks. The original
/// language-model experiments include 125M, 350M, 760M, and 1.3B configurations. These
/// options retain the library's 256-wide, four-layer defaults rather than claiming checkpoint
/// equivalence, and do not configure every block-mixture choice in the paper.</para>
/// </remarks>
/// <seealso href="https://arxiv.org/abs/2405.04517">Original xLSTM paper.</seealso>
public class XLSTMOptions : SequenceModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="XLSTMOptions"/> class carrying
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
    public XLSTMOptions()
    {
        VocabSize = 50277;
        ModelDimension = 256;
        NumLayers = 4;
        NumHeads = 8;
        MaxSequenceLength = 512;
    }

    /// <summary>Initializes an instance by copying every declared and inherited setting.</summary>
    /// <param name="other">The source options.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public XLSTMOptions(XLSTMOptions other) : base(other)
    {
        LearningRate = other.LearningRate;
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
        Require(LearningRate, nameof(LearningRate));
    }
    /// <summary>
    /// AdamW learning rate. Defaults to 1e-3, the rate used for the xLSTM language models in
    /// Beck et al., 2024 (S4.1).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How big a step the model takes each time it learns. The default is
    /// the value from the xLSTM paper. Lower it if training becomes unstable; raise it if the loss
    /// barely moves.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-3;
}
