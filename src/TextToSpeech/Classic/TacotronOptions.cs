namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Tacotron (attention-based seq2seq TTS).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Tacotron model. Default values follow the original paper settings.</para>
/// </remarks>
public class TacotronOptions : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TacotronOptions(TacotronOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        CbhgBankSize = other.CbhgBankSize;
        PrenetDim = other.PrenetDim;
    }

    public TacotronOptions() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 3; NumDecoderLayers = 2; NumHeads = 1; OutputsPerStep = 2; UsePostnet = true; }

    /// <summary>Gets or sets the CBHG encoder bank size.</summary>
    public int CbhgBankSize { get; set; } = 16;

    /// <summary>Gets or sets the prenet dim for decoder.</summary>
    public int PrenetDim { get; set; } = 256;
}
