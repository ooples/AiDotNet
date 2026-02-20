namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for Tacotron (attention-based seq2seq TTS).</summary>
public class TacotronOptions : AcousticModelOptions
{
    public TacotronOptions() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 3; NumDecoderLayers = 2; NumHeads = 1; OutputsPerStep = 2; UsePostnet = true; }

    /// <summary>Gets or sets the CBHG encoder bank size.</summary>
    public int CbhgBankSize { get; set; } = 16;

    /// <summary>Gets or sets the prenet dim for decoder.</summary>
    public int PrenetDim { get; set; } = 256;
}
