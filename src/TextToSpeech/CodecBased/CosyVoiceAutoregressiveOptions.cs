namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Shared settings for the standard CosyVoice language models and their repetition-aware sampler.</summary>
public abstract class CosyVoiceAutoregressiveOptions : CosyVoiceConformerOptions
{
    /// <summary>Creates the released sampler defaults.</summary>
    protected CosyVoiceAutoregressiveOptions() { }

    /// <summary>Copies the common components and standard language-model sampler.</summary>
    protected CosyVoiceAutoregressiveOptions(CosyVoiceAutoregressiveOptions other) : base(other)
    {
        TopP = other.TopP;
        TopK = other.TopK;
        RepetitionWindow = other.RepetitionWindow;
        RepetitionRatio = other.RepetitionRatio;
    }

    /// <summary>The LM sampler's nucleus (top_p).</summary>
    public double TopP { get; set; } = 0.8;

    /// <summary>The LM sampler's candidate count (top_k).</summary>
    public int TopK { get; set; } = 25;

    /// <summary>The repetition-aware sampler's window (win_size).</summary>
    public int RepetitionWindow { get; set; } = 10;

    /// <summary>The repetition-aware sampler's threshold ratio (tau_r).</summary>
    public double RepetitionRatio { get; set; } = 0.1;
}
