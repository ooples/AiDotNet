namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for SeedTTS.</summary>
/// <remarks>
/// <para>
/// The Seed-TTS technical report describes a large autoregressive transformer followed by a token
/// diffusion model and acoustic vocoder, but does not disclose reconstructible widths, depths,
/// codebook counts, or optimizer hyperparameters. These defaults therefore provide a practical
/// large native approximation; every exposed value can be customized for a released checkpoint.
/// </para>
/// <para><b>For Beginners:</b> These options control the size and training behavior of the native SeedTTS approximation.</para>
/// </remarks>
public class SeedTTSOptions : CodecTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    public SeedTTSOptions(SeedTTSOptions other)
        : base(other)
    {
    }

    public SeedTTSOptions()
    {
        SampleRate = 24000;
        NumCodebooks = 8;
        CodebookSize = 1024;
        CodecFrameRate = 50;
        LLMDim = 2048;
        NumLLMLayers = 24;
    }
}
