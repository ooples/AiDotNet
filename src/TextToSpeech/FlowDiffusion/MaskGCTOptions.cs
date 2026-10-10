using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for MaskGCT.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MaskGCT model. Default values follow the original paper settings.</para>
/// </remarks>
public class MaskGCTOptions : CodecTtsOptions
{
    public MaskGCTOptions()
    {
        SampleRate = 24000;
        NumCodebooks = 8;
        CodebookSize = 1024;
        CodecFrameRate = 50;
        LLMDim = 1024;
        NumLLMLayers = 16;
    }

    /// <summary>
    /// Gets or sets the number of steps over which the learning rate warms up.
    /// </summary>
    /// <value>Defaults to 32000, the paper's warmup (Wang et al. 2024, Sec. 4).</value>
    /// <remarks>
    /// The recipe ramps the learning rate up to 1e-4 over this many steps, so a run far shorter than the
    /// warmup barely trains. Lower it for short fine-tuning runs; leave it at the paper value to reproduce
    /// the paper.
    /// </remarks>
    public int WarmupSteps { get; set; } = 32000;
}
