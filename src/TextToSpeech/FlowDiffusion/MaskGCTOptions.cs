using AiDotNet.TextToSpeech.CodecBased;
namespace AiDotNet.TextToSpeech.FlowDiffusion;
/// <summary>Options for MaskGCT.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MaskGCT model. Default values follow the original paper settings.</para>
/// </remarks>
public class MaskGCTOptions : CodecTtsOptions { public MaskGCTOptions() { SampleRate = 24000; NumCodebooks = 8; CodebookSize = 1024; CodecFrameRate = 50; LLMDim = 1024; NumLLMLayers = 16; } }
