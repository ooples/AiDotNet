using AiDotNet.TextToSpeech.CodecBased;
namespace AiDotNet.TextToSpeech.FlowDiffusion;
/// <summary>Options for F5TTS.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the F5TTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class F5TTSOptions : CodecTtsOptions { public F5TTSOptions() { SampleRate = 24000; NumCodebooks = 1; CodebookSize = 4096; CodecFrameRate = 75; LLMDim = 1024; NumLLMLayers = 22; } }
