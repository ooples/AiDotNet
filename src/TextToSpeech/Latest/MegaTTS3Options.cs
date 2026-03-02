using AiDotNet.TextToSpeech.CodecBased;
namespace AiDotNet.TextToSpeech.Latest;
/// <summary>Options for MegaTTS3.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MegaTTS3 model. Default values follow the original paper settings.</para>
/// </remarks>
public class MegaTTS3Options : CodecTtsOptions { public MegaTTS3Options() { SampleRate = 24000; NumCodebooks = 1; CodebookSize = 4096; CodecFrameRate = 50; LLMDim = 2048; NumLLMLayers = 24; } }
