namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for Zonos.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Zonos model. Default values follow the original paper settings.</para>
/// </remarks>
public class ZonosOptions : CodecTtsOptions { public ZonosOptions() { SampleRate = 44100; NumCodebooks = 9; CodebookSize = 1024; CodecFrameRate = 86; LLMDim = 1024; NumLLMLayers = 16; } }
