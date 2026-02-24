namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for VALLEX.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VALLEX model. Default values follow the original paper settings.</para>
/// </remarks>
public class VALLEXOptions : CodecTtsOptions { public VALLEXOptions() { SampleRate = 24000; NumCodebooks = 8; CodebookSize = 1024; CodecFrameRate = 75; LLMDim = 1024; NumLLMLayers = 12; } }
