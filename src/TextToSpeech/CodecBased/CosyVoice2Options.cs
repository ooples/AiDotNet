namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for CosyVoice2.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the CosyVoice2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class CosyVoice2Options : CodecTtsOptions { public CosyVoice2Options() { SampleRate = 22050; NumCodebooks = 1; CodebookSize = 4096; CodecFrameRate = 50; LLMDim = 1024; NumLLMLayers = 14; } }
