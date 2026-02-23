namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>Options for Llasa.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Llasa model. Default values follow the original paper settings.</para>
/// </remarks>
public class LlasaOptions : CodecTtsOptions { public LlasaOptions() { SampleRate = 16000; NumCodebooks = 1; CodebookSize = 8192; CodecFrameRate = 50; LLMDim = 2048; NumLLMLayers = 24; LanguageModelName = "LLaMA"; } }
