using AiDotNet.TextToSpeech.CodecBased;
namespace AiDotNet.TextToSpeech.VoiceCloning;
/// <summary>Options for XTTS v2 (GPT-2 based multilingual zero-shot TTS with VQ-VAE audio tokens).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the XTTSv2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class XTTSv2Options : CodecTtsOptions { public XTTSv2Options() { SampleRate = 24000; NumCodebooks = 1; CodebookSize = 8192; CodecFrameRate = 22; LLMDim = 1024; NumLLMLayers = 30; LanguageModelName = "GPT-2"; } }
