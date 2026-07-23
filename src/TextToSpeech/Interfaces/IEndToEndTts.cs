namespace AiDotNet.TextToSpeech.Interfaces;

/// <summary>
/// Interface for end-to-end TTS models that generate waveforms directly from text
/// without a separate vocoder stage.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// End-to-end models combine acoustic modeling and waveform generation in a single model:
/// Text -> [Single Model] -> Waveform.
/// Architectures include:
/// <list type="bullet">
/// <item>VITS: VAE + normalizing flow + adversarial training</item>
/// <item>VITS2: improved alignment and duration prediction</item>
/// <item>YourTTS: multilingual zero-shot VITS variant</item>
/// <item>Piper: lightweight VITS for edge deployment</item>
/// </list>
/// </para>
/// </remarks>
public interface IEndToEndTts<T> : ITtsModel<T>
{
    /// <summary>
    /// Gets the hidden dimension of the model's internal representation.
    /// </summary>
    int HiddenDim { get; }

    /// <summary>
    /// Gets the number of flow steps used in the posterior encoder (for VAE-based models).
    /// </summary>
    int NumFlowSteps { get; }
}
