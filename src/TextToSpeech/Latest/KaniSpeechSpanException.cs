namespace AiDotNet.TextToSpeech.Latest;

/// <summary>The generated token sequence cannot be decoded as complete Kani speech frames.</summary>
internal sealed class KaniSpeechSpanException : InvalidOperationException
{
    internal KaniSpeechSpanException(string message) : base(message) { }
}
