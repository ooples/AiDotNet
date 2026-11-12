using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>DeepFilterNet's effective pinned sinc_fast 24-to-48 kHz filter.
/// Torchaudio 2.1's legacy alias reaches Kaiser rather than its modern Hann branch.</summary>
internal sealed class MetaVoiceSincFastResample<T>
{
    private readonly IEngine _engine;
    private readonly Tensor<T> _kernel;
    internal MetaVoiceSincFastResample(IEngine engine)
    {
        _engine = engine; var numeric = MathHelper.GetNumericOperations<T>();
        var values = typeof(T) == typeof(float) ? MetaVoiceSincFastCoefficients.SinglePrecision : MetaVoiceSincFastCoefficients.DoublePrecision;
        _kernel = new Tensor<T>(new[] { 2, 1, 1, 35 }, new Vector<T>(values.Select(x => numeric.FromDouble(x))));
    }

    internal Tensor<T> Forward(Tensor<T> audio)
    {
        if (audio.Rank != 1 || audio.Length == 0) throw new ArgumentException("Expected nonempty mono 24 kHz samples.", nameof(audio));
        var padded = VocoderOps.ZeroPad(_engine, _engine.Reshape(audio, new[] { 1, 1, audio.Length }), 17, 18);
        var filtered = _engine.Conv2D(_engine.Reshape(padded, new[] { 1, 1, 1, padded.Shape[2] }), _kernel,
            new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        int frames = filtered.Shape[3];
        var interleaved = _engine.Reshape(_engine.TensorTranspose(_engine.Reshape(filtered, new[] { 2, frames })), new[] { 2 * frames });
        return _engine.TensorSlice(interleaved, new[] { 0 }, new[] { 2 * audio.Length });
    }
}
