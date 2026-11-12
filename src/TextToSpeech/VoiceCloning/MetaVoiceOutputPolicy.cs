using AiDotNet.Audio.Effects;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>AudioCraft loudness/compressor and the release's two distinct PCM16 conversions.</summary>
internal static class MetaVoiceOutputPolicy
{
    internal static double[] Normalize(double[] audio)
    {
        if (audio.Length == 0) throw new ArgumentException("A nonempty waveform is required.", nameof(audio));
        double rms = Math.Sqrt(audio.Average(x => x * x));
        if (rms < .002) return audio.Select(x => Math.Max(-1, Math.Min(1, x))).ToArray();
        double loudness = LoudnessMeter.IntegratedTorchaudioLoudness(audio, 24000);
        double gain = Math.Pow(10, (-14 - loudness) / 20);
        var result = audio.Select(x => Math.Tanh(gain * x)).ToArray();
        if (result.Any(x => double.IsNaN(x) || double.IsInfinity(x))) throw new InvalidOperationException("The original loudness policy produced a nonfinite waveform.");
        return result;
    }

    /// <summary>AudioCraft writes float32 samples through ffmpeg's saturating, nearest PCM16 conversion.</summary>
    internal static short[] DecoderPcm(double[] audio) => audio.Select(x =>
        (short)Math.Max(short.MinValue, Math.Min(short.MaxValue, Math.Round((double)(float)x * 32768, MidpointRounding.ToEven)))).ToArray();

    /// <summary>DF save_audio multiplies float32 by 32768, truncates and wraps to int16.</summary>
    internal static short[] EnhancedPcm(double[] audio) => audio.Select(x => unchecked((short)(int)((float)x * 32768f))).ToArray();
}
