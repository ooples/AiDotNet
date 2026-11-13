using System.Text.Json;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.StyleEmotion;
using AiDotNet.TextToSpeech.VoiceCloning;

internal static class MetaVoiceOutputAudit
{
    internal static void Run(string fixture)
    {
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture)); var reference = document.RootElement;
        using var decoder = JsonDocument.Parse(File.ReadAllText(reference.GetProperty("decoder_reference").GetString()!));
        static double[] Values(JsonElement item) => item.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var normalized = MetaVoiceOutputPolicy.Normalize(Values(decoder.RootElement.GetProperty("waveform")));
        double Error(double[] actual, string name) => actual.Zip(Values(reference.GetProperty(name)), (a, b) => Math.Abs(a - b)).Max();
        Console.WriteLine($"Output loudness error {Error(normalized, "normalized"):R}.");
        var pcm = MetaVoiceOutputPolicy.DecoderPcm(normalized);
        var expectedPcm = reference.GetProperty("decoder_pcm").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        int pcmDifference = pcm.Zip(expectedPcm, (a, b) => Math.Abs(a - b)).Max();
        Console.WriteLine($"Output decoder PCM maximum code difference {pcmDifference}; mismatches {pcm.Zip(expectedPcm, (a, b) => a != b).Count(x => x)}.");
        if (pcmDifference != 0) throw new InvalidDataException("Decoder PCM differs.");
        var audio = new Tensor<double>(new[] { pcm.Length }, new Vector<double>(pcm.Select(x => x / 32768.0)));
        var resampled = new MetaVoiceSincFastResample<double>(AiDotNetEngine.Current).Forward(audio);
        double resampleError = Error(resampled.ToArray(), "resampled");
        Console.WriteLine($"Output resampler error {resampleError:R}.");
        if (resampleError > 2e-10) throw new InvalidDataException("Resampled audio differs.");
        var enhancer = new MetaVoiceDeepFilterEnhancer<double>(reference.GetProperty("df_checkpoint").GetString()!);
        var enhanced = enhancer.Enhance48k(resampled);
        double enhancedError = Error(enhanced.ToArray(), "enhanced");
        Console.WriteLine($"Output trained enhancement error {enhancedError:R}.");
        if (enhancedError > 2e-6) throw new InvalidDataException("Enhanced waveform differs.");
        var output = MetaVoiceOutputPolicy.EnhancedPcm(enhanced.ToArray()).Select(x => x / 32768.0).ToArray();
        double outputError = Error(output, "waveform");
        Console.WriteLine($"Output final PCM waveform error {outputError:R}.");
        if (outputError > 1.0 / 32768) throw new InvalidDataException("Final enhanced PCM waveform differs.");
    }
}
