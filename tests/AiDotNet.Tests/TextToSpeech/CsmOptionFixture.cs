using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Safety.Watermarking;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class CsmOptionFixture
{
    internal static bool Handles(string option) => option is "Temperature" or "TopK" or "MimiHeads"
        or "MimiContext" or "Watermark" or "WatermarkKey" or "WatermarkSdr";

    internal static void Prepare(CSMOptions options, string option)
    {
        if (option == "TopK") options.TopK = 3;
        if (option == "MimiContext") options.MimiContext = 2;
        if (option is "WatermarkKey" or "WatermarkSdr") options.Watermark = true;
    }

    internal static long Observe<T>(CSM<T> model, Tensor<T> input, string option)
    {
        var numeric = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        long Fingerprint(Tensor<T> value) => OptionSensitivityHarness.Fingerprint(value.Select(numeric.ToDouble));
        if (option is "MimiHeads" or "MimiContext")
        {
            // Observe the codec before nearest-codebook quantization can hide attention changes.
            var audio = new Tensor<T>(new[] { 1, 1, 9600 });
            for (int i = 0; i < audio.Length; i++)
                audio[i] = numeric.FromDouble(.2 * Math.Sin(i * .07) + .1 * Math.Cos(i * i * .0001));
            return Fingerprint(model.Codec.Latent(audio));
        }
        if (option is "Temperature" or "TopK")
        {
            var observations = new List<long>();
            model.SamplingProbabilities = probabilities => observations.Add(OptionSensitivityHarness.Fingerprint(probabilities));
            try { model.Predict(input); }
            finally { model.SamplingProbabilities = null; }
            return OptionSensitivityHarness.Combine(observations.ToArray());
        }
        // Fit the real compact SilentCipher network used by its independent reference test.
        var reference = SilentCipherReferenceTests.Load("silentcipher_reference.json");
        var configuration = new SilentCipherConfiguration
        {
            Fft = reference.GetProperty("fft").GetInt32(), Hop = reference.GetProperty("hop").GetInt32(),
            MessageBandSize = reference.GetProperty("band").GetInt32(),
            MessageDecoderChannels = reference.GetProperty("message_channels").GetInt32(),
            MessageDecoderLayers = reference.GetProperty("message_layers").GetInt32()
        };
        var network = new SilentCipherNetwork<T>(AiDotNetEngine.Current, configuration, new List<LayerBase<T>>());
        Func<string, double[]?> Source(string module)
        {
            var shapes = reference.GetProperty("shapes").GetProperty(module).EnumerateObject()
                .ToDictionary(p => p.Name, p => p.Value.EnumerateArray().Select(v => v.GetInt32()).ToArray());
            return name => shapes.TryGetValue(name, out var shape) ? SilentCipherReferenceTests.ClosedForm(name, shape) : null;
        }
        network.Load(Source("enc_c"), Source("dec_c"), Source("dec_m"));
        typeof(CSM<T>).GetField("_watermarker", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
            .SetValue(model, network);
        return Fingerprint(model.Predict(input));
    }
}
