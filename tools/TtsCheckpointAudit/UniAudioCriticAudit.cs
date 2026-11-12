using System.Reflection;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;

internal static class UniAudioCriticAudit
{
    internal static void Run(string checkpoint, string fixturePath)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixturePath));
        var fixture = document.RootElement.GetProperty("float64_operators_derived_buffers");
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var options = UniAudioOptions.OfficialV1();
        // This audit isolates the complete released critics. The unused LM is small.
        options.LLMDim = options.TextEncoderDim = 12;
        options.NumHeads = 3;
        options.NumEncoderLayers = options.NumLLMLayers = 1;
        options.GlobalFeedForwardDim = options.LocalFeedForwardDim = 24;
        using var model = new UniAudio<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 4215), options);
        model.DisableAutoStreaming();
        model.LoadAuthorCodecDiscriminatorWeights(checkpoint);
        model.SetTrainingMode(false);
        var gan = (UniAudioCodecGan<double>)typeof(UniAudio<double>).GetField("_codecGan",
            BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        var realValues = Values(fixture.GetProperty("real"));
        var fakeValues = Values(fixture.GetProperty("fake"));
        var real = new Tensor<double>(new[] { realValues.Length }, new Vector<double>(realValues));
        var fake = new Tensor<double>(new[] { fakeValues.Length }, new Vector<double>(fakeValues));
        void Check(string label, Tensor<double> actual, double[] expected, double tolerance)
        {
            if (actual.Length != expected.Length) throw new InvalidDataException(label + " length mismatch.");
            double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"{label}: maximum error {error:R}.");
            if (!double.IsFinite(error) || error > tolerance)
                throw new InvalidDataException($"{label}: {error:R} exceeds {tolerance:R}.");
        }
        UniAudioCriticOutput<double> realOutput;
        using (new NoGradScope<double>()) realOutput = gan.Forward(real);
        using var tape = new GradientTape<double>();
        var fakeOutput = gan.Forward(fake);
        for (int i = 0; i < 6; i++)
        {
            Check($"trained critic {i} real", realOutput.Scores[i], Values(fixture.GetProperty("real_scores")[i]), 1e-8);
            Check($"trained critic {i} fake", fakeOutput.Scores[i], Values(fixture.GetProperty("fake_scores")[i]), 1e-8);
        }
        var generator = gan.AdversarialGeneratorObjective(realOutput, fakeOutput);
        var discriminator = gan.DiscriminatorObjective(realOutput, fakeOutput);
        Check("trained generator adversarial/feature loss", generator, new[] { fixture.GetProperty("generator").GetDouble() }, 1e-8);
        Check("trained discriminator loss", discriminator, new[] { fixture.GetProperty("discriminator").GetDouble() }, 1e-8);
        Check("trained critic waveform gradient", tape.ComputeGradients(generator, sources: new[] { fake })[fake],
            Values(fixture.GetProperty("gradient")), 1e-7);
        Console.WriteLine("PASS: all six original trained UniAudio critics, G/D objectives and waveform gradient.");
    }
}
