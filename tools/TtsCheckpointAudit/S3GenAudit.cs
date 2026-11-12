using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.VoiceCloning;

internal static class S3GenAudit
{
    internal static void Cam(string directory, string fixture)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        var engine = AiDotNetEngine.Current;
        using var document = JsonDocument.Parse(File.ReadAllText(fixture + ".cam-precision.json"));
        var root = document.RootElement;
        double[] Values(string name) => root.GetProperty(name).EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var layers = new List<AiDotNet.NeuralNetworks.Layers.LayerBase<double>>();
        var model = new AiDotNet.Audio.Speaker.CamPlusPlusEncoder<double>(engine, layers,
            new AiDotNet.Audio.Speaker.CamPlusPlusConfiguration(), new Random(1));
        using var stream = File.OpenRead(Path.Combine(directory, "s3gen.safetensors"));
        var weights = AiDotNet.Agentic.Models.Local.SafetensorsReader.Read(stream);
        model.LoadTorchWeights((name, shape) =>
        {
            name = "speaker_encoder." + name;
            return weights.Get(name)!.DataType == "I64"
                ? new[] { (double)BitConverter.ToInt64(weights.GetRawBytes(name), 0) }
                : weights.ReadAsDouble(name);
        });
        model.SetTrainingMode(false);
        var audio = Values("audio16");
        using var originalDocument = JsonDocument.Parse(File.ReadAllText(fixture));
        var originalAudio = originalDocument.RootElement.GetProperty("audio").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var resampled = AiDotNet.Helpers.AudioHelper<double>.ResampleBandLimited(engine,
            new Tensor<double>(new[] { originalAudio.Length }, new Vector<double>(originalAudio)), 24000, 16000);
        double resampleError = resampled.ToArray().Zip(audio, (a,b) => Math.Abs(a-b)).Max();
        Console.WriteLine($"Original float64 resampling maximum error {resampleError:R}.");
        var fbank = new AiDotNet.Audio.Features.KaldiFbank<double>(engine, 16000, 80, AiDotNet.Audio.Features.WindowType.Povey);
        var features = fbank.Forward(resampled);
        features = engine.TensorSubtract(features, engine.TensorBroadcastTo(engine.ReduceMean(features, new[] { 0 }, true), features.Shape.ToArray()));
        var expected = Values("embedding");
        double Error(Tensor<double> actual, double[] target) => actual.ToArray().Zip(target, (a,b) => Math.Abs(a-b)).Max();
        double featureError = Error(features, Values("features"));
        var authorFeatures = new Tensor<double>(new[] { 1, root.GetProperty("frames").GetInt32(), 80 }, new Vector<double>(Values("features")));
        double networkError = Error(model.Forward(authorFeatures), expected);
        double chainError = Error(model.Forward(engine.Reshape(features, authorFeatures.Shape.ToArray())), expected);
        Console.WriteLine($"Original float64 Kaldi features {featureError:R}; released CAM network {networkError:R}; native feature chain {chainError:R}.");
        if (resampleError > 1e-10 || featureError > 1e-7 || networkError > 1e-7 || chainError > 1e-7) throw new InvalidDataException("CAM precision comparison failed.");
    }

    internal static void Mel(string fixture)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        double[] audio = document.RootElement.GetProperty("audio").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var mel = new AiDotNet.TextToSpeech.Vocoders.DifferentiableMel<double>(AiDotNetEngine.Current, 24000, 1920, 480, 1920, 80, 0, 8000);
        var result = mel.Forward(new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio)));
        File.WriteAllText(fixture + ".native-mel.json", JsonSerializer.Serialize(result.ToArray()));
        Console.WriteLine("Wrote native double-precision mel for frontend arithmetic diagnosis.");
    }

    internal static void Run(string directory, string fixture)
    {
        Environment.SetEnvironmentVariable("OMP_NUM_THREADS", "1");
        Environment.SetEnvironmentVariable("OPENBLAS_NUM_THREADS", "1");
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        var root = document.RootElement;
        var failures = new List<string>();
        using var melPrecision = JsonDocument.Parse(File.ReadAllText(fixture + ".mel-precision.json"));
        using var camPrecision = JsonDocument.Parse(File.ReadAllText(fixture + ".cam-precision.json"));
        double[] Values(string key) => root.GetProperty(key).EnumerateArray().Select(x => x.GetDouble()).ToArray();
        Tensor<double> Tensor(string key, params int[] shape) => new(shape, new Vector<double>(Values(key)));
        void Check(string label, Tensor<double> actual, double[] expected, double tolerance, bool reportOnly = false)
        {
            if (actual.Length != expected.Length) throw new InvalidDataException($"{label}: length {actual.Length} != {expected.Length}.");
            double error = actual.ToArray().Zip(expected, (a,b) => Math.Abs(a-b)).Max();
            Console.WriteLine($"{label}: maximum error {error:R}.");
            if (!double.IsFinite(error) || (!reportOnly && error > tolerance)) failures.Add($"{label}: {error:R} exceeds {tolerance:R}");
        }
        Console.WriteLine("Creating full released Chatterbox graph for S3Gen audit.");
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        using var model = new Chatterbox<double>(architecture, new ChatterboxOptions { TokenizerPath = Path.Combine(directory, "tokenizer.json") });
        // Numerical fidelity requires the original precision, not automatic inference-store quantization.
        model.DisableAutoStreaming();
        Console.WriteLine("Loading all original checkpoint files through the public loader.");
        model.LoadPretrainedWeights(directory);
        Console.WriteLine("Original checkpoint import complete; setting evaluation mode without streaming.");
        model.SetTrainingMode(false);
        GC.Collect(); GC.WaitForPendingFinalizers();
        Console.WriteLine("Preparing reference audio through the public CreateVoice API.");
        var voice = model.CreateVoice(Tensor("audio", Values("audio").Length), 24000);
        Check("S3 tokenizer IDs", voice.ReferenceCodecTokens!, Values("prompt_tokens"), 0);
        Check("S3 mel / released float32 (reported precision difference)", voice.Reference!, Values("prompt_mel"), 2e-5, reportOnly: true);
        Check("S3 CAM++ / released float32 (reported precision difference)", voice.SpeakerEmbedding!, Values("speaker"), 2e-4, reportOnly: true);
        Check("S3 mel / independently executed float64 author-function variant", voice.Reference!,
            melPrecision.RootElement.GetProperty("float64_with_float64_buffers").EnumerateArray().Select(x => x.GetDouble()).ToArray(), 2e-8);
        Check("S3 CAM++ / independently executed float64 author-network variant", voice.SpeakerEmbedding!,
            camPrecision.RootElement.GetProperty("embedding").EnumerateArray().Select(x => x.GetDouble()).ToArray(), 1e-7);
        int frames = root.GetProperty("prompt_frames").GetInt32();
        int[] tokens = Values("target_tokens").Select(x => (int)x).ToArray();
        int[] prompt = Values("prompt_tokens").Select(x => (int)x).ToArray();
        var noise = Tensor("noise", 1, 80, frames + tokens.Length * 2);
        var mel = model.FlowModel!.Inference(tokens, prompt, Tensor("prompt_mel", frames, 80), Tensor("speaker", 192), noise);
        var expectedMel = root.GetProperty("float32").GetProperty("mel").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        Check("Released full flow / author reference", mel, expectedMel, 2e-4);
        mel = model.FlowModel.Inference(tokens, prompt, voice.Reference, voice.SpeakerEmbedding!, noise);
        Check("Released full flow / native reference versus released float32 (reported precision difference)", mel, expectedMel, 2e-4, reportOnly: true);
        Check("Released full flow / native reference with precision-matched author conditioning", mel,
            root.GetProperty("float32_with_precision_matched_conditioning").GetProperty("mel").EnumerateArray().Select(x => x.GetDouble()).ToArray(), 2e-4);
        var authorMel = new Tensor<double>(new[] { 1, 80, tokens.Length * 2 }, new Vector<double>(expectedMel));
        Check("Released HiFT F0", model.Vocoder!.PredictF0(authorMel), Values("f0"), 2e-3);
        var waveform = model.Vocoder.Decode(authorMel, Tensor("excitation", 1, 1, Values("excitation").Length));
        Check("Released HiFT waveform / controlled author excitation", waveform, Values("waveform"), 2e-4);
        if (failures.Count > 0) throw new InvalidDataException(string.Join(Environment.NewLine, failures));
        Console.WriteLine("Released S3Gen checkpoint audit passed.");
    }
}
