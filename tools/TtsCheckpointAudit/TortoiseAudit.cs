using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;

internal static class TortoiseAudit
{
    internal static void Run(string directory, string fixture)
    {
        Environment.SetEnvironmentVariable("OMP_NUM_THREADS", "1");
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        var root = document.RootElement;
        Console.WriteLine($"TorToiSe original checkpoint {root.GetProperty("revision").GetString()}, code {root.GetProperty("code_revision").GetString()}.");
        var failures = new List<string>();
        static double[] Values(JsonElement array) => array.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        static Tensor<double> Tensor(JsonElement array, params int[] shape) => new(shape, new Vector<double>(Values(array)));
        void Check(string label, Tensor<double> actual, JsonElement expected, double tolerance)
        {
            var values = Values(expected);
            if (actual.Length != values.Length) throw new InvalidDataException($"{label}: output lengths differ.");
            double error = actual.ToArray().Zip(values, (a,b) => Math.Abs(a-b)).Max();
            Console.WriteLine($"{label}: maximum error {error:R}.");
            if (!double.IsFinite(error) || error > tolerance) failures.Add($"{label}: {error:R} exceeds {tolerance:R}");
        }
        Func<string,int[],double[]> Reader(string file, string prefix = "")
        {
            Console.WriteLine($"Loading original {file} with WeightLoader.");
            var state = new WeightLoader().LoadWeights(Path.Combine(directory, ".models", file));
            return (name, shape) =>
            {
                if (!state.TryGetValue(prefix + name, out var tensor)) throw new InvalidDataException($"Missing original {file} tensor {prefix + name}.");
                if (!tensor.Shape.ToArray().SequenceEqual(shape) && !(shape.Length == 0 && tensor.Length == 1))
                    throw new InvalidDataException($"Original {file} tensor {prefix + name} has incompatible axes [{string.Join(",", tensor.Shape)}] vs [{string.Join(",", shape)}].");
                return tensor.ToArray().Select(x => (double)x).ToArray();
            };
        }
        var engine = AiDotNetEngine.Current;
        var options = new TortoiseTTSOptions();
        var ar = root.GetProperty("ar");
        int[] text = ar.GetProperty("text").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        int[] speech = ar.GetProperty("speech").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        void Ar()
        {
            var layers = new List<LayerBase<double>>();
            var model = new TortoiseAutoregressive<double>(engine, layers, options);
            model.LoadTorchWeights(Reader("autoregressive.pth"));
            foreach (var layer in layers) layer.SetTrainingMode(false);
            GC.Collect(); GC.WaitForPendingFinalizers();
            var mel = Tensor(ar.GetProperty("mel"), 80, ar.GetProperty("frames").GetInt32());
            var condition = model.Conditioning(engine.TensorTranspose(mel), false, new Random(1));
            var output = model.TeacherForced(text, speech, condition, false, new Random(1));
            foreach (string precision in new[] { "float64", "float32" })
            {
                var expected = ar.GetProperty(precision);
                Check($"Released AR conditioner / {precision}", condition, expected.GetProperty("condition"), 2e-4);
                Check($"Released AR text logits / {precision}", output.TextLogits, expected.GetProperty("text_logits"), 2e-4);
                Check($"Released AR speech logits / {precision}", output.SpeechLogits, expected.GetProperty("speech_logits"), 2e-4);
                Check($"Released AR speech latents / {precision}", output.SpeechLatents, expected.GetProperty("speech_latents"), 2e-4);
            }
        }
        Ar(); GC.Collect(); GC.WaitForPendingFinalizers();
        void Clvp()
        {
            var layers = new List<LayerBase<double>>();
            var model = new TortoiseClvp<double>(engine, layers, options);
            model.LoadTorchWeights(Reader("clvp2.pth"));
            foreach (var layer in layers) layer.SetTrainingMode(false);
            var score = model.ScoreTensor(text, speech, false, new Random(1));
            Check("Released CLVP / float64", score, root.GetProperty("clvp").GetProperty("float64"), 2e-6);
            Check("Released CLVP / float32", score, root.GetProperty("clvp").GetProperty("float32"), 2e-4);
        }
        Clvp(); GC.Collect(); GC.WaitForPendingFinalizers();
        void Diffusion()
        {
            var layers = new List<LayerBase<double>>();
            var model = new TortoiseDiffusionCore<double>(engine, layers, options);
            model.LoadTorchWeights(Reader("diffusion_decoder.pth"));
            foreach (var layer in layers) layer.SetTrainingMode(false);
            var expected = root.GetProperty("diffusion");
            var mel = engine.TensorTranspose(Tensor(expected.GetProperty("mel"), 100, 12));
            var latents = Tensor(expected.GetProperty("latents"), 3, 1024);
            var noisy = Tensor(expected.GetProperty("noisy"), 1, 100, 9);
            var condition = model.Conditioning(mel);
            Check("Released diffusion conditioner", condition, expected.GetProperty("condition"), 2e-4);
            Check("Released diffusion latent alignment", model.Aligned(latents, condition, 9), expected.GetProperty("aligned"), 2e-4);
            Check("Released diffusion epsilon and variance", model.Forward(noisy, 137, latents, mel), expected.GetProperty("prediction"), 2e-4);
            Check("Released diffusion unconditional epsilon and variance", model.Forward(noisy, 137, latents, mel, conditioningFree: true), expected.GetProperty("unconditional"), 2e-4);
        }
        Diffusion(); GC.Collect(); GC.WaitForPendingFinalizers();
        void Vocoder()
        {
            var model = new AiDotNet.TextToSpeech.Vocoders.UnivNetGenerator<double>(engine, 100, 64, 32,
                options.UnivNetUpsampleRates, options.UnivNetDilations, .2, options.UnivNetKernelPredictorHidden, options.UnivNetKernelPredictorConvSize);
            model.LoadTorchWeights(Reader("vocoder.pth", "model_g."));
            foreach (var layer in model.Layers) layer.SetTrainingMode(false);
            var expected = root.GetProperty("vocoder");
            Check("Released UnivNet waveform", model.Forward(Tensor(expected.GetProperty("mel"), 1, 100, 5),
                Tensor(expected.GetProperty("noise"), 1, 64, 5)), expected.GetProperty("waveform"), 2e-6);
        }
        Vocoder();
        if (failures.Count > 0) throw new InvalidDataException(string.Join(Environment.NewLine, failures));
        Console.WriteLine("Original TorToiSe AR / CLVP / diffusion / UnivNet checkpoint audit passed.");
    }
}
