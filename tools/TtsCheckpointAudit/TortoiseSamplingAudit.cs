using System.Security.Cryptography;
using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;

internal static class TortoiseSamplingAudit
{
    internal static void Run(string directory, string fixture)
    {
        Environment.SetEnvironmentVariable("OMP_NUM_THREADS", "1");
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        var root = document.RootElement;
        if (!root.GetProperty("released").GetBoolean())
            throw new InvalidDataException("Expected complete released-checkpoint sampler reference.");
        string checkpoint = Path.Combine(directory, ".models", "diffusion_decoder.pth");
        using (var stream = File.OpenRead(checkpoint))
        {
            string hash = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
            if (hash != root.GetProperty("checkpoint_sha256").GetString())
                throw new InvalidDataException("Original diffusion checkpoint SHA-256 differs from the reference.");
        }
        Console.WriteLine($"Original author {root.GetProperty("code_revision").GetString()}, checkpoint {root.GetProperty("checkpoint_revision").GetString()}.");
        var state = new WeightLoader().LoadWeights(checkpoint);
        var options = new TortoiseTTSOptions
        {
            DiffusionInferenceSteps = root.GetProperty("inference_steps").GetInt32(),
            DropoutRate = 0
        };
        var layers = new List<LayerBase<double>>();
        var model = new TortoiseDiffusion<double>(AiDotNetEngine.Current, layers, options);
        model.LoadTorchWeights((name, shape) =>
        {
            if (!state.TryGetValue(name, out var tensor) || !tensor.Shape.ToArray().SequenceEqual(shape))
                throw new InvalidDataException($"Original diffusion tensor {name} has incompatible axes.");
            return tensor.ToArray().Select(x => (double)x).ToArray();
        });
        state.Clear();
        foreach (var layer in layers) layer.SetTrainingMode(false);
        var latents = Tensor(root.GetProperty("latents"), 3, 1024);
        var mel = AiDotNetEngine.Current.TensorTranspose(Tensor(root.GetProperty("mel"), 100, 12));
        foreach (var expected in root.GetProperty("cases").EnumerateArray())
        {
            options.ClassifierFreeGuidanceScale = expected.GetProperty("guidance").GetDouble();
            var random = new RecordedGaussian(expected);
            var actual = model.Sample(latents, mel, random);
            double error = actual.ToArray().Zip(Values(expected.GetProperty("mel")),
                (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"Complete released {options.DiffusionInferenceSteps}-step DDPM: maximum mel error {error:R}; replayed {random.Consumed} of {random.DrawCount} uniform draws.");
            if (!double.IsFinite(error) || error > 2e-3 || random.Consumed != random.DrawCount)
                throw new InvalidDataException("Complete original diffusion sampler comparison failed.");
        }
        Console.WriteLine("Original complete TorToiSe released-checkpoint DDPM sampler audit passed.");
    }

    private sealed class RecordedGaussian : Random
    {
        private readonly Queue<double> _uniforms = new();
        internal int Consumed { get; private set; }
        internal int DrawCount { get; }
        internal RecordedGaussian(JsonElement fixture)
        {
            var blocks = new List<double[]> { Values(fixture.GetProperty("initial")) };
            blocks.AddRange(fixture.GetProperty("noises").EnumerateArray().Select(Values));
            foreach (var values in blocks)
                for (int i = 0; i < values.Length; i += 2)
                {
                    double x = values[i], y = i + 1 < values.Length ? values[i + 1] : 0;
                    _uniforms.Enqueue(1 - Math.Exp(-(x * x + y * y) / 2));
                    double angle = Math.Atan2(y, x);
                    _uniforms.Enqueue((angle < 0 ? angle + 2 * Math.PI : angle) / (2 * Math.PI));
                }
            DrawCount = _uniforms.Count;
        }
        public override double NextDouble()
        {
            Consumed++;
            if (_uniforms.Count == 0) throw new InvalidDataException("Native sampler consumed extra noise.");
            return _uniforms.Dequeue();
        }
    }
    private static double[] Values(JsonElement array) => array.EnumerateArray().Select(x => x.GetDouble()).ToArray();
    private static Tensor<double> Tensor(JsonElement array, params int[] shape) => new(shape, new Vector<double>(Values(array)));
}
