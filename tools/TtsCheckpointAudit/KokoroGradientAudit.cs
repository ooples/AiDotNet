using System.Reflection;
using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.TextToSpeech.StyleEmotion;

internal static class KokoroGradientAudit
{
    internal static void Run(string checkpoint, string inferenceFixture, string directory)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var fixture = JsonDocument.Parse(File.ReadAllText(inferenceFixture));
        using var metadata = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "gradients.json")));
        using var stream = File.OpenRead(Path.Combine(directory, "gradients.safetensors")); using var reader = new BinaryReader(stream);
        long headerLength = reader.ReadInt64(); using var header = JsonDocument.Parse(reader.ReadBytes(checked((int)headerLength)));
        double[] Gradient(string name)
        {
            var entry = header.RootElement.GetProperty(name);
            if (entry.GetProperty("dtype").GetString() != "F64") throw new InvalidDataException("Expected float64 gradient.");
            var offsets = entry.GetProperty("data_offsets").EnumerateArray().Select(x => x.GetInt64()).ToArray();
            stream.Position = 8 + headerLength + offsets[0]; var result = new double[checked((int)((offsets[1]-offsets[0])/8))];
            for (int i = 0; i < result.Length; i++) result[i] = reader.ReadDouble(); return result;
        }
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var raw = new WeightLoader().LoadWeights(checkpoint);
        var state = raw.ToDictionary(pair => pair.Key.Replace(".module.", "."), pair => pair.Value);
        var consumed = new HashSet<string>();
        double[] Read(string name, int[] shape)
        {
            if (!state.TryGetValue(name, out var tensor) || !tensor.Shape.ToArray().SequenceEqual(shape))
                throw new InvalidDataException("Missing or incompatible Kokoro tensor " + name);
            if (!consumed.Add(name)) throw new InvalidDataException("Duplicate checkpoint read " + name);
            return tensor.ToArray().Select(x => (double)x).ToArray();
        }
        var engine = AiDotNetEngine.Current;
        // This secondary reference explicitly promotes the author's fixed DSP
        // buffers to double to isolate learned-network derivatives.
        var model = new KokoroCore<double>(engine, originalFloat32Buffers: false); model.LoadTorch(Read);
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(model.Layers.Cast<ILayer<double>>().ToList());
        if (parameters.Count != 546 || consumed.Count != 546) throw new InvalidDataException("Independent Kokoro parameter count differs.");
        foreach (var layer in model.Layers) layer.SetTrainingMode(false);
        static string Fingerprint(double[] values)
        {
            var parts = new long[12];
            for (int i = 0; i < parts.Length; i++) parts[i] = BitConverter.DoubleToInt64Bits(values[(long)i*(values.Length-1)/(parts.Length-1)]);
            return values.Length + ":" + string.Join(",", parts);
        }
        double[] Transform(string name, double[] values, bool transpose)
        {
            if (!transpose) return values;
            var shape = state[name].Shape.ToArray(); var result = new double[values.Length];
            for (int o = 0; o < shape[0]; o++) for (int i = 0; i < shape[1]; i++) result[i*shape[0]+o] = values[o*shape[1]+i];
            return result;
        }
        var candidates = new Dictionary<string, List<(string Name, bool Transpose)>>();
        foreach (string name in consumed)
        {
            var values = state[name].ToArray().Select(x => (double)x).ToArray();
            foreach (bool transpose in state[name].Shape.Length == 2 && state[name].Shape[0] > 1 && state[name].Shape[1] > 1
                ? new[] { false, true } : new[] { false })
            {
                string key = Fingerprint(Transform(name, values, transpose));
                if (!candidates.TryGetValue(key, out var bucket)) candidates.Add(key, bucket = new());
                bucket.Add((name, transpose));
            }
        }
        var mapping = new Dictionary<Tensor<double>, (string Name, bool Transpose)>(); var used = new HashSet<string>();
        foreach (var parameter in parameters)
        {
            var values = parameter.ToArray();
            if (!candidates.TryGetValue(Fingerprint(values), out var bucket)) throw new InvalidDataException("No checkpoint parameter matches native values.");
            var matches = bucket.Where(item => !used.Contains(item.Name) && Transform(item.Name,
                state[item.Name].ToArray().Select(x => (double)x).ToArray(), item.Transpose).SequenceEqual(values)).ToArray();
            if (matches.Length != 1) throw new InvalidDataException("Ambiguous native parameter: " + string.Join(",", matches.Select(x => x.Name)));
            mapping.Add(parameter, matches[0]); used.Add(matches[0].Name);
        }
        Console.WriteLine("Kokoro gradient audit: 546 checkpoint/native parameter identities mapped without folded biases.");
        var root = fixture.RootElement; var matched = root.GetProperty("precision_matched_double_buffers");
        var draws = matched.GetProperty("excitation_draws").EnumerateArray().ToArray();
        model.SetExcitationDraws(Values(draws.Single(x => x.GetProperty("kind").GetString() == "uniform").GetProperty("values")),
            Values(draws.Single(x => x.GetProperty("kind").GetString() == "normal" && x.GetProperty("shape").EnumerateArray().Last().GetInt32() == 9).GetProperty("values")));
        var decoder = (StyleTts2Decoder<double>)typeof(KokoroCore<double>).GetField("_decoder", BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(model)!;
        var spectrum = Values(matched.GetProperty("intermediates").GetProperty("harmonic_spectrum"));
        decoder.Generator.SourceSpectrumTransform = _ => new Tensor<double>(new[] { 1, 22, spectrum.Length/22 }, new Vector<double>(spectrum));
        Console.WriteLine("Feature-conditioned neural-gradient diagnostic: original detached harmonic spectrum supplied; raw FFT parity is not asserted.");
        var style = new Tensor<double>(new[] { 1, 256 }, new Vector<double>(Values(root.GetProperty("style"))));
        using var tape = new GradientTape<double>();
        var result = model.Forward(root.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray(), style, root.GetProperty("speed").GetDouble());
        var outputs = new[] { result.Bert, result.Projected, result.DurationLogits, result.Pitch, result.Energy, result.Text, result.Audio };
        string[] names = { "bert", "projected", "duration", "pitch", "energy", "text", "audio" };
        Tensor<double>? loss = null;
        for (int i = 0; i < outputs.Length; i++)
        {
            var output = outputs[i]; var expected = Values(metadata.RootElement.GetProperty("outputs").GetProperty(names[i]));
            if (output.Length != expected.Length || output.ToArray().Zip(expected, (x,y) => Math.Abs(x-y)).Max() > 1e-8)
                throw new InvalidDataException("Neural-gradient objective output differs: " + names[i]);
            var probe = new Tensor<double>(output.Shape.ToArray(), new Vector<double>(Enumerable.Range(0,output.Length).Select(n => -.7 + 1.6*n/(output.Length-1))));
            var term = engine.TensorMultiplyScalar(engine.Reshape(engine.ReduceSum(engine.TensorMultiply(output,probe),
                Enumerable.Range(0,output.Rank).ToArray(),true),new[] {1}), (i+1)/100.0);
            loss = loss is null ? term : engine.TensorAdd(loss,term);
        }
        double lossError = Math.Abs(loss![0]-metadata.RootElement.GetProperty("loss").GetDouble());
        if (lossError > 1e-8) throw new InvalidDataException("Kokoro full neural objective differs.");
        var gradients = tape.ComputeGradients(loss,sources:parameters.Concat(new[] {style}).ToList());
        double maximum = 0; int checkedCount = 0;
        foreach (var parameter in parameters)
        {
            var item = mapping[parameter];
            if (!header.RootElement.TryGetProperty(item.Name,out _))
            {
                if (item.Name != "decoder.generator.m_source.l_linear.weight" && item.Name != "decoder.generator.m_source.l_linear.bias")
                    throw new InvalidDataException("Unexpected absent author gradient " + item.Name);
                if (gradients.TryGetValue(parameter,out var frozen) && frozen.ToArray().Any(x => x != 0)) throw new InvalidDataException("Detached source acquired a gradient.");
                continue;
            }
            if (!gradients.TryGetValue(parameter,out var gradient)) throw new InvalidDataException("Missing native gradient " + item.Name);
            var expected = Transform(item.Name,Gradient(item.Name),item.Transpose); var actual = gradient.ToArray();
            if (actual.Length != expected.Length) throw new InvalidDataException("Gradient shape differs " + item.Name);
            for (int n = 0; n < actual.Length; n++)
            {
                double error = Math.Abs(actual[n]-expected[n]); maximum = Math.Max(maximum,error);
                if (!double.IsFinite(error) || error > 1e-7 + 1e-9*Math.Abs(expected[n])) throw new InvalidDataException($"{item.Name} gradient error {error:R} at {n}.");
            }
            checkedCount++;
        }
        if (checkedCount != 544) throw new InvalidDataException("Not all 544 original gradients were checked.");
        var expectedStyle = Gradient("input.style"); var actualStyle = gradients[style].ToArray();
        for (int n = 0; n < expectedStyle.Length; n++)
            if (Math.Abs(actualStyle[n]-expectedStyle[n]) > 1e-7 + 1e-9*Math.Abs(expectedStyle[n])) throw new InvalidDataException("Style-input gradient differs.");
        Console.WriteLine($"Kokoro: all 544 original neural parameter gradients and style-input gradient pass; maximum absolute parameter error {maximum:R}, loss error {lossError:R}.");
    }
}
