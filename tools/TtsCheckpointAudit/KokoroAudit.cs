using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.EndToEnd;

internal static class KokoroAudit
{
    internal static void Run(string checkpoint, string fixture, bool authorSpectrum = false)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        var root = document.RootElement;
        var expected = root.GetProperty("original_forced_float32_buffers");
        var raw = new WeightLoader().LoadWeights(checkpoint);
        var state = raw.ToDictionary(pair => pair.Key.Replace(".module.", "."), pair => pair.Value);
        var consumed = new HashSet<string>();
        double[] Read(string name, int[] shape)
        {
            if (!state.TryGetValue(name, out var tensor)) throw new InvalidDataException("Missing Kokoro tensor: " + name);
            if (!tensor.Shape.ToArray().SequenceEqual(shape)) throw new InvalidDataException("Incompatible Kokoro tensor: " + name);
            consumed.Add(name);
            return tensor.ToArray().Select(value => (double)value).ToArray();
        }
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        void Check(string name, Tensor<double> actual, JsonElement target)
        {
            var values = Values(target);
            if (actual.Length != values.Length) throw new InvalidDataException(name + ": output shape differs.");
            double error = actual.ToArray().Zip(values, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"Kokoro {name}: maximum error {error:R}");
            if (name == "harmonic spectrum" && error > 1e-7)
            {
                var a = actual.ToArray();
                int frames = actual.Shape[2], bins = actual.Shape[1] / 2;
                foreach (int index in Enumerable.Range(0, a.Length).Where(i => Math.Abs(a[i] - values[i]) > 1e-7).Take(24))
                    Console.WriteLine($"spectrum channel {index / frames}, frame {index % frames}, native {a[index]:R}, author {values[index]:R}, magnitude {a[(index / frames % bins) * frames + index % frames]:R}");
            }
            if (!double.IsFinite(error) || error > 1e-7) throw new InvalidDataException(name + ": numerical mismatch.");
        }
        var model = new KokoroCore<double>(AiDotNetEngine.Current);
        model.LoadTorch(Read);
        var unused = state.Keys.Where(key => !consumed.Contains(key)).OrderBy(key => key).ToArray();
        // CustomAlbert returns only last_hidden_state; the separately computed pooler is unused by Kokoro.
        if (!unused.SequenceEqual(new[] { "bert.pooler.bias", "bert.pooler.weight" }))
            throw new InvalidDataException("Unexpected unused Kokoro tensors: " + string.Join(",", unused));
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(
            model.Layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        Console.WriteLine($"Kokoro learned tensor ownership: {parameters.Count} native parameters for {consumed.Count} active checkpoint tensors.");
        if (parameters.Count != consumed.Count)
            throw new InvalidDataException("Kokoro checkpoint tensors must retain independent parameter ownership.");
        foreach (var layer in model.Layers) layer.SetTrainingMode(false);
        model.StochasticExcitation = true;
        var draws = expected.GetProperty("excitation_draws").EnumerateArray().ToArray();
        var phases = draws.Single(draw => draw.GetProperty("kind").GetString() == "uniform");
        var gaussian = draws.Single(draw => draw.GetProperty("kind").GetString() == "normal" &&
            draw.GetProperty("shape").EnumerateArray().Last().GetInt32() == 9);
        model.SetExcitationDraws(Values(phases.GetProperty("values")), Values(gaussian.GetProperty("values")));
        var decoder = (AiDotNet.TextToSpeech.StyleEmotion.StyleTts2Decoder<double>)typeof(KokoroCore<double>)
            .GetField("_decoder", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(model)!;
        if (authorSpectrum)
        {
            var originalSpectrum = Values(expected.GetProperty("intermediates").GetProperty("harmonic_spectrum"));
            decoder.Generator.SourceSpectrumTransform = _ => new Tensor<double>(new[] { 1, 22, originalSpectrum.Length / 22 }, new Vector<double>(originalSpectrum));
            Console.WriteLine("Diagnostic: waveform uses author harmonic-spectrum features; this is not end-to-end native FFT parity.");
        }
        var ids = root.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var result = model.Forward(ids, new Tensor<double>(new[] { 1, 256 }, new Vector<double>(Values(root.GetProperty("style")))),
            root.GetProperty("speed").GetDouble());
        double completeError = result.Audio.ToArray().Zip(Values(expected.GetProperty("waveform")), (a,b) => Math.Abs(a-b)).Max();
        Console.WriteLine($"Kokoro raw waveform error before intermediate checks: {completeError:R}");
        if (!result.Durations.SequenceEqual(expected.GetProperty("durations").EnumerateArray().Select(x => x.GetInt32())))
            throw new InvalidDataException("Kokoro durations differ.");
        var intermediate = expected.GetProperty("intermediates");
        Check("ALBERT", result.Bert, intermediate.GetProperty("bert"));
        Check("ALBERT projection", result.Projected, intermediate.GetProperty("bert_encoder"));
        Check("duration logits", result.DurationLogits, intermediate.GetProperty("duration"));
        Check("pitch", result.Pitch, intermediate.GetProperty("pitch"));
        Check("energy", result.Energy, intermediate.GetProperty("energy"));
        Check("text encoder", result.Text, intermediate.GetProperty("text_encoder"));
        var engine = AiDotNetEngine.Current;
        var style = new Tensor<double>(new[] { 1, 128 }, new Vector<double>(Values(root.GetProperty("style")).Take(128)));
        var random = new Random(1);
        var alignment = new Tensor<double>(new[] { ids.Length, result.Durations.Sum() });
        int offset = 0;
        for (int i = 0; i < ids.Length; i++) for (int frame = 0; frame < result.Durations[i]; frame++) alignment[i, offset++] = 1;
        var asr = engine.Reshape(engine.TensorMatMul(engine.Reshape(result.Text, new[] { 512, ids.Length }), alignment), new[] { 1, 512, offset });
        var pitch = decoder.F0Conv.Forward(engine.Reshape(result.Pitch, new[] { 1, 1, result.Pitch.Length }));
        var energy = decoder.NConv.Forward(engine.Reshape(result.Energy, new[] { 1, 1, result.Energy.Length }));
        var decoded = decoder.Encode.Forward(engine.TensorConcatenate(new[] { asr, pitch, energy }, 1), style, false, random);
        Check("decoder encode", decoded, intermediate.GetProperty("decoder_encode"));
        var residual = decoder.AsrResidual.Forward(asr);
        bool concatenate = true;
        for (int i = 0; i < decoder.Decode.Count; i++)
        {
            if (concatenate) decoded = engine.TensorConcatenate(new[] { decoded, residual, pitch, energy }, 1);
            decoded = decoder.Decode[i].Forward(decoded, style, false, random);
            if (decoder.Decode[i].Upsample) concatenate = false;
            Check("decoder block " + i, decoded, intermediate.GetProperty("decoder_block_" + i));
        }
        var generator = decoder.Generator;
        var harmonic = generator.Source.Forward(result.Pitch, random);
        Check("harmonic source", harmonic, intermediate.GetProperty("harmonic_source"));
        var spectrum = new KokoroHarmonicStft<double>(window: KokoroSignalBuffers.HannFloat32()).Forward(harmonic);
        var expectedSpectrum = Values(intermediate.GetProperty("harmonic_spectrum"));
        if (!authorSpectrum) Check("harmonic spectrum", spectrum, intermediate.GetProperty("harmonic_spectrum"));
        else
        {
            int bins = spectrum.Shape[1] / 2, frames = spectrum.Shape[2];
            var values = spectrum.ToArray(); double maximum = 0;
            for (int i = 0; i < values.Length; i++)
            {
                double difference = values[i] - expectedSpectrum[i];
                if (i >= bins * frames) difference = Math.Atan2(Math.Sin(difference), Math.Cos(difference));
                maximum = Math.Max(maximum, Math.Abs(difference));
            }
            Console.WriteLine($"Native harmonic spectrum modulo phase wrapping: {maximum:R}");
            if (maximum > 1e-7) throw new InvalidDataException("Harmonic spectrum differs beyond a phase branch choice.");
        }
        Check("complete waveform", result.Audio, expected.GetProperty("waveform"));
        Console.WriteLine($"Kokoro: all {consumed.Count} active tensors imported; two unused ALBERT pooler tensors explicitly excluded.");
    }
}
