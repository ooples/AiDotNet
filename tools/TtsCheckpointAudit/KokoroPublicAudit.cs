using System.Reflection;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.EndToEnd;

internal static class KokoroPublicAudit
{
    internal static void Run(string checkpoint, string config, string voiceBank, string fixture, bool checkClone = false, bool checkRestore = false)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        var root = document.RootElement;
        var reference = root.GetProperty("original_forced_float32_buffers");
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var architecture = new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression, inputSize: 16, outputSize: 1);
        using var model = new Kokoro<double>(architecture, new KokoroOptions { Speed = 20 });
        model.LoadAuthorWeights(checkpoint, config);
        model.LoadVoice(voiceBank);
        string phonemes = root.GetProperty("phonemes").GetString()!;
        var expectedIds = root.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        if (!model.EncodePhonemes(phonemes).SequenceEqual(expectedIds)) throw new InvalidDataException("Public Kokoro phoneme IDs differ.");
        var core = (KokoroCore<double>)typeof(Kokoro<double>).GetField("_native", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        var draws = reference.GetProperty("excitation_draws").EnumerateArray().ToArray();
        core.SetExcitationDraws(Values(draws.Single(x => x.GetProperty("kind").GetString() == "uniform").GetProperty("values")),
            Values(draws.Single(x => x.GetProperty("kind").GetString() == "normal" &&
                x.GetProperty("shape").EnumerateArray().Last().GetInt32() == 9).GetProperty("values")));
        var style = new Tensor<double>(new[] { 1, 256 }, new Vector<double>(Values(root.GetProperty("style"))));
        var explicitStyle = model.SynthesizePhonemes(phonemes, style, 20);
        bool rejectedApproximateFrontend = false;
        try { model.Synthesize("unconfigured raw text"); }
        catch (InvalidOperationException) { rejectedApproximateFrontend = true; }
        if (!rejectedApproximateFrontend) throw new InvalidDataException("Unconfigured Kokoro text must not silently substitute the approximate frontend.");
        model.TextPhonemizer = _ => phonemes;
        var bankStyle = model.Synthesize("public text frontend routing");
        double routeError = explicitStyle.ToArray().Zip(bankStyle.ToArray(), (a, b) => Math.Abs(a - b)).Max();
        if (routeError > 1e-12) throw new InvalidDataException("Voice-bank selection or public synthesis routing differs.");
        int expectedSamples = reference.GetProperty("waveform").GetArrayLength();
        if (bankStyle.Length != expectedSamples || model.SampleRate != 24000) throw new InvalidDataException("Public waveform geometry differs.");
        if (bankStyle.ToArray().Any(x => !double.IsFinite(x))) throw new InvalidDataException("Non-finite public waveform.");
        double waveformError = bankStyle.ToArray().Zip(Values(reference.GetProperty("waveform")), (a, b) => Math.Abs(a - b)).Max();
        Console.WriteLine($"Public Kokoro: exact phoneme IDs, original .pth/.pt loading, voice-bank/style routing error {routeError:R}, {bankStyle.Length} samples at 24000 Hz.");
        Console.WriteLine($"Original author waveform error with original float32 buffers: {waveformError:R}");
        if (!double.IsFinite(waveformError) || waveformError > 1e-7) throw new InvalidDataException("Public raw author waveform differs.");
        if (checkClone)
        {
            using var copy = (Kokoro<double>)model.DeepCopy();
            if (!copy.EncodePhonemes(phonemes).SequenceEqual(expectedIds)) throw new InvalidDataException("Cloned Kokoro vocabulary differs.");
            copy.TextPhonemizer = _ => phonemes;
            var copiedCore = (KokoroCore<double>)typeof(Kokoro<double>).GetField("_native", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(copy)!;
            copiedCore.SetExcitationDraws(Values(draws.Single(x => x.GetProperty("kind").GetString() == "uniform").GetProperty("values")),
                Values(draws.Single(x => x.GetProperty("kind").GetString() == "normal" &&
                    x.GetProperty("shape").EnumerateArray().Last().GetInt32() == 9).GetProperty("values")));
            var copiedWaveform = copy.Synthesize("cloned public voice-bank routing");
            if (copiedWaveform.Length != bankStyle.Length || copiedWaveform.ToArray().Zip(bankStyle.ToArray(),(a,b) => Math.Abs(a-b)).Max() > 1e-12)
                throw new InvalidDataException("Cloned Kokoro voice bank or native waveform differs.");
            Console.WriteLine("Public Kokoro DeepCopy preserves the loaded vocabulary, voice bank and native waveform with explicit source draws.");
        }
        if (checkRestore)
        {
            // Use the repository's existing test-only trial-file isolation, as
            // unit tests do; never consume or reset the caller's shared counter.
            string trialPath = Path.Combine(Path.GetTempPath(), "aidotnet-kokoro-audit-" + Guid.NewGuid().ToString("N") + ".json");
            try
            {
                using var trial = AiDotNet.Helpers.ModelPersistenceGuard.SetTestTrialFilePathOverride(trialPath);
                byte[] saved = model.Serialize();
                using var restored = new Kokoro<double>(architecture, new KokoroOptions { Speed = 20 });
                restored.Deserialize(saved);
                if (!restored.EncodePhonemes(phonemes).SequenceEqual(expectedIds)) throw new InvalidDataException("Restored Kokoro vocabulary differs.");
                restored.TextPhonemizer = _ => phonemes;
                var restoredCore = (KokoroCore<double>)typeof(Kokoro<double>).GetField("_native", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(restored)!;
                restoredCore.SetExcitationDraws(Values(draws.Single(x => x.GetProperty("kind").GetString() == "uniform").GetProperty("values")),
                    Values(draws.Single(x => x.GetProperty("kind").GetString() == "normal" &&
                        x.GetProperty("shape").EnumerateArray().Last().GetInt32() == 9).GetProperty("values")));
                var restoredWaveform = restored.Synthesize("restored public voice-bank routing");
                if (restoredWaveform.Length != bankStyle.Length || restoredWaveform.ToArray().Zip(bankStyle.ToArray(),(a,b) => Math.Abs(a-b)).Max() > 1e-12)
                    throw new InvalidDataException("Restored Kokoro voice bank or native waveform differs.");
                Console.WriteLine("Public Kokoro save/restore into a fresh instance preserves its loaded vocabulary, voice bank and native waveform.");
            }
            finally
            {
                foreach (string path in new[] { trialPath, trialPath + ".tombstone" }) if (File.Exists(path)) File.Delete(path);
            }
        }
    }
}
