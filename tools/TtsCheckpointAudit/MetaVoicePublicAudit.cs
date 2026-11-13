using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.VoiceCloning;

internal static class MetaVoicePublicAudit
{
    internal static void Run(string actorFixture, string decoderFixture, string? releaseOutputFixture = null)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var actors = JsonDocument.Parse(File.ReadAllText(actorFixture)); var reference = actors.RootElement;
        using var decoder = JsonDocument.Parse(File.ReadAllText(decoderFixture)); var waveReference = decoder.RootElement;
        using var releasedOutput = releaseOutputFixture is null ? null : JsonDocument.Parse(File.ReadAllText(releaseOutputFixture));
        var first = reference.GetProperty("first_config"); var second = reference.GetProperty("second_config");
        var speaker = reference.GetProperty("speaker_config");
        var options = new MetaVoice1BOptions
        {
            FirstStageDim = first.GetProperty("n_embd").GetInt32(), NumFirstStageLayers = first.GetProperty("n_layer").GetInt32(),
            NumHeads = first.GetProperty("n_head").GetInt32(), FirstStageContext = first.GetProperty("block_size").GetInt32(),
            SecondStageDim = second.GetProperty("n_embd").GetInt32(), NumSecondStageLayers = second.GetProperty("n_layer").GetInt32(),
            SecondStageHeads = second.GetProperty("n_head").GetInt32(), SecondStageContext = second.GetProperty("block_size").GetInt32(),
            SpeakerEmbeddingDim = speaker.GetProperty("dimension").GetInt32(), SpeakerHiddenDim = speaker.GetProperty("hidden").GetInt32(),
            SpeakerMelChannels = speaker.GetProperty("mels").GetInt32(), SpeakerEncoderLayers = speaker.GetProperty("layers").GetInt32(),
            SwiGLUMultipleOf = first.GetProperty("swiglu_multiple_of").GetInt32(),
            MaximumNewTokens = reference.TryGetProperty("maximum_new_tokens", out var maximum) ? maximum.GetInt32() : 5,
            FirstStageTopP = 1, SpeakerGuidance = 2, PromptGuidance = 2,
            TrainingDropout = 0, MinReferenceDurationSec = .001, RawDecoderOutput = releaseOutputFixture is null
        };
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = 2294 };
        using var model = new MetaVoice1B<double>(architecture, options);
        string directory = Path.GetDirectoryName(Path.GetFullPath(actorFixture))!;
        var files = reference.GetProperty("checkpoint_files");
        model.LoadLanguageModelCheckpoints(Path.Combine(directory, files[0].GetString()!), Path.Combine(directory, files[1].GetString()!));
        model.LoadSpeakerEncoderCheckpoint(Path.Combine(directory, files[2].GetString()!));
        // This is a numerical source audit: retain the original stored precision.
        // The framework's automatic inference policy may otherwise quantize weights.
        model.ConfigureWeightLifetime(new GpuOffloadOptions
        {
            StreamingBackingStorePath = Path.Combine(Path.GetTempPath(), "metavoice-public-audit-" + Guid.NewGuid().ToString("N")),
            StreamingPoolMaxResidentBytes = 64 * 1024 * 1024,
            StreamingStoreDtype = StreamingStoreDtype.FullPrecision
        });
        Console.WriteLine("Public MetaVoice: original actor/speaker PTH fixtures loaded; loading complete released decoder.");
        model.LoadDecoderCheckpoints(waveReference.GetProperty("codec_release").GetProperty("paths").GetProperty("model.safetensors").GetString()!,
            waveReference.GetProperty("release").GetProperty("checkpoint_path").GetString()!);
        Console.WriteLine("Public MetaVoice: all four original decoder bands and trained processors loaded.");
        if (releasedOutput is not null)
        {
            model.LoadEnhancerCheckpoint(releasedOutput.RootElement.GetProperty("df_checkpoint").GetString()!);
            model.ObserveOutputForAudit = (name, value) =>
            {
                var expectedStage = releasedOutput.RootElement.GetProperty(name).EnumerateArray().Select(x => x.GetDouble()).ToArray();
                if (value.Length != expectedStage.Length) throw new InvalidDataException("Released output stage length differs: " + name);
                double stageError = value.ToArray().Zip(expectedStage, (a, b) => Math.Abs(a - b)).Max();
                Console.WriteLine($"Public released output {name} error {stageError:R}.");
                if (!double.IsFinite(stageError) || stageError > (name == "enhanced" ? 2e-6 : 2e-10))
                    throw new InvalidDataException("Released output stage differs: " + name);
            };
        }
        var uniforms = reference.GetProperty("uniforms").EnumerateArray().Select(x => x.GetDouble()).ToArray(); int uniformCount = 0, gaussianCount = 0;
        model.SamplingUniformForAudit = () => uniforms[uniformCount++];
        model.DiffusionGaussianForAudit = shape => new Tensor<double>(shape, new Vector<double>(
            waveReference.GetProperty("gaussian_draws")[gaussianCount++].EnumerateArray().Select(x => x.GetDouble())));
        // DecoderForAudit remains null: the actual public method runs the full trained neural decoder.
        var samples = reference.GetProperty("reference_audio").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var actual = model.SynthesizeWithVoice(reference.GetProperty("text").GetString()!,
            new Tensor<double>(new[] { samples.Length }, new Vector<double>(samples)), 16000);
        if (uniformCount != uniforms.Length || gaussianCount != 80) throw new InvalidDataException("Public MetaVoice draw order/count differs.");
        var expected = (releasedOutput?.RootElement ?? waveReference).GetProperty("waveform").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        if (actual.Length != expected.Length) throw new InvalidDataException("Public MetaVoice waveform length differs.");
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Console.WriteLine($"Public MetaVoice: actual speaker+two GPTs+EnCodec+four-band diffusion+processors+re_eq,{uniformCount} uniform/80 Gaussian draws; waveform error {error:R}.");
        double tolerance = releasedOutput is null ? 1e-8 : 1.0 / 32768;
        if (!double.IsFinite(error) || error > tolerance) throw new InvalidDataException("Public MetaVoice complete waveform mismatch.");
        if (releasedOutput is not null && (model.SampleRate != 48000 || ((AiDotNet.TextToSpeech.Interfaces.ITtsModel<double>)model).SampleRate != 48000))
            throw new InvalidDataException("The enhanced release output must report 48 kHz through both public contracts.");
        Console.WriteLine("Reduced actor/speaker widths with original classes and real vocabularies; full released decoder. Full actor checkpoint proofs are separate.");
    }
}
