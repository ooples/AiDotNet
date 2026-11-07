using AiDotNet.Audio.Generation;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.VoiceCloning;

public partial class MetaVoice1B<T>
{
    /// <summary>Loads the original MetaVoice actor directory and its original external decoder assets.</summary>
    public void LoadPretrainedWeights(string modelDirectory, string encodecSafetensors, string multibandCheckpoint)
    {
        LoadLanguageModelCheckpoints(Path.Combine(modelDirectory, "first_stage.pt"), Path.Combine(modelDirectory, "second_stage.pt"));
        LoadSpeakerEncoderCheckpoint(Path.Combine(modelDirectory, "speaker_encoder.pt"));
        LoadDecoderCheckpoints(encodecSafetensors, multibandCheckpoint);
    }

    /// <summary>Loads all actors, the neural decoder and the release's DeepFilterNet3 enhancer.</summary>
    public void LoadPretrainedWeights(string modelDirectory, string encodecSafetensors, string multibandCheckpoint, string deepFilterCheckpoint)
    {
        LoadPretrainedWeights(modelDirectory, encodecSafetensors, multibandCheckpoint);
        LoadEnhancerCheckpoint(deepFilterCheckpoint);
    }

    /// <summary>Loads the unmodified DeepFilterNet3 model_120.ckpt.best archive.</summary>
    public void LoadEnhancerCheckpoint(string checkpoint)
    {
        ThrowIfDisposed();
        if (!HasPaperPipeline) throw new NotSupportedException("Released enhancement requires the native MetaVoice graph.");
        var enhancer = new MetaVoiceDeepFilterEnhancer<T>(checkpoint);
        InstallEnhancer(enhancer);
    }

    /// <summary>Loads both unmodified actor PTH files, including their trained byte-BPE metadata.</summary>
    public void LoadLanguageModelCheckpoints(string firstStage, string secondStage)
    {
        ThrowIfDisposed();
        if (_first is null || _second is null) throw new NotSupportedException("Original actors require the native MetaVoice graphs.");
        LoadActor(_first, firstStage, true);
        _firstTokenizer = MetaVoiceTokenizer.FromCheckpoint(firstStage);
        _firstTokenizerRanks = _firstTokenizer.ExportByteRanks();
        _firstLoaded = true;
        LoadActor(_second, secondStage, false);
        _secondTokenizer = MetaVoiceTokenizer.FromCheckpoint(secondStage);
        _secondTokenizerRanks = _secondTokenizer.ExportByteRanks();
        _secondLoaded = true;
    }

    private void LoadActor(MetaVoiceGpt<T> model, string checkpoint, bool first)
    {
        var metadata = MetaVoiceTokenizer.ReadMetadata(checkpoint);
        var args = metadata["model_args"] as Dictionary<string, object?> ?? throw new InvalidDataException("Missing original GPT arguments.");
        var meta = metadata["meta"] as Dictionary<string, object?> ?? throw new InvalidDataException("Missing original GPT metadata.");
        bool Flag(string name, bool fallback) => args.TryGetValue(name, out var value) ? Convert.ToBoolean(value) : fallback;
        string Text(string name, string fallback) => args.TryGetValue(name, out var value) ? Convert.ToString(value) ?? fallback : fallback;
        if (Flag("causal", true) != first || Flag("bias", false) || !Flag("spk_emb_on_text", true) ||
            Text("norm_type", "layernorm") != (first ? "rmsnorm" : "layernorm") ||
            Text("nonlinearity_type", "gelu") != (first ? "swiglu" : "gelu") ||
            !Convert.ToBoolean(meta["speaker_cond"]) || Convert.ToInt32(meta["speaker_emb_size"]) != SpeakerEmbeddingDim ||
            (first && Convert.ToDouble(args["rmsnorm_eps"]) != 1e-5))
            throw new InvalidDataException("Checkpoint GPT normalization, positions or speaker-conditioning contract differs.");
        var raw = new WeightLoader().LoadWeights(checkpoint, name => name.StartsWith("model.", StringComparison.Ordinal));
        var state = raw.ToDictionary(pair => pair.Key.Substring(6).Replace("_orig_mod.", ""), pair => pair.Value);
        model.LoadTorch((name, shape) =>
        {
            if (!state.TryGetValue(name, out var tensor) || !tensor.Shape.ToArray().SequenceEqual(shape))
                throw new InvalidDataException("MetaVoice GPT tensor missing or incompatible: " + name);
            state.Remove(name); return tensor.ToArray().Select(x => (double)x).ToArray();
        });
        if (state.Count != 0) throw new InvalidDataException("Unconsumed MetaVoice GPT tensors: " + state.Count);
    }

    /// <summary>Loads all 14 active speaker tensors; the author's two unused similarity scalars are explicitly recognized.</summary>
    public void LoadSpeakerEncoderCheckpoint(string checkpoint)
    {
        ThrowIfDisposed();
        if (_speaker is null) throw new NotSupportedException("The native speaker encoder is unavailable.");
        var raw = new WeightLoader().LoadWeights(checkpoint, name => name.StartsWith("model_state.", StringComparison.Ordinal));
        var state = raw.ToDictionary(pair => pair.Key.Substring(12), pair => pair.Value);
        _speaker.LoadTorch((name, shape) =>
        {
            if (!state.TryGetValue(name, out var tensor) || !tensor.Shape.ToArray().SequenceEqual(shape))
                throw new InvalidDataException("MetaVoice speaker tensor missing or incompatible: " + name);
            state.Remove(name); return tensor.ToArray().Select(x => (double)x).ToArray();
        });
        foreach (string unused in new[] { "similarity_weight", "similarity_bias" })
            if (state.TryGetValue(unused, out var tensor))
            {
                if (tensor.Length != 1) throw new InvalidDataException("Unused author similarity scalar has incompatible axes.");
                state.Remove(unused);
            }
        if (state.Count != 0) throw new InvalidDataException("Unconsumed active speaker tensors: " + state.Count);
        _speakerLoaded = true;
    }

    /// <summary>Loads actual HF EnCodec and all four original AudioCraft PTH bands.</summary>
    public void LoadDecoderCheckpoints(string encodecSafetensors, string multibandCheckpoint)
    {
        ThrowIfDisposed();
        if (!HasPaperPipeline) throw new NotSupportedException("The native MetaVoice graph is unavailable.");
        var codecOptions = EnCodecOptions.OfficialCheckpoint24kHz(); codecOptions.IncludeDiscriminators = false;
        var architecture = new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = Architecture.RandomSeed };
        var codec = new EnCodec<T>(architecture, codecOptions);
        try
        {
            codec.LoadPretrainedWeights(encodecSafetensors);
            var decoder = new MetaVoiceMultibandDecoder<T>(Engine, codec);
            for (int band = 0; band < 4; band++) decoder.LoadAuthorBand(multibandCheckpoint, band);
            InstallDecoder(codec, decoder, codecOptions);
        }
        catch { codec.Dispose(); throw; }
    }
}
