using System.Text.Json;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.EndToEnd;

public partial class Kokoro<T>
{
    /// <summary>Loads the original Kokoro-82M configuration and phoneme vocabulary, independently of its weights.</summary>
    public void LoadAuthorConfiguration(string configuration)
    {
        ThrowIfDisposed();
        if (_native is null) throw new InvalidOperationException("Native Kokoro is required.");
        using var document = JsonDocument.Parse(File.ReadAllText(configuration));
        var config = document.RootElement;
        if (config.GetProperty("hidden_dim").GetInt32() != 512 || config.GetProperty("style_dim").GetInt32() != 128 ||
            config.GetProperty("n_token").GetInt32() != 178 || config.GetProperty("n_layer").GetInt32() != 3 ||
            config.GetProperty("text_encoder_kernel_size").GetInt32() != 5 || config.GetProperty("max_dur").GetInt32() != 50)
            throw new InvalidDataException("The configuration does not describe the Kokoro-82M graph.");
        var bert = config.GetProperty("plbert");
        if (bert.GetProperty("hidden_size").GetInt32() != 768 || bert.GetProperty("num_attention_heads").GetInt32() != 12 ||
            bert.GetProperty("intermediate_size").GetInt32() != 2048 || bert.GetProperty("num_hidden_layers").GetInt32() != 12 ||
            bert.GetProperty("max_position_embeddings").GetInt32() != 512)
            throw new InvalidDataException("The configuration does not describe Kokoro-82M's shared ALBERT.");
        var vocoder = config.GetProperty("istftnet");
        int[] Integers(string name) => vocoder.GetProperty(name).EnumerateArray().Select(value => value.GetInt32()).ToArray();
        if (!Integers("upsample_rates").SequenceEqual(new[] { 10, 6 }) || !Integers("upsample_kernel_sizes").SequenceEqual(new[] { 20, 12 }) ||
            !Integers("resblock_kernel_sizes").SequenceEqual(new[] { 3, 7, 11 }) ||
            vocoder.GetProperty("upsample_initial_channel").GetInt32() != 512 ||
            vocoder.GetProperty("gen_istft_n_fft").GetInt32() != 20 || vocoder.GetProperty("gen_istft_hop_size").GetInt32() != 5 ||
            vocoder.GetProperty("resblock_dilation_sizes").GetArrayLength() != 3 ||
            vocoder.GetProperty("resblock_dilation_sizes").EnumerateArray().Any(row =>
                !row.EnumerateArray().Select(value => value.GetInt32()).SequenceEqual(new[] { 1, 3, 5 })))
            throw new InvalidDataException("The configuration does not describe Kokoro-82M's iSTFTNet.");
        var vocabulary = new Dictionary<char, int>();
        foreach (var symbol in config.GetProperty("vocab").EnumerateObject())
        {
            int id = symbol.Value.GetInt32();
            if (symbol.Name.Length != 1 || id < 0 || id >= 178)
                throw new InvalidDataException("Invalid Kokoro phoneme vocabulary entry: " + symbol.Name);
            vocabulary.Add(symbol.Name[0], id);
        }
        _vocabulary = vocabulary;
    }

    /// <summary>Loads the original Kokoro-82M component checkpoint and its exact phoneme vocabulary.</summary>
    public void LoadAuthorWeights(string checkpoint, string configuration)
    {
        LoadAuthorConfiguration(configuration);
        var raw = new ComputerVision.Weights.WeightLoader().LoadWeights(checkpoint);
        var state = raw.ToDictionary(pair => pair.Key.Replace(".module.", "."), pair => pair.Value);
        var consumed = new HashSet<string>();
        double[] Read(string name, int[] shape)
        {
            if (!state.TryGetValue(name, out var tensor)) throw new InvalidDataException("Missing Kokoro tensor: " + name);
            if (!tensor.Shape.ToArray().SequenceEqual(shape)) throw new InvalidDataException("Kokoro tensor shape differs: " + name);
            consumed.Add(name);
            return tensor.ToArray().Select(value => (double)value).ToArray();
        }
        _native!.LoadTorch(Read);
        var unused = state.Keys.Where(name => !consumed.Contains(name)).ToArray();
        // CustomAlbert computes a pooler but KModel returns only its last hidden state.
        if (unused.Any(name => name != "bert.pooler.weight" && name != "bert.pooler.bias"))
            throw new InvalidDataException("Unexpected Kokoro checkpoint tensors: " + string.Join(", ", unused));
    }

    /// <summary>Loads an original voice .pt bank [phoneme length, 1, 256]. Each sentence selects its length's style.</summary>
    public void LoadVoice(string path)
    {
        ThrowIfDisposed();
        if (_native is null) throw new InvalidOperationException("Native Kokoro is required.");
        var state = new ComputerVision.Weights.WeightLoader().LoadWeights(path);
        if (state.Count != 1) throw new InvalidDataException("Expected a single Kokoro voice-bank tensor.");
        var bank = state.Values.Single();
        if (bank.Rank != 3 || bank.Shape[1] != 1 || bank.Shape[2] != 256 || bank.Shape[0] <= 0)
            throw new InvalidDataException("Expected Kokoro voice-bank shape [length, 1, 256].");
        _voiceBank = new Tensor<T>(new[] { bank.Shape[0], 256 },
            new Vector<T>(bank.ToArray().Select(value => NumOps.FromDouble(value))));
    }

    /// <summary>Maps the exact configured phoneme symbols to IDs, dropping unknown symbols and adding both pad IDs.</summary>
    public int[] EncodePhonemes(string phonemes)
    {
        if (phonemes is null) throw new ArgumentNullException(nameof(phonemes));
        if (_vocabulary is null) throw new InvalidOperationException("Load the original Kokoro configuration with LoadAuthorWeights first.");
        var ids = new List<int> { 0 };
        foreach (char symbol in phonemes)
            if (_vocabulary.TryGetValue(symbol, out int id)) ids.Add(id);
        ids.Add(0);
        if (ids.Count > 512) throw new ArgumentException("Kokoro accepts at most 510 phoneme IDs plus two pad IDs.", nameof(phonemes));
        return ids.ToArray();
    }

    /// <summary>Runs the actual language-independent KModel graph with explicit phonemes and a 256-value voice style.</summary>
    public Tensor<T> SynthesizePhonemes(string phonemes, Tensor<T> voiceStyle, double speed = 1)
    {
        ThrowIfDisposed();
        if (_native is null) throw new InvalidOperationException("Native Kokoro is required.");
        using var scope = new NoGradScope<T>();
        _native.SynthesisSeed = _options.SynthesisSeed;
        return _native.Forward(EncodePhonemes(phonemes), voiceStyle, speed).Audio;
    }

    private Tensor<T> SelectVoiceStyle(int phonemeLength)
    {
        if (Voice?.Reference is { } style && style.Length == 256) return style;
        if (_voiceBank is null) throw new InvalidOperationException("Load a Kokoro voice bank or supply Voice.Reference with 256 style values.");
        int index = Math.Max(0, phonemeLength - 1);
        if (index >= _voiceBank.Shape[0]) throw new ArgumentException("The sentence exceeds the loaded voice bank's length range.");
        return Engine.TensorSlice(_voiceBank, new[] { index, 0 }, new[] { 1, 256 });
    }
}
