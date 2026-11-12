using AiDotNet.ModelLoading;

namespace AiDotNet.TextToSpeech.CodecBased;

public abstract partial class FishSpeechBase<T>
{
    /// <summary>The release's codec checkpoint file name in the model directory.</summary>
    public const string CodecCheckpointName = "firefly-gan-vq-fsq-8x1024-21hz-generator.pth";

    /// <summary>The checkpoint's state with a wrapper's prefix stripped, as the releases' loaders do: a Lightning
    /// <c>state_dict</c>, then <paramref name="prefixes"/> (<c>model.</c> for the transformer, <c>generator.</c> for the
    /// codec).</summary>
    private static TorchTensorSource Unwrapped(TorchCheckpointFile file, params string[] prefixes)
    {
        string prefix = "";
        foreach (var next in new[] { "state_dict." }.Concat(prefixes))
            if (file.Names.Any(k => k.StartsWith(prefix + next, StringComparison.Ordinal))) prefix += next;
        return file.WithPrefix(prefix);
    }

    /// <summary>
    /// Loads a release's directory (<c>fishaudio/fish-speech-1.5</c> or <c>-1.4</c>): the Dual-AR transformer
    /// (<c>model.pth</c>), Firefly-GAN (<see cref="CodecCheckpointName"/>) and the tokenizer (1.5:
    /// <c>tokenizer.tiktoken</c>; 1.4: <c>tokenizer.json</c>), unless <see cref="FishSpeechFamilyOptions.TokenizerPath"/>
    /// names another.
    /// </summary>
    public void LoadReferenceWeights(string directory)
    {
        ThrowIfDisposed();
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        if (!HasPaperLayers) throw NotNative();
        _options.TokenizerPath ??= Path.Combine(directory, _version == FishSpeechVersion.V15 ? "tokenizer.tiktoken" : "tokenizer.json");
        ResetTokenizer();
        _ = Tokenizer;
        LoadTransformer(Path.Combine(directory, "model.pth"));
        LoadCodec(Path.Combine(directory, CodecCheckpointName));
    }

    /// <summary>Loads the Dual-AR transformer from the release's <c>model.pth</c> (or any checkpoint with its names).</summary>
    public void LoadTransformer(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        using var checkpoint = TorchCheckpointFile.Open(path ?? throw new ArgumentNullException(nameof(path)));
        Transformer.Load(Unwrapped(checkpoint, "model."));
        _preferenceReference = null;
    }

    /// <summary>Loads Firefly-GAN from the release's generator checkpoint.</summary>
    public void LoadCodec(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        using var checkpoint = TorchCheckpointFile.Open(path ?? throw new ArgumentNullException(nameof(path)));
        Codec.Load(Unwrapped(checkpoint, "generator."));
    }
}
