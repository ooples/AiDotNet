using System;
using System.Collections.Concurrent;
using System.IO;
using AiDotNet.Tokenization.HuggingFace;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Associates a tokenizer with a served model by name. The generation engine is token-ID native;
/// the OpenAI-compatible API is text in/out, so it needs a tokenizer per model to bridge the two.
/// </summary>
public interface ITokenizerRegistry
{
    /// <summary>Tries to resolve the tokenizer registered for <paramref name="modelName"/>.</summary>
    bool TryGet(string modelName, out ITokenizer tokenizer);

    /// <summary>Loads a tokenizer from a local directory/file or a HuggingFace id and registers it.</summary>
    ITokenizer LoadAndRegister(string modelName, string pathOrName);

    /// <summary>Registers an already-constructed tokenizer for a model.</summary>
    void Register(string modelName, ITokenizer tokenizer);

    /// <summary>Removes a model's tokenizer (e.g. on unload).</summary>
    bool Remove(string modelName);
}

/// <summary>Thread-safe in-memory <see cref="ITokenizerRegistry"/>.</summary>
public sealed class TokenizerRegistry : ITokenizerRegistry
{
    private readonly ConcurrentDictionary<string, ITokenizer> _map = new(StringComparer.Ordinal);

    /// <inheritdoc/>
    public bool TryGet(string modelName, out ITokenizer tokenizer) => _map.TryGetValue(modelName, out tokenizer!);

    /// <inheritdoc/>
    public void Register(string modelName, ITokenizer tokenizer)
    {
        Guard.NotNullOrEmpty(modelName);
        Guard.NotNull(tokenizer);
        _map[modelName] = tokenizer;
    }

    /// <inheritdoc/>
    public bool Remove(string modelName) => _map.TryRemove(modelName, out _);

    /// <inheritdoc/>
    public ITokenizer LoadAndRegister(string modelName, string pathOrName)
    {
        Guard.NotNullOrEmpty(modelName);
        if (string.IsNullOrWhiteSpace(pathOrName))
            throw new ArgumentException("Tokenizer path or name is required.", nameof(pathOrName));

        // A file (e.g. tokenizer.json) resolves to its containing directory; a directory or a
        // HuggingFace id is passed through to AutoTokenizer as-is.
        string target = File.Exists(pathOrName)
            ? Path.GetDirectoryName(Path.GetFullPath(pathOrName)) ?? pathOrName
            : pathOrName;

        var tokenizer = AutoTokenizer.FromPretrained(target);
        Register(modelName, tokenizer);
        return tokenizer;
    }
}
