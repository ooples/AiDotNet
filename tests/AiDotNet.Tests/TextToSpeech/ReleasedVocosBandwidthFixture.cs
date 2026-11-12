using System;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Vocoders;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Complete learned bandwidth-normalization component state from the pinned Vocos release.</summary>
internal static class ReleasedVocosBandwidthFixture
{
    private static readonly ConditionalWeakTable<object, object> Loaded = new();
    private static readonly Lazy<JObject> State = new(Read);

    private static JObject Read()
    {
        const string name = "vocos_released_bandwidth_state.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && directory is not null; directory = directory.Parent)
            path = Path.Combine(directory.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", name);
        var bytes = File.ReadAllBytes(path);
        using var sha = SHA256.Create();
        string digest = BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
        if (digest != "a93a373d4f74626809b2208e2b84009a961669ab4010d10dbc4f1c3b742d963b")
            throw new InvalidDataException("Released Vocos bandwidth component state SHA-256 mismatch.");
        return JObject.Parse(System.Text.Encoding.UTF8.GetString(bytes));
    }

    internal static void Load<T>(VocosEncodecDecoder<T> decoder)
    {
        Loaded.GetValue(decoder, _ =>
        {
            var c = decoder.Configuration;
            if (c.Dim != 384 || c.IntermediateDim != 1152 || c.Layers != 8 || c.Bandwidths.Length != 4)
                throw new InvalidDataException("Released normalization state requires the full Vocos hidden geometry; tensors are never sliced or reshaped.");
            var tensors = State.Value["tensors"]!.ToArray();
            var tables = decoder.Layers.OfType<TiedEmbeddingLayer<T>>().Where(table => !ReferenceEquals(table, decoder.Codebooks)).ToArray();
            if (tables.Length != 18 || tensors.Length != tables.Length)
                throw new InvalidDataException("Every released adaptive normalization table must be consumed.");
            var source = TorchTensorSource.From((name, shape) =>
            {
                var tensor = tensors.Single(value => (string)value["name"]! == name);
                if (!tensor["shape"]!.Select(value => (int)value).SequenceEqual(shape))
                    throw new InvalidDataException($"Released normalization shape mismatch for {name}.");
                return tensor["values"]!.Select(value => (double)value).ToArray();
            });
            for (int i = 0; i < tables.Length; i++) tables[i].LoadTable(source, (string)tensors[i]["name"]!);
            return new object();
        });
    }
}
