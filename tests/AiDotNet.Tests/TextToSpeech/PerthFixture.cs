using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>A compact Perth watermark checkpoint (the hidden-8 PerthNet of perth_reference.json, which resemble-perth
/// itself was checked against) written once as a safetensors file, for models that watermark their output.</summary>
internal static class PerthFixture
{
    private static readonly Lazy<string> Checkpoint = new(Write);

    /// <summary>The checkpoint's path.</summary>
    internal static string TinyCheckpointPath => Checkpoint.Value;

    private static string Write()
    {
        var state = (JObject)WhisperSpeechT2SReferenceTests.Load("perth_reference.json")["tiny_state"]!;
        string path = Path.Combine(Path.GetTempPath(), $"aidotnet-perth-tiny-{Guid.NewGuid():N}.safetensors");
        var header = new JObject();
        long offset = 0;
        var tensors = new List<float[]>();
        foreach (var property in state.Properties())
        {
            var values = property.Value.Select(v => (float)v).ToArray();
            // Biases are [hidden]; the loader reads the other tensors by their element count.
            header[property.Name] = new JObject
            {
                ["dtype"] = "F32",
                ["shape"] = new JArray(values.Length),
                ["data_offsets"] = new JArray(offset, offset + 4L * values.Length),
            };
            offset += 4L * values.Length;
            tensors.Add(values);
        }
        var headerBytes = Encoding.UTF8.GetBytes(header.ToString(Newtonsoft.Json.Formatting.None));
        using (var writer = new BinaryWriter(File.Create(path)))
        {
            writer.Write((long)headerBytes.Length);
            writer.Write(headerBytes);
            foreach (var values in tensors)
                foreach (var v in values) writer.Write(v);
        }
        return path;
    }
}
