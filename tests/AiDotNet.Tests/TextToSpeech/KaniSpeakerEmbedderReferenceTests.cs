using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Foundations;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// KaniTTS 2's speaker embedder against the release's EmbeddingsModel over transformers' WavLMModel in WavLM-Large's
/// layout (tools/reference-data/kani_speaker_embedder_reference.py): the WavLM's last hidden state and the embedding.
/// </summary>
public class KaniSpeakerEmbedderReferenceTests
{
    [Fact(Timeout = 300000)]
    public async Task HiddenStateAndEmbedding_MatchTheRelease()
    {
        await Task.Yield();
        const string fileName = "kani_speaker_embedder_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        var f = JObject.Parse(File.ReadAllText(path));
        var w = f["wavlm"]!;
        int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();
        var embedder = new KaniSpeakerEmbedder<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), new KaniSpeakerEmbedderConfiguration
        {
            WavLm = new WavLmConfiguration
            {
                ConvDims = Ints(w["conv_dim"]!), ConvKernels = Ints(w["conv_kernel"]!), ConvStrides = Ints(w["conv_stride"]!),
                HiddenSize = (int)w["hidden_size"]!, Layers = (int)w["num_hidden_layers"]!, Heads = (int)w["num_attention_heads"]!,
                IntermediateSize = (int)w["intermediate_size"]!, PositionalConvKernel = (int)w["num_conv_pos_embeddings"]!,
                PositionalConvGroups = (int)w["num_conv_pos_embedding_groups"]!, RelativePositionBuckets = (int)w["num_buckets"]!,
                RelativePositionMaxDistance = (int)w["max_bucket_distance"]!, LayerNormFeatureEncoder = true, StableLayerNorm = true,
            },
            IntermediateSize = (int)w["top_interm_size"]!, EmbeddingSize = (int)w["embd_size"]!,
        });
        var state = (JObject)f["state"]!;
        var used = new HashSet<string>();
        embedder.LoadTorchWeights(name =>
        {
            if (state[name] is not JArray values) return null;
            used.Add(name);
            return values.Select(v => (double)v).ToArray();
        });
        // masked_spec_embed replaces masked frames in training only; the frozen embedder never masks.
        Assert.Empty(state.Properties().Select(p => p.Name).Where(n => !used.Contains(n) && n != "wavlm.masked_spec_embed"));

        using var _ = new NoGradScope<double>();
        var audio = f["audio"]!.Select(v => (double)v).ToArray();
        var input = new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio));
        void Close(JToken expected, Tensor<double> actual, string what)
        {
            var e = expected.Select(v => (double)v).ToArray();
            Assert.Equal(e.Length, actual.Length);
            for (int i = 0; i < e.Length; i++)
                Assert.True(Math.Abs(e[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {actual[i]}");
        }
        // The release normalizes the waveform before WavLM; the embedder does the same inside Embed.
        double mean = audio.Average(), std = Math.Sqrt(audio.Sum(v => (v - mean) * (v - mean)) / (audio.Length - 1));
        var normed = new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio.Select(v => (v - mean) / (std + 1e-10)).ToArray()));
        Close(f["hidden"]!, embedder.WavLm.HiddenStates(normed)[^1], "hidden");
        Close(f["embedding"]!, embedder.Embed(input), "embedding");
    }
}
