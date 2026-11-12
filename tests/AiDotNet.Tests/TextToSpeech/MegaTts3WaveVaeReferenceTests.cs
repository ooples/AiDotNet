using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MegaTTS 3's WaveVAE against the release's own code (tools/reference-data/megatts3_wavvae_reference.py) on a tiny model:
/// the SEANet encoder's posterior (mean and log-variance), a latent drawn with a given draw, and the decoded audio.
/// </summary>
public class MegaTts3WaveVaeReferenceTests
{
    [Fact(Timeout = 300000)]
    public async Task PosteriorAndDecoding_MatchTheRelease()
    {
        await Task.Yield();
        const string fileName = "megatts3_wavvae_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        var f = JObject.Parse(File.ReadAllText(path));
        double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();
        var engine = AiDotNetEngine.Current;
        var vae = new MegaTts3WaveVae<double>(engine, new List<LayerBase<double>>(), new List<LayerBase<double>>(), new MegaTts3WaveVaeConfiguration
        {
            EncoderRatios = new[] { 4, 6 }, EncoderFilters = 4, EncoderDimension = 16, LatentChannels = 4, UpsamplerChannels = 20,
            LatentUpsampling = 4, GeneratorFilters = 2, GeneratorRatios = new[] { 3, 2 },
        });
        var state = (JObject)f["state"]!;
        var used = new HashSet<string>();
        vae.LoadTorch(engine, name =>
        {
            if (state[name] is not JArray v) return null;
            used.Add(name);
            return Doubles(v);
        }, includeEncoder: true);
        Assert.Empty(state.Properties().Select(p => p.Name).Where(n => !used.Contains(n)));

        void Close(JToken expected, Tensor<double> actual, string what)
        {
            var e = Doubles(expected);
            Assert.Equal(e.Length, actual.Length);
            for (int i = 0; i < e.Length; i++)
                Assert.True(Math.Abs(e[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {actual[i]}");
        }
        using var _ = new NoGradScope<double>();
        var audio = Doubles(f["audio"]!);
        var (mean, logVariance) = vae.Encode(new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio)));
        // The reference's posterior is [1, latent, frames]; ours is [1, frames, latent].
        Close(f["mean"]!, engine.TensorPermute(mean, new[] { 0, 2, 1 }), "mean");
        Close(f["logvar"]!, engine.TensorPermute(logVariance, new[] { 0, 2, 1 }), "log-variance");
        var eps = engine.TensorPermute(new Tensor<double>(new[] { 1, 4, mean.Shape[1] }, new Vector<double>(Doubles(f["eps"]!))), new[] { 0, 2, 1 });
        var latent = vae.EncodeLatent(new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio)), eps);
        Close(f["latent"]!, latent, "latent");
        Close(f["wav"]!, vae.Decode(latent), "audio");
    }
}
