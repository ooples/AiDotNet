using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MegaTTS 3's grapheme-to-phoneme model against the release's code path (tools/reference-data/megatts3_g2p_reference.py)
/// on a tiny Qwen2: the tokens generated from the text and phone-start tokens, and the phones and tones split from them.
/// </summary>
public class MegaTts3G2pReferenceTests
{
    [Fact(Timeout = 300000)]
    public async Task Generation_MatchesTheRelease()
    {
        await Task.Yield();
        const string fileName = "megatts3_g2p_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        var f = JObject.Parse(File.ReadAllText(path));
        int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();
        var engine = AiDotNetEngine.Current;
        var g2p = new MegaTts3G2p<double>(engine, new List<LayerBase<double>>(), new LlamaConfiguration
        {
            VocabSize = 1200, Hidden = 32, Intermediate = 64, Layers = 2, Heads = 4, KeyValueHeads = 2, RmsEpsilon = 1e-6, RopeTheta = 1000000,
            AttentionBias = true, TieWordEmbeddings = true,
        }, null, (int)f["speech_start"]!);
        var state = (JObject)f["state"]!;
        var used = new HashSet<string>();
        g2p.Network.LoadTorchWeights((name, _) =>
        {
            used.Add(name);
            return state[name]!.Select(v => (double)v).ToArray();
        });
        Assert.Empty(state.Properties().Select(p => p.Name).Where(n => !used.Contains(n)));
        var text = Ints(f["txt_token"]!);
        var generated = g2p.Generate(text);
        Assert.Equal(Ints(f["outputs"]!), generated);
        var (phones, tones) = g2p.Split(generated, text.Length);
        Assert.Equal(Ints(f["ph"]!), phones);
        Assert.Equal(Ints(f["tone"]!), tones);
    }
}
