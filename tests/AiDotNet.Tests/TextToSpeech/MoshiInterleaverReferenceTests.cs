using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Moshi's inner-monologue text stream (Eq. 5) against the release's fine-tuning <c>Interleaver</c> with Helium's
/// tokenizer (<c>tools/reference-data/moshi_interleaver_reference.py</c>).</summary>
public class MoshiInterleaverReferenceTests
{
    private static JObject Load(string name)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", name);
        return JObject.Parse(File.ReadAllText(path));
    }

    [Fact(Timeout = 120000)]
    public async Task TextStream_MatchesTheRelease()
    {
        await Task.Yield();
        string tokenizer = Path.Combine(Path.GetTempPath(), $"aidotnet_helium_{System.Diagnostics.Process.GetCurrentProcess().Id}.model");
        File.WriteAllBytes(tokenizer, Convert.FromBase64String((string)Load("moshi_tokenizer_reference.json")["model_base64"]!));
        // Mimi's real 12.5 Hz frame (24 kHz, ratios 8·6·5·4, stride 2) around tiny networks.
        var options = new MoshiOptions
        {
            HiddenDim = 8, NumHeads = 1, Layers = 1, FeedForward = 16, Context = 16, DepthDim = 8, DepthHeads = 1, DepthLayers = 1,
            DepthFeedForward = 16, Codebooks = 2, CodebookSize = 4, MimiFilters = 2, MimiDimension = 8, MimiTransformerLayers = 1,
            MimiHeads = 1, MimiFeedForward = 8, MimiContext = 4, MimiCodebooks = 2, MimiCodebookDim = 4, TokenizerPath = tokenizer,
        };
        using var model = new Moshi<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1), options);
        Assert.Equal(12.5, model.Core.FrameRate, 12);
        foreach (var c in Load("moshi_interleaver_reference.json")["cases"]!)
        {
            var words = c["words"]!.Select(w => new MoshiWord((string)w[0]!, (double)w[1]!, (double)w[2]!)).ToList();
            var stream = model.Core.TextStream(words, (int)c["frames"]!, (double)c["delay"]!);
            Assert.Equal(c["stream"]!.Select(v => (int)v).ToArray(), stream);
        }
    }
}
