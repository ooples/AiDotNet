using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Whisper;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MegaTTS 3's aligner against the release's own code (tools/reference-data/megatts3_aligner_reference.py) on a tiny
/// model: the log-mel, the encoder's output, the greedy decoding (768 steps) and its last logits, and align()'s
/// post-processing of two token sequences (durations short of and over the prompt's frames).
/// </summary>
public class MegaTts3AlignerReferenceTests
{
    [Fact(Timeout = 600000)]
    public async Task DecodingAndPostProcessing_MatchTheRelease()
    {
        await Task.Yield();
        const string fileName = "megatts3_aligner_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        var f = JObject.Parse(File.ReadAllText(path));
        double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();
        int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();
        var engine = AiDotNetEngine.Current;
        var aligner = new MegaTts3Aligner<double>(engine, new List<LayerBase<double>>(),
            new WhisperDimensions(80, 3000, 16, 2, 2, 6800, 2048, 16, 2, 2) { UntiedOutput = true, VariableAudioLength = true });
        var state = (JObject)f["state"]!;
        aligner.LoadTorch(name => name == "encoder.positional_embedding" ? OpenAiWhisperNetwork<double>.Sinusoids(3000, 16) : Doubles(state[name]!));

        void Close(JToken expected, Tensor<double> actual, string what)
        {
            var e = Doubles(expected);
            Assert.Equal(e.Length, actual.Length);
            for (int i = 0; i < e.Length; i++)
                Assert.True(Math.Abs(e[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {actual[i]}");
        }
        using var _ = new NoGradScope<double>();
        var mel = aligner.LogMel(Doubles(f["audio"]!));
        Assert.Equal((int)f["frames"]!, mel.Shape[1]);
        Close(f["mel"]!, mel, "mel");
        Close(f["features"]!, aligner.Network.Encode(mel), "features");
        var (tokens, last) = aligner.Decode(mel);
        Assert.Equal(Ints(f["tokens"]!), tokens);
        Close(f["last_logits"]!, last, "last logits");
        foreach (var post in ((JObject)f["posts"]!).Properties().Select(p => p.Value))
        {
            var (phones, tones, map) = MegaTts3Aligner<double>.Split(Ints(post["tokens"]!), (int)post["frames"]!);
            Assert.Equal(Ints(post["ph"]!), phones);
            Assert.Equal(Ints(post["tone"]!), tones);
            Assert.Equal(Ints(post["mel2ph"]!), map);
        }
    }
}
