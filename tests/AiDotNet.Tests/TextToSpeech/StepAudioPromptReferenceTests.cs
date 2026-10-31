using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.MultiModal;
using AiDotNet.Tokenization.Algorithms;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Step-Audio's text as the release encodes it under transformers 4.48.3: plain encodings, the TTS prompt
/// (tts.py's tokenize) and the chat prompt (stepaudio.py's apply_chat_template), with the system prompts
/// (<c>tools/reference-data/stepaudio_prompt_reference.py</c>).</summary>
public class StepAudioPromptReferenceTests
{
    private static JObject Load(string name)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", name);
        return JObject.Parse(File.ReadAllText(path));
    }

    [Fact(Timeout = 120000)]
    public async Task Prompts_MatchTheRelease()
    {
        await Task.Yield();
        var model = new SentencePieceModel(Convert.FromBase64String((string)Load("stepaudio_tokenizer_reference.json")["model_base64"]!));
        var tokens = new StepAudioTokens(model);
        var f = Load("stepaudio_prompt_reference.json");
        int[] Ids(JToken t) => t.Select(v => (int)v).ToArray();
        foreach (var p in ((JObject)f["system_prompts"]!).Properties())
            Assert.Equal((string)p.Value!, StepAudioTokens.SystemPrompts[p.Name]);
        foreach (var e in f["encodings"]!)
            Assert.Equal(Ids(e["ids"]!), tokens.Encode((string)e["text"]!));
        var code = Ids(f["prompt_code"]!);
        foreach (var c in f["tts"]!)
            Assert.Equal(Ids(c["ids"]!), tokens.TtsPrompt((string)c["text"]!, (string)c["prompt_text"]!, (string)c["speaker"]!, code));
        string audio = (string)f["audio_tokens"]!;
        foreach (var c in f["chat"]!)
        {
            var turns = c["messages"]!.Select(m => ((string)m["role"]!, m["content"] is JObject ? audio : (string)m["content"]!)).ToList();
            Assert.Equal(Ids(c["ids"]!), tokens.ChatPrompt(turns));
        }
        Assert.Equal("RAP", StepAudioTokens.InstructionName("(RAP)我们一起唱"));
        Assert.Equal("哼唱", StepAudioTokens.InstructionName("（哼唱）啦啦啦"));
        Assert.Equal(new[] { 65536 + 1, 65536 + 2, 65536 + 1024 + 5, 65536 + 1024 + 6, 65536 + 1024 + 7 },
            StepAudioTokens.AudioIds(new[] { 1, 2, 9 }, new[] { 5, 6, 7, 8 }));
    }
}
