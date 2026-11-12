using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.EndToEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MeloTTS's text side as its model reads it, against MeloTTS's own code (tools/reference-data/melo_tts_text_reference.py):
/// get_text_for_tts_infer's symbols, tones, languages and blank-adjusted phones per piece, the BERT tokenizer's input ids,
/// and split_sentence, for every language.
/// </summary>
public class MeloTextPipelineTests
{
    private static JsonElement Load()
    {
        const string name = "melo_tts_text_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static MeloLanguage Language(string code) => code switch
    {
        "EN" => MeloLanguage.English,
        "ZH" => MeloLanguage.Chinese,
        "ZH_MIX_EN" => MeloLanguage.ChineseMixEnglish,
        "JP" => MeloLanguage.Japanese,
        "KR" => MeloLanguage.Korean,
        "ES" => MeloLanguage.Spanish,
        "FR" => MeloLanguage.French,
        _ => throw new ArgumentException(code),
    };

    private static string Join(JsonElement array) => string.Join(" ", array.EnumerateArray().Select(v => v.ToString()));

    [Fact(Timeout = 300000)]
    public async Task ModelInputs_MatchMeloTTS()
    {
        await Task.Yield();
        foreach (var language in Load().GetProperty("languages").EnumerateObject())
        {
            var lang = Language(language.Name);
            foreach (var s in language.Value.GetProperty("sentences").EnumerateArray())
            {
                string text = s.GetProperty("text").GetString()!;
                var prepared = MeloTextPipeline.Prepare(lang, text);
                Assert.Equal(s.GetProperty("normalized").GetString(), prepared.Normalized);
                Assert.Equal(Join(s.GetProperty("symbols")), string.Join(" ", prepared.Symbols));
                Assert.Equal(Join(s.GetProperty("tones")), string.Join(" ", prepared.Tones));
                Assert.Equal(Join(s.GetProperty("languages")), string.Join(" ", prepared.Languages));
                Assert.Equal(Join(s.GetProperty("word2ph")), string.Join(" ", prepared.PhonesPerPiece));
                Assert.True(Join(s.GetProperty("bert_ids")) == string.Join(" ", MeloTextPipeline.BertTokenIds(lang, prepared.Normalized)),
                    $"{language.Name} {text}: BERT ids");
            }
            var split = language.Value.GetProperty("split");
            if (split.ValueKind == JsonValueKind.Null) continue;
            var sentences = MeloTextPipeline.SplitSentence(split.GetProperty("text").GetString()!, lang);
            Assert.Equal(split.GetProperty("sentences").EnumerateArray().Select(v => v.GetString()), sentences);
        }
    }
}
