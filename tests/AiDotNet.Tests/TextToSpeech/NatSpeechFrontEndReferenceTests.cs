using System;
using System.IO;
using System.Linq;
using AiDotNet.TextToSpeech.FrontEnd;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// NATSpeech's English front end against its own code (tools/reference-data/natspeech_frontend_reference.py): the cleaned
/// text, the words and phones with separators and <c>&lt;BOS&gt;</c>/<c>&lt;EOS&gt;</c>, and the token ids.
/// </summary>
public class NatSpeechFrontEndReferenceTests
{
    [Fact]
    public void English_MatchesNatSpeech()
    {
        const string fileName = "natspeech_frontend_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        var f = JObject.Parse(File.ReadAllText(path));
        var vocabulary = NatSpeechFrontEnd.Vocabulary(NatSpeechFrontEnd.EnglishPhones());
        Assert.Equal(f["vocabulary"]!.Select(v => (string)v!), vocabulary);
        foreach (var c in f["cases"]!)
        {
            string text = (string)c["text"]!;
            Assert.Equal((string)c["clean"]!, NatSpeechFrontEnd.PreprocessEnglish(text).Trim());
            var words = NatSpeechFrontEnd.EnglishWords(text);
            Assert.Equal(c["words"]!.Select(v => (string)v!), words.Select(w => w.Word));
            var phones = words.SelectMany(w => w.Phones).ToList();
            Assert.Equal(c["phones"]!.Select(v => (string)v!), phones);
            Assert.Equal(c["ids"]!.Select(v => (int)v), NatSpeechFrontEnd.Encode(phones, vocabulary));
        }
    }
}
