using System.Text.Json;
using AiDotNet.Tokenization.Algorithms;
using Xunit;

namespace AiDotNet.Tests.Tokenization;

public sealed class HuggingFaceWhitespaceBpeTests
{
    [Fact]
    public void UnicodeWhitespaceAndExistingByteLevel_MatchHuggingFaceIdsAndDecoding()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "Tokenization", "ReferenceData", "whitespace_bpe_reference.json")));
        foreach (var model in document.RootElement.GetProperty("models").EnumerateArray())
        {
            var tokenizer = ByteLevelBpeTokenizer.FromJson(model.GetProperty("tokenizer").GetRawText());
            foreach (var item in model.GetProperty("cases").EnumerateArray())
            {
                var expected = item.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray();
                var actual = tokenizer.Encode(item.GetProperty("text").GetString()!);
                Assert.True(expected.SequenceEqual(actual), $"{model.GetProperty("mode").GetString()} IDs differ for {item.GetProperty("text").GetString()}.");
                Assert.Equal(item.GetProperty("decoded").GetString(), tokenizer.Decode(actual));
                Assert.Equal(item.GetProperty("decoded_skip_special").GetString(), tokenizer.Decode(actual, skipSpecial: true));
            }
        }
    }
}
