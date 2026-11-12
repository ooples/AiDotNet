using System.Text.Json;
using AiDotNet.TextToSpeech.VoiceCloning;

internal static class MetaVoiceTokenizerAudit
{
    internal static void Run(string fixture)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        foreach (var stage in document.RootElement.GetProperty("stages").EnumerateObject())
        {
            var root = stage.Value;
            string checkpoint = root.GetProperty("checkpoint_path").GetString()!;
            var tokenizer = MetaVoiceTokenizer.FromCheckpoint(checkpoint);
            foreach (var example in root.GetProperty("tokenizer_examples").EnumerateArray())
            {
                var expected = example.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray();
                if (!tokenizer.Encode(example.GetProperty("text").GetString()!).SequenceEqual(expected))
                    throw new InvalidDataException("Original MetaVoice checkpoint token IDs differ: " + stage.Name);
            }
            var metadata = MetaVoiceTokenizer.ReadMetadata(checkpoint);
            var arguments = (Dictionary<string, object?>)metadata["model_args"]!;
            if (stage.Name == "first" && Convert.ToDouble(arguments["spkemb_dropout"]) != .1)
                throw new InvalidDataException("MetaVoice checkpoint floating-point metadata differs.");
            Console.WriteLine($"MetaVoice {stage.Name}: original PTH metadata, all512 byte ranks, offset{tokenizer.Offset}, end{tokenizer.EndToken}, and author token IDs verified.");
        }
    }
}
