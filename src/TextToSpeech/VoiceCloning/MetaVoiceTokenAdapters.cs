namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>The author's flattened first-stage and tilted second-stage extraction rules,
/// including removal of the final text token and truncation to the shortest audio hierarchy.</summary>
internal static class MetaVoiceTokenAdapters
{
    internal static (int[] Text, int[][] Audio) Decode(int[][] streams, bool flattened, int codebookSize = 1024)
    {
        if (streams is null || (flattened ? streams.Length != 1 : streams.Length < 2))
            throw new ArgumentException("MetaVoice adapter stream count differs.", nameof(streams));
        var text = new List<int>();
        var audio = Enumerable.Range(0, flattened ? 2 : streams.Length).Select(_ => new List<int>()).ToArray();
        if (flattened)
        {
            foreach (int token in streams[0])
            {
                if (token < codebookSize) audio[0].Add(token);
                else if (token < 2 * codebookSize) audio[1].Add(token - codebookSize);
                else if (token > 2 * codebookSize) text.Add(token);
            }
        }
        else
        {
            for (int i = 0; i < streams.Length; i++) foreach (int token in streams[i])
            {
                if (token < codebookSize) audio[i].Add(token);
                else if (i == 0 && token > codebookSize) text.Add(token);
            }
        }
        int length = audio.Min(x => x.Count);
        return (text.Take(Math.Max(0, text.Count - 1)).ToArray(), audio.Select(x => x.Take(length).ToArray()).ToArray());
    }
}
