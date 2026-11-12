namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The released API's candidate padding/repair and post-ranking latent cut.</summary>
internal static class TortoiseCandidateTail
{
    internal static int[] Prepare(IReadOnlyList<int> generated, int maximum, int stopToken, int codebookSize)
    {
        var result = new int[maximum];
        for (int i = 0; i < maximum; i++) result[i] = i < generated.Count ? generated[i] : stopToken;
        int first = Array.IndexOf(result, stopToken);
        if (first < 0) return result;
        // Literal release codes are 83, 45, 248. Modulo only supports explicitly
        // reduced experimental codebooks; it is the identity for the release.
        for (int i = first; i < maximum; i++) result[i] = 83 % codebookSize;
        if (maximum >= 3)
        {
            result[maximum - 3] = 45 % codebookSize;
            result[maximum - 2] = 45 % codebookSize;
            result[maximum - 1] = 248 % codebookSize;
        }
        return result;
    }

    internal static int LatentLength(IReadOnlyList<int> codes, int codebookSize)
    {
        int run = 0;
        for (int i = 0; i < codes.Count; i++)
        {
            run = codes[i] == 83 % codebookSize ? run + 1 : 0;
            if (run > 8) return i;
        }
        return codes.Count;
    }
}
