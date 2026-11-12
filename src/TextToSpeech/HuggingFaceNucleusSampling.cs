namespace AiDotNet.TextToSpeech;

/// <summary>HF's ascending-tail nucleus rule, preserving at least one candidate.</summary>
internal static class HuggingFaceNucleusSampling
{
    internal static void Retain(List<(int Token, double Weight)> candidates, double topP)
    {
        if (!(topP >= 0 && topP < 1)) return;
        var ascending = candidates.OrderBy(item => item.Weight).ThenBy(item => item.Token).ToList();
        double total = ascending.Sum(item => item.Weight), cumulative = 0;
        var removed = new HashSet<int>();
        for (int i = 0; i < ascending.Count - 1; i++)
        {
            cumulative += ascending[i].Weight / total;
            if (cumulative <= 1 - topP) removed.Add(ascending[i].Token);
        }
        candidates.RemoveAll(item => removed.Contains(item.Token));
    }
}
