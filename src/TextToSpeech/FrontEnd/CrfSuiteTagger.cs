using System.Linq;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// CRFsuite's first-order linear-chain tagger (<c>crf1d</c>, as python-crfsuite's <c>Tagger.tag</c> runs it): each
/// item's state scores are the sum, over its attributes in order, of the attribute's value times each of its feature
/// weights; the labels are the Viterbi path over those scores plus the label-to-label transition weights, ties going to
/// the lower label id as CRFsuite's strict comparisons do. Attributes the model does not know are dropped.
/// </summary>
internal sealed class CrfSuiteTagger
{
    private readonly string[] _labels;
    private readonly Dictionary<string, (int Label, double Weight)[]> _attributes;
    private readonly double[][] _transitions;

    private CrfSuiteTagger(string[] labels, Dictionary<string, (int, double)[]> attributes, double[][] transitions)
    {
        _labels = labels;
        _attributes = attributes;
        _transitions = transitions;
    }

    /// <summary>Loads a model exported by <c>tools/reference-data/melo_gruut_resources.py</c>.</summary>
    public static CrfSuiteTagger Load(string resource)
    {
        var model = JObject.Parse(string.Join("\n", ChineseText.ReadLines(resource)));
        var labels = ((JArray)model["labels"]!).Select(t => (string)t!).ToArray();
        var attributes = new Dictionary<string, (int, double)[]>(StringComparer.Ordinal);
        foreach (var entry in (JObject)model["attributes"]!)
            attributes[entry.Key] = ((JArray)entry.Value!).Select(f => ((int)f[0]!, (double)f[1]!)).ToArray();
        var transitions = ((JArray)model["transitions"]!).Select(row => ((JArray)row).Select(w => (double)w!).ToArray()).ToArray();
        return new CrfSuiteTagger(labels, attributes, transitions);
    }

    /// <summary>The most probable label sequence for the items, each a list of (attribute, value) in insertion order.</summary>
    public string[] Tag(IReadOnlyList<IReadOnlyList<(string Attribute, double Value)>> items)
    {
        int count = items.Count, labelCount = _labels.Length;
        if (count == 0) return Array.Empty<string>();
        var state = new double[count][];
        for (int t = 0; t < count; t++)
        {
            var scores = new double[labelCount];
            foreach (var (attribute, value) in items[t])
            {
                if (!_attributes.TryGetValue(attribute, out var features)) continue;
                foreach (var (label, weight) in features) scores[label] += weight * value;
            }
            state[t] = scores;
        }

        var alpha = new double[count][];
        var back = new int[count][];
        alpha[0] = (double[])state[0].Clone();
        for (int t = 1; t < count; t++)
        {
            alpha[t] = new double[labelCount];
            back[t] = new int[labelCount];
            for (int j = 0; j < labelCount; j++)
            {
                double maxScore = -double.MaxValue;
                int argmax = -1;
                for (int i = 0; i < labelCount; i++)
                {
                    double score = alpha[t - 1][i] + _transitions[i][j];
                    if (maxScore < score)
                    {
                        maxScore = score;
                        argmax = i;
                    }
                }
                if (argmax >= 0) back[t][j] = argmax;
                alpha[t][j] = maxScore + state[t][j];
            }
        }

        var path = new int[count];
        double best = -double.MaxValue;
        for (int i = 0; i < labelCount; i++)
        {
            if (best < alpha[count - 1][i])
            {
                best = alpha[count - 1][i];
                path[count - 1] = i;
            }
        }
        for (int t = count - 1; t > 0; t--) path[t - 1] = back[t][path[t]];
        return path.Select(p => _labels[p]).ToArray();
    }
}
