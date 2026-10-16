using AiDotNet.Helpers;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>A box that a caption's phrase names, for training a grounding model.</summary>
/// <param name="Phrase">The phrase, one of the caption's period-separated phrases (for "cat . remote control ." either
/// "cat" or "remote control").</param>
/// <param name="X1">Left edge, as a fraction of the image width.</param>
/// <param name="Y1">Top edge, as a fraction of the image height.</param>
/// <param name="X2">Right edge, as a fraction of the image width.</param>
/// <param name="Y2">Bottom edge, as a fraction of the image height.</param>
public sealed record GroundingBox(string Phrase, double X1, double Y1, double X2, double Y2);

/// <summary>A region a grounding model found for a caption.</summary>
/// <param name="X1">Left edge, as a fraction of the image width.</param>
/// <param name="Y1">Top edge, as a fraction of the image height.</param>
/// <param name="X2">Right edge, as a fraction of the image width.</param>
/// <param name="Y2">Bottom edge, as a fraction of the image height.</param>
/// <param name="Score">The highest token probability of the region's query.</param>
/// <param name="Phrase">The caption's words whose tokens scored above the text threshold.</param>
public sealed record GroundingDetection(double X1, double Y1, double X2, double Y2, double Score, string Phrase);

/// <summary>
/// A caption as Grounding DINO reads it (the official <c>preprocess_caption</c>: lowercased, trimmed and ended with a
/// period), tokenized by BERT-base-uncased, with its phrases: the runs of tokens between the [CLS], [SEP], "." and "?"
/// delimiters.
/// </summary>
internal sealed class GroundingDinoPrompt
{
    private static readonly Lazy<WordPieceTokenizer> Tokenizer = new(WordPieceTokenizer.BertBaseUncased);

    private GroundingDinoPrompt(string caption, int[] ids)
    {
        Caption = caption;
        Ids = ids;
    }

    public string Caption { get; }

    /// <summary>The token ids, [CLS] first and [SEP] last.</summary>
    public int[] Ids { get; }

    public int[,] IdMatrix
    {
        get
        {
            var m = new int[1, Ids.Length];
            for (int j = 0; j < Ids.Length; j++) m[0, j] = Ids[j];
            return m;
        }
    }

    public bool[,] Mask
    {
        get
        {
            var m = new bool[1, Ids.Length];
            for (int j = 0; j < Ids.Length; j++) m[0, j] = true;
            return m;
        }
    }

    public static string Preprocess(string caption)
    {
        if (string.IsNullOrWhiteSpace(caption)) throw new ArgumentException("The caption is empty.", nameof(caption));
        var result = caption.ToLowerInvariant().Trim();
        return result.EndsWith(".", StringComparison.Ordinal) ? result : result + ".";
    }

    public static GroundingDinoPrompt Create(string caption)
    {
        var text = Preprocess(caption);
        return new GroundingDinoPrompt(text, Tokenizer.Value.Encode(text).TokenIds.ToArray());
    }

    /// <summary>The index of the caption's phrase whose tokens are exactly <paramref name="phrase"/>'s.</summary>
    public int PhraseIndex(string phrase)
    {
        var tokens = Tokenizer.Value.Encode(phrase).TokenIds.Where(id => Array.IndexOf(GroundingDinoLoss<double>.Delimiters, id) < 0).ToArray();
        var groups = Phrases();
        for (int i = 0; i < groups.Count; i++)
            if (groups[i].Select(j => Ids[j]).SequenceEqual(tokens)) return i;
        throw new ArgumentException(
            $"\"{phrase}\" is not one of the caption's phrases: {string.Join(", ", groups.Select(g => "\"" + Decode(g.Select(j => Ids[j])) + "\""))}.",
            nameof(phrase));
    }

    /// <summary>Each phrase's token positions, in caption order.</summary>
    public List<int[]> Phrases()
    {
        var groups = new List<int[]>();
        var current = new List<int>();
        for (int j = 0; j <= Ids.Length; j++)
        {
            if (j == Ids.Length || Array.IndexOf(GroundingDinoLoss<double>.Delimiters, Ids[j]) >= 0)
            {
                if (current.Count > 0) groups.Add(current.ToArray());
                current.Clear();
            }
            else current.Add(j);
        }
        return groups;
    }

    /// <summary>The official <c>get_phrases_from_posmap</c>: the tokens above the threshold, past [CLS], decoded, with
    /// periods removed.</summary>
    public string PhraseAbove(double[] probabilities, double threshold)
    {
        var ids = new List<int>();
        for (int j = 1; j < Math.Min(Ids.Length, probabilities.Length); j++)
            if (probabilities[j] > threshold) ids.Add(Ids[j]);
        return Decode(ids).Replace(".", "");
    }

    /// <summary>BERT's decode: tokens joined by spaces, word pieces merged, and the tokenizer's clean-up of spaces before
    /// punctuation and contractions (Hugging Face <c>clean_up_tokenization</c>).</summary>
    internal static string Decode(IEnumerable<int> ids)
    {
        var text = string.Join(" ", ids.Select(id => Tokenizer.Value.Vocabulary.GetToken(id) ?? "[UNK]")).Replace(" ##", "");
        foreach (var (from, to) in new[] { (" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"), (" n't", "n't"),
                     (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re") })
            text = text.Replace(from, to);
        return text;
    }

    /// <summary>
    /// The official <c>predict</c>: every query whose highest token probability exceeds <paramref name="boxThreshold"/>,
    /// with that probability as its score, its box as corners, and the phrase of its tokens above
    /// <paramref name="textThreshold"/>. The queries keep their order; there is no non-maximum suppression.
    /// </summary>
    public List<GroundingDetection> Detections<T>(GroundingDinoOutput<T> output, int image, double boxThreshold, double textThreshold)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var logits = output.Logits[^1];
        var boxes = output.Boxes[^1];
        int queries = logits.Shape[1], width = logits.Shape[2];
        var result = new List<GroundingDetection>();
        for (int q = 0; q < queries; q++)
        {
            var probabilities = new double[width];
            double best = 0;
            for (int j = 0; j < width; j++)
            {
                probabilities[j] = 1 / (1 + Math.Exp(-ops.ToDouble(logits[image, q, j])));
                best = Math.Max(best, probabilities[j]);
            }
            if (best <= boxThreshold) continue;
            double cx = ops.ToDouble(boxes[image, q, 0]), cy = ops.ToDouble(boxes[image, q, 1]);
            double w = ops.ToDouble(boxes[image, q, 2]), h = ops.ToDouble(boxes[image, q, 3]);
            result.Add(new GroundingDetection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, best, PhraseAbove(probabilities, textThreshold)));
        }
        return result;
    }

    /// <summary>The official <c>phrases2classes</c>: the first class whose name the phrase contains, or -1.</summary>
    public static int ClassOf(string phrase, IReadOnlyList<string> classes)
    {
        for (int c = 0; c < classes.Count; c++)
            if (phrase.Contains(classes[c])) return c;
        return -1;
    }
}
