using AiDotNet.ComputerVision.Segmentation.Common;
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
/// <typeparam name="T">The numeric type of the model that found it.</typeparam>
/// <param name="X1">Left edge, as a fraction of the image width.</param>
/// <param name="Y1">Top edge, as a fraction of the image height.</param>
/// <param name="X2">Right edge, as a fraction of the image width.</param>
/// <param name="Y2">Bottom edge, as a fraction of the image height.</param>
/// <param name="Score">The highest token probability of the region's query.</param>
/// <param name="Phrase">The caption's words whose tokens scored above the text threshold.</param>
public sealed record GroundingDetection<T>(T X1, T Y1, T X2, T Y2, T Score, string Phrase)
{
    /// <summary>
    /// The query's highest token logit, whose sigmoid is <see cref="Score"/>. The logit is the exact model output:
    /// Score rounds to 1 in Half above a logit of about 8.3 (and in float above about 17), so detections are ranked
    /// by it where their scores tie. Zero for a detection built outside the model.
    /// </summary>
    internal T Logit { get; init; } = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// The model's box (cx, cy, w, h) the corners came from, or <c>null</c> for a detection built outside the model. Grounded
    /// SAM 2 scales the corners from it in double, rounding to T once, as the official transform_boxes does.
    /// </summary>
    internal T[]? CenterSize { get; init; }
}

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
        return Tokenization.TokenizationCleanup.CleanUpSpaces(text);
    }

    /// <summary>
    /// The official <c>predict</c>: every query whose highest token probability exceeds <paramref name="boxThreshold"/>,
    /// with that probability as its score, its box as corners, and the phrase of its tokens above
    /// <paramref name="textThreshold"/>. The queries keep their order; there is no non-maximum suppression.
    /// </summary>
    public List<GroundingDetection<T>> Detections<T>(GroundingDinoOutput<T> output, int image, double boxThreshold, double textThreshold)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var logits = output.Logits[^1];
        var boxes = output.Boxes[^1];
        int queries = logits.Shape[1], width = logits.Shape[2];
        var tokenLogits = engine.Reshape(
            engine.TensorSlice(logits, new[] { image, 0, 0 }, new[] { 1, queries, width }), new[] { queries, width });
        // Both thresholds are decided on the logits (sigmoid(x) > p is x > ln(p / (1 - p))): a sigmoid rounded to T
        // lands on the threshold near the boundary and saturates at 1 for confident queries. The sigmoid is monotone,
        // so a query's score is the sigmoid of its highest logit.
        var bestLogit = engine.ReduceMax(tokenLogits, new[] { 1 }, false);
        var kept = MaskStatistics.SigmoidGreaterThan(bestLogit, boxThreshold).ToArray();
        var above = MaskStatistics.SigmoidGreaterThan(tokenLogits, textThreshold).ToArray();

        // (cx, cy, w, h) to corners: (cx, cy) -/+ (w, h) / 2.
        var box = engine.Reshape(engine.TensorSlice(boxes, new[] { image, 0, 0 }, new[] { 1, queries, 4 }), new[] { queries, 4 });
        var centre = engine.TensorSlice(box, new[] { 0, 0 }, new[] { queries, 2 });
        var half = engine.TensorMultiplyScalar(engine.TensorSlice(box, new[] { 0, 2 }, new[] { queries, 2 }), ops.FromDouble(0.5));
        var corners = engine.TensorConcatenate(new[] { engine.TensorSubtract(centre, half), engine.TensorAdd(centre, half) }, 1).ToArray();
        var scores = engine.TensorSigmoid(bestLogit).ToArray();
        var maxLogits = bestLogit.ToArray();
        var centreSize = box.ToArray();

        var zero = ops.Zero;
        var comparer = EqualityComparer<T>.Default;
        int tokens = Math.Min(Ids.Length, width);
        var result = new List<GroundingDetection<T>>();
        for (int q = 0; q < queries; q++)
        {
            if (comparer.Equals(kept[q], zero)) continue;
            // The official get_phrases_from_posmap: the tokens above the threshold, past [CLS], decoded, periods removed.
            var ids = new List<int>();
            for (int j = 1; j < tokens; j++)
                if (!comparer.Equals(above[q * width + j], zero)) ids.Add(Ids[j]);
            result.Add(new GroundingDetection<T>(corners[4 * q], corners[4 * q + 1], corners[4 * q + 2], corners[4 * q + 3], scores[q],
                Decode(ids).Replace(".", "")) { Logit = maxLogits[q], CenterSize = centreSize.AsSpan(4 * q, 4).ToArray() });
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

/// <summary>Detections as the tensors <see cref="Interfaces.IVisualGroundingModel{T}"/> returns.</summary>
internal static class GroundingRows<T>
{
    /// <summary>Rows <c>[n, 5]</c> of (x1, y1, x2, y2, score).</summary>
    public static Tensor<T> Boxes(IReadOnlyList<GroundingDetection<T>> detections)
        => Rows(detections, 5, d => new[] { d.X1, d.Y1, d.X2, d.Y2, d.Score });

    /// <summary>Rows <c>[n, 6]</c> of (x1, y1, x2, y2, class, score), the class by the official <c>phrases2classes</c>.</summary>
    public static Tensor<T> Classified(IReadOnlyList<GroundingDetection<T>> detections, IReadOnlyList<string> categories)
    {
        var classes = detections.Select(d => GroundingDinoPrompt.ClassOf(d.Phrase, categories)).ToArray();
        var classValues = AiDotNetEngine.Current.TensorArange<T>(-1, Math.Max(categories.Count, 1), 1).ToArray();
        // classValues[c + 1] is the class index c as T (from -1).
        return Rows(detections.Select((d, i) => (d, Class: classValues[classes[i] + 1])).ToList(), 6,
            r => new[] { r.d.X1, r.d.Y1, r.d.X2, r.d.Y2, r.Class, r.d.Score });
    }

    private static Tensor<T> Rows<TRow>(IReadOnlyList<TRow> rows, int width, Func<TRow, T[]> values)
        => new(new[] { rows.Count, width }, Vector<T>.FromArray(rows.SelectMany(values).ToArray()));
}
