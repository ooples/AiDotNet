using System.Globalization;
using System.Text.RegularExpressions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tokenization.Algorithms;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// IndexTTS2's text-to-emotion module (paper § Text-to-Emotion; the release's <c>QwenEmotion</c>): a Qwen3 language model
/// distilled to answer a sentence with a JSON object of emotion probabilities, prompted with LLaMA-Factory's default
/// template (<c>System: 文本情感分类&lt;|endoftext|&gt;\nHuman: {text}&lt;|endoftext|&gt;\nAssistant:</c>) and decoded
/// greedily; the answer parsed into one score per emotion, clamped to [0, 1.2], with "calm"/"neutral" taking all of it
/// when nothing is detected.
/// </summary>
internal sealed class IndexTtsTextToEmotion<T>
{
    internal const string SystemPrompt = "文本情感分类";
    private const string EndOfText = "<|endoftext|>";
    private const int ThinkEnd = 151668;

    // The release's Chinese keys and their English names, in the release's vector order.
    private static readonly (string Chinese, string English)[] ReleaseKeys =
    {
        ("高兴", "happy"), ("愤怒", "angry"), ("悲伤", "sad"), ("恐惧", "afraid"), ("反感", "disgusted"), ("低落", "melancholic"),
        ("惊讶", "surprised"), ("自然", "calm"),
    };

    private static readonly string[] MelancholicWords = { "低落", "melancholy", "melancholic", "depression", "depressed", "gloomy" };
    private static readonly Regex Pairs = new("([^\\s\":.,]+?)\"?\\s*:\\s*([\\d.]+)", RegexOptions.CultureInvariant);

    private readonly IEngine _engine;
    private readonly string[] _labels;

    public IndexTtsTextToEmotion(IEngine engine, List<LayerBase<T>> layers, LlamaConfiguration configuration, ByteLevelBpeTokenizer tokenizer,
        string[] labels, Random random)
    {
        _engine = engine;
        _labels = labels;
        Tokenizer = tokenizer;
        Model = new LlamaNetwork<T>(engine, layers, configuration);
        Model.InitializeLikeReference(random);
    }

    public LlamaNetwork<T> Model { get; }
    public ByteLevelBpeTokenizer Tokenizer { get; }

    /// <summary>The chat prompt of <paramref name="text"/> (<c>apply_chat_template</c> with the release's template).</summary>
    public static string Prompt(string text) => $"System: {SystemPrompt}{EndOfText}\nHuman: {text}{EndOfText}\nAssistant:";

    /// <summary>The assistant's answer, greedily decoded until <c>&lt;|endoftext|&gt;</c> or <paramref name="maxTokens"/>,
    /// past any <c>&lt;/think&gt;</c>, special tokens dropped.</summary>
    public string Answer(string text, int maxTokens)
    {
        int end = Tokenizer.TokenId(EndOfText);
        var prompt = Tokenizer.Encode(Prompt(text));
        var cache = new LlamaCache<T>();
        var logits = Model.Forward(Model.Embed(Row(prompt)), null, null, cache, lastLogitsOnly: true).Logits!;
        var output = new List<int>();
        for (int step = 0; step < maxTokens; step++)
        {
            int next = ArgMax(logits);
            output.Add(next);
            if (next == end) break;
            logits = Model.Forward(Model.Embed(Row(new[] { next })), null, null, cache, lastLogitsOnly: true).Logits!;
        }
        int think = output.LastIndexOf(ThinkEnd);
        return Tokenizer.Decode(output.Skip(think + 1), skipSpecial: true);
    }

    private static int[,] Row(IReadOnlyList<int> ids)
    {
        var row = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) row[0, i] = ids[i];
        return row;
    }

    private static int ArgMax(Tensor<T> logits)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int best = 0;
        double max = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            double v = ops.ToDouble(logits.GetFlat(i));
            if (v > max)
            {
                max = v;
                best = i;
            }
        }
        return best;
    }

    /// <summary>The emotion scores of <paramref name="text"/>, one per label (the release's <c>inference</c>).</summary>
    public double[] Scores(string text, int maxTokens) => Parse(Answer(text, maxTokens), text);

    /// <summary>The release's parsing: JSON (or <c>key: value</c> pairs), the melancholic swap, the label aliases, the
    /// clamp and the calm/neutral default.</summary>
    internal double[] Parse(string answer, string text)
    {
        var content = new Dictionary<string, object?>(StringComparer.Ordinal);
        string? single = null;
        try
        {
            var token = JToken.Parse(answer);
            if (token is JObject json)
                foreach (var p in json.Properties()) content[p.Name] = p.Value.Type switch
                {
                    JTokenType.Integer or JTokenType.Float => (double)p.Value,
                    JTokenType.String => (string)p.Value!,
                    _ => p.Value.ToString(),
                };
            else if (token.Type == JTokenType.String) single = (string)token!;
        }
        catch (Newtonsoft.Json.JsonException)
        {
            foreach (Match m in Pairs.Matches(answer))
                content[m.Groups[1].Value] = double.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture);
        }

        string lowered = text.ToLowerInvariant();
        if (MelancholicWords.Any(w => lowered.Contains(w)))
        {
            object sad = content.TryGetValue("悲伤", out var s) ? s! : 0.0, low = content.TryGetValue("低落", out var l) ? l! : 0.0;
            content["悲伤"] = low;
            content["低落"] = sad;
        }

        // normalize_content: a bare label (or one under an alias key) becomes that emotion at 1.0, and a key whose value
        // is another label moves its 1.0 there.
        string? detected = single is not null ? Key(single) : null;
        if (detected is null)
            foreach (var alias in new[] { "emotion", "emotion_label", "label", "情感", "情绪" })
                if (content.TryGetValue(alias, out var v) && v is string sv && (detected = Key(sv)) is not null) break;
        var keys = _labels.Select(Canonical).ToArray();
        if (detected is not null && keys.All(k => !content.ContainsKey(k))) content[detected] = 1.0;
        foreach (var key in keys)
            if (content.TryGetValue(key, out var v) && v is string sv && Key(sv) is string named)
            {
                content[key] = named == key ? 1.0 : 0.0;
                if (named != key) content[named] = 1.0;
            }

        var scores = new double[_labels.Length];
        for (int i = 0; i < _labels.Length; i++)
        {
            if (!content.TryGetValue(keys[i], out var v) && !content.TryGetValue(_labels[i], out v)) continue;
            double value = v switch
            {
                double d => d,
                string str when double.TryParse(str, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) => parsed,
                _ => throw new InvalidOperationException($"The emotion model returned a non-numeric score {v} for '{_labels[i]}'."),
            };
            scores[i] = Math.Max(0.0, Math.Min(1.2, value));
        }
        if (scores.All(v => v <= 0)) scores[NeutralIndex()] = 1.0;
        return scores;
    }

    private int NeutralIndex()
    {
        int i = Array.FindIndex(_labels, l => l is "calm" or "neutral");
        return i >= 0 ? i : _labels.Length - 1;
    }

    // The key a label's score sits under in the answer: the release's Chinese key for its English names.
    private static string Canonical(string label)
    {
        foreach (var (chinese, english) in ReleaseKeys)
            if (string.Equals(english, label, StringComparison.OrdinalIgnoreCase)) return chinese;
        return label;
    }

    private string? Key(string value)
    {
        value = value.Trim();
        foreach (var label in _labels)
        {
            string key = Canonical(label);
            if (value == key || string.Equals(value, label, StringComparison.OrdinalIgnoreCase)) return key;
        }
        return null;
    }

    /// <summary>
    /// The distillation loss (Eq. 4) for one sentence: the cross-entropy of the teacher's answer — its distribution as
    /// JSON, probabilities to two decimals, then <c>&lt;|endoftext|&gt;</c> — given the prompt, over the answer's tokens
    /// (LLaMA-Factory's supervised fine-tuning).
    /// </summary>
    public Tensor<T> Loss(string text, IReadOnlyList<double> distribution)
    {
        if (distribution.Count != _labels.Length) throw new ArgumentException($"Give one probability per emotion ({_labels.Length}).", nameof(distribution));
        var answer = new JObject();
        for (int i = 0; i < _labels.Length; i++) answer[Canonical(_labels[i])] = Math.Round(distribution[i], 2);
        var prompt = Tokenizer.Encode(Prompt(text));
        var response = Tokenizer.Encode(answer.ToString(Newtonsoft.Json.Formatting.None) + EndOfText);
        var ids = prompt.Concat(response).ToArray();
        var labels = new int[1, ids.Length];
        for (int i = 0; i < ids.Length; i++) labels[0, i] = i < prompt.Length ? -100 : ids[i];
        var (_, logits) = Model.ForwardWithLogits(Model.Embed(Row(ids)));
        return Model.CausalLmLoss(logits, labels);
    }
}
