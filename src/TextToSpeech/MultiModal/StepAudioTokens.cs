using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// Step-1's text as the release encodes it (transformers 4.48.3's <c>LlamaTokenizerFast</c> converted from
/// <c>tokenizer.model</c>, <c>legacy = false</c>): &lt;s&gt; first; control and user-defined pieces (&lt;|BOT|&gt;,
/// &lt;|EOT|&gt;, &lt;audio_N&gt;, …) matched as tokens, leftmost-longest; each stretch between them with spaces as "▁" and, for
/// the stretch at the very start only, "▁" prepended unless it already starts with one (Metaspace, prepend scheme
/// "first"); then the unigram model with byte fallback.
/// </summary>
internal sealed class StepAudioTokens
{
    public const int Bos = 1, Eos = 3, BeginTurn = 4, EndTurn = 3, AudioBase = 65536, LinguisticCodes = 1024;
    private const string Space = "▁";
    private readonly List<(string Text, int Id)> _specials;

    public StepAudioTokens(SentencePieceModel model)
    {
        Model = model ?? throw new ArgumentNullException(nameof(model));
        _specials = Enumerable.Range(0, model.Count)
            .Where(i => model.IsControl(i) || model.IsUserDefined(i) || i == model.UnknownId)
            .Select(i => (Text: model.IdToPiece(i), Id: i))
            .Where(p => p.Text.Length > 0)
            .OrderByDescending(p => p.Text.Length)
            .ToList();
    }

    public SentencePieceModel Model { get; }

    /// <summary>The ids of text, &lt;s&gt; first when <paramref name="addBos"/> (the release's <c>tokenizer.encode</c>).</summary>
    public int[] Encode(string text, bool addBos = true)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var ids = new List<int>();
        if (addBos) ids.Add(Bos);
        int start = 0, position = 0;
        while (position < text.Length)
        {
            int match = -1, length = 0;
            foreach (var (piece, id) in _specials)
                if (string.CompareOrdinal(text, position, piece, 0, piece.Length) == 0)
                {
                    match = id;
                    length = piece.Length;
                    break;
                }
            if (match < 0)
            {
                position++;
                continue;
            }
            Segment(text.Substring(start, position - start), start == 0, ids);
            ids.Add(match);
            position += length;
            start = position;
        }
        Segment(text.Substring(start), start == 0, ids);
        return ids.ToArray();
    }

    private void Segment(string segment, bool first, List<int> ids)
    {
        if (segment.Length == 0) return;
        string normalized = segment.Replace(" ", Space);
        if (first && !normalized.StartsWith(Space, StringComparison.Ordinal)) normalized = Space + normalized;
        ids.AddRange(Model.EncodeNormalized(normalized).Select(p => p.Id));
    }

    /// <summary>The text of ids, special and audio tokens left out.</summary>
    public string Decode(IEnumerable<int> ids) =>
        Model.Decode(ids.Where(id => id >= 0 && id < Model.Count && !Model.IsControl(id) && !Model.IsUserDefined(id)));

    /// <summary>The release's system prompts (tts.py's <c>sys_prompt_dict</c>).</summary>
    public static readonly IReadOnlyDictionary<string, string> SystemPrompts = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["sys_prompt_for_rap"] = "请参考对话历史里的音色，用RAP方式将文本内容大声说唱出来。",
        ["sys_prompt_for_vocal"] = "请参考对话历史里的音色，用哼唱的方式将文本内容大声唱出来。",
        ["sys_prompt_wo_spk"] = "作为一名卓越的声优演员，你的任务是根据文本中（）或()括号内标注的情感、语种或方言、音乐哼唱、语音调整等标签，以丰富细腻的情感和自然顺畅的语调来朗读文本。\n# 情感标签涵盖了多种情绪状态，包括但不限于：\n- \"高兴1\"\n- \"高兴2\"\n- \"生气1\"\n- \"生气2\"\n- \"悲伤1\"\n- \"撒娇1\"\n\n# 语种或方言标签包含多种语言或方言，包括但不限于：\n- \"中文\"\n- \"英文\"\n- \"韩语\"\n- \"日语\"\n- \"四川话\"\n- \"粤语\"\n- \"广东话\"\n\n# 音乐哼唱标签包含多种类型歌曲哼唱，包括但不限于：\n- \"RAP\"\n- \"哼唱\"\n\n# 语音调整标签，包括但不限于：\n- \"慢速1\"\n- \"慢速2\"\n- \"快速1\"\n- \"快速2\"\n\n请在朗读时，根据这些情感标签的指示，调整你的情感、语气、语调和哼唱节奏，以确保文本的情感和意义得到准确而生动的传达，如果没有()或（）括号，则根据文本语义内容自由演绎。",
        ["sys_prompt_with_spk"] = "作为一名卓越的声优演员，你的任务是根据文本中（）或()括号内标注的情感、语种或方言、音乐哼唱、语音调整等标签，以丰富细腻的情感和自然顺畅的语调来朗读文本。\n# 情感标签涵盖了多种情绪状态，包括但不限于：\n- \"高兴1\"\n- \"高兴2\"\n- \"生气1\"\n- \"生气2\"\n- \"悲伤1\"\n- \"撒娇1\"\n\n# 语种或方言标签包含多种语言或方言，包括但不限于：\n- \"中文\"\n- \"英文\"\n- \"韩语\"\n- \"日语\"\n- \"四川话\"\n- \"粤语\"\n- \"广东话\"\n\n# 音乐哼唱标签包含多种类型歌曲哼唱，包括但不限于：\n- \"RAP\"\n- \"哼唱\"\n\n# 语音调整标签，包括但不限于：\n- \"慢速1\"\n- \"慢速2\"\n- \"快速1\"\n- \"快速2\"\n\n请在朗读时，使用[{}]的声音，根据这些情感标签的指示，调整你的情感、语气、语调和哼唱节奏，以确保文本的情感和意义得到准确而生动的传达，如果没有()或（）括号，则根据文本语义内容自由演绎。",
    };

    /// <summary>The leading instruction tag of TTS text, "(高兴1)…" or "（哼唱）…" (tts.py's <c>detect_instruction_name</c>):
    /// what is inside the first brackets when the text starts with them, else empty.</summary>
    public static string InstructionName(string text)
    {
        var match = System.Text.RegularExpressions.Regex.Match(text, @"^([（\(][^\(\)()]*[）\)]).*$", System.Text.RegularExpressions.RegexOptions.Singleline);
        return match.Success ? match.Groups[1].Value.Trim('(', ')', '（', '）') : string.Empty;
    }

    /// <summary>Whether the text's tag selects the music decoder and its system prompts (RAP, 哼唱).</summary>
    public static bool IsMusic(string text) => InstructionName(text) is "RAP" or "哼唱";

    /// <summary>
    /// The TTS prompt (tts.py's <c>tokenize</c>): &lt;s&gt;, then the system turn, the prompt speaker's human turn (their
    /// transcript) and assistant turn (their dual-codebook audio tokens), the human turn with the text, and the open
    /// assistant turn — each turn &lt;|BOT|&gt; … &lt;|EOT|&gt; with the role's ids as encode() gives them (each with its
    /// own &lt;s&gt;). The system prompt is the RAP or humming one for those tags, else the one naming the speaker, or
    /// the one without.
    /// </summary>
    public int[] TtsPrompt(string text, string promptText, string speaker, IReadOnlyList<int> promptCode)
    {
        string system = IsMusic(text)
            ? SystemPrompts[text.Contains("哼唱") ? "sys_prompt_for_vocal" : "sys_prompt_for_rap"]
            : speaker.Length > 0 ? SystemPrompts["sys_prompt_with_spk"].Replace("{}", speaker) : SystemPrompts["sys_prompt_wo_spk"];
        var prefix = Encode("\n");
        int[] AfterNewline(string s) => Encode("\n" + s).Skip(prefix.Length).ToArray();
        var human = Encode("human\n");
        var assistant = Encode("assistant\n");
        var ids = new List<int> { Bos, BeginTurn };
        ids.AddRange(Encode("system\n" + system));
        ids.Add(EndTurn);
        ids.Add(BeginTurn);
        ids.AddRange(human);
        ids.AddRange(AfterNewline(promptText));
        ids.Add(EndTurn);
        ids.Add(BeginTurn);
        ids.AddRange(assistant);
        ids.AddRange(promptCode);
        ids.Add(EndTurn);
        ids.Add(BeginTurn);
        ids.AddRange(human);
        ids.AddRange(AfterNewline(text));
        ids.Add(EndTurn);
        ids.Add(BeginTurn);
        ids.AddRange(assistant);
        return ids.ToArray();
    }

    /// <summary>The chat prompt (stepaudio.py's <c>apply_chat_template</c>, then encode()): each turn
    /// "&lt;|BOT|&gt;{role}\n{content}&lt;|EOT|&gt;" with "user" as "human", then "&lt;|BOT|&gt;assistant\n".</summary>
    public int[] ChatPrompt(IReadOnlyList<(string Role, string Content)> turns)
    {
        var text = new System.Text.StringBuilder();
        foreach (var (role, content) in turns)
            text.Append("<|BOT|>").Append(role == "user" ? "human" : role).Append('\n').Append(content).Append("<|EOT|>");
        string s = text.ToString();
        if (!s.EndsWith("<|BOT|>assistant\n", StringComparison.Ordinal)) s += "<|BOT|>assistant\n";
        return Encode(s);
    }

    /// <summary>The dual-codebook string of linguistic and semantic codes (tokenizer.py's
    /// <c>merge_vq0206_to_token_str</c>): two linguistic then three semantic, while both last.</summary>
    public static string AudioString(IReadOnlyList<int> linguistic, IReadOnlyList<int> semantic)
    {
        var text = new System.Text.StringBuilder();
        int i = 0, j = 0;
        while (i < linguistic.Count - 1 && j < semantic.Count - 2)
        {
            text.Append("<audio_").Append(linguistic[i]).Append('>').Append("<audio_").Append(linguistic[i + 1]).Append('>');
            for (int k = 0; k < 3; k++) text.Append("<audio_").Append(LinguisticCodes + semantic[j + k]).Append('>');
            i += 2;
            j += 3;
        }
        return text.ToString();
    }

    /// <summary>The dual-codebook ids (tokenizer.py's <c>wav2token</c>): 65536 + linguistic, 65536 + 1024 + semantic,
    /// two then three, over min(semantic / 3, linguistic / 2) groups.</summary>
    public static int[] AudioIds(IReadOnlyList<int> linguistic, IReadOnlyList<int> semantic)
    {
        int groups = Math.Min(semantic.Count / 3, linguistic.Count / 2);
        var ids = new List<int>(groups * 5);
        for (int g = 0; g < groups; g++)
        {
            ids.Add(AudioBase + linguistic[2 * g]);
            ids.Add(AudioBase + linguistic[2 * g + 1]);
            for (int k = 0; k < 3; k++) ids.Add(AudioBase + LinguisticCodes + semantic[3 * g + k]);
        }
        return ids.ToArray();
    }
}
