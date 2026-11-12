using System.Text;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// russian_text_stresser 0.6.0's <c>RussianTextStresser.stress_text</c> (Vuizur/add-stress-to-epub): Russian text with
/// its stress marks (U+0301) and the letter ё restored, from Russian Wiktionary data chosen by spaCy's part of speech,
/// case and number. The multilingual Chatterbox marks Russian text with it before tokenizing.
/// </summary>
/// <remarks>
/// <para>Each token spaCy (<see cref="SpacyRussianTagger"/>) does not tag as punctuation, symbol, X or space is looked up:
/// first in the list of unambiguous words (<c>simple_cases.pkl</c>), else its ё-spelling is settled and its stressed
/// canonical form read from <c>russian_dict.db</c>, narrowing homographs by part of speech, then by case and number.
/// A word it cannot settle is left as written. Two words joined by a hyphen are tried as one first.</para>
/// <para>The release's code and data (MIT; the database is built from Wiktionary, CC BY-SA) are downloaded by
/// <see cref="DefaultModelDirectory"/>.</para>
/// </remarks>
internal sealed class RussianTextStresser : IDisposable
{
    private const string Commit = "9e064c815373ef6f0360dd97b0d0fc37b52d256f";
    private const string DictionaryUrl = "https://github.com/Vuizur/add-stress-to-epub/releases/download/v1.0.1/russian_dict.zip";
    private const string DictionarySha256 = "833a256704b3fe849234799f2d15780bc5b56551e6c92e2e4e46e10630f5df3d";
    private static readonly (string File, string Sha256)[] SpacyFiles =
    {
        ("tokenizer", "2227a2106f96afd2a302f3a71f5ef8068df8cf871efd3e1672ea375bde057366"),
        ("tok2vec/model", "831172259ebcc925a122a1ae7b8c353c4028c23981f74585a6954a3df35592c0"),
        ("morphologizer/model", "4e3654b06fd31e7df73294024a3f341147ea0e1694306ddb9ecf4acbdcae2cbf"),
        ("morphologizer/cfg", "94577e702cbab1510878b3335de7310d5ef35c05dae0dda34e8df2fbeb60c7a4"),
    };

    private static readonly Dictionary<string, string> PosMapping = new()
    {
        ["NOUN"] = "noun", ["VERB"] = "verb", ["PROPN"] = "name", ["ADV"] = "adv", ["SCONJ"] = "conj", ["ADP"] = "prep",
        ["ADJ"] = "adj", ["PRON"] = "pron", ["CCONJ"] = "conj", ["PART"] = "particle", ["DET"] = "pron", ["INTJ"] = "intj",
        ["NUM"] = "num", ["AUX"] = "verb",
    };
    private static readonly Dictionary<string, string> NumberMapping = new() { ["Sing"] = "singular", ["Plur"] = "plural" };
    private static readonly Dictionary<string, string> CaseMapping = new()
    {
        ["Nom"] = "nominative", ["Gen"] = "genitive", ["Dat"] = "dative", ["Acc"] = "accusative", ["Ins"] = "instrumental",
        ["Loc"] = "prepositional", ["Par"] = "partitive",
    };
    private static readonly string[] VeryOftenWrongWords = { "\u0437\u0430\u043C\u0435\u0440", "\u0443\u0442\u0440\u0430", "\u0447\u0430\u0441\u0430", "\u043F\u043E\u0442\u043E\u043C" };

    // stressed_cyrillic_tools.ACCENT_MAPPING, applied in order after NFKC.
    private static readonly (string From, string To)[] AccentMapping =
    {
        ("\u0301", ""), ("\u0300", ""), ("\u0430\u0301", "\u0430"), ("\u0430\u0300", "\u0430"), ("\u0435\u0301", "\u0435"),
        ("\u0450", "\u0435"), ("\u0438\u0301", "\u0438"), ("\u045D", "\u0438"), ("\u043E\u0301", "\u043E"), ("\u043E\u0300", "\u043E"),
        ("\u0443\u0301", "\u0443"), ("\u0443\u0300", "\u0443"), ("\u044B\u0301", "\u044B"), ("\u044B\u0300", "\u044B"),
        ("\u044D\u0301", "\u044D"), ("\u044D\u0300", "\u044D"), ("\u044E\u0301", "\u044E"), ("\u0300\u044E", "\u044E"),
        ("\u044F\u0301\u0301", "\u044F"), ("\u044F\u0300", "\u044F"),
    };

    private const string Yo = "\u0451", YoUpper = "\u0401", Ye = "\u0435", YeUpper = "\u0415", Acute = "\u0301";

    /// <summary>A failure the release raises (and the multilingual Chatterbox catches, keeping the text unstressed): a
    /// part of speech or case its mappings lack, or a dictionary form that does not line up with the word.</summary>
    private sealed class StressFailure : Exception
    {
        public StressFailure(string message) : base(message) { }
    }

    private readonly SpacyRussianTagger _tagger;
    private readonly SqliteFile _database;
    private readonly SimpleCases _simpleCases;
    private readonly string[] _wordColumns, _caseTagColumns;

    private RussianTextStresser(SpacyRussianTagger tagger, SqliteFile database, SimpleCases simpleCases)
    {
        _tagger = tagger;
        _database = database;
        _simpleCases = simpleCases;
        _wordColumns = database.Columns("word");
        _caseTagColumns = database.Columns("case_tags");
    }

    /// <summary>Loads a directory holding <c>russian_dict.db</c>, <c>simple_cases.pkl</c> and the
    /// <c>ru_core_news_sm-3.6.0</c> pipeline files (<c>tokenizer</c>, <c>tok2vec/model</c>, <c>morphologizer/*</c>).</summary>
    public static RussianTextStresser Load(string directory) => new(SpacyRussianTagger.Load(directory),
        SqliteFile.Open(Path.Combine(directory, "russian_dict.db")), SimpleCases.Load(Path.Combine(directory, "simple_cases.pkl")));

    /// <summary>The release's data in <c>~/.aidotnet/models/russian-text-stresser</c>, downloaded and checked first if
    /// needed: its dictionary release (356 MB, 1.5 GB unpacked) and its bundled spaCy pipeline's files.</summary>
    public static string DefaultModelDirectory()
    {
        string directory = VerifiedDownload.CacheDirectory("russian-text-stresser");
        VerifiedDownload.Zip(DictionaryUrl, directory, DictionarySha256, 2L << 30, "russian_dict.db", "simple_cases.pkl");
        foreach (var (file, sha256) in SpacyFiles)
            VerifiedDownload.File($"https://raw.githubusercontent.com/Vuizur/add-stress-to-epub/{Commit}/russian_text_stresser/ru_core_news_sm-3.6.0/{file}",
                Path.Combine(directory, file.Replace('/', Path.DirectorySeparatorChar)), sha256);
        return directory;
    }

    public void Dispose() => _database.Dispose();

    /// <summary>The text with its stress marks, or the text unchanged when the release's stresser fails on it (the
    /// multilingual Chatterbox's <c>add_russian_stress</c> keeps the text then).</summary>
    public string StressText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        try
        {
            return Stress(text);
        }
        catch (StressFailure)
        {
            return text;
        }
    }

    private static bool IsUnimportant(SpacyRussianTagger.Token token) => token.Pos is "PUNCT" or "SYM" or "X" or "SPACE" || token.Text == "-";

    // RussianTextStresser.stress_text (without the optional LLM and StressRNN passes, which it does not run by default).
    private string Stress(string text)
    {
        if (text.Length == 0) return "";
        var doc = _tagger.Tag(text);
        var result = new StringBuilder();
        for (int i = 0; i < doc.Count; i++)
        {
            var token = doc[i];
            if (IsUnimportant(token))
            {
                result.Append(token.TextWithWhitespace);
                continue;
            }
            if (doc.Count >= i + 3 && doc[i + 1].Text == "-" && !token.Space && !doc[i + 1].Space)
            {
                string fused = token.Text + doc[i + 1].Text + doc[i + 2].Text;
                string stressed = StressedWordAndYo(fused, null, null);
                if (stressed.Contains(Acute))
                {
                    result.Append(stressed).Append(doc[i + 2].Space ? " " : "");
                    i += 2;
                    continue;
                }
            }
            result.Append(StressedWordAndYo(token.Text, token.Pos, token)).Append(token.Space ? " " : "");
        }
        return result.ToString();
    }

    // stressed_cyrillic_tools.get_lower_and_without_yo.
    private static string LowerAndWithoutYo(string word)
    {
        string s = word.Normalize(NormalizationForm.FormKC);
        foreach (var (from, to) in AccentMapping) s = s.Replace(from, to);
        return PythonString.Lower(s.Replace(Yo, Ye).Replace(YoUpper, YeUpper));
    }

    // RussianDictionary.get_stressed_word_and_set_yo.
    private string StressedWordAndYo(string word, string? pos, SpacyRussianTagger.Token? token)
    {
        if (VeryOftenWrongWords.Contains(PythonString.Lower(word))) return word;
        if (pos == "PUNCT") return word;
        if (_simpleCases.TryGet(LowerAndWithoutYo(word), out var simple)) return WriteStressedWord(word, simple);
        var (withYo, unique) = CorrectYoForm(word, pos, token);
        return unique ? StressedWord(withYo, pos, token) : word;
    }

    private string Mapped(Dictionary<string, string> mapping, string key, string what) =>
        mapping.TryGetValue(key, out var value) ? value : throw new StressFailure($"No {what} mapping for '{key}'.");

    private string WordColumn(object?[] row, string column) => row[Array.IndexOf(_wordColumns, column)] as string ?? "";
    private object? WordValue(object?[] row, string column) => row[Array.IndexOf(_wordColumns, column)];

    private IEnumerable<object?[]> Words(string index, params object?[] key) =>
        _database.Seek(index, key).Select(id => _database.Row("word", id) ?? throw new InvalidDataException($"word {id} is missing."));

    // The form_of_word ids of the word rows, each with its row's canonical form and its tags (the release's join).
    private Dictionary<long, (string? Canonical, HashSet<string> Tags)> TaggedForms(IEnumerable<object?[]> words)
    {
        var forms = new Dictionary<long, (string? Canonical, HashSet<string> Tags)>();
        int tagColumn = Array.IndexOf(_caseTagColumns, "tag_text");
        foreach (var row in words)
        {
            long wordId = (long)WordValue(row, "word_id")!;
            foreach (long formId in _database.Seek("word_id_index", wordId))
                foreach (long tagRow in _database.Seek("case_tags_fow_id_index", formId))
                {
                    var tag = _database.Row("case_tags", tagRow)![tagColumn] as string;
                    if (!forms.TryGetValue(formId, out var entry)) forms[formId] = entry = (WordValue(row, "canonical_form") as string, new HashSet<string>());
                    if (tag is not null) entry.Tags.Add(tag);
                }
        }
        return forms;
    }

    // The forms whose tags hold the token's case (and number, when it has one); the release's case-and-number narrowing.
    private static HashSet<string?> FittingForms(Dictionary<long, (string? Canonical, HashSet<string> Tags)> forms, string caseTag,
        string? number, bool locative)
    {
        var fitting = new HashSet<string?>();
        foreach (var (canonical, tags) in forms.Values)
            if ((tags.Contains(caseTag) && (number is null || tags.Contains(number))) || (locative && tags.Contains("locative")))
                fitting.Add(canonical);
        return fitting;
    }

    // RussianDictionary.get_correct_yo_form: the word with its ё, and whether its spelling was settled.
    private (string Word, bool Unique) CorrectYoForm(string word, string? pos, SpacyRussianTagger.Token? token)
    {
        string lower = PythonString.Lower(word);
        var candidates = Words("word_lower_and_without_yo_index", lower).Select(r => WordColumn(r, "word")).ToList();
        if (candidates.Count == 0) return (word, false);
        bool withYo = candidates.Any(w => w.Contains(Yo)), withoutYo = candidates.Any(w => !w.Contains(Yo));
        if (withoutYo && withYo)
        {
            if (pos is not null)
            {
                string mapped = Mapped(PosMapping, pos, "part-of-speech");
                var filtered = Words("wordlo_noyo_index", lower, mapped).Select(r => WordColumn(r, "word")).Distinct().ToList();
                if (filtered.Count == 0) return (word, false);
                if (filtered.Count == 1) return (WriteWordWithYo(word, filtered[0]), true);
                var features = token!.MorphFeatures();
                if (mapped is "noun" or "adj" or "name" or "pron" && features.TryGetValue("Case", out var caseValue))
                {
                    string caseTag = Mapped(CaseMapping, caseValue, "case");
                    string? number = features.TryGetValue("Number", out var n) ? Mapped(NumberMapping, n, "number") : null;
                    var fitting = FittingForms(TaggedForms(Words("wordlo_noyo_index", lower, mapped)), caseTag, number, locative: false);
                    if (fitting.Count == 1) return (WriteWordWithYo(word, NotNull(fitting.Single())), true);
                }
            }
            return (word, false);
        }
        if (withoutYo) return (word, true);
        return (WriteWordWithYo(word, candidates[0]), true);
    }

    // A form without its canonical spelling: the release fails on it (None has no characters).
    private static string NotNull(string? form) => form ?? throw new StressFailure("The only fitting form has no canonical spelling.");

    // RussianDictionary.write_word_with_yo.
    private static string WriteWordWithYo(string word, string yoWord)
    {
        var letters = ChineseText.Codepoints(word);
        var dictionary = ChineseText.Codepoints(yoWord);
        for (int i = 0; i < dictionary.Count; i++)
        {
            if (dictionary[i] != Yo && dictionary[i] != YoUpper) continue;
            if (i >= letters.Count) throw new StressFailure("The \u0451-form is longer than the word.");
            letters[i] = letters[i] == YeUpper ? YoUpper : Yo;
        }
        return string.Concat(letters);
    }

    // RussianDictionary.write_stressed_word: the word's letters with the dictionary form's stress marks and ё.
    private static string WriteStressedWord(string word, string stressed)
    {
        stressed = stressed.Replace(Yo + "\u0300", Yo);
        var letters = ChineseText.Codepoints(word);
        var result = new StringBuilder();
        int index = 0;
        foreach (var c in ChineseText.Codepoints(stressed))
        {
            if (c == Yo || c == YoUpper)
            {
                if (index >= letters.Count) throw new StressFailure("The stressed form is longer than the word.");
                string letter = letters[index];
                if (letter == Ye || letter == Yo) result.Append(Yo);
                else if (letter == YeUpper || letter == YoUpper) result.Append(YoUpper);
                index++;
            }
            else if (c != Acute)
            {
                if (index >= letters.Count) break;
                result.Append(letters[index]);
                index++;
            }
            else
            {
                result.Append(Acute);
            }
        }
        return result.ToString();
    }

    // The canonical forms of word rows, lower-cased, keeping only lower-case ones for a lower-case word.
    private static HashSet<string> Canonical(IEnumerable<string?> forms, bool isLower) =>
        new(forms.Where(f => f is not null && !(isLower && !PythonString.IsLower(f))).Select(f => PythonString.Lower(f!)), StringComparer.Ordinal);

    // RussianDictionary.get_stressed_word: the word stressed, or as written when its stress is unclear.
    private string StressedWord(string word, string? pos, SpacyRussianTagger.Token? token)
    {
        bool isLower = PythonString.IsLower(word);
        string lower = PythonString.Lower(word);
        var forms = Canonical(Words("word_lowercase_index", lower).Select(r => WordValue(r, "canonical_form") as string), isLower);
        if (forms.Count == 1) return WriteStressedWord(word, forms.Single());
        if (forms.Count == 0 || pos is null) return word;
        string mapped = Mapped(PosMapping, pos, "part-of-speech");
        forms = Canonical(Words("wordlo_pos_index", lower, mapped).Select(r => WordValue(r, "canonical_form") as string), isLower);
        if (forms.Count == 0 && mapped == "name")
        {
            var nouns = Words("wordlo_noyo_index", lower, "noun").Select(r => WordValue(r, "word") as string).Distinct();
            forms = Canonical(nouns, isLower);
        }
        if (forms.Count == 0) return word;
        if (forms.Count == 1) return WriteStressedWord(word, forms.Single());
        var features = token!.MorphFeatures();
        if (mapped is "noun" or "adj" or "name" or "pron" && features.TryGetValue("Case", out var caseValue))
        {
            string caseTag = Mapped(CaseMapping, caseValue, "case");
            string? number = features.TryGetValue("Number", out var n) ? Mapped(NumberMapping, n, "number") : null;
            var fitting = FittingForms(TaggedForms(Words("wordlo_pos_index", lower, mapped)), caseTag, number, locative: caseValue == "Loc");
            return fitting.Count == 1 ? WriteStressedWord(word, NotNull(fitting.Single())) : word;
        }
        return word;
    }

    /// <summary>The release's <c>simple_cases.pkl</c> (a pickled dict of unaccented lower-case words to their stressed
    /// forms), kept as its UTF-8 bytes with a sorted index rather than three million .NET strings.</summary>
    private sealed class SimpleCases
    {
        private readonly byte[] _data;
        private readonly int[] _keyStart, _keyLength, _valueStart, _valueLength;

        private SimpleCases(byte[] data, int[] keyStart, int[] keyLength, int[] valueStart, int[] valueLength)
        {
            _data = data;
            _keyStart = keyStart;
            _keyLength = keyLength;
            _valueStart = valueStart;
            _valueLength = valueLength;
        }

        // The pickle opcodes a protocol-4 dict of strings uses: PROTO, FRAME, EMPTY_DICT, MEMOIZE, MARK,
        // SHORT_BINUNICODE, BINUNICODE, SETITEMS, STOP.
        public static SimpleCases Load(string path)
        {
            var data = File.ReadAllBytes(path);
            var strings = new List<(int Start, int Length)>();
            var mark = new Stack<int>();
            var pairs = new List<(int KeyStart, int KeyLength, int ValueStart, int ValueLength)>();
            for (int p = 0; p < data.Length;)
            {
                byte op = data[p++];
                switch (op)
                {
                    case 0x80: p++; break;                                         // PROTO
                    case 0x95: p += 8; break;                                      // FRAME
                    case 0x7D: case 0x94: break;                                   // EMPTY_DICT, MEMOIZE
                    case 0x28: mark.Push(strings.Count); break;                    // MARK
                    case 0x8C:                                                     // SHORT_BINUNICODE
                        strings.Add((p + 1, data[p]));
                        p += 1 + data[p];
                        break;
                    case 0x58:                                                     // BINUNICODE
                    {
                        int length = BitConverter.ToInt32(data, p);
                        strings.Add((p + 4, length));
                        p += 4 + length;
                        break;
                    }
                    case 0x75:                                                     // SETITEMS
                    {
                        int from = mark.Pop();
                        for (int i = from; i + 1 < strings.Count; i += 2)
                            pairs.Add((strings[i].Start, strings[i].Length, strings[i + 1].Start, strings[i + 1].Length));
                        strings.RemoveRange(from, strings.Count - from);
                        break;
                    }
                    case 0x2E: p = data.Length; break;                             // STOP
                    default: throw new InvalidDataException($"simple_cases.pkl uses pickle opcode 0x{op:X2}, which this reader does not read.");
                }
            }
            // A later key replaces an earlier one, as a dict's SETITEMS does.
            var order = Enumerable.Range(0, pairs.Count).ToArray();
            int Compare(int a, int b)
            {
                int c = CompareBytes(data, pairs[a].KeyStart, pairs[a].KeyLength, data, pairs[b].KeyStart, pairs[b].KeyLength);
                return c != 0 ? c : a.CompareTo(b);
            }
            System.Array.Sort(order, Compare);
            var kept = new List<int>(order.Length);
            for (int i = 0; i < order.Length; i++)
            {
                if (i + 1 < order.Length && CompareBytes(data, pairs[order[i]].KeyStart, pairs[order[i]].KeyLength,
                        data, pairs[order[i + 1]].KeyStart, pairs[order[i + 1]].KeyLength) == 0) continue;
                kept.Add(order[i]);
            }
            return new SimpleCases(data, kept.Select(i => pairs[i].KeyStart).ToArray(), kept.Select(i => pairs[i].KeyLength).ToArray(),
                kept.Select(i => pairs[i].ValueStart).ToArray(), kept.Select(i => pairs[i].ValueLength).ToArray());
        }

        private static int CompareBytes(byte[] a, int aStart, int aLength, byte[] b, int bStart, int bLength)
        {
            int n = Math.Min(aLength, bLength);
            for (int i = 0; i < n; i++)
                if (a[aStart + i] != b[bStart + i]) return a[aStart + i].CompareTo(b[bStart + i]);
            return aLength.CompareTo(bLength);
        }

        public bool TryGet(string key, out string value)
        {
            var bytes = Encoding.UTF8.GetBytes(key);
            int low = 0, high = _keyStart.Length - 1;
            while (low <= high)
            {
                int mid = (low + high) >> 1;
                int c = CompareBytes(_data, _keyStart[mid], _keyLength[mid], bytes, 0, bytes.Length);
                if (c == 0)
                {
                    value = Encoding.UTF8.GetString(_data, _valueStart[mid], _valueLength[mid]);
                    return true;
                }
                if (c < 0) low = mid + 1;
                else high = mid - 1;
            }
            value = "";
            return false;
        }
    }
}
