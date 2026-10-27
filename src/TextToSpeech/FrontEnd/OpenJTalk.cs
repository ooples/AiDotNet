using System.Globalization;
using static AiDotNet.TextToSpeech.FrontEnd.OpenJTalkRules;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// Open JTalk 1.11's text front end as pyopenjtalk 0.3.4 drives it: <see cref="RunFrontend"/> is
/// <c>pyopenjtalk.run_frontend</c> (text2mecab, MeCab with the Open JTalk dictionary, mecab2njd and the NJD passes)
/// and <see cref="MakeLabel"/> is <c>pyopenjtalk.make_label</c> (njd2jpcommon and JPCommon's full-context labels).
/// </summary>
/// <remarks>
/// The dictionary (<c>open_jtalk_dic_utf_8-1.11</c>, the one pyopenjtalk downloads: naist-jdic with Open JTalk's
/// accent fields, Modified BSD) is read from its directory at run time; it is ~100 MB and is not embedded.
/// </remarks>
internal sealed class OpenJTalk
{
    private readonly MeCab _mecab;

    /// <summary>Opens the MeCab dictionary in <paramref name="dictionaryDirectory"/> (sys.dic, unk.dic, matrix.bin, char.bin).</summary>
    public OpenJTalk(string dictionaryDirectory)
    {
        if (string.IsNullOrEmpty(dictionaryDirectory)) throw new ArgumentException("The Open JTalk dictionary directory is required.", nameof(dictionaryDirectory));
        if (!File.Exists(Path.Combine(dictionaryDirectory, "sys.dic")))
            throw new FileNotFoundException($"No Open JTalk dictionary (sys.dic) in {dictionaryDirectory}; use open_jtalk_dic_utf_8-1.11.");
        _mecab = MeCab.Open(dictionaryDirectory);
    }

    /// <summary><c>run_frontend(text)</c>: the words with their readings, pronunciations, accents and chain flags.</summary>
    public List<NjdFeature> RunFrontend(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var njd = new Njd();
        // Mecab_analysis + mecab2njd: each word's "surface,feature" loaded into a node
        foreach (var (surface, feature) in _mecab.ParseFeatures(OpenJTalkPasses.Text2Mecab(text)))
        {
            var node = new NjdNode();
            node.Load(surface + "," + feature);
            njd.Push(node);
        }
        OpenJTalkPasses.SetPronunciation(njd);
        OpenJTalkPasses.SetDigit(njd);
        OpenJTalkPasses.SetAccentPhrase(njd);
        OpenJTalkPasses.SetAccentType(njd);
        OpenJTalkPasses.SetUnvoicedVowel(njd);
        return njd.ToFeatures();
    }

    /// <summary><c>make_label(features)</c>: HTS full-context labels, one per phoneme with silences at both ends.</summary>
    public static List<string> MakeLabel(IReadOnlyList<NjdFeature> features)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        var label = new JpCommonLabel();
        foreach (var feature in features)
        {
            var node = NjdNode.FromFeature(feature);
            // njd2jpcommon: the part of speech, conjugation type and form in JPCommon's terms
            label.PushWord(node.Pron, ConvertPos(node), Convert("njd2jpcommon_ctype_list", node.CType),
                Convert("njd2jpcommon_cform_list", node.CForm), node.Acc, node.ChainFlag);
        }
        return label.Make();
    }

    private static string ConvertPos(NjdNode node)
    {
        var list = L("njd2jpcommon_pos_list");
        for (int i = 0; list[i] is not null; i += 5)
            if (list[i] == node.Pos && list[i + 1] == node.PosGroup1 && list[i + 2] == node.PosGroup2 && list[i + 3] == node.PosGroup3)
                return list[i + 4]!;
        return list[4]!;
    }

    private static string Convert(string listName, string value)
    {
        var list = L(listName);
        for (int i = 0; list[i] is not null; i += 2)
            if (list[i] == value) return list[i + 1]!;
        return list[1]!;
    }
}

/// <summary>JPCommon's label builder (jpcommon/jpcommon_label.c): phonemes, moras, words, accent phrases and breath groups.</summary>
internal sealed class JpCommonLabel
{
    private const int MaxS = 19, MaxM = 49, MaxL = 99, MaxLL = 199;

    private sealed class Phoneme
    {
        public string Name = "";
        public Phoneme? Prev, Next;
        public Mora? Up;
    }

    private sealed class Mora
    {
        public string Text = "";
        public Phoneme Head = null!, Tail = null!;
        public Mora? Prev, Next;
        public Word Up = null!;
    }

    private sealed class Word
    {
        public string Pos = "", CType = "", CForm = "";
        public Mora Head = null!, Tail = null!;
        public Word? Prev, Next;
        public AccentPhrase Up = null!;
    }

    private sealed class AccentPhrase
    {
        public int Accent;
        public string? Emotion;
        public Word Head = null!, Tail = null!;
        public AccentPhrase? Prev, Next;
        public BreathGroup Up = null!;
    }

    private sealed class BreathGroup
    {
        public AccentPhrase Head = null!, Tail = null!;
        public BreathGroup? Prev, Next;
    }

    private bool _shortPause;
    private BreathGroup? _breathHead, _breathTail;
    private AccentPhrase? _accentHead, _accentTail;
    private Word? _wordHead, _wordTail;
    private Mora? _moraHead, _moraTail;
    private Phoneme? _phonemeHead, _phonemeTail;

    private static string J(string name) => D("JPCOMMON_" + name);

    private static string Lookup(string listName, string key)
    {
        var list = L(listName);
        for (int i = 0; list[i] is not null; i += 2)
            if (list[i] == key) return list[i + 1]!;
        return list[1]!;
    }

    private static Word NewWord(string pos, string ctype, string cform, Mora mora, Word? prev) => new()
    {
        Pos = Lookup("jpcommon_pos_list", pos), CType = Lookup("jpcommon_ctype_list", ctype),
        CForm = Lookup("jpcommon_cform_list", cform), Head = mora, Tail = mora, Prev = prev,
    };

    private void InsertPause()
    {
        if (!_shortPause) return;
        if (_phonemeTail is not null)
        {
            if (_phonemeTail.Name == J("PHONEME_SHORT_PAUSE")) return;   // "Short pause should not be chained."
            _phonemeTail.Next = new Phoneme { Name = J("PHONEME_SHORT_PAUSE"), Prev = _phonemeTail };
            _phonemeTail = _phonemeTail.Next;
        }
        _shortPause = false;
    }

    private static void Unvoice(Phoneme p)
    {
        var list = L("jpcommon_unvoice_list");
        for (int i = 0; list[i] is not null; i += 2)
            if (list[i] == p.Name)
            {
                p.Name = list[i + 1]!;
                return;
            }
    }

    /// <summary>JPCommonLabel_push_word.</summary>
    public void PushWord(string pron, string pos, string ctype, string cform, int acc, int chainFlag)
    {
        bool firstWord = true;
        if (pron == J("MORA_SHORT_PAUSE"))
        {
            _shortPause = true;
            return;
        }
        // set emotion flag
        if (pron == J("MORA_QUESTION"))
        {
            if (_phonemeTail is not null)
            {
                var phrase = _phonemeTail.Name == J("PHONEME_SHORT_PAUSE") ? _phonemeTail.Prev!.Up!.Up.Up : _phonemeTail.Up!.Up.Up;
                phrase.Emotion ??= J("FLAG_QUESTION");
            }
            _shortPause = true;
            return;
        }

        var moras = L("jpcommon_mora_list");
        int at = 0;
        while (at < pron.Length)
        {
            int find = StrTopCmp(pron, at, J("MORA_LONG_VOWEL"));
            if (find != -1)
            {
                // for long vowel
                if (_phonemeTail is not null && !_shortPause)
                {
                    InsertPause();
                    var mora = new Mora { Text = J("MORA_LONG_VOWEL"), Prev = _moraTail, Up = _moraTail!.Up };
                    var phoneme = new Phoneme { Name = _phonemeTail.Name, Prev = _phonemeTail, Up = mora };
                    mora.Head = mora.Tail = phoneme;
                    _phonemeTail.Next = phoneme;
                    _moraTail.Next = mora;
                    _phonemeTail = phoneme;
                    _moraTail = mora;
                    _wordTail!.Tail = mora;
                }
                at += find;
                continue;
            }
            find = StrTopCmp(pron, at, J("MORA_UNVOICE"));
            if (find != -1)
            {
                // for unvoice
                if (_phonemeTail is not null && !firstWord) Unvoice(_phonemeTail);
                at += find;
                continue;
            }
            // for normal word
            int i = 0;
            for (; moras[i] is not null; i += 3)
            {
                find = StrTopCmp(pron, at, moras[i]!);
                if (find != -1) break;
            }
            if (find == -1) break;   // "wrong mora list"
            InsertPause();
            var newMora = new Mora { Text = moras[i]!, Prev = _moraTail };
            var newPhoneme = new Phoneme { Name = moras[i + 1]!, Prev = _phonemeTail, Up = newMora };
            newMora.Head = newMora.Tail = newPhoneme;
            if (_phonemeTail is null)
            {
                _wordTail = NewWord(pos, ctype, cform, newMora, null);
                newMora.Up = _wordTail;
                _phonemeHead = _phonemeTail = newPhoneme;
                _moraHead = _moraTail = newMora;
                _wordHead = _wordTail;
                firstWord = false;
            }
            else if (firstWord)
            {
                var word = NewWord(pos, ctype, cform, newMora, _wordTail);
                newMora.Up = word;
                _phonemeTail.Next = newPhoneme;
                _moraTail!.Next = newMora;
                _wordTail!.Next = word;
                _phonemeTail = newPhoneme;
                _moraTail = newMora;
                _wordTail = word;
                firstWord = false;
            }
            else
            {
                newMora.Up = _moraTail!.Up;
                _phonemeTail.Next = newPhoneme;
                _moraTail.Next = newMora;
                _phonemeTail = newPhoneme;
                _moraTail = newMora;
                _wordTail!.Tail = newMora;
            }
            if (moras[i + 2] is not null)
            {
                InsertPause();
                var vowel = new Phoneme { Name = moras[i + 2]!, Prev = _phonemeTail, Up = _moraTail };
                _phonemeTail.Next = vowel;
                _phonemeTail = vowel;
                _moraTail.Tail = vowel;
            }
            at += find;
        }

        // check
        if (firstWord || _phonemeTail is null || _phonemeTail.Name == J("PHONEME_SHORT_PAUSE")) return;

        // make accent, phrase
        if (_wordHead == _wordTail)
        {
            // first word
            _accentTail = new AccentPhrase { Accent = acc, Head = _wordTail!, Tail = _wordTail! };
            _breathTail = new BreathGroup { Head = _accentTail, Tail = _accentTail };
            _accentTail.Up = _breathTail;
            _wordTail!.Up = _accentTail;
            _accentHead = _accentTail;
            _breathHead = _breathTail;
        }
        else if (chainFlag == 1)
        {
            // common accent phrase and common phrase
            _wordTail!.Up = _accentTail!;
            _accentTail!.Tail = _wordTail;
        }
        else if (_wordTail!.Prev!.Tail.Tail.Next!.Name != J("PHONEME_SHORT_PAUSE"))
        {
            // different accent phrase && common phrase
            var phrase = new AccentPhrase { Accent = acc, Head = _wordTail, Tail = _wordTail, Prev = _accentTail, Up = _breathTail! };
            _accentTail!.Next = phrase;
            _wordTail.Up = phrase;
            _breathTail!.Tail = phrase;
            _accentTail = phrase;
        }
        else
        {
            // different accent phrase && different phrase
            var phrase = new AccentPhrase { Accent = acc, Head = _wordTail, Tail = _wordTail, Prev = _accentTail };
            var breath = new BreathGroup { Head = phrase, Tail = phrase, Prev = _breathTail };
            phrase.Up = breath;
            _accentTail!.Next = phrase;
            _breathTail!.Next = breath;
            _wordTail.Up = phrase;
            _accentTail = phrase;
            _breathTail = breath;
        }
    }

    private static int Limit(int value, int min, int max) => value <= min ? min : value >= max ? max : value;

    private static int IndexMoraInAccentPhrase(Mora m)
    {
        int i = 0;
        for (var index = m.Up.Up.Head.Head; index is not null; index = index.Next)
        {
            i++;
            if (index == m) break;
        }
        return i;
    }

    private static int CountMoraInAccentPhrase(Mora m)
    {
        int i = 0;
        for (var index = m.Up.Up.Head.Head; index is not null; index = index.Next)
        {
            i++;
            if (index == m.Up.Up.Tail.Tail) break;
        }
        return i;
    }

    private static int IndexAccentPhraseInBreathGroup(AccentPhrase a)
    {
        int i = 0;
        for (var index = a.Up.Head; index is not null; index = index.Next)
        {
            i++;
            if (index == a) break;
        }
        return i;
    }

    private static int CountAccentPhraseInBreathGroup(AccentPhrase a)
    {
        int i = 0;
        for (var index = a.Up.Head; index is not null; index = index.Next)
        {
            i++;
            if (index == a.Up.Tail) break;
        }
        return i;
    }

    private static int IndexMoraInBreathGroup(Mora m)
    {
        int i = 0;
        for (var index = m.Up.Up.Up.Head.Head.Head; index is not null; index = index.Next)
        {
            i++;
            if (index == m) break;
        }
        return i;
    }

    private static int CountMoraInBreathGroup(Mora m)
    {
        int i = 0;
        for (var index = m.Up.Up.Up.Head.Head.Head; index is not null; index = index.Next)
        {
            i++;
            if (index == m.Up.Up.Up.Tail.Tail.Tail) break;
        }
        return i;
    }

    private static int IndexBreathGroupInUtterance(BreathGroup b)
    {
        int i = 0;
        for (BreathGroup? index = b; index is not null; index = index.Prev) i++;
        return i;
    }

    private static int CountBreathGroupInUtterance(BreathGroup b)
    {
        int i = 0;
        for (var index = b.Next; index is not null; index = index.Next) i++;
        return IndexBreathGroupInUtterance(b) + i;
    }

    private static int IndexAccentPhraseInUtterance(AccentPhrase a)
    {
        int i = 0;
        for (AccentPhrase? index = a; index is not null; index = index.Prev) i++;
        return i;
    }

    private static int CountAccentPhraseInUtterance(AccentPhrase a)
    {
        int i = 0;
        for (var index = a.Next; index is not null; index = index.Next) i++;
        return IndexAccentPhraseInUtterance(a) + i;
    }

    private static int IndexMoraInUtterance(Mora m)
    {
        int i = 0;
        for (Mora? index = m; index is not null; index = index.Prev) i++;
        return i;
    }

    private static int CountMoraInUtterance(Mora m)
    {
        int i = 0;
        for (var index = m.Next; index is not null; index = index.Next) i++;
        return IndexMoraInUtterance(m) + i;
    }

    private static string I(int value) => value.ToString(CultureInfo.InvariantCulture);

    private static string AccentOf(AccentPhrase a) =>
        I(Limit(CountMoraInAccentPhrase(a.Head.Head), 1, MaxM)) + "_"
        + I(Limit(a.Accent == 0 ? CountMoraInAccentPhrase(a.Head.Head) : a.Accent, 1, MaxM));

    /// <summary>JPCommonLabel_make: the labels (empty when there is no phoneme).</summary>
    public List<string> Make()
    {
        var labels = new List<string>();
        int size = 0;
        for (var q = _phonemeHead; q is not null; q = q.Next) size++;
        if (size < 1) return labels;   // "No phoneme."
        size += 2;

        var phonemes = new string[size + 4];
        phonemes[0] = phonemes[1] = J("PHONEME_UNKNOWN");
        phonemes[2] = J("PHONEME_SILENT");
        phonemes[size + 1] = J("PHONEME_SILENT");
        phonemes[size + 2] = phonemes[size + 3] = J("PHONEME_UNKNOWN");
        int k = 3;
        for (var q = _phonemeHead; q is not null; q = q.Next) phonemes[k++] = q.Name;

        string pause = J("PHONEME_SHORT_PAUSE");
        var p = _phonemeHead!;
        for (int i = 0; i < size; i++)
        {
            bool shortPause = p.Name == pause;
            bool edge = i == 0 || i == size - 1;
            var f = new System.Text.StringBuilder();
            f.Append(phonemes[i]).Append('^').Append(phonemes[i + 1]).Append('-').Append(phonemes[i + 2])
                .Append('+').Append(phonemes[i + 3]).Append('=').Append(phonemes[i + 4]);

            // A:
            if (edge || shortPause) f.Append("/A:xx+xx+xx");
            else
            {
                int tmp1 = IndexMoraInAccentPhrase(p.Up!);
                int tmp2 = p.Up!.Up.Up.Accent == 0 ? CountMoraInAccentPhrase(p.Up) : p.Up.Up.Up.Accent;
                f.Append("/A:").Append(I(Limit(tmp1 - tmp2, -MaxM, MaxM))).Append('+').Append(I(Limit(tmp1, 1, MaxM)))
                    .Append('+').Append(I(Limit(CountMoraInAccentPhrase(p.Up) - tmp1 + 1, 1, MaxM)));
            }

            // B:
            Word? w = shortPause ? p.Prev!.Up!.Up : p.Up!.Up.Prev is null ? null : i == size - 1 ? p.Up.Up : p.Up.Up.Prev;
            f.Append(w is null ? "/B:xx-xx_xx" : "/B:" + w.Pos + "-" + w.CType + "_" + w.CForm);

            // C:
            f.Append(edge || shortPause ? "/C:xx_xx+xx" : "/C:" + p.Up!.Up.Pos + "_" + p.Up.Up.CType + "+" + p.Up.Up.CForm);

            // D:
            w = shortPause ? p.Next!.Up!.Up : p.Up!.Up.Next is null ? null : i == 0 ? p.Up.Up : p.Up.Up.Next;
            f.Append(w is null ? "/D:xx+xx_xx" : "/D:" + w.Pos + "+" + w.CType + "_" + w.CForm);

            // E:
            AccentPhrase? a = shortPause ? p.Prev!.Up!.Up.Up : i == size - 1 ? p.Up!.Up.Up : p.Up!.Up.Up.Prev;
            f.Append(a is null ? "/E:xx_xx!xx_xx" : "/E:" + AccentOf(a) + "!" + (a.Emotion ?? "0") + "_xx");
            if (edge || shortPause || a is null) f.Append("-xx");
            else f.Append('-').Append(a.Tail.Tail.Tail.Next!.Name == pause ? "0" : "1");

            // F:
            a = edge || shortPause ? null : p.Up!.Up.Up;
            if (a is null) f.Append("/F:xx_xx#xx_xx@xx_xx|xx_xx");
            else
            {
                int tmp1 = IndexAccentPhraseInBreathGroup(a);
                int tmp2 = IndexMoraInBreathGroup(a.Head.Head);
                f.Append("/F:").Append(AccentOf(a)).Append('#').Append(a.Emotion ?? "0").Append("_xx@")
                    .Append(I(Limit(tmp1, 1, MaxM))).Append('_').Append(I(Limit(CountAccentPhraseInBreathGroup(a) - tmp1 + 1, 1, MaxM)))
                    .Append('|').Append(I(Limit(tmp2, 1, MaxL))).Append('_')
                    .Append(I(Limit(CountMoraInBreathGroup(a.Head.Head) - tmp2 + 1, 1, MaxL)));
            }

            // G:
            a = shortPause ? p.Next!.Up!.Up.Up : i == 0 ? p.Up!.Up.Up : p.Up!.Up.Up.Next;
            f.Append(a is null ? "/G:xx_xx%xx_xx" : "/G:" + AccentOf(a) + "%" + (a.Emotion ?? "0") + "_xx");
            if (edge || shortPause || a is null) f.Append("_xx");
            else f.Append('_').Append(a.Head.Head.Head.Prev!.Name == pause ? "0" : "1");

            // H:
            BreathGroup? b = shortPause ? p.Prev!.Up!.Up.Up.Up : i == size - 1 ? p.Up!.Up.Up.Up : p.Up!.Up.Up.Up.Prev;
            f.Append(b is null ? "/H:xx_xx" : "/H:" + I(Limit(CountAccentPhraseInBreathGroup(b.Head), 1, MaxM)) + "_"
                + I(Limit(CountMoraInBreathGroup(b.Head.Head.Head), 1, MaxL)));

            // I:
            b = edge || shortPause ? null : p.Up!.Up.Up.Up;
            if (b is null) f.Append("/I:xx-xx@xx+xx&xx-xx|xx+xx");
            else
            {
                int tmp1 = IndexBreathGroupInUtterance(b);
                int tmp2 = IndexAccentPhraseInUtterance(b.Head);
                int tmp3 = IndexMoraInUtterance(b.Head.Head.Head);
                f.Append("/I:").Append(I(Limit(CountAccentPhraseInBreathGroup(b.Head), 1, MaxM))).Append('-')
                    .Append(I(Limit(CountMoraInBreathGroup(b.Head.Head.Head), 1, MaxL))).Append('@')
                    .Append(I(Limit(tmp1, 1, MaxS))).Append('+').Append(I(Limit(CountBreathGroupInUtterance(b) - tmp1 + 1, 1, MaxS)))
                    .Append('&').Append(I(Limit(tmp2, 1, MaxM))).Append('-')
                    .Append(I(Limit(CountAccentPhraseInUtterance(b.Head) - tmp2 + 1, 1, MaxM))).Append('|')
                    .Append(I(Limit(tmp3, 1, MaxLL))).Append('+').Append(I(Limit(CountMoraInUtterance(b.Head.Head.Head) - tmp3 + 1, 1, MaxLL)));
            }

            // J:
            b = shortPause ? p.Next!.Up!.Up.Up.Up : i == 0 ? p.Up!.Up.Up.Up : p.Up!.Up.Up.Up.Next;
            f.Append(b is null ? "/J:xx_xx" : "/J:" + I(Limit(CountAccentPhraseInBreathGroup(b.Head), 1, MaxM)) + "_"
                + I(Limit(CountMoraInBreathGroup(b.Head.Head.Head), 1, MaxL)));

            // K:
            f.Append("/K:").Append(I(Limit(CountBreathGroupInUtterance(_breathHead!), 1, MaxS))).Append('+')
                .Append(I(Limit(CountAccentPhraseInUtterance(_accentHead!), 1, MaxM))).Append('-')
                .Append(I(Limit(CountMoraInUtterance(_moraHead!), 1, MaxLL)));

            labels.Add(f.ToString());
            if (0 < i && i < size - 2) p = p.Next!;
        }
        return labels;
    }
}
