using System.Text;
using static AiDotNet.TextToSpeech.FrontEnd.OpenJTalkRules;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// Open JTalk 1.11's text processing (pyopenjtalk 0.3.4's vendored copy, Modified BSD), function for function:
/// <c>text2mecab</c>, then the NJD passes <c>njd_set_pronunciation</c>, <c>njd_set_digit</c>,
/// <c>njd_set_accent_phrase</c>, <c>njd_set_accent_type</c> and <c>njd_set_unvoiced_vowel</c>
/// (<c>njd_set_long_vowel</c> returns at once in 1.11). Their tables are the C headers' (<see cref="OpenJTalkRules"/>).
/// </summary>
internal static class OpenJTalkPasses
{
    // ------------------------------------------------------------------ text2mecab

    private static bool BytesStartWith(byte[] s, int at, byte[] pattern)
    {
        if (s.Length - at < pattern.Length) return false;
        for (int i = 0; i < pattern.Length; i++)
            if (s[at + i] != pattern[i]) return false;
        return true;
    }

    private static readonly Lazy<(byte[] From, byte[] To)[]> ConversionList = new(() =>
    {
        var list = L("text2mecab_conv_list");
        var pairs = new List<(byte[], byte[])>();
        for (int i = 0; list[i] is not null; i += 2) pairs.Add((Encoding.UTF8.GetBytes(list[i]!), Encoding.UTF8.GetBytes(list[i + 1]!)));
        return pairs.ToArray();
    }, isThreadSafe: true);

    /// <summary>
    /// <c>text2mecab</c>: the text's UTF-8 bytes through the conversion list (half-width to full-width and the
    /// like, first matching entry), ASCII control and other single bytes not in the list dropped, other characters
    /// copied by their lead byte's length.
    /// </summary>
    public static string Text2Mecab(string input)
    {
        var s = Encoding.UTF8.GetBytes(input);
        var output = new List<byte>(s.Length * 2);
        for (int p = 0; p < s.Length;)
        {
            int e = -1;
            byte[]? replacement = null;
            foreach (var (from, to) in ConversionList.Value)
                if (from.Length > 0 && BytesStartWith(s, p, from))
                {
                    e = from.Length;
                    replacement = to;
                    break;
                }
            if (replacement is not null)
            {
                p += e;
                output.AddRange(replacement);
            }
            else if (s[p] <= 0x7F)
            {
                p++;   // control character (text2mecab_control_range: 0x00-0x7F)
            }
            else
            {
                // text2mecab_kanji_range for UTF-8: lead bytes C0-DF, E0-EF, F0-F7 start 2-, 3- and 4-byte characters
                int length = s[p] >= 0xC0 && s[p] <= 0xDF ? 2 : s[p] >= 0xE0 && s[p] <= 0xEF ? 3 : s[p] >= 0xF0 && s[p] <= 0xF7 ? 4 : -1;
                if (length > 0)
                    for (int j = 0; j < length && p < s.Length; j++) output.Add(s[p++]);
                else
                    p++;   // "Wrong character."
            }
        }
        return Encoding.UTF8.GetString(output.ToArray());
    }

    // ------------------------------------------------------------------ njd_set_pronunciation

    public static void SetPronunciation(Njd njd)
    {
        var list = L("njd_set_pronunciation_list");
        var symbols = L("njd_set_pronunciation_symbol_list");
        for (var node = njd.Head; node is not null; node = node.Next)
        {
            if (node.MoraSize != 0) continue;
            node.Read = null!;
            node.Pron = null!;
            // if the word is kana, set them as filler
            string str = node.String;
            for (int pos = 0; pos < str.Length;)
            {
                int i = 0, j = 0;
                for (; list[i] is not null; i += 3)
                {
                    j = StrTopCmp(str, pos, list[i]!);
                    if (j > 0) break;
                }
                if (j > 0)
                {
                    pos += j;
                    node.AddRead(list[i + 1]);
                    node.AddPron(list[i + 1]);
                    node.AddMoraSize(Atoi(list[i + 2]));
                }
                else
                {
                    pos++;
                }
            }
            if (node.MoraSize != 0)
            {
                node.Pos = D("NJD_SET_PRONUNCIATION_FILLER");
                node.PosGroup1 = null!;
                node.PosGroup2 = null!;
                node.PosGroup3 = null!;
            }
            if (node.Orig == "*") node.Orig = str;
            // if known symbol, set the pronunciation
            if (node.Pron == "*")
                for (int i = 0; symbols[i] is not null; i += 2)
                    if (node.String == symbols[i])
                    {
                        node.Read = symbols[i + 1]!;
                        node.Pron = symbols[i + 1]!;
                        break;
                    }
            // if the word is not kana, set pause symbol
            if (node.Pron == "*")
            {
                node.Read = D("NJD_SET_PRONUNCIATION_TOUTEN");
                node.Pron = D("NJD_SET_PRONUNCIATION_TOUTEN");
                node.Pos = D("NJD_SET_PRONUNCIATION_KIGOU");
            }
        }
        njd.RemoveSilentNodes();

        // chain kana sequence
        NjdNode? head = null;
        for (var node = njd.Head; node is not null; node = node.Next)
        {
            if (node.Pos == D("NJD_SET_PRONUNCIATION_FILLER"))
            {
                bool find = false;
                for (int i = 0; list[i] is not null; i += 3)
                    if (node.String == list[i])
                    {
                        find = true;
                        if (head is null) head = node;
                        else
                        {
                            head.AddString(node.String);
                            head.AddOrig(node.Orig);
                            head.AddRead(node.Read);
                            head.AddPron(node.Pron);
                            head.AddMoraSize(node.MoraSize);
                            node.Pron = null!;
                        }
                        break;
                    }
                if (!find) head = null;
            }
            else
            {
                head = null;
            }
        }
        njd.RemoveSilentNodes();

        for (var node = njd.Head; node is not null; node = node.Next)
        {
            if (node.Next is not null && node.Next.Pron == D("NJD_SET_PRONUNCIATION_U")
                && node.Next.Pos == D("NJD_SET_PRONUNCIATION_JODOUSHI")
                && (node.Pos == D("NJD_SET_PRONUNCIATION_DOUSHI") || node.Pos == D("NJD_SET_PRONUNCIATION_JODOUSHI"))
                && node.MoraSize > 0)
                node.Next.Pron = D("NJD_SET_PRONUNCIATION_CHOUON");
            if (node.Next is not null && node.Pos == D("NJD_SET_PRONUNCIATION_JODOUSHI")
                && node.Next.String == D("NJD_SET_PRONUNCIATION_QUESTION"))
            {
                if (node.String == D("NJD_SET_PRONUNCIATION_DESU_STR")) node.Pron = D("NJD_SET_PRONUNCIATION_DESU_PRON");
                else if (node.String == D("NJD_SET_PRONUNCIATION_MASU_STR")) node.Pron = D("NJD_SET_PRONUNCIATION_MASU_PRON");
            }
        }
    }

    // ------------------------------------------------------------------ njd_set_digit

    private static int GetDigit(NjdNode node, bool convert)
    {
        if (node.String == "*") return -1;
        if (node.PosGroup1 == D("NJD_SET_DIGIT_KAZU"))
        {
            var list = L("njd_set_digit_rule_numeral_list1");
            for (int i = 0; list[i] is not null; i += 3)
                if (list[i] == node.String)
                {
                    if (convert)
                    {
                        node.String = list[i + 2]!;
                        node.Orig = list[i + 2]!;
                    }
                    return Atoi(list[i + 1]);
                }
        }
        return -1;
    }

    private static bool IsPeriod(string? s) => s is not null && (s == D("NJD_SET_DIGIT_TEN1") || s == D("NJD_SET_DIGIT_TEN2"));
    private static bool IsComma(string? s) => s is not null && s == D("NJD_SET_DIGIT_COMMA");

    private static bool IsHyphen(string s) =>
        s == D("NJD_SET_DIGIT_HAIHUN1") || s == D("NJD_SET_DIGIT_HAIHUN2") || s == D("NJD_SET_DIGIT_HAIHUN3")
        || s == D("NJD_SET_DIGIT_HAIHUN4") || s == D("NJD_SET_DIGIT_HAIHUN5");

    private static int DigitSequenceScore(NjdNode start, NjdNode end)
    {
        int score = 0;
        string kazu = D("NJD_SET_DIGIT_KAZU");
        if (start.Prev is not null)
        {
            string group1 = start.Prev.PosGroup1, group2 = start.Prev.PosGroup2, str = start.Prev.String;
            if (group1 == D("NJD_SET_DIGIT_SUUSETSUZOKU")) score += 2;
            if (group2 == D("NJD_SET_DIGIT_JOSUUSHI") || group1 == D("NJD_SET_DIGIT_FUKUSHIKANOU")) score += 1;
            if (IsPeriod(str))
            {
                if (start.Prev.Prev is not null && start.Prev.Prev.PosGroup1 == kazu) score -= 5;
            }
            else if (IsHyphen(str)) score -= 2;
            else if (str == D("NJD_SET_DIGIT_KAKKO1"))
            {
                if (start.Prev.Prev is not null && start.Prev.Prev.PosGroup1 == kazu) score -= 2;
            }
            else if (str == D("NJD_SET_DIGIT_KAKKO2")) score -= 2;
            else if (str == D("NJD_SET_DIGIT_BANGOU")) score -= 2;
            if (start.Prev.Prev is not null && start.Prev.Prev.String == D("NJD_SET_DIGIT_BANGOU")) score -= 2;
        }
        if (end.Next is not null)
        {
            string group1 = end.Next.PosGroup1, group2 = end.Next.PosGroup2, str = end.Next.String;
            if (group2 == D("NJD_SET_DIGIT_JOSUUSHI") || group1 == D("NJD_SET_DIGIT_FUKUSHIKANOU")) score += 2;
            if (IsHyphen(str)) score -= 2;
            else if (str == D("NJD_SET_DIGIT_KAKKO1")) score -= 2;
            else if (str == D("NJD_SET_DIGIT_KAKKO2"))
            {
                if (end.Next.Next is not null && end.Next.Next.PosGroup1 == kazu) score -= 2;
            }
            else if (str == D("NJD_SET_DIGIT_BANGOU")) score -= 2;
            else if (IsPeriod(str)) score += 4;
        }
        return score;
    }

    private static void ConvertForNonNumericalReading(NjdNode start, NjdNode end)
    {
        int size = 0;
        for (NjdNode node = start; node != end.Next; node = node.Next!) size++;
        if (size <= 1) return;
        size = 0;
        for (NjdNode node = start; node != end.Next; node = node.Next!)
        {
            if (node.String == D("NJD_SET_DIGIT_ZERO1") || node.String == D("NJD_SET_DIGIT_ZERO2"))
            {
                node.Pron = D("NJD_SET_DIGIT_ZERO_AFTER_DP");
                node.MoraSize = 2;
            }
            else if (node.String == D("NJD_SET_DIGIT_TWO"))
            {
                node.Pron = D("NJD_SET_DIGIT_TWO_AFTER_DP");
                node.MoraSize = 2;
            }
            else if (node.String == D("NJD_SET_DIGIT_FIVE"))
            {
                node.Pron = D("NJD_SET_DIGIT_FIVE_AFTER_DP");
                node.MoraSize = 2;
            }
            node.ChainRule = null!;
            if (size % 2 == 0) node.ChainFlag = 0;
            else
            {
                node.ChainFlag = 1;
                node.Prev!.Acc = 3;
            }
            size++;
        }
    }

    private static void ConvertForNumericalReading(NjdNode start, NjdNode end)
    {
        int size = 0;
        for (NjdNode node = start; node != end.Next; node = node.Next!) size++;
        if (size <= 1) return;
        int index = size % 4;
        if (index == 0) index = 4;
        int place = size > index ? (size - index) / 4 : 0;
        index--;
        if (place > 17) return;
        var list2 = L("njd_set_digit_rule_numeral_list2");
        var list3 = L("njd_set_digit_rule_numeral_list3");
        bool have = false;
        for (NjdNode node = start; node != end.Next; node = node.Next!)
        {
            int digit = GetDigit(node, false);
            if (index == 0)
            {
                if (digit == 0)
                {
                    node.Pron = null!;
                    node.Acc = 0;
                    node.MoraSize = 0;
                }
                else
                {
                    have = true;
                }
                if (have)
                {
                    if (place > 0)
                    {
                        var created = new NjdNode();
                        created.Load(list3[place]!);
                        node = NjdNode.Insert(node, node.Next, created);
                    }
                    have = false;
                }
                place--;
            }
            else
            {
                if (digit <= 0)
                {
                    node.Pron = null!;
                    node.Acc = 0;
                    node.MoraSize = 0;
                }
                else if (digit == 1)
                {
                    node.Load(list2[index]!);
                    have = true;
                }
                else
                {
                    var created = new NjdNode();
                    created.Load(list2[index]!);
                    node = NjdNode.Insert(node, node.Next, created);
                    have = true;
                }
            }
            index--;
            if (index < 0) index = 4 - 1;
        }
    }

    private static bool SearchNumerativeClass(string?[] list, NjdNode node)
    {
        string str = node.String;
        if (str == "*") return false;
        for (int i = 0; list[i] is not null; i++)
            if (list[i] == str) return true;
        return false;
    }

    private static void ConvertDigitPron(string?[] list, NjdNode node)
    {
        string str = node.String;
        if (str == "*") return;
        for (int i = 0; list[i] is not null; i += 4)
            if (list[i] == str)
            {
                node.Pron = list[i + 1]!;
                node.Acc = Atoi(list[i + 2]);
                node.MoraSize = Atoi(list[i + 3]);
                return;
            }
    }

    private static void ConvertNumerativePron(string?[] list, NjdNode node1, NjdNode node2)
    {
        string str = node1.String;
        if (str == "*") return;
        int type = 0;
        for (int i = 0; list[i] is not null; i += 2)
            if (list[i] == str)
            {
                type = Atoi(list[i + 1]);
                break;
            }
        string? table = type == 1 ? "njd_set_digit_rule_voiced_sound_symbol_list" : type == 2 ? "njd_set_digit_rule_semivoiced_sound_symbol_list" : null;
        if (table is null) return;
        var symbols = L(table);
        for (int i = 0; symbols[i] is not null; i += 2)
        {
            string pron = node2.Pron;
            int j = StrTopCmp(pron, 0, symbols[i]!);
            if (j >= 0)
            {
                node2.Pron = symbols[i + 1] + pron.Substring(j);
                break;
            }
        }
    }

    private static void ConvertDigitSequence(Njd njd, NjdNode? s, NjdNode? e)
    {
        if (s is null || e is null) return;
        // skip head marks
        if (IsComma(s.String) || IsPeriod(s.String))
        {
            if (s != e) ConvertDigitSequence(njd, s.Next, e);
            return;
        }

        int numericalReading = 1;   // 1: numerical, 0: unknown, -1: non-numerical
        int commas = 0;
        NjdNode? firstCommaBeforePeriod = null;
        var finalDigitBeforePeriod = s;
        while (finalDigitBeforePeriod.Next is not null && finalDigitBeforePeriod != e && !IsPeriod(finalDigitBeforePeriod.Next.String))
            finalDigitBeforePeriod = finalDigitBeforePeriod.Next;
        while (finalDigitBeforePeriod.Prev is not null && finalDigitBeforePeriod != s && IsComma(finalDigitBeforePeriod.String))
            finalDigitBeforePeriod = finalDigitBeforePeriod.Prev;

        // check commas
        {
            int rindex = 0;
            for (var node = finalDigitBeforePeriod; ; node = node.Prev!, rindex++)
            {
                if (IsComma(node.String))
                {
                    firstCommaBeforePeriod = node;
                    commas++;
                    if (numericalReading == 1 && rindex % 4 != 3) numericalReading = 0;
                }
                else if (numericalReading == 1 && rindex % 4 == 3)
                {
                    numericalReading = 0;
                }
                if (node == s) break;
            }
        }

        // check zero-start
        if (s != finalDigitBeforePeriod && GetDigit(s, false) == 0) numericalReading = -1;
        // if no info, set unknown flag
        if (numericalReading == 1 && commas == 0) numericalReading = 0;

        if (numericalReading == 1)
        {
            // numerical reading until period
            if (commas > 0)
                for (var node = s; node != finalDigitBeforePeriod;)
                    node = IsComma(node!.String) ? njd.Remove(node) : node.Next;
            ConvertForNumericalReading(s, finalDigitBeforePeriod);
            if (finalDigitBeforePeriod != e) ConvertDigitSequence(njd, finalDigitBeforePeriod.Next, e);
        }
        else
        {
            var finalDigit = firstCommaBeforePeriod is null ? finalDigitBeforePeriod : firstCommaBeforePeriod.Prev!;
            if (numericalReading == 0) numericalReading = DigitSequenceScore(s, finalDigit) >= 0 ? 1 : -1;
            if (numericalReading == 1) ConvertForNumericalReading(s, finalDigit);
            else ConvertForNonNumericalReading(s, finalDigit);
            if (finalDigit != e) ConvertDigitSequence(njd, finalDigit.Next, e);
        }
    }

    public static void SetDigit(Njd njd)
    {
        string kazu = D("NJD_SET_DIGIT_KAZU");
        NjdNode? s = null, e = null;
        bool find = false;

        // convert digit sequence
        for (var node = njd.Head; node is not null; node = node.Next)
        {
            if (!find && node.PosGroup1 == kazu) find = true;
            if (GetDigit(node, true) >= 0 || (node.PosGroup1 == kazu && (IsPeriod(node.String) || IsComma(node.String))))
            {
                s ??= node;
                if (node == njd.Tail) e = node;
            }
            else if (s is not null)
            {
                e = node.Prev;
            }
            if (s is not null && e is not null)
            {
                ConvertDigitSequence(njd, s, e);
                s = e = null;
            }
        }
        if (!find) return;
        njd.RemoveSilentNodes();
        if (njd.Head is null) return;

        for (var node = njd.Head.Next; node is not null && node.Next is not null;)
        {
            if (node.String != "*" && node.Prev!.String != "*" && IsPeriod(node.String)
                && node.Prev.PosGroup1 == kazu && node.Next.PosGroup1 == kazu)
            {
                node.Load(D("NJD_SET_DIGIT_TEN_FEATURE"));
                node.ChainFlag = 1;
                var prev = node.Prev!;
                if (prev.String == D("NJD_SET_DIGIT_ZERO1") || prev.String == D("NJD_SET_DIGIT_ZERO2"))
                {
                    prev.Pron = D("NJD_SET_DIGIT_ZERO_BEFORE_DP");
                    prev.MoraSize = 2;
                }
                else if (prev.String == D("NJD_SET_DIGIT_TWO"))
                {
                    prev.Pron = D("NJD_SET_DIGIT_TWO_BEFORE_DP");
                    prev.MoraSize = 2;
                }
                else if (prev.String == D("NJD_SET_DIGIT_FIVE"))
                {
                    prev.Pron = D("NJD_SET_DIGIT_FIVE_BEFORE_DP");
                    prev.MoraSize = 2;
                }
                else if (prev.String == D("NJD_SET_DIGIT_SIX"))
                {
                    prev.Acc = 1;
                }
                // skip digit sequence
                node = node.Next;
                while (node is not null && node.Pos == D("NJD_SET_DIGIT_MEISHI")) node = node.Next;
                if (node is not null) node = node.Next;
            }
            else
            {
                node = node.Next;
            }
        }

        string[] classes1 = { "1b", "1c1", "1c2", "1d", "1e", "1f", "1g", "1h", "1i", "1j", "1k" };
        string[] classes2 = { "2b", "2c", "2d", "2e", "2f" };
        for (var node = njd.Head.Next; node is not null; node = node.Next)
        {
            if (node.Prev!.PosGroup1 != kazu) continue;
            if (node.PosGroup2 != D("NJD_SET_DIGIT_JOSUUSHI") && node.PosGroup1 != D("NJD_SET_DIGIT_FUKUSHIKANOU")) continue;
            // convert digit pron
            foreach (var c in classes1)
                if (SearchNumerativeClass(L("njd_set_digit_rule_numerative_class" + c), node))
                {
                    ConvertDigitPron(L("njd_set_digit_rule_conv_table" + c), node.Prev);
                    break;
                }
            // convert numerative pron
            foreach (var c in classes2)
                if (SearchNumerativeClass(L("njd_set_digit_rule_numerative_class" + c), node))
                {
                    ConvertNumerativePron(L("njd_set_digit_rule_conv_table" + c), node.Prev, node);
                    break;
                }
            // modify accent phrase
            node.Prev.ChainFlag = 0;
            node.ChainFlag = 1;
        }

        var list4 = L("njd_set_digit_rule_numeral_list4");
        var list5 = L("njd_set_digit_rule_numeral_list5");
        for (var node = njd.Head.Next; node is not null; node = node.Next)
        {
            if (node.Prev!.PosGroup1 != kazu) continue;
            if (node.PosGroup1 == kazu)
            {
                // modify accent phrase
                find = false;
                for (int i = 0; list4[i] is not null; i++)
                    if (node.Prev.String == list4[i])
                    {
                        for (int j = 0; list5[j] is not null; j++)
                            if (node.String == list5[j])
                            {
                                node.Prev.ChainFlag = 0;
                                node.ChainFlag = 1;
                                find = true;
                                break;
                            }
                        break;
                    }
                if (!find)
                    for (int i = 0; list5[i] is not null; i++)
                        if (node.Prev.String == list5[i])
                        {
                            for (int j = 0; list4[j] is not null; j++)
                                if (node.String == list4[j])
                                {
                                    node.ChainFlag = 0;
                                    break;
                                }
                            break;
                        }
            }
            if (SearchNumerativeClass(L("njd_set_digit_rule_numeral_list8"), node)) ConvertDigitPron(L("njd_set_digit_rule_numeral_list9"), node.Prev);
            if (SearchNumerativeClass(L("njd_set_digit_rule_numeral_list10"), node)) ConvertDigitPron(L("njd_set_digit_rule_numeral_list11"), node.Prev);
            if (SearchNumerativeClass(L("njd_set_digit_rule_numeral_list6"), node)) ConvertNumerativePron(L("njd_set_digit_rule_numeral_list7"), node.Prev, node);
        }

        var class3 = L("njd_set_digit_rule_numerative_class3");
        var table3 = L("njd_set_digit_rule_conv_table3");
        for (var node = njd.Head; node is not null; node = node.Next)
        {
            if (node.Next is null || node.Next.String == "*" || node.PosGroup1 != kazu) continue;
            if (!(node.Prev is null || node.Prev.Pos == D("NJD_SET_DIGIT_KIGOU") || node.Prev.PosGroup1 != kazu)) continue;
            if (node.Next.PosGroup2 != D("NJD_SET_DIGIT_JOSUUSHI") && node.Next.PosGroup1 != D("NJD_SET_DIGIT_FUKUSHIKANOU")) continue;
            // convert class3
            for (int i = 0; class3[i] is not null; i += 2)
                if (node.Next.String == class3[i] && node.Next.Read == class3[i + 1])
                {
                    for (int j = 0; table3[j] is not null; j += 4)
                        if (node.String == table3[j])
                        {
                            node.Read = table3[j + 1]!;
                            node.Pron = table3[j + 1]!;
                            node.Acc = Atoi(table3[j + 2]);
                            node.MoraSize = Atoi(table3[j + 3]);
                            break;
                        }
                    break;
                }
            // person
            if (node.Next.String == D("NJD_SET_DIGIT_NIN"))
            {
                var table4 = L("njd_set_digit_rule_conv_table4");
                for (int i = 0; table4[i] is not null; i += 2)
                    if (node.String == table4[i])
                    {
                        node.Load(table4[i + 1]!);
                        node.Next!.Pron = null!;
                        break;
                    }
            }
            // the day of month
            if (node.Next!.String == D("NJD_SET_DIGIT_NICHI") && node.String != "*")
            {
                if (node.Prev is not null && node.Prev.String.Contains(D("NJD_SET_DIGIT_GATSU")) && node.String == D("NJD_SET_DIGIT_ONE"))
                {
                    node.Load(D("NJD_SET_DIGIT_TSUITACHI"));
                    node.Next!.Pron = null!;
                }
                else
                {
                    var table5 = L("njd_set_digit_rule_conv_table5");
                    for (int i = 0; table5[i] is not null; i += 2)
                        if (node.String == table5[i])
                        {
                            node.Load(table5[i + 1]!);
                            node.Next!.Pron = null!;
                            break;
                        }
                }
            }
            else if (node.Next.String == D("NJD_SET_DIGIT_NICHIKAN"))
            {
                var table6 = L("njd_set_digit_rule_conv_table6");
                for (int i = 0; table6[i] is not null; i += 2)
                    if (node.String == table6[i])
                    {
                        node.Load(table6[i + 1]!);
                        node.Next!.Pron = null!;
                        break;
                    }
            }
        }

        for (var node = njd.Head; node is not null; node = node.Next)
        {
            if (!((node.Prev is null || node.Prev.PosGroup1 != kazu) && node.Next is not null && node.Next.Next is not null)) continue;
            var n1 = node.Next;
            var n2 = n1.Next!;
            if (node.String == D("NJD_SET_DIGIT_TEN") && n1.String == D("NJD_SET_DIGIT_FOUR"))
            {
                if (n2.String == D("NJD_SET_DIGIT_NICHI"))
                {
                    node.Load(D("NJD_SET_DIGIT_JUYOKKA"));
                    node.Next!.Pron = null!;
                    node.Next.Next!.Pron = null!;
                }
                else if (n2.String == D("NJD_SET_DIGIT_NICHIKAN"))
                {
                    node.Load(D("NJD_SET_DIGIT_JUYOKKAKAN"));
                    node.Next!.Pron = null!;
                    node.Next.Next!.Pron = null!;
                }
            }
            else if (node.String == D("NJD_SET_DIGIT_TWO") && n1.String == D("NJD_SET_DIGIT_TEN"))
            {
                if (n2.String == D("NJD_SET_DIGIT_NICHI"))
                {
                    node.Load(D("NJD_SET_DITIT_HATSUKA"));
                    node.Next!.Pron = null!;
                    node.Next.Next!.Pron = null!;
                }
                else if (n2.String == D("NJD_SET_DIGIT_NICHIKAN"))
                {
                    node.Load(D("NJD_SET_DIGIT_HATSUKAKAN"));
                    node.Next!.Pron = null!;
                    node.Next.Next!.Pron = null!;
                }
                else if (n2.String == D("NJD_SET_DIGIT_FOUR") && n2.Next is not null)
                {
                    if (n2.Next.String == D("NJD_SET_DIGIT_NICHI"))
                    {
                        node.Load(D("NJD_SET_DIGIT_NIJU"));
                        node.Next!.Load(D("NJD_SET_DITIT_YOKKA"));
                        node.Next.Next!.Pron = null!;
                        node.Next.Next.Next!.Pron = null!;
                    }
                    else if (n2.Next.String == D("NJD_SET_DIGIT_NICHIKAN"))
                    {
                        node.Load(D("NJD_SET_DIGIT_NIJU"));
                        node.Next!.Load(D("NJD_SET_DIGIT_YOKKAKAN"));
                        node.Next.Next!.Pron = null!;
                        node.Next.Next.Next!.Pron = null!;
                    }
                }
            }
        }

        njd.RemoveSilentNodes();
    }

    // ------------------------------------------------------------------ njd_set_accent_phrase

    public static void SetAccentPhrase(Njd njd)
    {
        if (njd.Head is null) return;
        string P(string name) => D("NJD_SET_ACCENT_PHRASE_" + name);
        for (var node = njd.Head.Next; node is not null; node = node.Next)
        {
            if (node.ChainFlag >= 0) continue;
            var prev = node.Prev!;
            // Rule 01
            node.ChainFlag = 1;
            // Rule 02
            if (prev.Pos == P("MEISHI") && node.Pos == P("MEISHI")) node.ChainFlag = 1;
            // Rule 03
            if (prev.Pos == P("KEIYOUSHI") && node.Pos == P("MEISHI")) node.ChainFlag = 0;
            // Rule 04
            if (prev.Pos == P("MEISHI") && prev.PosGroup1 == P("KEIYOUDOUSHI_GOKAN") && node.Pos == P("MEISHI")) node.ChainFlag = 0;
            // Rule 05
            if (prev.Pos == P("DOUSHI"))
            {
                if (node.Pos == P("KEIYOUSHI")) node.ChainFlag = 0;
                else if (node.Pos == P("MEISHI")) node.ChainFlag = 0;
            }
            // Rule 06
            if (node.Pos == P("FUKUSHI") || prev.Pos == P("FUKUSHI") || node.Pos == P("SETSUZOKUSHI")
                || prev.Pos == P("SETSUZOKUSHI") || node.Pos == P("RENTAISHI") || prev.Pos == P("RENTAISHI"))
                node.ChainFlag = 0;
            // Rule 07
            if (prev.Pos == P("MEISHI") && prev.PosGroup1 == P("FUKUSHI_KANOU")) node.ChainFlag = 0;
            if (node.Pos == P("MEISHI") && node.PosGroup1 == P("FUKUSHI_KANOU")) node.ChainFlag = 0;
            // Rule 08
            if (node.Pos == P("JODOUSHI")) node.ChainFlag = 1;
            if (node.Pos == P("JOSHI")) node.ChainFlag = 1;
            // Rule 09
            if (prev.Pos == P("JODOUSHI") && node.Pos != P("JODOUSHI") && node.Pos != P("JOSHI")) node.ChainFlag = 0;
            if (prev.Pos == P("JOSHI") && node.Pos != P("JODOUSHI") && node.Pos != P("JOSHI")) node.ChainFlag = 0;
            // Rule 10
            if (prev.PosGroup1 == P("SETSUBI") && node.Pos == P("MEISHI")) node.ChainFlag = 0;
            // Rule 11
            if (node.Pos == P("KEIYOUSHI") && node.PosGroup1 == P("HIJIRITSU"))
            {
                if (prev.Pos == P("DOUSHI"))
                {
                    if (StrTopCmp(prev.CForm, 0, P("RENYOU")) != -1) node.ChainFlag = 1;
                }
                else if (prev.Pos == P("KEIYOUSHI"))
                {
                    if (StrTopCmp(prev.CForm, 0, P("RENYOU")) != -1) node.ChainFlag = 1;
                }
                else if (prev.Pos == P("JOSHI"))
                {
                    if (prev.PosGroup1 == P("SETSUZOKUJOSHI"))
                    {
                        if (prev.String == P("TE")) node.ChainFlag = 1;
                        else if (prev.String == P("DE")) node.ChainFlag = 1;
                    }
                }
            }
            // Rule 12
            if (node.Pos == P("DOUSHI") && node.PosGroup1 == P("HIJIRITSU"))
            {
                if (prev.Pos == P("DOUSHI"))
                {
                    if (StrTopCmp(prev.CForm, 0, P("RENYOU")) != -1) node.ChainFlag = 1;
                }
                else if (prev.Pos == P("MEISHI"))
                {
                    if (prev.PosGroup1 == P("SAHEN_SETSUZOKU")) node.ChainFlag = 1;
                }
            }
            // Rule 13
            if (prev.Pos == P("MEISHI") && (node.Pos == P("DOUSHI") || node.Pos == P("KEIYOUSHI") || node.PosGroup1 == P("KEIYOUDOUSHI_GOKAN")))
                node.ChainFlag = 0;
            // Rule 14
            if (node.Pos == P("KIGOU") || prev.Pos == P("KIGOU")) node.ChainFlag = 0;
            // Rule 15
            if (node.Pos == P("SETTOUSHI")) node.ChainFlag = 0;
            // Rule 16
            if (prev.PosGroup3 == P("SEI") && node.Pos == P("MEISHI")) node.ChainFlag = 0;
            // Rule 17
            if (prev.Pos == P("MEISHI") && node.PosGroup3 == P("MEI")) node.ChainFlag = 0;
            // Rule 18
            if (node.PosGroup1 == P("SETSUBI")) node.ChainFlag = 1;
        }
    }

    // ------------------------------------------------------------------ njd_set_accent_type

    // njd_set_accent_type.c's get_token_from_string: up to '%', '@' or '/', the delimiter returned and consumed.
    private static char RuleToken(string str, ref int index, out string buff)
    {
        int start = index;
        char c = index < str.Length ? str[index] : '\0';
        if (c != '\0')
        {
            while (c != '%' && c != '@' && c != '/' && c != '\0')
            {
                index++;
                c = index < str.Length ? str[index] : '\0';
            }
            buff = str.Substring(start, index - start);
            if (c == '%' || c == '@' || c == '/') index++;
            return c;
        }
        buff = "";
        return c;
    }

    // get_rule: the chain rule that applies after a word of the previous part of speech, and its added accent.
    private static string GetRule(string inputRule, string prevPos, out int addType)
    {
        int index = 0;
        char c = ' ';
        while (c != '\0')
        {
            c = RuleToken(inputRule, ref index, out string buff);
            if ((c == '%' && prevPos.Contains(buff)) || c == '@' || c == '/' || c == '\0')
            {
                string rule = buff;
                if (c == '%') c = RuleToken(inputRule, ref index, out rule);
                if (c == '@')
                {
                    RuleToken(inputRule, ref index, out string add);
                    addType = Atoi(add);
                }
                else
                {
                    addType = 0;
                }
                return rule;
            }
            while (c == '%' || c == '@') c = RuleToken(inputRule, ref index, out _);
        }
        addType = 0;
        return "*";
    }

    public static void SetAccentType(Njd njd)
    {
        if (njd.Head is null) return;
        string T(string name) => D("NJD_SET_ACCENT_TYPE_" + name);
        NjdNode? top = null;
        int moraSize = 0;
        for (var node = njd.Head; node is not null; node = node.Next)
        {
            if (node == njd.Head || node.ChainFlag != 1)
            {
                // store the top node
                top = node;
                moraSize = 0;
            }
            else if (node.Prev is not null && node.ChainFlag == 1)
            {
                string rule = GetRule(node.ChainRule, node.Prev.Pos, out int addType);
                switch (rule)
                {
                    case "F2":
                        if (top!.Acc == 0) top.Acc = moraSize + addType;
                        break;
                    case "F3":
                        if (top!.Acc != 0) top.Acc = moraSize + addType;
                        break;
                    case "F4":
                        top!.Acc = moraSize + addType;
                        break;
                    case "F5":
                        top!.Acc = 0;
                        break;
                    case "C1":
                        top!.Acc = moraSize + node.Acc;
                        break;
                    case "C2":
                        top!.Acc = moraSize + 1;
                        break;
                    case "C3":
                        top!.Acc = moraSize;
                        break;
                    case "C4":
                        top!.Acc = 0;
                        break;
                    case "P1":
                        top!.Acc = node.Acc == 0 ? 0 : moraSize + node.Acc;
                        break;
                    case "P2":
                        top!.Acc = node.Acc == 0 ? moraSize + 1 : moraSize + node.Acc;
                        break;
                    case "P6":
                        top!.Acc = 0;
                        break;
                    case "P14":
                        if (node.Acc != 0) top!.Acc = moraSize + node.Acc;
                        break;
                }
            }

            // change accent type for digit
            if (node.Prev is not null && node.ChainFlag == 1 && node.Prev.PosGroup1 == T("KAZU") && node.PosGroup1 == T("KAZU"))
            {
                var prev = node.Prev;
                string ps = prev.String;
                if (node.String == T("JYUU"))
                {
                    prev.Acc = 1;   // both branches of the C code set 1
                    if (ps == T("GO") || ps == T("ROKU") || ps == T("HACHI"))
                    {
                        string? ns = node.Next?.String;
                        if (ns is not null && (ns == T("ICHI") || ns == T("NI") || ns == T("SAN") || ns == T("YON") || ns == T("GO")
                            || ns == T("ROKU") || ns == T("NANA") || ns == T("HACHI") || ns == T("KYUU")))
                            prev.Acc = 0;
                    }
                }
                else if (node.String == T("HYAKU"))
                {
                    if (ps == T("NANA")) prev.Acc = 2;
                    else if (ps == T("SAN") || ps == T("YON") || ps == T("KYUU") || ps == T("NAN")) prev.Acc = 1;
                    else prev.Acc = prev.MoraSize + node.MoraSize;
                }
                else if (node.String == T("SEN"))
                {
                    prev.Acc = prev.MoraSize + 1;
                }
                else if (node.String == T("MAN"))
                {
                    prev.Acc = prev.MoraSize + 1;
                }
                else if (node.String == T("OKU"))
                {
                    prev.Acc = ps == T("ICHI") || ps == T("ROKU") || ps == T("NANA") || ps == T("HACHI") || ps == T("IKU") ? 2 : 1;
                }
                else if (node.String == T("CHOU"))
                {
                    prev.Acc = ps == T("ROKU") || ps == T("NANA") ? 2 : 1;
                }
            }

            if (node.String == T("JYUU") && node.ChainFlag != 1 && node.Next is not null && node.Next.PosGroup1 == T("KAZU"))
                node.Acc = 0;

            moraSize += node.MoraSize;
        }
    }

    // ------------------------------------------------------------------ njd_set_unvoiced_vowel

    private sealed class MoraInfo
    {
        public string? Mora;
        public NjdNode? Link;
        public int Flag = -1, Size, Index, AccentType;
    }

    private static void GetMoraInformation(NjdNode node, int index, MoraInfo m)
    {
        string str = node.Pron;
        int len = str.Length;
        // find next word
        if (index >= len)
        {
            if (node.Next is not null) GetMoraInformation(node.Next, index - len, m);
            else
            {
                m.Mora = null;
                m.Link = null;
                m.Flag = -1;
                m.Size = 0;
                m.Index = 0;
                m.AccentType = 0;
            }
            return;
        }
        m.Link = node;
        // reset mora index and accent type for new word
        if (index == 0 && node.ChainFlag != 1)
        {
            m.Index = 0;
            m.AccentType = node.Acc;
        }
        // special symbol
        string touten = D("NJD_SET_UNVOICED_VOWEL_TOUTEN"), question = D("NJD_SET_UNVOICED_VOWEL_QUESTION");
        if (str == touten || str == question)
        {
            m.Mora = str == touten ? touten : question;
            m.Flag = 0;
            m.Size = m.Mora.Length;
            return;
        }
        m.Mora = null;
        m.Flag = -1;
        m.Size = 0;
        var moras = L("njd_set_unvoiced_vowel_mora_list");
        for (int i = 0; moras[i] is not null; i++)
        {
            int matched = StrTopCmp(str, index, moras[i]!);
            if (matched > 0)
            {
                m.Mora = moras[i];
                m.Size = matched;
                break;
            }
        }
        // get unvoice flag
        int quotation = StrTopCmp(str, index + m.Size, D("NJD_SET_UNVOICED_VOWEL_QUOTATION"));
        if (quotation > 0)
        {
            m.Flag = 1;
            m.Size += quotation;
        }
    }

    private static int ApplyUnvoiceRule(string current, string? next)
    {
        if (next is null) return 0;
        for (int list = 1; list <= 3; list++)
        {
            var candidates = L("njd_set_unvoiced_vowel_candidate_list" + list);
            for (int i = 0; candidates[i] is not null; i++)
                if (current == candidates[i])
                {
                    var nextMoras = L("njd_set_unvoiced_vowel_next_mora_list" + list);
                    for (int j = 0; nextMoras[j] is not null; j++)
                        if (StrTopCmp(next, 0, nextMoras[j]!) > 0) return 1;
                    return 0;
                }
        }
        return -1;   // unknown
    }

    public static void SetUnvoicedVowel(Njd njd)
    {
        string U(string name) => D("NJD_SET_UNVOICED_VOWEL_" + name);
        // mora information for the current, next and next-next moras
        var m1 = new MoraInfo { Index = 0 };
        var m2 = new MoraInfo { Index = 1 };
        var m3 = new MoraInfo { Index = 2 };
        for (var node = njd.Head; node is not null; node = node.Next)
        {
            var buff = new StringBuilder();
            string str = node.Pron;
            int len = str.Length;
            for (int index = 0; index < len;)
            {
                if (m1.Mora is null) GetMoraInformation(node, index, m1);
                if (m1.Mora is null) return;   // "Wrong pron."
                if (m2.Mora is null)
                {
                    m2.Index = m1.Index + 1;
                    m2.AccentType = m1.AccentType;
                    GetMoraInformation(node, index + m1.Size, m2);
                }
                if (m3.Mora is null)
                {
                    m3.Index = m2.Index + 1;
                    m3.AccentType = m2.AccentType;
                    GetMoraInformation(node, index + m1.Size + m2.Size, m3);
                }

                // rule 1: look-ahead for 'masu' and 'desu'
                if (m2.Mora is not null && m3.Mora is not null && m1.Link == m2.Link && m2.Link != m3.Link
                    && (m1.Mora == U("MA") || m1.Mora == U("DE")) && m2.Mora == U("SU")
                    && (m2.Link!.Pos == U("DOUSHI") || m2.Link.Pos == U("JODOUSHI") || m2.Link.Pos == U("KANDOUSHI")))
                    m2.Flag = m3.Link!.Pron == U("QUESTION") || m3.Link.Pron == U("CHOUON") ? 0 : 1;

                // rule 2: look-ahead for 'shi'
                if (m1.Flag != 1 && m2.Flag == -1 && m3.Flag != 1 && m2.Mora is not null && m2.Link!.Pron == U("SHI")
                    && (m2.Link.Pos == U("DOUSHI") || m2.Link.Pos == U("JODOUSHI") || m2.Link.Pos == U("JOSHI")))
                {
                    // rule 4, else rule 5
                    m2.Flag = m2.AccentType == m2.Index + 1 ? 0 : ApplyUnvoiceRule(m2.Mora, m3.Mora);
                    if (m2.Flag == 1)
                    {
                        if (m1.Flag == -1) m1.Flag = 0;
                        if (m3.Flag == -1) m3.Flag = 0;
                    }
                }

                // estimate unvoice
                if (m1.Flag == -1)
                {
                    if (m1.Link!.Pos == U("FILLER")) m1.Flag = 0;                  // rule 0
                    else if (m2.Flag == 1) m1.Flag = 0;                            // rule 3
                    else if (m1.AccentType == m1.Index + 1) m1.Flag = 0;           // rule 4
                    else m1.Flag = ApplyUnvoiceRule(m1.Mora, m2.Mora);           // rule 5
                }
                if (m1.Flag == 1 && m2.Flag == -1) m2.Flag = 0;

                // store pronunciation
                buff.Append(m1.Mora);
                if (m1.Flag == 1) buff.Append(U("QUOTATION"));

                // prepare next step
                index += m1.Size;
                var recycled = m1;
                m1 = m2;
                m2 = m3;
                m3 = recycled;
                m3.Mora = null;
                m3.Link = null;
                m3.Size = 0;
                m3.Flag = -1;
                m3.Index = 0;
                m3.AccentType = 0;
            }
            node.Pron = buff.ToString();
        }
    }
}
