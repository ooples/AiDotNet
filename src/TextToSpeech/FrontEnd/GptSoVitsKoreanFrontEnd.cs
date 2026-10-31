using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// GPT-SoVITS's Korean front end (RVC-Boss/GPT-SoVITS at b4570b8, MIT; <c>text/korean.py</c>, after ORI-Muchim's
/// MB-iSTFT-VITS-Korean): Latin letters read out in Hangul, g2pk2 0.0.3's pronunciation, the syllables written as
/// compatibility jamo with compound vowels split, g2pk2's \u3139-linking corrected, a final jamo followed by ".", and one
/// phone per character ("\u7a7a" for a space, "\u505c" for anything outside GPT-SoVITS's symbols).
/// </summary>
internal static class GptSoVitsKoreanFrontEnd
{
    private static readonly string[] LatinNames =
    {
        "\uc5d0\uc774", "\ube44", "\uc2dc", "\ub514", "\uc774", "\uc5d0\ud504", "\uc9c0", "\uc5d0\uc774\uce58", "\uc544\uc774", "\uc81c\uc774", "\ucf00\uc774", "\uc5d8", "\uc5e0", "\uc5d4", "\uc624", "\ud53c", "\ud050", "\uc544\ub974", "\uc5d0\uc2a4", "\ud2f0",
        "\uc720", "\ube0c\uc774", "\ub354\ube14\uc720", "\uc5d1\uc2a4", "\uc640\uc774", "\uc81c\ud2b8",
    };

    // jamo 0.4.1's _jamo_char_to_hcj over U+1100\u2013U+11FF: the conjoining jamo that have a compatibility jamo of the same
    // name (by Unicode name), as code point pairs.
    private static readonly Lazy<Dictionary<char, char>> ConjoiningToCompatibility = new(() =>
    {
        const string pairs =
            "1100:3131,1101:3132,1102:3134,1103:3137,1104:3138,1105:3139,1106:3141,1107:3142,1108:3143,1109:3145,110a:3146,110b:3147," +
            "110c:3148,110d:3149,110e:314a,110f:314b,1110:314c,1111:314d,1112:314e,1114:3165,1115:3166,111a:3140,111c:316e,111d:3171," +
            "111e:3172,1120:3173,1121:3144,1122:3174,1123:3175,1127:3176,1129:3177,112b:3178,112c:3179,112d:317a,112e:317b,112f:317c," +
            "1132:317d,1136:317e,1140:317f,1147:3180,114c:3181,1157:3184,1158:3185,1159:3186,115b:3167,115c:3135,115d:3136,1161:314f," +
            "1162:3150,1163:3151,1164:3152,1165:3153,1166:3154,1167:3155,1168:3156,1169:3157,116a:3158,116b:3159,116c:315a,116d:315b," +
            "116e:315c,116f:315d,1170:315e,1171:315f,1172:3160,1173:3161,1174:3162,1175:3163,1184:3187,1185:3188,1188:3189,1191:318a," +
            "1192:318b,1194:318c,119e:318d,11a8:3131,11a9:3132,11aa:3133,11ab:3134,11ac:3135,11ad:3136,11ae:3137,11af:3139,11b0:313a," +
            "11b1:313b,11b2:313c,11b3:313d,11b4:313e,11b5:313f,11b6:3140,11b7:3141,11b8:3142,11b9:3144,11ba:3145,11bb:3146,11bc:3147," +
            "11bd:3148,11be:314a,11bf:314b,11c0:314c,11c1:314d,11c2:314e,11c6:3166,11c7:3167,11c8:3168,11cc:3169,11ce:316a,11d3:316b," +
            "11d7:316c,11d9:316d,11dc:316e,11dd:316f,11df:3170,11e2:3171,11e6:3178,11e7:317a,11e8:317c,11ea:317d,11eb:317f,11ee:3180," +
            "11f0:3181,11f1:3182,11f2:3183,11f4:3184,11f9:3186,11ff:3165";
        var map = new Dictionary<char, char>();
        foreach (var pair in pairs.Split(','))
        {
            var parts = pair.Split(':');
            map[(char)int.Parse(parts[0], NumberStyles.HexNumber, CultureInfo.InvariantCulture)] =
                (char)int.Parse(parts[1], NumberStyles.HexNumber, CultureInfo.InvariantCulture);
        }
        return map;
    }, isThreadSafe: true);

    private static readonly (string From, string To)[] DividedVowels =
    {
        ("\u3158", "\u3157\u314f"), ("\u3159", "\u3157\u3150"), ("\u315a", "\u3157\u3163"), ("\u315d", "\u315c\u3153"),
        ("\u315e", "\u315c\u3154"), ("\u315f", "\u315c\u3163"), ("\u3162", "\u3161\u3163"), ("\u3151", "\u3163\u314f"),
        ("\u3152", "\u3163\u3150"), ("\u3155", "\u3163\u3153"), ("\u3156", "\u3163\u3154"), ("\u315b", "\u3163\u3157"),
        ("\u3160", "\u3163\u315c"),
    };

    /// <summary><c>latin_to_hangul</c>: each Latin letter (either case) replaced by its Korean name, a to z in turn.</summary>
    internal static string LatinToHangul(string text)
    {
        for (int i = 0; i < 26; i++)
            text = text.Replace(((char)('a' + i)).ToString(), LatinNames[i]).Replace(((char)('A' + i)).ToString(), LatinNames[i]);
        return text;
    }

    /// <summary><c>divide_hangul</c>: <c>j2hcj(h2j(text))</c>, then the compound vowels split.</summary>
    internal static string DivideHangul(string text)
    {
        var map = ConjoiningToCompatibility.Value;
        var output = new StringBuilder(text.Length * 3);
        foreach (char c in G2pKk.HangulToJamo(text)) output.Append(map.TryGetValue(c, out char h) ? h : c);
        string result = output.ToString();
        foreach (var (from, to) in DividedVowels) result = result.Replace(from, to);
        return result;
    }

    /// <summary><c>fix_g2pk2_error</c>: "\u3147\u3161\u3139 \u3139" and "\u3139\u3161\u3139 \u3139" read with \u3134.</summary>
    internal static string FixG2pk2Error(string text)
    {
        var output = new StringBuilder();
        int i = 0;
        while (i < text.Length - 4)
        {
            string three = text.Substring(i, 3);
            if ((three == "\u3147\u3161\u3139" || three == "\u3139\u3161\u3139") && text[i + 3] == ' ' && text[i + 4] == '\u3139')
            {
                output.Append(three).Append(' ').Append('\u3134');
                i += 5;
            }
            else
            {
                output.Append(text[i]);
                i++;
            }
        }
        output.Append(text.Substring(Math.Min(i, text.Length)));
        return output.ToString();
    }

    private static readonly Regex FinalJamo = new("([\u3131-\u3163])$");

    /// <summary><c>korean.g2p(text)</c>.</summary>
    public static List<string> G2P(string text)
    {
        var symbols = new HashSet<string>(GptSoVitsChineseData.Instance.SymbolsV2, StringComparer.Ordinal);
        text = LatinToHangul(text);
        text = G2pKk.G2pk2.Convert(text);
        text = DivideHangul(text);
        text = FixG2pk2Error(text);
        text = FinalJamo.Replace(text, "$1.");
        var phones = new List<string>();
        foreach (var c in ChineseText.Codepoints(text))
        {
            string ph = c switch
            {
                "\uff1a" or "\uff1b" or "\uff0c" or "\u00b7" or "\u3001" => ",",
                "\u3002" or "\n" => ".",
                "\uff01" => "!",
                "\uff1f" => "?",
                " " => "\u7a7a",
                _ => c,
            };
            phones.Add(symbols.Contains(ph) ? ph : "\u505c");
        }
        return phones;
    }
}
