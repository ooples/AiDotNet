using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Generators;

/// <summary>
/// Minimal hand-rolled parser for the ModelPerfProbe baseline manifest emitted by
/// <c>ModelPerfProbe --profile-all --output artifacts/perf-baseline.json</c>.
/// Pulls out exactly the data the test scaffold needs — the model name and the
/// flagged flag — and ignores everything else, so a netstandard2.0 source
/// generator can read it without pulling in a heavyweight JSON dependency.
/// </summary>
/// <remarks>
/// The manifest is a JSON array of ProbeResult records:
/// <code>
/// [
///   { "model": "LayoutXLM`1", "flagged": true, "flagReason": "..." },
///   { "model": "ACGAN`1",     "flagged": false, ... }
/// ]
/// </code>
/// The parser walks the source by hand, no allocations beyond the returned set.
/// It tolerates field reordering, whitespace, and extra fields the generator
/// doesn't care about — so adding fields to ProbeResult never breaks scaffold
/// emission.
/// </remarks>
internal static class PerfBaselineParser
{
    /// <summary>
    /// Returns the set of model names whose probe-result has <c>"flagged": true</c>.
    /// Both backtick-suffixed (<c>LayoutXLM`1</c>) and stripped (<c>LayoutXLM</c>)
    /// variants are added so the caller can look up either form. Returns an empty
    /// set on any parse failure — the scaffold falls back to "no models tagged
    /// slow", matching the pre-manifest behavior.
    /// </summary>
    public static HashSet<string> ParseSlowModels(string json)
    {
        var slow = new HashSet<string>(System.StringComparer.Ordinal);
        if (string.IsNullOrWhiteSpace(json))
            return slow;

        int i = 0;
        // Walk through each top-level array element and look for {"model": "...", ..., "flagged": true}
        // The probe writes a flat JSON array; one record per { ... } block.
        while (i < json.Length)
        {
            int openBrace = json.IndexOf('{', i);
            if (openBrace < 0) break;
            int closeBrace = FindMatchingBrace(json, openBrace);
            if (closeBrace < 0) break;

            string record = json.Substring(openBrace, closeBrace - openBrace + 1);
            string? modelName = ExtractStringField(record, "model");
            bool flagged = ExtractBoolField(record, "flagged");
            if (flagged && !string.IsNullOrEmpty(modelName))
            {
                slow.Add(modelName!);
                int tick = modelName!.IndexOf('`');
                if (tick > 0) slow.Add(modelName.Substring(0, tick));
            }

            i = closeBrace + 1;
        }
        return slow;
    }

    /// <summary>
    /// Returns the offset of the <c>}</c> that closes the <c>{</c> at <paramref name="start"/>,
    /// respecting brace nesting and skipping over string literals. Returns -1 if no
    /// matching brace is found before the input ends.
    /// </summary>
    private static int FindMatchingBrace(string json, int start)
    {
        int depth = 0;
        bool inString = false;
        for (int i = start; i < json.Length; i++)
        {
            char c = json[i];
            if (inString)
            {
                if (c == '\\' && i + 1 < json.Length) { i++; continue; }
                if (c == '"') inString = false;
            }
            else
            {
                if (c == '"') { inString = true; continue; }
                if (c == '{') depth++;
                else if (c == '}') { depth--; if (depth == 0) return i; }
            }
        }
        return -1;
    }

    /// <summary>
    /// Pulls the value of <c>"fieldName": "value"</c> out of a single JSON record.
    /// Returns null if not found. Unescapes only the basic <c>\\</c> and <c>\"</c>
    /// sequences — the probe never writes anything fancier in the model name.
    /// </summary>
    private static string? ExtractStringField(string record, string fieldName)
    {
        string needle = "\"" + fieldName + "\"";
        int key = record.IndexOf(needle, System.StringComparison.Ordinal);
        if (key < 0) return null;
        int colon = record.IndexOf(':', key + needle.Length);
        if (colon < 0) return null;
        int firstQuote = record.IndexOf('"', colon + 1);
        if (firstQuote < 0) return null;
        var sb = new StringBuilder();
        for (int i = firstQuote + 1; i < record.Length; i++)
        {
            char c = record[i];
            if (c == '\\' && i + 1 < record.Length)
            {
                char next = record[i + 1];
                sb.Append(next == '"' ? '"' : next == '\\' ? '\\' : next);
                i++;
            }
            else if (c == '"')
            {
                return sb.ToString();
            }
            else
            {
                sb.Append(c);
            }
        }
        return null;
    }

    /// <summary>
    /// Pulls the value of <c>"fieldName": true | false</c> out of a single JSON
    /// record. Returns false if not found or if the literal isn't <c>true</c>.
    /// </summary>
    private static bool ExtractBoolField(string record, string fieldName)
    {
        string needle = "\"" + fieldName + "\"";
        int key = record.IndexOf(needle, System.StringComparison.Ordinal);
        if (key < 0) return false;
        int colon = record.IndexOf(':', key + needle.Length);
        if (colon < 0) return false;
        // Skip whitespace after the colon.
        int j = colon + 1;
        while (j < record.Length && (record[j] == ' ' || record[j] == '\t' || record[j] == '\n' || record[j] == '\r'))
            j++;
        return j + 3 < record.Length
            && record[j] == 't' && record[j + 1] == 'r' && record[j + 2] == 'u' && record[j + 3] == 'e';
    }
}
