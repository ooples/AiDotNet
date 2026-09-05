using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.Evolution.Programs;

/// <summary>Infers a program's language and maps languages to file extensions, comment prefixes, and fence labels.</summary>
/// <remarks>
/// <para>
/// The reference OpenEvolve implementation detects a language with six ordered regular expressions and returns a
/// bare string, so <c>public class Main</c> is reported as Python because the Python rule tests <c>^class\s</c>
/// first, and C#, TypeScript, and Go have no rule at all. This detector scores every language in
/// <see cref="ProgramLanguage"/> against weighted, mutually distinguishing signals and returns the highest scorer,
/// resolving ties by enumeration order so the result is reproducible on every machine and every run.
/// </para>
/// <para>
/// Detection is purely lexical and performs no I/O, so it costs one pass over the source and never depends on the
/// file system, the current culture, or an installed toolchain. The mapping helpers are the companion piece: they
/// give the fence label to ask a model for, the comment prefix an evolve-block marker must use, and the file
/// extension a sandbox should write, all from the same single source of truth.
/// </para>
/// <para><b>For Beginners:</b> When a model hands back a block of code you often have to work out what language it
/// is written in — to pick the right file extension, the right comment character, or the right sandbox. This class
/// makes that guess by looking for telltale signs: <c>#include</c> means C or C++, <c>fn main</c> means Rust,
/// <c>SELECT ... FROM</c> means SQL. It also answers the reverse questions, such as "what file extension does C#
/// use?" and "does the label <c>py</c> mean Python?" (it does).</para>
/// </remarks>
public static class ProgramLanguageDetector
{
    private static readonly Dictionary<string, ProgramLanguage> FenceLabels = BuildFenceLabels();
    private static readonly Dictionary<string, ProgramLanguage> FileExtensions = BuildFileExtensions();

    /// <summary>Infers the most likely language of a source snippet.</summary>
    /// <param name="source">The source text to inspect.</param>
    /// <param name="fallback">The language returned when no signal is found.</param>
    /// <returns>The highest scoring language, or <paramref name="fallback"/> when nothing matched.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="fallback"/> is not a defined value.</exception>
    public static ProgramLanguage Detect(string source, ProgramLanguage fallback = ProgramLanguage.Generic)
    {
        if (!Enum.IsDefined(typeof(ProgramLanguage), fallback)) throw new ArgumentOutOfRangeException(nameof(fallback));
        return TryDetect(source, out ProgramLanguage language) ? language : fallback;
    }

    /// <summary>Attempts to infer the language of a source snippet.</summary>
    /// <param name="source">The source text to inspect.</param>
    /// <param name="language">The highest scoring language when at least one signal was found.</param>
    /// <returns><c>true</c> when a language could be inferred.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is <c>null</c>.</exception>
    public static bool TryDetect(string source, out ProgramLanguage language)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        language = ProgramLanguage.Generic;

        List<string> lines = ProgramText.SplitLines(source);
        var scores = new Dictionary<ProgramLanguage, int>();
        foreach (string rawLine in lines)
        {
            string line = rawLine.Trim();
            if (line.Length == 0) continue;
            ScoreLine(line, scores);
        }

        int best = 0;
        bool found = false;
        foreach (ProgramLanguage candidate in EnumerateLanguages())
        {
            if (!scores.TryGetValue(candidate, out int score) || score <= 0) continue;
            if (score > best)
            {
                best = score;
                language = candidate;
                found = true;
            }
        }

        return found;
    }

    /// <summary>Returns the conventional file extension for a language, including the leading dot.</summary>
    /// <param name="language">The language whose extension is wanted.</param>
    /// <returns>An extension such as <c>.py</c> or <c>.cs</c>; <c>.txt</c> for <see cref="ProgramLanguage.Generic"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="language"/> is not a defined value.</exception>
    public static string GetFileExtension(ProgramLanguage language)
    {
        switch (language)
        {
            case ProgramLanguage.Python: return ".py";
            case ProgramLanguage.CSharp: return ".cs";
            case ProgramLanguage.Java: return ".java";
            case ProgramLanguage.JavaScript: return ".js";
            case ProgramLanguage.TypeScript: return ".ts";
            case ProgramLanguage.CPlusPlus: return ".cpp";
            case ProgramLanguage.C: return ".c";
            case ProgramLanguage.Go: return ".go";
            case ProgramLanguage.Rust: return ".rs";
            case ProgramLanguage.SQL: return ".sql";
            case ProgramLanguage.Generic: return ".txt";
            default: throw new ArgumentOutOfRangeException(nameof(language));
        }
    }

    /// <summary>Returns the single-line comment prefix a language uses.</summary>
    /// <param name="language">The language whose comment syntax is wanted.</param>
    /// <returns><c>#</c> for Python and generic text, <c>--</c> for SQL, and <c>//</c> for the C-like languages.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="language"/> is not a defined value.</exception>
    public static string GetLineCommentPrefix(ProgramLanguage language)
    {
        switch (language)
        {
            case ProgramLanguage.Python:
            case ProgramLanguage.Generic:
                return "#";
            case ProgramLanguage.SQL:
                return "--";
            case ProgramLanguage.CSharp:
            case ProgramLanguage.Java:
            case ProgramLanguage.JavaScript:
            case ProgramLanguage.TypeScript:
            case ProgramLanguage.CPlusPlus:
            case ProgramLanguage.C:
            case ProgramLanguage.Go:
            case ProgramLanguage.Rust:
                return "//";
            default:
                throw new ArgumentOutOfRangeException(nameof(language));
        }
    }

    /// <summary>Returns the markdown fence label conventionally used for a language.</summary>
    /// <param name="language">The language whose fence label is wanted.</param>
    /// <returns>A label such as <c>python</c> or <c>csharp</c>; an empty string for <see cref="ProgramLanguage.Generic"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="language"/> is not a defined value.</exception>
    public static string GetFenceLabel(ProgramLanguage language)
    {
        switch (language)
        {
            case ProgramLanguage.Python: return "python";
            case ProgramLanguage.CSharp: return "csharp";
            case ProgramLanguage.Java: return "java";
            case ProgramLanguage.JavaScript: return "javascript";
            case ProgramLanguage.TypeScript: return "typescript";
            case ProgramLanguage.CPlusPlus: return "cpp";
            case ProgramLanguage.C: return "c";
            case ProgramLanguage.Go: return "go";
            case ProgramLanguage.Rust: return "rust";
            case ProgramLanguage.SQL: return "sql";
            case ProgramLanguage.Generic: return string.Empty;
            default: throw new ArgumentOutOfRangeException(nameof(language));
        }
    }

    /// <summary>Resolves a markdown fence label, including common aliases, to a language.</summary>
    /// <param name="label">The fence label, such as <c>py</c>, <c>c#</c>, or <c>golang</c>; matching ignores case.</param>
    /// <param name="language">The resolved language when the label is recognized.</param>
    /// <returns><c>true</c> when the label maps to a known language.</returns>
    public static bool TryGetLanguageForFenceLabel(string? label, out ProgramLanguage language)
    {
        language = ProgramLanguage.Generic;
        if (label is null) return false;
        string key = label.Trim();
        if (key.Length == 0) return false;
        if (!FenceLabels.TryGetValue(key, out ProgramLanguage resolved)) return false;
        language = resolved;
        return true;
    }

    /// <summary>Resolves a file extension, with or without its leading dot, to a language.</summary>
    /// <param name="extension">The extension, such as <c>.py</c> or <c>rs</c>; matching ignores case.</param>
    /// <param name="language">The resolved language when the extension is recognized.</param>
    /// <returns><c>true</c> when the extension maps to a known language.</returns>
    public static bool TryGetLanguageForFileExtension(string? extension, out ProgramLanguage language)
    {
        language = ProgramLanguage.Generic;
        if (extension is null) return false;
        string key = extension.Trim();
        if (key.Length == 0) return false;
        if (key.Length > 0 && key[0] == '.') key = key.Substring(1);
        if (!FileExtensions.TryGetValue(key, out ProgramLanguage resolved)) return false;
        language = resolved;
        return true;
    }

    private static IEnumerable<ProgramLanguage> EnumerateLanguages()
    {
        yield return ProgramLanguage.Python;
        yield return ProgramLanguage.CSharp;
        yield return ProgramLanguage.Java;
        yield return ProgramLanguage.JavaScript;
        yield return ProgramLanguage.TypeScript;
        yield return ProgramLanguage.CPlusPlus;
        yield return ProgramLanguage.C;
        yield return ProgramLanguage.Go;
        yield return ProgramLanguage.Rust;
        yield return ProgramLanguage.SQL;
    }

    private static void ScoreLine(string line, Dictionary<ProgramLanguage, int> scores)
    {
        Add(scores, ProgramLanguage.Python, StartsWith(line, "def ") ? 3 : 0);
        Add(scores, ProgramLanguage.Python, StartsWith(line, "async def ") ? 3 : 0);
        Add(scores, ProgramLanguage.Python, StartsWith(line, "elif ") ? 3 : 0);
        Add(scores, ProgramLanguage.Python, StartsWith(line, "from ") && Contains(line, " import ") ? 4 : 0);
        Add(scores, ProgramLanguage.Python, StartsWith(line, "if __name__") ? 5 : 0);
        Add(scores, ProgramLanguage.Python, Contains(line, "self.") ? 2 : 0);
        Add(scores, ProgramLanguage.Python, StartsWith(line, "import ") && !line.EndsWith(";", StringComparison.Ordinal) ? 2 : 0);

        Add(scores, ProgramLanguage.CSharp, StartsWith(line, "using System") ? 5 : 0);
        Add(scores, ProgramLanguage.CSharp, StartsWith(line, "namespace ") ? 4 : 0);
        Add(scores, ProgramLanguage.CSharp, Contains(line, "Console.WriteLine") ? 5 : 0);
        Add(scores, ProgramLanguage.CSharp, Contains(line, "public sealed class") ? 4 : 0);
        Add(scores, ProgramLanguage.CSharp, Contains(line, "static void Main(string") ? 5 : 0);
        Add(scores, ProgramLanguage.CSharp, Contains(line, " var ") || StartsWith(line, "var ") ? 1 : 0);

        Add(scores, ProgramLanguage.Java, StartsWith(line, "package ") && line.EndsWith(";", StringComparison.Ordinal) ? 4 : 0);
        Add(scores, ProgramLanguage.Java, StartsWith(line, "import java") ? 5 : 0);
        Add(scores, ProgramLanguage.Java, Contains(line, "System.out.println") ? 5 : 0);
        Add(scores, ProgramLanguage.Java, Contains(line, "public static void main(String") ? 5 : 0);

        Add(scores, ProgramLanguage.JavaScript, Contains(line, "console.log") ? 4 : 0);
        Add(scores, ProgramLanguage.JavaScript, StartsWith(line, "function ") ? 3 : 0);
        Add(scores, ProgramLanguage.JavaScript, StartsWith(line, "const ") || StartsWith(line, "let ") ? 2 : 0);
        Add(scores, ProgramLanguage.JavaScript, StartsWith(line, "module.exports") || StartsWith(line, "require(") ? 4 : 0);

        Add(scores, ProgramLanguage.TypeScript, StartsWith(line, "interface ") ? 3 : 0);
        Add(scores, ProgramLanguage.TypeScript, StartsWith(line, "type ") && Contains(line, " = ") ? 3 : 0);
        Add(scores, ProgramLanguage.TypeScript, Contains(line, ": string") || Contains(line, ": number") || Contains(line, ": boolean") ? 4 : 0);
        Add(scores, ProgramLanguage.TypeScript, StartsWith(line, "export ") ? 2 : 0);

        Add(scores, ProgramLanguage.CPlusPlus, Contains(line, "std::") ? 5 : 0);
        Add(scores, ProgramLanguage.CPlusPlus, StartsWith(line, "#include <iostream>") ? 5 : 0);
        Add(scores, ProgramLanguage.CPlusPlus, StartsWith(line, "template<") || StartsWith(line, "template <") ? 4 : 0);
        Add(scores, ProgramLanguage.CPlusPlus, Contains(line, "using namespace std") ? 5 : 0);

        Add(scores, ProgramLanguage.C, StartsWith(line, "#include <stdio.h>") ? 5 : 0);
        Add(scores, ProgramLanguage.C, StartsWith(line, "#include <stdlib.h>") ? 4 : 0);
        Add(scores, ProgramLanguage.C, StartsWith(line, "#include ") ? 1 : 0);
        Add(scores, ProgramLanguage.C, Contains(line, "printf(") ? 3 : 0);
        Add(scores, ProgramLanguage.C, StartsWith(line, "int main(") ? 2 : 0);

        Add(scores, ProgramLanguage.Go, StartsWith(line, "package main") ? 5 : 0);
        Add(scores, ProgramLanguage.Go, StartsWith(line, "func ") ? 4 : 0);
        Add(scores, ProgramLanguage.Go, Contains(line, "fmt.") ? 4 : 0);
        Add(scores, ProgramLanguage.Go, StartsWith(line, ":=") || Contains(line, " := ") ? 3 : 0);

        Add(scores, ProgramLanguage.Rust, StartsWith(line, "fn ") ? 4 : 0);
        Add(scores, ProgramLanguage.Rust, Contains(line, "let mut ") ? 5 : 0);
        Add(scores, ProgramLanguage.Rust, StartsWith(line, "impl ") ? 4 : 0);
        Add(scores, ProgramLanguage.Rust, Contains(line, "println!") ? 5 : 0);
        Add(scores, ProgramLanguage.Rust, StartsWith(line, "use ") && line.EndsWith(";", StringComparison.Ordinal) ? 2 : 0);

        string upper = line.ToUpperInvariant();
        Add(scores, ProgramLanguage.SQL, StartsWith(upper, "SELECT ") ? 4 : 0);
        Add(scores, ProgramLanguage.SQL, Contains(upper, " FROM ") ? 2 : 0);
        Add(scores, ProgramLanguage.SQL, StartsWith(upper, "CREATE TABLE") ? 5 : 0);
        Add(scores, ProgramLanguage.SQL, StartsWith(upper, "INSERT INTO") ? 5 : 0);
        Add(scores, ProgramLanguage.SQL, StartsWith(upper, "UPDATE ") && Contains(upper, " SET ") ? 4 : 0);
    }

    private static void Add(Dictionary<ProgramLanguage, int> scores, ProgramLanguage language, int amount)
    {
        if (amount == 0) return;
        scores[language] = scores.TryGetValue(language, out int current) ? current + amount : amount;
    }

    private static bool StartsWith(string line, string prefix) => line.StartsWith(prefix, StringComparison.Ordinal);

    private static bool Contains(string line, string value) => line.IndexOf(value, StringComparison.Ordinal) >= 0;

    private static Dictionary<string, ProgramLanguage> BuildFenceLabels()
    {
        var map = new Dictionary<string, ProgramLanguage>(StringComparer.OrdinalIgnoreCase)
        {
            ["py"] = ProgramLanguage.Python,
            ["py3"] = ProgramLanguage.Python,
            ["python"] = ProgramLanguage.Python,
            ["python3"] = ProgramLanguage.Python,
            ["cs"] = ProgramLanguage.CSharp,
            ["c#"] = ProgramLanguage.CSharp,
            ["csharp"] = ProgramLanguage.CSharp,
            ["dotnet"] = ProgramLanguage.CSharp,
            ["java"] = ProgramLanguage.Java,
            ["js"] = ProgramLanguage.JavaScript,
            ["jsx"] = ProgramLanguage.JavaScript,
            ["node"] = ProgramLanguage.JavaScript,
            ["javascript"] = ProgramLanguage.JavaScript,
            ["ts"] = ProgramLanguage.TypeScript,
            ["tsx"] = ProgramLanguage.TypeScript,
            ["typescript"] = ProgramLanguage.TypeScript,
            ["cpp"] = ProgramLanguage.CPlusPlus,
            ["c++"] = ProgramLanguage.CPlusPlus,
            ["cxx"] = ProgramLanguage.CPlusPlus,
            ["cc"] = ProgramLanguage.CPlusPlus,
            ["c"] = ProgramLanguage.C,
            ["go"] = ProgramLanguage.Go,
            ["golang"] = ProgramLanguage.Go,
            ["rs"] = ProgramLanguage.Rust,
            ["rust"] = ProgramLanguage.Rust,
            ["sql"] = ProgramLanguage.SQL,
            ["mysql"] = ProgramLanguage.SQL,
            ["postgres"] = ProgramLanguage.SQL,
            ["postgresql"] = ProgramLanguage.SQL,
            ["tsql"] = ProgramLanguage.SQL,
            ["plsql"] = ProgramLanguage.SQL,
            ["text"] = ProgramLanguage.Generic,
            ["txt"] = ProgramLanguage.Generic,
            ["plaintext"] = ProgramLanguage.Generic
        };

        return map;
    }

    private static Dictionary<string, ProgramLanguage> BuildFileExtensions()
    {
        var map = new Dictionary<string, ProgramLanguage>(StringComparer.OrdinalIgnoreCase)
        {
            ["py"] = ProgramLanguage.Python,
            ["pyw"] = ProgramLanguage.Python,
            ["cs"] = ProgramLanguage.CSharp,
            ["java"] = ProgramLanguage.Java,
            ["js"] = ProgramLanguage.JavaScript,
            ["mjs"] = ProgramLanguage.JavaScript,
            ["cjs"] = ProgramLanguage.JavaScript,
            ["jsx"] = ProgramLanguage.JavaScript,
            ["ts"] = ProgramLanguage.TypeScript,
            ["tsx"] = ProgramLanguage.TypeScript,
            ["cpp"] = ProgramLanguage.CPlusPlus,
            ["cxx"] = ProgramLanguage.CPlusPlus,
            ["cc"] = ProgramLanguage.CPlusPlus,
            ["hpp"] = ProgramLanguage.CPlusPlus,
            ["hh"] = ProgramLanguage.CPlusPlus,
            ["c"] = ProgramLanguage.C,
            ["h"] = ProgramLanguage.C,
            ["go"] = ProgramLanguage.Go,
            ["rs"] = ProgramLanguage.Rust,
            ["sql"] = ProgramLanguage.SQL,
            ["txt"] = ProgramLanguage.Generic
        };

        return map;
    }
}
