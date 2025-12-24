using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.Serving.Sandboxing.Execution;

public static class ProgramLanguageDetector
{
    public static ProgramLanguage? Detect(
        string sourceCode,
        IReadOnlyCollection<ProgramLanguage> allowedLanguages,
        ProgramLanguage? preferredLanguage)
    {
        if (string.IsNullOrWhiteSpace(sourceCode))
        {
            return null;
        }

        var candidates = allowedLanguages.Count > 0
            ? allowedLanguages
            : new[]
            {
                ProgramLanguage.SQL,
                ProgramLanguage.Python,
                ProgramLanguage.CSharp,
                ProgramLanguage.Java,
                ProgramLanguage.TypeScript,
                ProgramLanguage.JavaScript,
                ProgramLanguage.C,
                ProgramLanguage.CPlusPlus,
                ProgramLanguage.Go,
                ProgramLanguage.Rust
            };

        var scores = new Dictionary<ProgramLanguage, int>();
        foreach (var language in candidates)
        {
            scores[language] = Score(language, sourceCode);
        }

        var max = scores.Values.DefaultIfEmpty(0).Max();
        if (max <= 0)
        {
            return preferredLanguage.HasValue && candidates.Contains(preferredLanguage.Value)
                ? preferredLanguage.Value
                : null;
        }

        var best = scores.Where(kvp => kvp.Value == max).Select(kvp => kvp.Key).ToList();
        if (best.Count == 1)
        {
            return best[0];
        }

        if (preferredLanguage.HasValue && best.Contains(preferredLanguage.Value))
        {
            return preferredLanguage.Value;
        }

        return null;
    }

    private static int Score(ProgramLanguage language, string sourceCode)
    {
        var s = sourceCode;
        int score = 0;

        bool Has(string token) => s.Contains(token, StringComparison.OrdinalIgnoreCase);

        switch (language)
        {
            case ProgramLanguage.SQL:
                if (Has("select ") || Has("insert ") || Has("update ") || Has("create table") || Has("from "))
                    score += 3;
                if (Has("where ") || Has("join ") || Has("group by") || Has("order by"))
                    score += 2;
                break;

            case ProgramLanguage.Python:
                if (Has("def ") || Has("import ") || Has("print("))
                    score += 3;
                if (Has("elif") || Has("self") || Has("None"))
                    score += 2;
                break;

            case ProgramLanguage.CSharp:
                if (Has("using System") || Has("namespace ") || Has("Console.WriteLine"))
                    score += 3;
                if (Has("public class") || Has("static void Main") || Has("var "))
                    score += 2;
                break;

            case ProgramLanguage.Java:
                if (Has("public class") || Has("System.out") || Has("package "))
                    score += 3;
                if (Has("static void main") || Has("import java"))
                    score += 2;
                break;

            case ProgramLanguage.JavaScript:
                if (Has("console.log") || Has("function ") || Has("=>"))
                    score += 3;
                if (Has("let ") || Has("const ") || Has("var "))
                    score += 2;
                break;

            case ProgramLanguage.TypeScript:
                if (Has("interface ") || Has("type ") || Has(": string") || Has(": number"))
                    score += 3;
                if (Has("enum ") || Has("implements ") || Has("readonly "))
                    score += 2;
                break;

            case ProgramLanguage.C:
                if (Has("#include") || Has("printf(") || Has("int main"))
                    score += 3;
                break;

            case ProgramLanguage.CPlusPlus:
                if (Has("#include") && (Has("std::") || Has("<iostream>")))
                    score += 3;
                if (Has("cout") || Has("using namespace std"))
                    score += 2;
                break;

            case ProgramLanguage.Go:
                if (Has("package main") || Has("func main") || Has("fmt."))
                    score += 3;
                break;

            case ProgramLanguage.Rust:
                if (Has("fn main") || Has("println!") || Has("use std::"))
                    score += 3;
                break;
        }

        return score;
    }
}

