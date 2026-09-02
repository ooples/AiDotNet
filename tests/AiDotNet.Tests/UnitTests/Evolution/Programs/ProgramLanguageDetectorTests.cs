using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramLanguageDetectorTests
{
    [Fact]
    public void DetectsPython()
    {
        const string Source = "import math\n\ndef solve(x):\n    return x * 2\n\nif __name__ == '__main__':\n    print(solve(1))\n";
        Assert.Equal(ProgramLanguage.Python, ProgramLanguageDetector.Detect(Source));
    }

    [Fact]
    public void DetectsCSharpWhichTheReferenceHeuristicMisclassifies()
    {
        const string Source =
            "using System;\n\nnamespace Demo;\n\npublic sealed class Solver\n{\n" +
            "    public static void Main(string[] args) => Console.WriteLine(1);\n}\n";

        Assert.Equal(ProgramLanguage.CSharp, ProgramLanguageDetector.Detect(Source));
    }

    [Fact]
    public void DetectsJava()
    {
        const string Source =
            "package demo;\nimport java.util.List;\npublic class Main {\n" +
            "    public static void main(String[] args) { System.out.println(1); }\n}\n";

        Assert.Equal(ProgramLanguage.Java, ProgramLanguageDetector.Detect(Source));
    }

    [Fact]
    public void DetectsJavaScriptAndTypeScriptSeparately()
    {
        const string JavaScript = "function add(a, b) {\n  return a + b;\n}\nconsole.log(add(1, 2));\n";
        const string TypeScript =
            "export interface Point {\n  x: number;\n}\n" +
            "export function add(a: number, b: number): number {\n  return a + b;\n}\n";

        Assert.Equal(ProgramLanguage.JavaScript, ProgramLanguageDetector.Detect(JavaScript));
        Assert.Equal(ProgramLanguage.TypeScript, ProgramLanguageDetector.Detect(TypeScript));
    }

    [Fact]
    public void DetectsCAndCPlusPlusSeparately()
    {
        const string C = "#include <stdio.h>\nint main(void) {\n    printf(\"hi\");\n    return 0;\n}\n";
        const string CPlusPlus = "#include <iostream>\nint main() {\n    std::cout << 1;\n    return 0;\n}\n";

        Assert.Equal(ProgramLanguage.C, ProgramLanguageDetector.Detect(C));
        Assert.Equal(ProgramLanguage.CPlusPlus, ProgramLanguageDetector.Detect(CPlusPlus));
    }

    [Fact]
    public void DetectsGoAndRust()
    {
        const string Go = "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"hi\")\n}\n";
        const string Rust = "fn main() {\n    let mut total = 0;\n    println!(\"{}\", total);\n}\n";

        Assert.Equal(ProgramLanguage.Go, ProgramLanguageDetector.Detect(Go));
        Assert.Equal(ProgramLanguage.Rust, ProgramLanguageDetector.Detect(Rust));
    }

    [Fact]
    public void DetectsSql()
    {
        Assert.Equal(ProgramLanguage.SQL, ProgramLanguageDetector.Detect("SELECT id, name FROM users WHERE id = 1;"));
        Assert.Equal(ProgramLanguage.SQL, ProgramLanguageDetector.Detect("CREATE TABLE t (id INT);"));
    }

    [Fact]
    public void UnrecognizedSourceFallsBackWithoutGuessing()
    {
        Assert.Equal(ProgramLanguage.Generic, ProgramLanguageDetector.Detect("hello world"));
        Assert.Equal(ProgramLanguage.Java, ProgramLanguageDetector.Detect("hello world", ProgramLanguage.Java));
        Assert.False(ProgramLanguageDetector.TryDetect("hello world", out _));
    }

    [Fact]
    public void DetectionIsRepeatable()
    {
        const string Source = "def solve():\n    return 1\n";
        ProgramLanguage first = ProgramLanguageDetector.Detect(Source);
        for (int index = 0; index < 5; index++) Assert.Equal(first, ProgramLanguageDetector.Detect(Source));
    }

    [Theory]
    [InlineData(ProgramLanguage.Python, ".py", "#", "python")]
    [InlineData(ProgramLanguage.CSharp, ".cs", "//", "csharp")]
    [InlineData(ProgramLanguage.Java, ".java", "//", "java")]
    [InlineData(ProgramLanguage.JavaScript, ".js", "//", "javascript")]
    [InlineData(ProgramLanguage.TypeScript, ".ts", "//", "typescript")]
    [InlineData(ProgramLanguage.CPlusPlus, ".cpp", "//", "cpp")]
    [InlineData(ProgramLanguage.C, ".c", "//", "c")]
    [InlineData(ProgramLanguage.Go, ".go", "//", "go")]
    [InlineData(ProgramLanguage.Rust, ".rs", "//", "rust")]
    [InlineData(ProgramLanguage.SQL, ".sql", "--", "sql")]
    [InlineData(ProgramLanguage.Generic, ".txt", "#", "")]
    public void MapsCoverEveryLanguage(ProgramLanguage language, string extension, string comment, string fence)
    {
        Assert.Equal(extension, ProgramLanguageDetector.GetFileExtension(language));
        Assert.Equal(comment, ProgramLanguageDetector.GetLineCommentPrefix(language));
        Assert.Equal(fence, ProgramLanguageDetector.GetFenceLabel(language));
    }

    [Theory]
    [InlineData("py", ProgramLanguage.Python)]
    [InlineData("Python3", ProgramLanguage.Python)]
    [InlineData("C#", ProgramLanguage.CSharp)]
    [InlineData("cs", ProgramLanguage.CSharp)]
    [InlineData("JS", ProgramLanguage.JavaScript)]
    [InlineData("tsx", ProgramLanguage.TypeScript)]
    [InlineData("c++", ProgramLanguage.CPlusPlus)]
    [InlineData("golang", ProgramLanguage.Go)]
    [InlineData("rs", ProgramLanguage.Rust)]
    [InlineData("postgresql", ProgramLanguage.SQL)]
    [InlineData(" plaintext ", ProgramLanguage.Generic)]
    public void FenceLabelAliasesResolve(string label, ProgramLanguage expected)
    {
        Assert.True(ProgramLanguageDetector.TryGetLanguageForFenceLabel(label, out ProgramLanguage language));
        Assert.Equal(expected, language);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("brainfuck")]
    public void UnknownFenceLabelsDoNotResolveToPython(string label)
    {
        Assert.False(ProgramLanguageDetector.TryGetLanguageForFenceLabel(label, out ProgramLanguage language));
        Assert.Equal(ProgramLanguage.Generic, language);
    }

    [Theory]
    [InlineData(".py", ProgramLanguage.Python)]
    [InlineData("py", ProgramLanguage.Python)]
    [InlineData(".CS", ProgramLanguage.CSharp)]
    [InlineData(".hpp", ProgramLanguage.CPlusPlus)]
    [InlineData(".h", ProgramLanguage.C)]
    [InlineData(".sql", ProgramLanguage.SQL)]
    public void FileExtensionsResolve(string extension, ProgramLanguage expected)
    {
        Assert.True(ProgramLanguageDetector.TryGetLanguageForFileExtension(extension, out ProgramLanguage language));
        Assert.Equal(expected, language);
    }

    [Fact]
    public void UnknownFileExtensionsDoNotResolveToPython()
    {
        Assert.False(ProgramLanguageDetector.TryGetLanguageForFileExtension(".bf", out ProgramLanguage language));
        Assert.Equal(ProgramLanguage.Generic, language);
        Assert.False(ProgramLanguageDetector.TryGetLanguageForFileExtension(null, out _));
    }

    [Fact]
    public void EveryLanguageRoundTripsThroughItsOwnFenceLabel()
    {
        foreach (ProgramLanguage language in Enum.GetValues(typeof(ProgramLanguage)))
        {
            string label = ProgramLanguageDetector.GetFenceLabel(language);
            if (label.Length == 0) continue;
            Assert.True(ProgramLanguageDetector.TryGetLanguageForFenceLabel(label, out ProgramLanguage resolved));
            Assert.Equal(language, resolved);
        }
    }

    [Fact]
    public void EveryLanguageRoundTripsThroughItsOwnFileExtension()
    {
        foreach (ProgramLanguage language in Enum.GetValues(typeof(ProgramLanguage)))
        {
            string extension = ProgramLanguageDetector.GetFileExtension(language);
            Assert.True(ProgramLanguageDetector.TryGetLanguageForFileExtension(extension, out ProgramLanguage resolved));
            Assert.Equal(language, resolved);
        }
    }

    [Fact]
    public void UndefinedLanguagesAreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ProgramLanguageDetector.GetFileExtension((ProgramLanguage)999));
        Assert.Throws<ArgumentOutOfRangeException>(() => ProgramLanguageDetector.GetLineCommentPrefix((ProgramLanguage)999));
        Assert.Throws<ArgumentOutOfRangeException>(() => ProgramLanguageDetector.GetFenceLabel((ProgramLanguage)999));
        Assert.Throws<ArgumentOutOfRangeException>(() => ProgramLanguageDetector.Detect("x", (ProgramLanguage)999));
    }
}
