using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.Serving.Sandboxing.Execution;
using Xunit;

namespace AiDotNet.Serving.Tests.Sandboxing;

public sealed class ProgramLanguageDetectorTests
{
    [Fact]
    public void Detect_EmptySource_ReturnsNull()
    {
        var detected = ProgramLanguageDetector.Detect("   ", Array.Empty<ProgramLanguage>(), preferredLanguage: null);
        Assert.Null(detected);
    }

    [Fact]
    public void Detect_SqlSource_ReturnsSql()
    {
        var detected = ProgramLanguageDetector.Detect(
            "SELECT * FROM users WHERE id = 1",
            new[] { ProgramLanguage.SQL, ProgramLanguage.CSharp },
            preferredLanguage: null);

        Assert.Equal(ProgramLanguage.SQL, detected);
    }

    [Fact]
    public void Detect_CSharpSource_ReturnsCSharp()
    {
        var detected = ProgramLanguageDetector.Detect(
            "using System; class C { static void Main(){ Console.WriteLine(\"hi\"); } }",
            new[] { ProgramLanguage.CSharp, ProgramLanguage.Java },
            preferredLanguage: null);

        Assert.Equal(ProgramLanguage.CSharp, detected);
    }

    [Fact]
    public void Detect_Tie_UsesPreferredLanguage()
    {
        var detected = ProgramLanguageDetector.Detect(
            "public class C { public static void main(String[] args) {} }",
            new[] { ProgramLanguage.CSharp, ProgramLanguage.Java },
            preferredLanguage: ProgramLanguage.Java);

        Assert.Equal(ProgramLanguage.Java, detected);
    }

    [Fact]
    public void Detect_NoStrongSignal_FallsBackToPreferredWhenAllowed()
    {
        var detected = ProgramLanguageDetector.Detect(
            "some text without language keywords",
            new[] { ProgramLanguage.Python },
            preferredLanguage: ProgramLanguage.Python);

        Assert.Equal(ProgramLanguage.Python, detected);
    }
}

