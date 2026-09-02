using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramSandboxOptionsTests
{
    [Fact]
    public void DefaultsMirrorTheServingFreeTierLimits()
    {
        var limits = new ProgramSandboxLimitOptions();

        Assert.Equal(5, limits.TimeLimitSeconds);
        Assert.Equal(256, limits.MemoryLimitMb);
        Assert.Equal(1.0, limits.CpuLimit);
        Assert.Equal(200_000, limits.MaxSourceCodeChars);
        Assert.Equal(100_000, limits.MaxStdInChars);
        Assert.Equal(64_000, limits.MaxStdOutChars);
        Assert.Equal(64_000, limits.MaxStdErrChars);
        Assert.Equal(4, limits.MaxConcurrentExecutions);
        Assert.Equal(TimeSpan.FromSeconds(5), limits.GetTimeLimit());
        Assert.Equal(256L * 1024L * 1024L, limits.GetMemoryLimitBytes());
    }

    [Fact]
    public void TheDefaultModeIsTheSafeOne()
    {
        var options = new ProgramSandboxOptions();

        Assert.Equal(ProgramSandboxMode.OutOfProcessWorker, options.Mode);
        Assert.False(options.AllowUnsafeInProcessExecution);
        options.Validate();
    }

    [Fact]
    public void InProcessUnsafeThrowsUntilItIsExplicitlyAllowed()
    {
        var options = new ProgramSandboxOptions { Mode = ProgramSandboxMode.InProcessUnsafe };

        ArgumentException failure = Assert.Throws<ArgumentException>(() => options.Validate());
        Assert.Contains("AllowUnsafeInProcessExecution", failure.Message, StringComparison.Ordinal);

        options.AllowUnsafeInProcessExecution = true;
        options.Validate();
    }

    [Theory]
    [InlineData(0, 256, 1.0, 4)]
    [InlineData(-1, 256, 1.0, 4)]
    [InlineData(5, 0, 1.0, 4)]
    [InlineData(5, -8, 1.0, 4)]
    [InlineData(5, 256, 0.0, 4)]
    [InlineData(5, 256, double.NaN, 4)]
    [InlineData(5, 256, 1.0, 0)]
    [InlineData(ProgramSandboxLimitOptions.MaxTimeLimitSeconds + 1, 256, 1.0, 4)]
    public void ImpossibleLimitsAreRejected(int seconds, int memoryMb, double cpu, int concurrency)
    {
        var limits = new ProgramSandboxLimitOptions
        {
            TimeLimitSeconds = seconds,
            MemoryLimitMb = memoryMb,
            CpuLimit = cpu,
            MaxConcurrentExecutions = concurrency
        };

        Assert.Throws<ArgumentOutOfRangeException>(() => limits.Validate());
    }

    [Fact]
    public void NegativeOutputCapsAreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ProgramSandboxLimitOptions { MaxStdOutChars = -1 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ProgramSandboxLimitOptions { MaxStdErrChars = -1 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ProgramSandboxLimitOptions { MaxStdInChars = -1 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ProgramSandboxLimitOptions { MaxSourceCodeChars = 0 }.Validate());
    }

    [Fact]
    public void DefaultInterpretersCoverPythonJavaScriptAndCSharp()
    {
        var options = new ProgramSandboxOptions();

        ProgramInterpreterSpecification python = Require(options, ProgramLanguage.Python);
        Assert.True(python.SupportsCompileOnly);
        Assert.Contains("python", python.Executable, StringComparison.Ordinal);

        ProgramInterpreterSpecification node = Require(options, ProgramLanguage.JavaScript);
        Assert.Equal("node", node.Executable);
        Assert.True(node.SupportsCompileOnly);

        ProgramInterpreterSpecification csharp = Require(options, ProgramLanguage.CSharp);
        Assert.Equal("dotnet", csharp.Executable);
        Assert.False(csharp.SupportsCompileOnly);

        Assert.False(options.TryGetInterpreter(ProgramLanguage.Rust, out _));
    }

    private static ProgramInterpreterSpecification Require(ProgramSandboxOptions options, ProgramLanguage language)
    {
        Assert.True(options.TryGetInterpreter(language, out ProgramInterpreterSpecification? specification));
        Assert.NotNull(specification);
        return specification ?? new ProgramInterpreterSpecification("missing", "{source}");
    }

    [Fact]
    public void AnInterpreterWithoutTheSourcePlaceholderIsRejected()
    {
        var options = new ProgramSandboxOptions();
        options.SetInterpreter(ProgramLanguage.Python, new ProgramInterpreterSpecification("python", "--version"));

        Assert.Throws<ArgumentException>(() => options.Validate());

        var compileMissing = new ProgramSandboxOptions();
        compileMissing.SetInterpreter(
            ProgramLanguage.Python,
            new ProgramInterpreterSpecification("python", "{source}", "--version"));

        Assert.Throws<ArgumentException>(() => compileMissing.Validate());
    }

    [Fact]
    public void CloneIsIndependentOfTheOriginal()
    {
        var options = new ProgramSandboxOptions { WorkingDirectory = "/tmp/original" };
        options.Limits.TimeLimitSeconds = 11;
        options.SetInterpreter(ProgramLanguage.Go, new ProgramInterpreterSpecification("go", "run {source}"));

        ProgramSandboxOptions clone = options.Clone();
        options.Limits.TimeLimitSeconds = 1;
        options.WorkingDirectory = "/tmp/changed";
        options.Interpreters.Remove(ProgramLanguage.Go);

        Assert.Equal(11, clone.Limits.TimeLimitSeconds);
        Assert.Equal("/tmp/original", clone.WorkingDirectory);
        Assert.True(clone.TryGetInterpreter(ProgramLanguage.Go, out _));
    }

    [Fact]
    public void WhiteSpaceWorkingDirectoryIsRejectedButNullIsAllowed()
    {
        Assert.Throws<ArgumentException>(() => new ProgramSandboxOptions { WorkingDirectory = "   " }.Validate());
        new ProgramSandboxOptions { WorkingDirectory = null }.Validate();
    }

    [Fact]
    public void SetInterpreterValidatesItsArguments()
    {
        var options = new ProgramSandboxOptions();
#pragma warning disable CS8625
        Assert.Throws<ArgumentNullException>(() => options.SetInterpreter(ProgramLanguage.Python, null));
#pragma warning restore CS8625
        Assert.Throws<ArgumentOutOfRangeException>(
            () => options.SetInterpreter((ProgramLanguage)99, new ProgramInterpreterSpecification("x", "{source}")));
    }

    [Fact]
    public void InterpreterSpecificationValidatesItsArguments()
    {
#pragma warning disable CS8625
        Assert.Throws<ArgumentNullException>(() => new ProgramInterpreterSpecification(null, "{source}"));
        Assert.Throws<ArgumentNullException>(() => new ProgramInterpreterSpecification("python", null));
#pragma warning restore CS8625
        Assert.Throws<ArgumentException>(() => new ProgramInterpreterSpecification("  ", "{source}"));
    }

    [Theory]
    [InlineData("plain", "\"plain\"")]
    [InlineData("with space", "\"with space\"")]
    [InlineData("C:\\dir\\file.py", "\"C:\\dir\\file.py\"")]
    [InlineData("C:\\dir\\", "\"C:\\dir\\\\\"")]
    [InlineData("quote\"inside", "\"quote\\\"inside\"")]
    public void CommandLineQuotingSurvivesTheParser(string value, string expected) =>
        Assert.Equal(expected, ProgramInterpreterSpecification.QuoteForCommandLine(value));

    [Theory]
    [InlineData("plain", "'plain'")]
    [InlineData("/tmp/a b", "'/tmp/a b'")]
    [InlineData("it's", "'it'\\''s'")]
    public void PosixQuotingEscapesEmbeddedQuotes(string value, string expected) =>
        Assert.Equal(expected, ProgramInterpreterSpecification.QuoteForPosixShell(value));

    [Fact]
    public void ExpandReplacesBothPlaceholders()
    {
        string expanded = ProgramInterpreterSpecification.Expand(
            "run {source} in {workspace}", "/w/program.py", "/w", quoteWithDoubleQuotes: false);

        Assert.Equal("run '/w/program.py' in '/w'", expanded);
    }
}
