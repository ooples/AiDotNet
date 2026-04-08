using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.Sandboxing.Execution;
using Xunit;

namespace AiDotNet.Serving.Tests.Sandboxing;

public sealed class CompilationDiagnosticParserTests
{
    [Fact]
    public void Parse_GccJson_DocumentRootAndLineRoots()
    {
        var json = """
        {
          "diagnostics": [
            {
              "kind": "warning",
              "message": "unused variable",
              "option": "-Wunused-variable",
              "locations": [
                { "caret": { "file": "main.c", "line": 3, "column": 5 } }
              ]
            }
          ]
        }
        """;

        var diagnostics = CompilationDiagnosticParser.Parse(ProgramLanguage.C, json);
        Assert.Single(diagnostics);
        Assert.Equal("gcc", diagnostics[0].Tool);
        Assert.Equal(CompilationDiagnosticSeverity.Warning, diagnostics[0].Severity);
        Assert.Equal("unused variable", diagnostics[0].Message);
        Assert.Equal("-Wunused-variable", diagnostics[0].Code);
        Assert.Equal("main.c", diagnostics[0].FilePath);
        Assert.Equal(3, diagnostics[0].Line);
        Assert.Equal(5, diagnostics[0].Column);

        var linePayload = "{\"kind\":\"error\",\"message\":\"boom\"}\nnot json\n{\"level\":\"note\",\"message\":\"FYI\"}\n";
        var multi = CompilationDiagnosticParser.Parse(ProgramLanguage.CPlusPlus, linePayload);
        Assert.Equal(2, multi.Count);
        Assert.Equal("g++", multi[0].Tool);
        Assert.Equal(CompilationDiagnosticSeverity.Error, multi[0].Severity);
        Assert.Equal(CompilationDiagnosticSeverity.Info, multi[1].Severity);
    }

    [Fact]
    public void Parse_RustcJson_ExtractsPrimarySpan()
    {
        var payload = """
        {"message":"mismatched types","level":"error","code":{"code":"E0308"},"spans":[{"file_name":"main.rs","line_start":10,"column_start":2,"is_primary":true}]}
        """;

        var diagnostics = CompilationDiagnosticParser.Parse(ProgramLanguage.Rust, payload);
        Assert.Single(diagnostics);
        Assert.Equal("rustc", diagnostics[0].Tool);
        Assert.Equal("E0308", diagnostics[0].Code);
        Assert.Equal("main.rs", diagnostics[0].FilePath);
        Assert.Equal(10, diagnostics[0].Line);
        Assert.Equal(2, diagnostics[0].Column);
    }

    [Fact]
    public void Parse_JavacText_ParsesWarningsAndErrors()
    {
        var text = """
        Main.java:12: error: cannot find symbol
        Main.java:14: warning: unchecked conversion
        """;

        var diagnostics = CompilationDiagnosticParser.Parse(ProgramLanguage.Java, text);
        Assert.Equal(2, diagnostics.Count);
        Assert.Equal("javac", diagnostics[0].Tool);
        Assert.Equal(CompilationDiagnosticSeverity.Error, diagnostics[0].Severity);
        Assert.Equal("Main.java", diagnostics[0].FilePath);
        Assert.Equal(12, diagnostics[0].Line);
        Assert.Equal(CompilationDiagnosticSeverity.Warning, diagnostics[1].Severity);
    }

    [Fact]
    public void Parse_DotNetBuildText_ParsesCSharpDiagnostics()
    {
        var text = "Program.cs(3,5): error CS1002: ; expected\nProgram.cs(4,1): warning CS0168: The variable 'x' is declared but never used";

        var diagnostics = CompilationDiagnosticParser.Parse(ProgramLanguage.CSharp, text);
        Assert.Equal(2, diagnostics.Count);
        Assert.Equal("dotnet", diagnostics[0].Tool);
        Assert.Equal(CompilationDiagnosticSeverity.Error, diagnostics[0].Severity);
        Assert.Equal("CS1002", diagnostics[0].Code);
        Assert.Equal("Program.cs", diagnostics[0].FilePath);
        Assert.Equal(3, diagnostics[0].Line);
        Assert.Equal(5, diagnostics[0].Column);
        Assert.Equal(CompilationDiagnosticSeverity.Warning, diagnostics[1].Severity);
    }
}
