using System.Net;
using System.Net.Http.Json;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using Xunit;

namespace AiDotNet.Serving.Tests;

public class ProgramSandboxTests : IClassFixture<ProgramSandboxTestFactory>
{
    private readonly HttpClient _client;

    public ProgramSandboxTests(ProgramSandboxTestFactory factory)
    {
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task ExecuteProgram_WithMissingSource_ReturnsBadRequest()
    {
        var request = new ProgramExecuteRequest
        {
            Language = ProgramLanguage.Python,
            SourceCode = string.Empty
        };

        var response = await _client.PostAsJsonAsync("/api/program-synthesis/program/execute", request);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);

        var result = await response.Content.ReadFromJsonAsync<ProgramExecuteResponse>();
        Assert.NotNull(result);
        Assert.False(result.Success);
        Assert.Equal(ProgramLanguage.Python, result.Language);
        Assert.Contains("SourceCode is required", result.Error ?? string.Empty, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task ExecuteProgram_WithValidRequest_ReturnsOk()
    {
        var request = new ProgramExecuteRequest
        {
            Language = ProgramLanguage.Python,
            SourceCode = "print('hi')"
        };

        var response = await _client.PostAsJsonAsync("/api/program-synthesis/program/execute", request);

        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<ProgramExecuteResponse>();
        Assert.NotNull(result);
        Assert.True(result.Success);
        Assert.Equal(ProgramLanguage.Python, result.Language);
        Assert.Equal(0, result.ExitCode);
        Assert.Equal("ok", result.StdOut);
    }

    [Fact]
    public async Task ExecuteProgram_WithCompileOnly_ReturnsCompilationMetadata()
    {
        var request = new ProgramExecuteRequest
        {
            Language = ProgramLanguage.CSharp,
            SourceCode = "public class Program { public static void Main() { } }",
            CompileOnly = true
        };

        var response = await _client.PostAsJsonAsync("/api/program-synthesis/program/execute", request);

        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<ProgramExecuteResponse>();
        Assert.NotNull(result);
        Assert.True(result.Success);
        Assert.Equal(ProgramLanguage.CSharp, result.Language);
        Assert.True(result.CompilationAttempted);
        Assert.True(result.CompilationSucceeded);
    }
}
