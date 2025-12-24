using System.Net;
using System.Text;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Serialization;
using Xunit;

namespace AiDotNet.Serving.Tests;

public class ProgramSandboxTests : IClassFixture<ProgramSandboxTestFactory>
{
    private readonly HttpClient _client;
    private static readonly JsonSerializerSettings JsonSettings = new()
    {
        ContractResolver = new CamelCasePropertyNamesContractResolver(),
        Converters = { new StringEnumConverter(new CamelCaseNamingStrategy(), allowIntegerValues: false) }
    };

    public ProgramSandboxTests(ProgramSandboxTestFactory factory)
    {
        _client = factory.CreateClient();
    }

    private async Task<HttpResponseMessage> PostAsJsonAsync<T>(string requestUri, T value)
    {
        var json = JsonConvert.SerializeObject(value, JsonSettings);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        return await _client.PostAsync(requestUri, content);
    }

    private static async Task<T?> ReadFromJsonAsync<T>(HttpContent content)
    {
        var json = await content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<T>(json, JsonSettings);
    }

    [Fact]
    public async Task ExecuteProgram_WithMissingSource_ReturnsBadRequest()
    {
        var request = new ProgramExecuteRequest
        {
            Language = ProgramLanguage.Python,
            SourceCode = string.Empty
        };

        var response = await PostAsJsonAsync("/api/program-synthesis/program/execute", request);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);

        var result = await ReadFromJsonAsync<ProgramExecuteResponse>(response.Content);
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

        var response = await PostAsJsonAsync("/api/program-synthesis/program/execute", request);

        response.EnsureSuccessStatusCode();

        var result = await ReadFromJsonAsync<ProgramExecuteResponse>(response.Content);
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

        var response = await PostAsJsonAsync("/api/program-synthesis/program/execute", request);

        response.EnsureSuccessStatusCode();

        var result = await ReadFromJsonAsync<ProgramExecuteResponse>(response.Content);
        Assert.NotNull(result);
        Assert.True(result.Success);
        Assert.Equal(ProgramLanguage.CSharp, result.Language);
        Assert.True(result.CompilationAttempted);
        Assert.True(result.CompilationSucceeded);
    }
}
