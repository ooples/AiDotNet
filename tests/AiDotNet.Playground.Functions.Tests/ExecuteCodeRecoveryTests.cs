using System.Text;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace AiDotNet.Playground.Functions.Tests;

public sealed class ExecuteCodeRecoveryTests
{
    [Fact(Timeout = 60000)]
    public async Task ExistingExecutionRouteStillRunsSimpleCode()
    {
        var response = await Execute("System.Console.WriteLine(6 * 7);");

        Assert.True(response.Success, response.Error);
        Assert.NotNull(response.Output);
        Assert.Equal("42", response.Output.Trim());
    }

    [Theory(Timeout = 60000)]
    [InlineData("class Program { static void Main() { Console.WriteLine(\"ok\"); } }", "ok")]
    [InlineData("using System;\nnamespace Demo { public static class App { public static void Main(string[] args) { Console.WriteLine(args.Length); } } }", "0")]
    [InlineData("class Program { static async Task Main() { await Task.Delay(1); Console.WriteLine(\"async\"); } }", "async")]
    [InlineData("class Program { static int Main() { Console.WriteLine(new List<int> { 1, 2 }.Sum()); return 0; } }", "3")]
    public async Task CompleteProgramRunsFromItsOwnEntryPoint(string code, string expected)
    {
        var response = await Execute(code);

        Assert.True(response.Success, response.Error + string.Join("; ", response.CompilationErrors ?? new List<string>()));
        Assert.NotNull(response.Output);
        Assert.Equal(expected, response.Output.Trim());
    }

    [Fact(Timeout = 60000)]
    public async Task SnippetThatDeclaresALocalFunctionNamedMainIsStillASnippet()
    {
        var response = await Execute("static int Main() => 7;\nSystem.Console.WriteLine(Main());");

        Assert.True(response.Success, response.Error + string.Join("; ", response.CompilationErrors ?? new List<string>()));
        Assert.NotNull(response.Output);
        Assert.Equal("7", response.Output.Trim());
    }

    private static async Task<CodeExecutionResponse> Execute(string code)
    {
        var context = new DefaultHttpContext();
        context.Request.Method = "POST";
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(System.Text.Json.JsonSerializer.Serialize(new { code })));

        var result = await new ExecuteCode(NullLogger<ExecuteCode>.Instance).Run(context.Request);

        return Assert.IsType<CodeExecutionResponse>(Assert.IsType<OkObjectResult>(result).Value);
    }
}
