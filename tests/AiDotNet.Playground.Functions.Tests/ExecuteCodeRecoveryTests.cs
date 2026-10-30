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
        var context = new DefaultHttpContext();
        context.Request.Method = "POST";
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes("{\"code\":\"System.Console.WriteLine(6 * 7);\"}"));

        var result = await new ExecuteCode(NullLogger<ExecuteCode>.Instance).Run(context.Request);

        var response = Assert.IsType<CodeExecutionResponse>(Assert.IsType<OkObjectResult>(result).Value);
        Assert.True(response.Success, response.Error);
        Assert.NotNull(response.Output);
        Assert.Equal("42", response.Output.Trim());
    }
}
