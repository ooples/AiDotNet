using AiDotNet.Serving.Sandboxing.Execution;
using AiDotNet.Serving.Tests.Fakes;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace AiDotNet.Serving.Tests;

public sealed class ProgramSandboxTestFactory : WebApplicationFactory<Program>
{
    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.ConfigureServices(services =>
        {
            services.RemoveAll<IProgramSandboxExecutor>();
            services.AddSingleton<IProgramSandboxExecutor, FakeProgramSandboxExecutor>();
        });
    }
}

