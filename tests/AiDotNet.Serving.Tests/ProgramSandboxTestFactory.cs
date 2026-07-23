using AiDotNet.Serving.Sandboxing.Execution;
using AiDotNet.Serving.Security.ApiKeys;
using AiDotNet.Serving.Tests.Fakes;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
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

        // Register a test auth handler under its own scheme name and
        // make it the default — needed after PR #1384 made
        // authentication the default for previously-public controllers.
        builder.ConfigureTestServices(services =>
        {
            services.AddAuthentication(TestAuthenticationHandler.Scheme)
                .AddScheme<ApiKeyAuthenticationOptions, TestAuthenticationHandler>(
                    TestAuthenticationHandler.Scheme,
                    _ => { });
            services.Configure<AuthenticationOptions>(options =>
            {
                options.DefaultScheme = TestAuthenticationHandler.Scheme;
                options.DefaultAuthenticateScheme = TestAuthenticationHandler.Scheme;
                options.DefaultChallengeScheme = TestAuthenticationHandler.Scheme;
            });
        });
    }
}

