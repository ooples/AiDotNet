using AiDotNet.Serving.Security.ApiKeys;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// WebApplicationFactory derivative that replaces the production
/// <see cref="ApiKeyAuthenticationHandler"/> with
/// <see cref="TestAuthenticationHandler"/> so integration tests can
/// drive the serving host without provisioning a real API key.
/// </summary>
/// <remarks>
/// PR #1384 added a <c>FallbackPolicy = RequireAuthenticatedUser</c> to
/// the serving host so previously-unprotected controllers (Inference,
/// Embeddings, Federated, Models, ProgramSynthesis/*) now require
/// authentication. Tests bound to this factory get a stubbed auth
/// handler that succeeds for every request, so they exercise the
/// production endpoint code without needing the real API-key
/// provisioning flow. Tests that specifically want to verify the
/// unauthenticated rejection path should construct
/// <see cref="WebApplicationFactory{Program}"/> directly instead of
/// using this factory.
/// </remarks>
public sealed class ServingTestWebApplicationFactory : WebApplicationFactory<Program>
{
    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.ConfigureTestServices(services =>
        {
            // Register the test handler under its own scheme name, then
            // override the AuthenticationOptions to make TestAuth the
            // default for every challenge. The production scheme stays
            // registered (so a re-set-default can't accidentally call
            // AddScheme twice on the same name), it just becomes a
            // dormant alternative the test code doesn't drive.
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
