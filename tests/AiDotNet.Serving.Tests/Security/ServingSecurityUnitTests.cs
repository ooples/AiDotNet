using System.Security.Claims;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Persistence;
using AiDotNet.Serving.Security;
using AiDotNet.Serving.Security.ApiKeys;
using AiDotNet.Serving.Services;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Http;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Xunit;

namespace AiDotNet.Serving.Tests.Security;

/// <summary>
/// Unit coverage for the #2427 wiring that the host-level tests do not reach directly: the JWT tier claim, the
/// scheme selection, option validation and the migration switch.
/// </summary>
public sealed class ServingSecurityUnitTests
{
    [Fact]
    public void TierResolver_ReadsTheConfiguredJwtTierClaim()
    {
        var security = new ServingSecurityOptions();
        security.Jwt.TierClaimType = "plan";
        var resolver = new ClaimsTierResolver(Options.Create(new TierEnforcementOptions()), Options.Create(security));

        Assert.Equal(SubscriptionTier.Pro, resolver.ResolveTier(ContextWith(new Claim("plan", "pro"))));
        // An unknown or numeric value gets the default tier rather than an undefined one.
        Assert.Equal(SubscriptionTier.Free, resolver.ResolveTier(ContextWith(new Claim("plan", "platinum"))));
        Assert.Equal(SubscriptionTier.Free, resolver.ResolveTier(ContextWith(new Claim("plan", "999"))));
        Assert.Equal(SubscriptionTier.Free, resolver.ResolveTier(ContextWith(new Claim("tier", "Enterprise"))));
    }

    [Fact]
    public void SchemeSelection_SendsOnlyNonApiKeyBearerTokensToJwt()
    {
        const string header = "X-AiDotNet-ApiKey";

        Assert.Equal(JwtBearerDefaults.AuthenticationScheme, ServingAuthenticationSchemes.Select(Request("Authorization", "Bearer eyJ.a.b"), header, jwtEnabled: true));
        Assert.Equal(ApiKeyAuthenticationDefaults.Scheme, ServingAuthenticationSchemes.Select(Request("Authorization", "Bearer aidn.id.secret"), header, jwtEnabled: true));
        Assert.Equal(ApiKeyAuthenticationDefaults.Scheme, ServingAuthenticationSchemes.Select(Request(header, "aidn.id.secret"), header, jwtEnabled: true));
        Assert.Equal(ApiKeyAuthenticationDefaults.Scheme, ServingAuthenticationSchemes.Select(Request("Authorization", "Bearer eyJ.a.b"), header, jwtEnabled: false));
    }

    [Fact]
    public void Validation_NamesTheInvalidSetting()
    {
        var hmacWithoutSecrets = new ServingSecurityOptions();
        var (_, entry) = StaticApiKeys.Generate(ServingTier.Free, 1_000, hmacSecret: "secret");
        hmacWithoutSecrets.ApiKeys.StaticApiKeys.Add(entry);
        Assert.Contains("StaticApiKeys:0:HmacIdBase64", Assert.Throws<InvalidOperationException>(hmacWithoutSecrets.Validate).Message, StringComparison.Ordinal);

        var shortKey = new ServingSecurityOptions();
        shortKey.JwtAuthentication.Enabled = true;
        shortKey.JwtAuthentication.Audience = "aud";
        shortKey.JwtAuthentication.Issuer = "iss";
        shortKey.JwtAuthentication.SigningKey = "too-short";
        Assert.Contains("SigningKey", Assert.Throws<InvalidOperationException>(shortKey.Validate).Message, StringComparison.Ordinal);

        var emptyAdmin = new ServingSecurityOptions();
        emptyAdmin.Admin.ClaimValue = " ";
        Assert.Contains("ServingSecurity:Admin", Assert.Throws<InvalidOperationException>(emptyAdmin.Validate).Message, StringComparison.Ordinal);

        new ServingSecurityOptions().Validate();
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task MigrationService_FollowsMigrateOnStartup(bool migrate)
    {
        using var connection = new SqliteConnection("Data Source=:memory:");
        connection.Open();
        var services = new ServiceCollection();
        // As Program configures it: the pending-model-changes check is logged, not thrown.
        services.AddDbContext<ServingDbContext>(options => options.UseSqlite(connection).ConfigureWarnings(w =>
            w.Log(Microsoft.EntityFrameworkCore.Diagnostics.RelationalEventId.PendingModelChangesWarning)));
        using var provider = services.BuildServiceProvider();

        var service = new ServingDatabaseMigrationService(
            provider.GetRequiredService<IServiceScopeFactory>(),
            Options.Create(new ServingPersistenceOptions { MigrateOnStartup = migrate }),
            NullLogger<ServingDatabaseMigrationService>.Instance);
        await service.StartAsync(CancellationToken.None);

        using var scope = provider.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<ServingDbContext>();
        Assert.Equal(migrate, (await db.Database.GetAppliedMigrationsAsync()).Any());
    }

    private static DefaultHttpContext ContextWith(Claim claim) =>
        new() { User = new ClaimsPrincipal(new ClaimsIdentity(new[] { claim }, "Bearer")) };

    private static HttpRequest Request(string header, string value)
    {
        var context = new DefaultHttpContext();
        context.Request.Headers[header] = value;
        return context.Request;
    }
}
