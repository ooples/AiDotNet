using System.Net;
using System.Security.Claims;
using System.Text;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Persistence;
using AiDotNet.Serving.Security;
using AiDotNet.Serving.Security.ApiKeys;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.IdentityModel.JsonWebTokens;
using Microsoft.IdentityModel.Tokens;
using Xunit;

namespace AiDotNet.Serving.Tests.Security;

/// <summary>
/// The ServingSecurity configuration drives the real authentication and authorization pipeline (#2427): static
/// API keys, the API key header, JWT bearer callers, the JWT tier claim and the admin claim. Each test runs the host
/// with the production schemes, not the test handler.
/// </summary>
[Collection("ServingIntegrationTests")]
public sealed class ServingSecurityConfigurationTests
{
    private const string ProtectedEndpoint = "/api/models";
    private const string AdminEndpoint = "/api/admin/api-keys/no-such-key/revoke";
    private const int FastIterations = 1_000;
    private const string Issuer = "https://issuer.test";
    private const string Audience = "aidotnet-serving";
    private static readonly string SigningKey = Convert.ToBase64String(Encoding.UTF8.GetBytes("serving-tests-signing-key-0123456789-abcdef"));

    [Fact(Timeout = 120000)]
    public async Task StaticKey_IsAccepted_AndAWrongSecretIsRefused()
    {
        var (apiKey, entry) = StaticApiKeys.Generate(ServingTier.Premium, FastIterations);
        using var factory = CreateFactory(StaticKeySettings(entry));
        using var client = factory.CreateClient();

        Assert.Equal(HttpStatusCode.OK, await GetAsync(client, ProtectedEndpoint, ServingApiKeyOptions.DefaultHeaderName, apiKey));

        string tampered = apiKey.Substring(0, apiKey.Length - 2) + (apiKey.EndsWith("AA", StringComparison.Ordinal) ? "BB" : "AA");
        Assert.Equal(HttpStatusCode.Unauthorized, await GetAsync(client, ProtectedEndpoint, ServingApiKeyOptions.DefaultHeaderName, tampered));
        Assert.Equal(HttpStatusCode.Unauthorized, await GetAsync(client, ProtectedEndpoint, null, null));
    }

    [Fact(Timeout = 120000)]
    public async Task RevokedOrExpiredStaticKey_IsRefused()
    {
        var (revokedKey, revoked) = StaticApiKeys.Generate(ServingTier.Free, FastIterations);
        revoked.Revoked = true;
        var (expiredKey, expired) = StaticApiKeys.Generate(ServingTier.Free, FastIterations, expiresAt: DateTimeOffset.UtcNow.AddMinutes(-1));
        var settings = StaticKeySettings(revoked, expired);
        using var factory = CreateFactory(settings);
        using var client = factory.CreateClient();

        Assert.Equal(HttpStatusCode.Unauthorized, await GetAsync(client, ProtectedEndpoint, ServingApiKeyOptions.DefaultHeaderName, revokedKey));
        Assert.Equal(HttpStatusCode.Unauthorized, await GetAsync(client, ProtectedEndpoint, ServingApiKeyOptions.DefaultHeaderName, expiredKey));
    }

    [Fact(Timeout = 120000)]
    public async Task StaticKey_WithHmacSecrets_NeedsAMatchingHmacId_UnderAnyListedSecret()
    {
        var (currentKey, current) = StaticApiKeys.Generate(ServingTier.Free, FastIterations, hmacSecret: "old-secret");
        var (forgedKey, forged) = StaticApiKeys.Generate(ServingTier.Free, FastIterations, hmacSecret: "not-a-listed-secret");
        var settings = StaticKeySettings(current, forged);
        settings["ServingSecurity:ApiKeys:HmacSecrets:0"] = "new-secret";
        settings["ServingSecurity:ApiKeys:HmacSecrets:1"] = "old-secret";
        using var factory = CreateFactory(settings);
        using var client = factory.CreateClient();

        Assert.Equal(HttpStatusCode.OK, await GetAsync(client, ProtectedEndpoint, ServingApiKeyOptions.DefaultHeaderName, currentKey));
        Assert.Equal(HttpStatusCode.Unauthorized, await GetAsync(client, ProtectedEndpoint, ServingApiKeyOptions.DefaultHeaderName, forgedKey));
    }

    [Fact(Timeout = 120000)]
    public async Task ConfiguredHeaderName_CarriesTheKey()
    {
        var (apiKey, entry) = StaticApiKeys.Generate(ServingTier.Free, FastIterations);
        var settings = StaticKeySettings(entry);
        settings["ServingSecurity:ApiKeys:HeaderName"] = "X-Custom-Key";
        using var factory = CreateFactory(settings);
        using var client = factory.CreateClient();

        Assert.Equal(HttpStatusCode.OK, await GetAsync(client, ProtectedEndpoint, "X-Custom-Key", apiKey));
        Assert.Equal(HttpStatusCode.Unauthorized, await GetAsync(client, ProtectedEndpoint, ServingApiKeyOptions.DefaultHeaderName, apiKey));
    }

    [Fact(Timeout = 120000)]
    public async Task LegacyHeaderName_StillOverridesTheNewSetting()
    {
        var (apiKey, entry) = StaticApiKeys.Generate(ServingTier.Free, FastIterations);
        var settings = StaticKeySettings(entry);
        settings["ServingSecurity:ApiKeys:HeaderName"] = "X-Custom-Key";
        settings["ApiKeyAuthenticationOptions:HeaderName"] = "X-Legacy-Key";
        using var factory = CreateFactory(settings);
        using var client = factory.CreateClient();

        Assert.Equal(HttpStatusCode.OK, await GetAsync(client, ProtectedEndpoint, "X-Legacy-Key", apiKey));
    }

    [Fact(Timeout = 120000)]
    public async Task CreatedKeys_AreHashedWithTheConfiguredIterations()
    {
        using var factory = CreateFactory(new Dictionary<string, string?> { ["ServingSecurity:ApiKeys:Pbkdf2Iterations"] = "12345" });
        using var scope = factory.Services.CreateScope();
        var apiKeys = scope.ServiceProvider.GetRequiredService<IApiKeyService>();

        var created = await apiKeys.CreateAsync(new ApiKeyCreateRequest { Name = $"iterations-{Guid.NewGuid():N}", Tier = SubscriptionTier.Free });

        var db = scope.ServiceProvider.GetRequiredService<ServingDbContext>();
        var entity = await db.ApiKeys.AsNoTracking().SingleAsync(k => k.KeyId == created.KeyId);
        Assert.Equal(12345, entity.Pbkdf2Iterations);
        Assert.NotNull(await apiKeys.AuthenticateAsync(created.ApiKey));
    }

    [Fact(Timeout = 120000)]
    public async Task Jwt_IsAccepted_AndAWrongIssuerOrAudienceIsRefused()
    {
        using var factory = CreateFactory(JwtSettings());
        using var client = factory.CreateClient();

        Assert.Equal(HttpStatusCode.OK, await GetAsync(client, ProtectedEndpoint, "Authorization", "Bearer " + CreateToken(Issuer, Audience)));
        Assert.Equal(HttpStatusCode.Unauthorized, await GetAsync(client, ProtectedEndpoint, "Authorization", "Bearer " + CreateToken("https://other.test", Audience)));
        Assert.Equal(HttpStatusCode.Unauthorized, await GetAsync(client, ProtectedEndpoint, "Authorization", "Bearer " + CreateToken(Issuer, "other-audience")));
    }

    [Fact(Timeout = 120000)]
    public async Task ApiKeys_StillAuthenticate_WhenJwtIsEnabled()
    {
        var (apiKey, entry) = StaticApiKeys.Generate(ServingTier.Free, FastIterations);
        var settings = JwtSettings();
        foreach (var setting in StaticKeySettings(entry))
        {
            settings[setting.Key] = setting.Value;
        }

        using var factory = CreateFactory(settings);
        using var client = factory.CreateClient();

        Assert.Equal(HttpStatusCode.OK, await GetAsync(client, ProtectedEndpoint, ServingApiKeyOptions.DefaultHeaderName, apiKey));
        Assert.Equal(HttpStatusCode.OK, await GetAsync(client, ProtectedEndpoint, "Authorization", "Bearer " + apiKey));
    }

    [Fact(Timeout = 120000)]
    public async Task AdminEndpoints_NeedTheConfiguredAdminClaim()
    {
        var settings = JwtSettings();
        settings["ServingSecurity:Admin:ClaimType"] = "groups";
        settings["ServingSecurity:Admin:ClaimValue"] = "serving-admins";
        using var factory = CreateFactory(settings);
        using var client = factory.CreateClient();

        string user = CreateToken(Issuer, Audience, new Claim("groups", "users"));
        string admin = CreateToken(Issuer, Audience, new Claim("groups", "serving-admins"));

        Assert.Equal(HttpStatusCode.Forbidden, await PostAsync(client, AdminEndpoint, user));
        // An admin passes authorization; the key does not exist, so the controller answers 404.
        Assert.Equal(HttpStatusCode.NotFound, await PostAsync(client, AdminEndpoint, admin));
    }

    [Fact]
    public void InvalidSecurityConfiguration_StopsTheHost()
    {
        using var factory = CreateFactory(new Dictionary<string, string?> { ["ServingSecurity:JwtAuthentication:Enabled"] = "true" });

        var error = Assert.Throws<InvalidOperationException>(() => factory.CreateClient());
        Assert.Contains("ServingSecurity:JwtAuthentication", error.Message, StringComparison.Ordinal);
    }

    private static WebApplicationFactory<Program> CreateFactory(IReadOnlyDictionary<string, string?> settings) =>
        new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
        {
            foreach (var setting in settings)
            {
                builder.UseSetting(setting.Key, setting.Value);
            }
        });

    private static Dictionary<string, string?> StaticKeySettings(params StaticApiKeyOptions[] entries)
    {
        var settings = new Dictionary<string, string?>();
        for (int i = 0; i < entries.Length; i++)
        {
            string path = $"ServingSecurity:ApiKeys:StaticApiKeys:{i}:";
            settings[path + nameof(StaticApiKeyOptions.Prefix)] = entries[i].Prefix;
            settings[path + nameof(StaticApiKeyOptions.Pbkdf2SaltBase64)] = entries[i].Pbkdf2SaltBase64;
            settings[path + nameof(StaticApiKeyOptions.Pbkdf2HashBase64)] = entries[i].Pbkdf2HashBase64;
            settings[path + nameof(StaticApiKeyOptions.Pbkdf2Iterations)] = entries[i].Pbkdf2Iterations.ToString(System.Globalization.CultureInfo.InvariantCulture);
            settings[path + nameof(StaticApiKeyOptions.Tier)] = entries[i].Tier.ToString();
            settings[path + nameof(StaticApiKeyOptions.Revoked)] = entries[i].Revoked ? "true" : "false";
            if (entries[i].ExpiresAt is DateTimeOffset expiresAt)
            {
                settings[path + nameof(StaticApiKeyOptions.ExpiresAt)] = expiresAt.ToString("O", System.Globalization.CultureInfo.InvariantCulture);
            }

            if (!string.IsNullOrEmpty(entries[i].HmacIdBase64))
            {
                settings[path + nameof(StaticApiKeyOptions.HmacIdBase64)] = entries[i].HmacIdBase64;
            }
        }

        return settings;
    }

    private static Dictionary<string, string?> JwtSettings() => new()
    {
        ["ServingSecurity:JwtAuthentication:Enabled"] = "true",
        ["ServingSecurity:JwtAuthentication:Issuer"] = Issuer,
        ["ServingSecurity:JwtAuthentication:Audience"] = Audience,
        ["ServingSecurity:JwtAuthentication:SigningKey"] = SigningKey
    };

    private static string CreateToken(string issuer, string audience, params Claim[] claims)
    {
        var allClaims = new List<Claim> { new("sub", "serving-test-user") };
        allClaims.AddRange(claims);
        return new JsonWebTokenHandler().CreateToken(new SecurityTokenDescriptor
        {
            Issuer = issuer,
            Audience = audience,
            Subject = new ClaimsIdentity(allClaims),
            Expires = DateTime.UtcNow.AddMinutes(5),
            SigningCredentials = new SigningCredentials(
                new SymmetricSecurityKey(Convert.FromBase64String(SigningKey)), SecurityAlgorithms.HmacSha256)
        });
    }

    private static async Task<HttpStatusCode> GetAsync(HttpClient client, string path, string? header, string? value)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, path);
        if (header is not null && value is not null)
        {
            request.Headers.TryAddWithoutValidation(header, value);
        }

        using var response = await client.SendAsync(request);
        return response.StatusCode;
    }

    private static async Task<HttpStatusCode> PostAsync(HttpClient client, string path, string token)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, path);
        request.Headers.TryAddWithoutValidation("Authorization", "Bearer " + token);
        using var response = await client.SendAsync(request);
        return response.StatusCode;
    }
}
