using System.Security.Cryptography;
using System.Text;
using AiDotNet.Serving.Persistence;
using AiDotNet.Serving.Persistence.Entities;
using AiDotNet.Serving.Security;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace AiDotNet.Serving.Security.ApiKeys;

/// <summary>
/// EF Core-backed API key service.
/// </summary>
public sealed class ApiKeyService : IApiKeyService
{
    private const int KeyIdBytes = 8;
    private const int SecretBytes = 32;
    private const int SaltBytes = 16;
    private const int DefaultPbkdf2Iterations = 210_000;
    private const int HashBytes = 32;

    private readonly ServingDbContext _db;
    private readonly ILogger<ApiKeyService> _logger;

    public ApiKeyService(ServingDbContext db, ILogger<ApiKeyService> logger)
    {
        _db = db ?? throw new ArgumentNullException(nameof(db));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public async Task<ApiKeyAuthenticationResult?> AuthenticateAsync(string apiKey, CancellationToken cancellationToken = default)
    {
        if (!ApiKeyFormat.TryParse(apiKey, out var keyId, out var secretText))
        {
            return null;
        }

        if (!Base64Url.TryDecode(secretText, out var secretBytes))
        {
            return null;
        }

        var entity = await _db.ApiKeys
            .AsNoTracking()
            .SingleOrDefaultAsync(k => k.KeyId == keyId, cancellationToken)
            .ConfigureAwait(false);

        if (entity == null)
        {
            return null;
        }

        if (entity.RevokedAt != null)
        {
            return null;
        }

        if (entity.ExpiresAt != null && entity.ExpiresAt <= DateTimeOffset.UtcNow)
        {
            return null;
        }

        var computed = Pbkdf2(secretBytes, entity.Salt, entity.Pbkdf2Iterations, HashBytes);
        bool ok = entity.Hash.Length == computed.Length &&
                  CryptographicOperations.FixedTimeEquals(entity.Hash, computed);

        CryptographicOperations.ZeroMemory(computed);
        CryptographicOperations.ZeroMemory(secretBytes);

        if (!ok)
        {
            return null;
        }

        return new ApiKeyAuthenticationResult(entity.KeyId, entity.Tier, entity.Scopes);
    }

    public async Task<ApiKeyCreateResponse> CreateAsync(ApiKeyCreateRequest request, CancellationToken cancellationToken = default)
    {
        if (request == null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        if (string.IsNullOrWhiteSpace(request.Name))
        {
            throw new ArgumentException("Name is required.", nameof(request));
        }

        if (request.ExpiresAt != null && request.ExpiresAt <= DateTimeOffset.UtcNow)
        {
            throw new ArgumentException("ExpiresAt must be in the future.", nameof(request));
        }

        var keyId = Base64Url.Encode(RandomNumberGenerator.GetBytes(KeyIdBytes));
        var secretBytes = RandomNumberGenerator.GetBytes(SecretBytes);
        var secretText = Base64Url.Encode(secretBytes);
        var apiKey = ApiKeyFormat.Create(keyId, secretText);

        var salt = RandomNumberGenerator.GetBytes(SaltBytes);
        int iterations = DefaultPbkdf2Iterations;
        var hash = Pbkdf2(secretBytes, salt, iterations, HashBytes);

        var scopes = request.Scopes ?? DefaultScopesForTier(request.Tier);

        var entity = new ApiKeyEntity
        {
            Id = Guid.NewGuid(),
            KeyId = keyId,
            Name = request.Name.Trim(),
            Tier = request.Tier,
            Scopes = scopes,
            Salt = salt,
            Hash = hash,
            Pbkdf2Iterations = iterations,
            CreatedAt = DateTimeOffset.UtcNow,
            ExpiresAt = request.ExpiresAt
        };

        _db.ApiKeys.Add(entity);
        await _db.SaveChangesAsync(cancellationToken).ConfigureAwait(false);

        CryptographicOperations.ZeroMemory(secretBytes);

        _logger.LogInformation("Created API key {KeyId} (Tier={Tier}, Scopes={Scopes})", entity.KeyId, entity.Tier, entity.Scopes);

        return new ApiKeyCreateResponse
        {
            KeyId = entity.KeyId,
            ApiKey = apiKey,
            Tier = entity.Tier,
            Scopes = entity.Scopes,
            CreatedAt = entity.CreatedAt,
            ExpiresAt = entity.ExpiresAt
        };
    }

    public async Task<bool> RevokeAsync(string keyId, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(keyId))
        {
            return false;
        }

        var entity = await _db.ApiKeys.SingleOrDefaultAsync(k => k.KeyId == keyId.Trim(), cancellationToken).ConfigureAwait(false);
        if (entity == null)
        {
            return false;
        }

        if (entity.RevokedAt != null)
        {
            return true;
        }

        entity.RevokedAt = DateTimeOffset.UtcNow;
        await _db.SaveChangesAsync(cancellationToken).ConfigureAwait(false);
        _logger.LogInformation("Revoked API key {KeyId}", entity.KeyId);
        return true;
    }

    private static ApiKeyScopes DefaultScopesForTier(SubscriptionTier tier)
    {
        return tier switch
        {
            SubscriptionTier.Free => ApiKeyScopes.Inference | ApiKeyScopes.Federated,
            SubscriptionTier.Pro => ApiKeyScopes.Inference | ApiKeyScopes.Federated | ApiKeyScopes.ArtifactDownload | ApiKeyScopes.ArtifactKeyRelease,
            _ => ApiKeyScopes.All
        };
    }

    private static byte[] Pbkdf2(byte[] secret, byte[] salt, int iterations, int bytes)
    {
        return Rfc2898DeriveBytes.Pbkdf2(
            password: secret,
            salt: salt,
            iterations: iterations,
            hashAlgorithm: HashAlgorithmName.SHA256,
            outputLength: bytes);
    }
}

