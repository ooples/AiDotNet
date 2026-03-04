using System.Security.Cryptography;
using AiDotNet.Serving.Persistence;
using AiDotNet.Serving.Persistence.Entities;
using AiDotNet.Serving.Security.ApiKeys;
using AiDotNet.Validation;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace AiDotNet.Serving.Security.Licensing;

/// <summary>
/// EF Core-backed license key service for creating, validating, and managing license keys.
/// </summary>
public sealed class LicenseService : ILicenseService
{
    private const int KeyIdBytes = 8;
    private const int SecretBytes = 32;
    private const int SaltBytes = 16;
    private const int DefaultPbkdf2Iterations = 210_000;
    private const int HashBytes = 32;

    private readonly ServingDbContext _db;
    private readonly ILogger<LicenseService> _logger;

    public LicenseService(ServingDbContext db, ILogger<LicenseService> logger)
    {
        Guard.NotNull(db);
        Guard.NotNull(logger);
        _db = db;
        _logger = logger;
    }

    public async Task<LicenseCreateResponse> CreateAsync(LicenseCreateRequest request, CancellationToken cancellationToken = default)
    {
        if (request is null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        if (string.IsNullOrWhiteSpace(request.CustomerName))
        {
            throw new ArgumentException("CustomerName is required.", nameof(request));
        }

        if (request.ExpiresAt is not null && request.ExpiresAt <= DateTimeOffset.UtcNow)
        {
            throw new ArgumentException("ExpiresAt must be in the future.", nameof(request));
        }

        // Generate key in the same format as API keys: aidn.{keyId}.{secret}
        var keyId = Base64Url.Encode(RandomNumberGenerator.GetBytes(KeyIdBytes));
        var secretBytes = RandomNumberGenerator.GetBytes(SecretBytes);
        var secretText = Base64Url.Encode(secretBytes);
        var licenseKey = ApiKeyFormat.Create(keyId, secretText);

        // Hash the secret for storage
        var salt = RandomNumberGenerator.GetBytes(SaltBytes);
        int iterations = DefaultPbkdf2Iterations;
        var hash = Pbkdf2(secretBytes, salt, iterations, HashBytes);

        // Generate escrow secret for Layer 2 key escrow
        var escrowSecret = RandomNumberGenerator.GetBytes(32);

        var entity = new LicenseKeyEntity
        {
            Id = Guid.NewGuid(),
            KeyId = keyId,
            Salt = salt,
            Hash = hash,
            Pbkdf2Iterations = iterations,
            CustomerName = request.CustomerName.Trim(),
            CustomerEmail = request.CustomerEmail?.Trim(),
            Tier = request.Tier,
            MaxSeats = request.MaxSeats,
            CreatedAt = DateTimeOffset.UtcNow,
            ExpiresAt = request.ExpiresAt,
            Environment = request.Environment?.Trim(),
            Notes = request.Notes?.Trim(),
            EscrowSecret = escrowSecret
        };

        _db.LicenseKeys.Add(entity);
        await _db.SaveChangesAsync(cancellationToken).ConfigureAwait(false);

        CryptographicOperations.ZeroMemory(secretBytes);

        _logger.LogInformation(
            "Created license key {KeyId} for {CustomerName} (Tier={Tier}, MaxSeats={MaxSeats})",
            entity.KeyId, entity.CustomerName, entity.Tier, entity.MaxSeats);

        return new LicenseCreateResponse
        {
            Id = entity.Id,
            LicenseKey = licenseKey,
            Tier = entity.Tier,
            MaxSeats = entity.MaxSeats,
            ExpiresAt = entity.ExpiresAt,
            CreatedAt = entity.CreatedAt
        };
    }

    public async Task<LicenseValidationResponse> ValidateAsync(LicenseValidateRequest request, CancellationToken cancellationToken = default)
    {
        if (request is null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        if (!ApiKeyFormat.TryParse(request.Key, out var keyId, out var secretText))
        {
            return InvalidResponse("Invalid license key format.");
        }

        if (!Base64Url.TryDecode(secretText, out var secretBytes))
        {
            return InvalidResponse("Invalid license key encoding.");
        }

        var entity = await _db.LicenseKeys
            .SingleOrDefaultAsync(k => k.KeyId == keyId, cancellationToken)
            .ConfigureAwait(false);

        if (entity is null)
        {
            CryptographicOperations.ZeroMemory(secretBytes);
            return InvalidResponse("License key not found.");
        }

        // Verify secret with constant-time comparison
        var computed = Pbkdf2(secretBytes, entity.Salt, entity.Pbkdf2Iterations, HashBytes);
        bool secretOk = entity.Hash.Length == computed.Length &&
                        CryptographicOperations.FixedTimeEquals(entity.Hash, computed);

        CryptographicOperations.ZeroMemory(computed);

        // Compute decryption token from escrow secret BEFORE zeroing secretBytes (Layer 2)
        string? decryptionToken = null;
        if (secretOk && entity.EscrowSecret is not null && entity.EscrowSecret.Length > 0)
        {
            using var hmac = new HMACSHA256(entity.EscrowSecret);
            var tokenBytes = hmac.ComputeHash(secretBytes);
            decryptionToken = Convert.ToBase64String(tokenBytes);
        }

        CryptographicOperations.ZeroMemory(secretBytes);

        if (!secretOk)
        {
            return InvalidResponse("Invalid license key.");
        }

        // Check revocation
        if (entity.RevokedAt is not null)
        {
            return new LicenseValidationResponse
            {
                Status = "Revoked",
                Tier = entity.Tier.ToString(),
                Message = "This license key has been revoked."
            };
        }

        // Check expiration
        if (entity.ExpiresAt is not null && entity.ExpiresAt <= DateTimeOffset.UtcNow)
        {
            return new LicenseValidationResponse
            {
                Status = "Expired",
                Tier = entity.Tier.ToString(),
                ExpiresAt = entity.ExpiresAt,
                Message = "This license key has expired."
            };
        }

        // Count distinct active machine activations
        int seatsUsed = await _db.LicenseActivations
            .CountAsync(a => a.LicenseKeyId == entity.Id && a.IsActive, cancellationToken)
            .ConfigureAwait(false);

        // Upsert activation record (advisory — does not block)
        if (!string.IsNullOrWhiteSpace(request.MachineId))
        {
            await UpsertActivationAsync(entity.Id, request.MachineId, request.MachineName, request.Environment, cancellationToken)
                .ConfigureAwait(false);

            // Re-count after upsert to include this machine
            seatsUsed = await _db.LicenseActivations
                .CountAsync(a => a.LicenseKeyId == entity.Id && a.IsActive, cancellationToken)
                .ConfigureAwait(false);
        }

        // Check seat limit
        if (seatsUsed > entity.MaxSeats)
        {
            return new LicenseValidationResponse
            {
                Status = "SeatLimitReached",
                Tier = entity.Tier.ToString(),
                ExpiresAt = entity.ExpiresAt,
                SeatsUsed = seatsUsed,
                SeatsMax = entity.MaxSeats,
                Message = $"Seat limit reached ({seatsUsed}/{entity.MaxSeats}). Contact your administrator."
            };
        }

        return new LicenseValidationResponse
        {
            Status = "Active",
            Tier = entity.Tier.ToString(),
            ExpiresAt = entity.ExpiresAt,
            SeatsUsed = seatsUsed,
            SeatsMax = entity.MaxSeats,
            Message = "License is valid.",
            DecryptionToken = decryptionToken
        };
    }

    public async Task<bool> RevokeAsync(Guid licenseId, CancellationToken cancellationToken = default)
    {
        var entity = await _db.LicenseKeys
            .SingleOrDefaultAsync(k => k.Id == licenseId, cancellationToken)
            .ConfigureAwait(false);

        if (entity is null)
        {
            return false;
        }

        if (entity.RevokedAt is not null)
        {
            return true;
        }

        entity.RevokedAt = DateTimeOffset.UtcNow;
        await _db.SaveChangesAsync(cancellationToken).ConfigureAwait(false);
        _logger.LogInformation("Revoked license key {KeyId}", entity.KeyId);
        return true;
    }

    public async Task<List<LicenseInfo>> ListAsync(CancellationToken cancellationToken = default)
    {
        // Single query with group join to avoid N+1 activation count queries
        var query = from k in _db.LicenseKeys.AsNoTracking()
                    join a in _db.LicenseActivations.AsNoTracking().Where(a => a.IsActive)
                        on k.Id equals a.LicenseKeyId into activations
                    orderby k.CreatedAt descending
                    select new { Key = k, SeatsUsed = activations.Count() };

        var rows = await query.ToListAsync(cancellationToken).ConfigureAwait(false);

        var result = new List<LicenseInfo>(rows.Count);
        foreach (var row in rows)
        {
            result.Add(MapToInfo(row.Key, row.SeatsUsed));
        }

        return result;
    }

    public async Task<LicenseInfo?> GetAsync(Guid licenseId, CancellationToken cancellationToken = default)
    {
        var entity = await _db.LicenseKeys
            .AsNoTracking()
            .SingleOrDefaultAsync(k => k.Id == licenseId, cancellationToken)
            .ConfigureAwait(false);

        if (entity is null)
        {
            return null;
        }

        var activations = await _db.LicenseActivations
            .AsNoTracking()
            .Where(a => a.LicenseKeyId == entity.Id)
            .OrderByDescending(a => a.LastSeenAt)
            .ToListAsync(cancellationToken)
            .ConfigureAwait(false);

        int seatsUsed = activations.Count(a => a.IsActive);
        var info = MapToInfo(entity, seatsUsed);
        info.Activations = activations.Select(a => new LicenseActivationInfo
        {
            MachineId = a.MachineId,
            MachineName = a.MachineName,
            Environment = a.Environment,
            FirstSeenAt = a.FirstSeenAt,
            LastSeenAt = a.LastSeenAt,
            IsActive = a.IsActive
        }).ToList();

        return info;
    }

    private async Task UpsertActivationAsync(
        Guid licenseKeyId,
        string machineId,
        string? machineName,
        string? environment,
        CancellationToken cancellationToken)
    {
        var existing = await _db.LicenseActivations
            .SingleOrDefaultAsync(
                a => a.LicenseKeyId == licenseKeyId && a.MachineId == machineId,
                cancellationToken)
            .ConfigureAwait(false);

        if (existing is not null)
        {
            existing.LastSeenAt = DateTimeOffset.UtcNow;
            existing.IsActive = true;
            if (machineName is not null)
            {
                existing.MachineName = machineName;
            }

            if (environment is not null)
            {
                existing.Environment = environment;
            }
        }
        else
        {
            _db.LicenseActivations.Add(new LicenseActivationEntity
            {
                Id = Guid.NewGuid(),
                LicenseKeyId = licenseKeyId,
                MachineId = machineId,
                MachineName = machineName,
                Environment = environment,
                FirstSeenAt = DateTimeOffset.UtcNow,
                LastSeenAt = DateTimeOffset.UtcNow,
                IsActive = true
            });
        }

        await _db.SaveChangesAsync(cancellationToken).ConfigureAwait(false);
    }

    private static LicenseInfo MapToInfo(LicenseKeyEntity entity, int seatsUsed)
    {
        return new LicenseInfo
        {
            Id = entity.Id,
            KeyId = entity.KeyId,
            CustomerName = entity.CustomerName,
            CustomerEmail = entity.CustomerEmail,
            Tier = entity.Tier,
            MaxSeats = entity.MaxSeats,
            SeatsUsed = seatsUsed,
            CreatedAt = entity.CreatedAt,
            ExpiresAt = entity.ExpiresAt,
            RevokedAt = entity.RevokedAt,
            Environment = entity.Environment,
            Notes = entity.Notes
        };
    }

    private static LicenseValidationResponse InvalidResponse(string message)
    {
        return new LicenseValidationResponse
        {
            Status = "Invalid",
            Message = message
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
