using System.Security.Cryptography;
using AiDotNet.Enums;
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
    private readonly TimeProvider _clock;

    /// <summary>
    /// How long an activation holds a seat after its machine was last seen. A seat is a LEASE, renewed by every
    /// validation from that machine, not a permanent claim: an activation that has not validated within the lease
    /// stops counting, so machines that are gone (a rebuilt laptop, every ephemeral CI runner, each of which has a
    /// fresh machine id) free their seat without an administrator. A lapsed machine that comes back re-activates if a
    /// seat is free. This is how heartbeat/lease licensing (Keygen machine heartbeats, FlexLM and JetBrains floating
    /// leases) bounds seat use; counting every activation ever seen exhausts a CI key in days.
    /// </summary>
    public static readonly TimeSpan ActivationLease = TimeSpan.FromDays(7);

    public LicenseService(ServingDbContext db, ILogger<LicenseService> logger)
        : this(db, logger, TimeProvider.System)
    {
    }

    /// <summary>Creates the service on an explicit clock, so lease expiry is testable.</summary>
    public LicenseService(ServingDbContext db, ILogger<LicenseService> logger, TimeProvider clock)
    {
        _clock = clock ?? throw new ArgumentNullException(nameof(clock));
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

        if (request.ExpiresAt is not null && request.ExpiresAt <= _clock.GetUtcNow())
        {
            throw new ArgumentException("ExpiresAt must be in the future.", nameof(request));
        }

        // Generate key in the same format as API keys: aidn.{keyId}.{secret}
        var keyId = Base64Url.Encode(RandomNumberGenerator.GetBytes(KeyIdBytes));
        var secretBytes = RandomNumberGenerator.GetBytes(SecretBytes);
        try
        {
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
                CreatedAt = _clock.GetUtcNow(),
                ExpiresAt = request.ExpiresAt,
                Environment = request.Environment?.Trim(),
                Notes = request.Notes?.Trim(),
                EscrowSecret = escrowSecret
            };

            _db.LicenseKeys.Add(entity);
            await _db.SaveChangesAsync(cancellationToken).ConfigureAwait(false);

            _logger.LogInformation(
                "Created license key {KeyId} for {CustomerName} (Tier={Tier}, MaxSeats={MaxSeats})",
                entity.KeyId, LogSanitizer.Sanitize(entity.CustomerName), entity.Tier, entity.MaxSeats);

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
        finally
        {
            CryptographicOperations.ZeroMemory(secretBytes);
        }
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

        // Revocation and expiry: checked here to answer without a lock, and again by AdmitSeatAsync under the lock.
        if (InactiveStatus(entity) is { } inactive)
        {
            return InactiveResponse(entity, inactive);
        }

        var (admitted, seatsUsed, inactiveUnderLock) =
            await AdmitSeatAsync(entity, request, cancellationToken).ConfigureAwait(false);
        if (inactiveUnderLock is { } lockedStatus)
        {
            return InactiveResponse(entity, lockedStatus);
        }

        if (!admitted)
        {
            return new LicenseValidationResponse
            {
                Status = LicenseKeyStatus.SeatLimitReached,
                Tier = entity.Tier.ToString(),
                ExpiresAt = entity.ExpiresAt,
                SeatsUsed = seatsUsed,
                SeatsMax = entity.MaxSeats,
                Message = $"Seat limit reached ({seatsUsed}/{entity.MaxSeats}). Contact your administrator."
            };
        }

        return new LicenseValidationResponse
        {
            Status = LicenseKeyStatus.Active,
            Tier = entity.Tier.ToString(),
            ExpiresAt = entity.ExpiresAt,
            SeatsUsed = seatsUsed,
            SeatsMax = entity.MaxSeats,
            Message = "License is valid.",
            DecryptionToken = decryptionToken
        };
    }

    /// <summary>Revoked or Expired when the key can no longer be admitted, otherwise null.</summary>
    private LicenseKeyStatus? InactiveStatus(LicenseKeyEntity entity)
    {
        if (entity.RevokedAt is not null)
        {
            return LicenseKeyStatus.Revoked;
        }

        if (entity.ExpiresAt is not null && entity.ExpiresAt <= _clock.GetUtcNow())
        {
            return LicenseKeyStatus.Expired;
        }

        return null;
    }

    private static LicenseValidationResponse InactiveResponse(LicenseKeyEntity entity, LicenseKeyStatus status) =>
        status == LicenseKeyStatus.Revoked
            ? new LicenseValidationResponse
            {
                Status = LicenseKeyStatus.Revoked,
                Tier = entity.Tier.ToString(),
                Message = "This license key has been revoked."
            }
            : new LicenseValidationResponse
            {
                Status = LicenseKeyStatus.Expired,
                Tier = entity.Tier.ToString(),
                ExpiresAt = entity.ExpiresAt,
                Message = "This license key has expired."
            };

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

        entity.RevokedAt = _clock.GetUtcNow();
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

    /// <summary>
    /// Decides whether this request holds a seat and, only if it does, renews its machine's lease.
    /// </summary>
    /// <remarks>
    /// Serialized per license key. The count, the decision and the renewal run in one transaction that first takes
    /// a write lock on the key's own row, so a second admission for the same key waits until the first has
    /// committed. Without it, two lapsed machines on a one-seat key could both read a free seat and both persist a
    /// lease. The seat limit is checked BEFORE the renewal is written, so a rejected request never extends its
    /// lease: a machine over the limit (for instance after an administrator lowered MaxSeats) lapses instead of
    /// holding its seat by validating repeatedly.
    /// </remarks>
    /// <returns>Whether a seat is held, the leased seats counted for the response, and Revoked or Expired when the
    /// key, re-read under the lock, can no longer be admitted (a revocation or expiry while this request waited).</returns>
    private async Task<(bool Admitted, int SeatsUsed, LicenseKeyStatus? Inactive)> AdmitSeatAsync(
        LicenseKeyEntity entity,
        LicenseValidateRequest request,
        CancellationToken cancellationToken)
    {
        var strategy = _db.Database.CreateExecutionStrategy();
        return await strategy.ExecuteAsync(async () =>
        {
            // A retrying strategy re-runs this whole block; an activation tracked by a failed attempt must not be
            // saved twice.
            foreach (var tracked in _db.ChangeTracker.Entries<LicenseActivationEntity>().ToList())
            {
                tracked.State = EntityState.Detached;
            }

            await using var transaction = await _db.Database.BeginTransactionAsync(cancellationToken).ConfigureAwait(false);
            await LockLicenseKeyAsync(entity.Id, cancellationToken).ConfigureAwait(false);

            // Decide on the key as it is now, not as read before the lock: a revocation, expiry or MaxSeats change
            // committed while this request waited applies to it.
            await _db.Entry(entity).ReloadAsync(cancellationToken).ConfigureAwait(false);
            if (InactiveStatus(entity) is { } inactive)
            {
                await transaction.RollbackAsync(cancellationToken).ConfigureAwait(false);
                return (false, 0, inactive);
            }

            // Seats held: active activations whose lease is current (see ActivationLease). Read after the lock, so
            // a waiting request counts the lease the request ahead of it just committed.
            int seatsUsed = await CountLeasedSeatsAsync(entity.Id, cancellationToken).ConfigureAwait(false);

            if (string.IsNullOrWhiteSpace(request.MachineId))
            {
                // Nothing to renew: the request only reports whether the key is within its limit.
                await transaction.RollbackAsync(cancellationToken).ConfigureAwait(false);
                return (seatsUsed <= entity.MaxSeats, seatsUsed, null);
            }

            // A machine whose lease is current does not take a new seat; a lapsed or new one does.
            var leaseStart = _clock.GetUtcNow() - ActivationLease;
            bool alreadyActive = (await _db.LicenseActivations
                .Where(a => a.LicenseKeyId == entity.Id && a.MachineId == request.MachineId && a.IsActive)
                .Select(a => a.LastSeenAt)
                .ToListAsync(cancellationToken)
                .ConfigureAwait(false))
                .Any(lastSeen => lastSeen >= leaseStart);

            int seatsAfter = alreadyActive ? seatsUsed : seatsUsed + 1;
            if (seatsAfter > entity.MaxSeats)
            {
                await transaction.RollbackAsync(cancellationToken).ConfigureAwait(false);
                return (false, seatsUsed, null);
            }

            await UpsertActivationAsync(entity.Id, request.MachineId, request.MachineName, request.Environment, cancellationToken)
                .ConfigureAwait(false);
            await transaction.CommitAsync(cancellationToken).ConfigureAwait(false);
            return (true, seatsAfter, null);
        }).ConfigureAwait(false);
    }

    /// <summary>
    /// Takes a write lock on one license key's row, held until the enclosing transaction ends.
    /// </summary>
    /// <remarks>
    /// A no-op UPDATE rather than a provider-specific lock statement: every supported provider takes a write lock
    /// for it that is held to commit -- a row lock on PostgreSQL, SQL Server and MySQL (InnoDB), the database write
    /// lock on SQLite -- so one statement serializes admissions on all four.
    /// </remarks>
    private Task<int> LockLicenseKeyAsync(Guid licenseKeyId, CancellationToken cancellationToken)
    {
        return _db.LicenseKeys
            .Where(k => k.Id == licenseKeyId)
            .ExecuteUpdateAsync(set => set.SetProperty(k => k.MaxSeats, k => k.MaxSeats), cancellationToken);
    }

    /// <summary>
    /// Active activations seen within <see cref="ActivationLease"/>. Evaluated in memory: a key has a handful of
    /// activations, and SQLite cannot compare <see cref="DateTimeOffset"/> columns in SQL.
    /// </summary>
    private async Task<int> CountLeasedSeatsAsync(Guid licenseKeyId, CancellationToken cancellationToken)
    {
        var leaseStart = _clock.GetUtcNow() - ActivationLease;
        var lastSeen = await _db.LicenseActivations
            .Where(a => a.LicenseKeyId == licenseKeyId && a.IsActive)
            .Select(a => a.LastSeenAt)
            .ToListAsync(cancellationToken)
            .ConfigureAwait(false);
        return lastSeen.Count(seen => seen >= leaseStart);
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
            existing.LastSeenAt = _clock.GetUtcNow();
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
                FirstSeenAt = _clock.GetUtcNow(),
                LastSeenAt = _clock.GetUtcNow(),
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
            Status = LicenseKeyStatus.Invalid,
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
