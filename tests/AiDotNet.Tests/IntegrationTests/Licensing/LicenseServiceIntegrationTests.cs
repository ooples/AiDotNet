using System.Threading.Tasks;
#if NET10_0_OR_GREATER
namespace AiDotNet.Tests.IntegrationTests.Licensing;

using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using AiDotNet.Helpers;
using AiDotNet.Serving.Persistence;
using AiDotNet.Serving.Security;
using AiDotNet.Tests.UnitTests.Serialization;
using AiDotNet.Serving.Security.Licensing;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

using AiDotNet.Enums;

/// <summary>
/// Integration tests for LicenseService using in-memory SQLite.
/// </summary>
public class LicenseServiceIntegrationTests : IDisposable
{
    private readonly ServingDbContext _db;
    private readonly LicenseService _svc;

    public LicenseServiceIntegrationTests()
    {
        var options = new DbContextOptionsBuilder<ServingDbContext>()
            .UseSqlite("DataSource=:memory:")
            .Options;

        _db = new ServingDbContext(options);
        _db.Database.OpenConnection();
        _db.Database.EnsureCreated();

        _svc = new LicenseService(_db, NullLogger<LicenseService>.Instance);
    }

    public void Dispose()
    {
        _db.Database.CloseConnection();
        _db.Dispose();
    }

    [Fact(Timeout = 120000)]
    public async Task CreateAsync_GeneratesValidKeyFormat()
    {
        var response = await _svc.CreateAsync(new LicenseCreateRequest
        {
            CustomerName = "Test Customer",
            Tier = SubscriptionTier.Pro,
            MaxSeats = 5
        });

        Assert.NotEqual(Guid.Empty, response.Id);
        Assert.StartsWith("aidn.", response.LicenseKey);

        // Format is aidn.{keyId}.{secret}
        var parts = response.LicenseKey.Split('.');
        Assert.Equal(3, parts.Length);
        Assert.Equal("aidn", parts[0]);
        Assert.False(string.IsNullOrWhiteSpace(parts[1]));
        Assert.False(string.IsNullOrWhiteSpace(parts[2]));
    }

    [Fact(Timeout = 120000)]
    public async Task CreateAsync_StoresHashNotPlaintext()
    {
        var response = await _svc.CreateAsync(new LicenseCreateRequest
        {
            CustomerName = "Hash Test"
        });

        var entity = await _db.LicenseKeys.SingleAsync(k => k.Id == response.Id);

        // Hash and salt must be populated
        Assert.NotEmpty(entity.Hash);
        Assert.NotEmpty(entity.Salt);
        Assert.True(entity.Pbkdf2Iterations >= 100_000);

        // The license key secret should NOT appear in stored fields
        var secret = response.LicenseKey.Split('.')[2];
        var secretBytes = System.Text.Encoding.UTF8.GetBytes(secret);

        // Hash should not equal the raw secret
        Assert.NotEqual(secretBytes, entity.Hash);
    }

    [Fact(Timeout = 120000)]
    public async Task ValidateAsync_ActiveKey_ReturnsActive()
    {
        var createResponse = await _svc.CreateAsync(new LicenseCreateRequest
        {
            CustomerName = "Active Test",
            Tier = SubscriptionTier.Pro,
            MaxSeats = 3
        });

        var validateResponse = await _svc.ValidateAsync(new LicenseValidateRequest
        {
            Key = createResponse.LicenseKey,
            MachineId = "machine-001"
        });

        Assert.Equal(LicenseKeyStatus.Active, validateResponse.Status);
        Assert.Equal("Pro", validateResponse.Tier);
        Assert.Equal(1, validateResponse.SeatsUsed);
        Assert.Equal(3, validateResponse.SeatsMax);
    }

    [Fact(Timeout = 120000)]
    public async Task ValidateAsync_RevokedKey_ReturnsRevoked()
    {
        var createResponse = await _svc.CreateAsync(new LicenseCreateRequest
        {
            CustomerName = "Revoke Test"
        });

        await _svc.RevokeAsync(createResponse.Id);

        var validateResponse = await _svc.ValidateAsync(new LicenseValidateRequest
        {
            Key = createResponse.LicenseKey
        });

        Assert.Equal(LicenseKeyStatus.Revoked, validateResponse.Status);
    }

    [Fact(Timeout = 120000)]
    public async Task ValidateAsync_ExpiredKey_ReturnsExpired()
    {
        var createResponse = await _svc.CreateAsync(new LicenseCreateRequest
        {
            CustomerName = "Expiry Test",
            ExpiresAt = DateTimeOffset.UtcNow.AddYears(1) // must be future for creation
        });

        // Manually set expiry to the past
        var entity = await _db.LicenseKeys.SingleAsync(k => k.Id == createResponse.Id);
        entity.ExpiresAt = DateTimeOffset.UtcNow.AddHours(-1);
        await _db.SaveChangesAsync();

        var validateResponse = await _svc.ValidateAsync(new LicenseValidateRequest
        {
            Key = createResponse.LicenseKey
        });

        Assert.Equal(LicenseKeyStatus.Expired, validateResponse.Status);
    }

    [Fact(Timeout = 120000)]
    public async Task ValidateAsync_InvalidKey_ReturnsInvalid()
    {
        var validateResponse = await _svc.ValidateAsync(new LicenseValidateRequest
        {
            Key = "aidn.bogus.key123"
        });

        Assert.Equal(LicenseKeyStatus.Invalid, validateResponse.Status);
    }

    [Fact(Timeout = 120000)]
    public async Task ValidateAsync_ReturnsDecryptionToken()
    {
        var createResponse = await _svc.CreateAsync(new LicenseCreateRequest
        {
            CustomerName = "Token Test",
            Tier = SubscriptionTier.Enterprise
        });

        var validateResponse = await _svc.ValidateAsync(new LicenseValidateRequest
        {
            Key = createResponse.LicenseKey,
            MachineId = "machine-token"
        });

        Assert.Equal(LicenseKeyStatus.Active, validateResponse.Status);
        Assert.False(string.IsNullOrWhiteSpace(validateResponse.DecryptionToken));

        // Token should be valid base64
        var tokenBytes = Convert.FromBase64String(validateResponse.DecryptionToken);
        Assert.Equal(32, tokenBytes.Length); // HMAC-SHA256 output
    }

    [Fact(Timeout = 120000)]
    public async Task SeatEnforcement_ExceedsMaxSeats_ReturnsSeatLimitReached()
    {
        var createResponse = await _svc.CreateAsync(new LicenseCreateRequest
        {
            CustomerName = "Seat Test",
            MaxSeats = 1
        });

        // First machine should succeed
        var result1 = await _svc.ValidateAsync(new LicenseValidateRequest
        {
            Key = createResponse.LicenseKey,
            MachineId = "machine-A"
        });
        Assert.Equal(LicenseKeyStatus.Active, result1.Status);

        // Second machine should hit seat limit
        var result2 = await _svc.ValidateAsync(new LicenseValidateRequest
        {
            Key = createResponse.LicenseKey,
            MachineId = "machine-B"
        });
        Assert.Equal(LicenseKeyStatus.SeatLimitReached, result2.Status);
    }

    [Fact(Timeout = 120000)]
    public async Task SeatLease_A_machine_unseen_for_the_lease_frees_its_seat_and_cannot_reclaim_a_taken_one()
    {
        var clock = new ManualClock(new DateTimeOffset(2026, 11, 1, 0, 0, 0, TimeSpan.Zero));
        var svc = new LicenseService(_db, NullLogger<LicenseService>.Instance, clock);
        var key = (await svc.CreateAsync(new LicenseCreateRequest { CustomerName = "Lease", MaxSeats = 1 })).LicenseKey;

        Assert.Equal(LicenseKeyStatus.Active, (await Validate(svc, key, "A")).Status);
        Assert.Equal(LicenseKeyStatus.SeatLimitReached, (await Validate(svc, key, "B")).Status);

        clock.Advance(LicenseService.ActivationLease + TimeSpan.FromMinutes(1));
        var b = await Validate(svc, key, "B");
        Assert.Equal(LicenseKeyStatus.Active, b.Status);
        Assert.Equal(1, b.SeatsUsed);

        // A comes back after its lease lapsed: B now holds the only seat.
        Assert.Equal(LicenseKeyStatus.SeatLimitReached, (await Validate(svc, key, "A")).Status);
    }

    [Fact(Timeout = 120000)]
    public async Task SeatLease_Every_validation_renews_the_lease()
    {
        var clock = new ManualClock(new DateTimeOffset(2026, 11, 1, 0, 0, 0, TimeSpan.Zero));
        var svc = new LicenseService(_db, NullLogger<LicenseService>.Instance, clock);
        var key = (await svc.CreateAsync(new LicenseCreateRequest { CustomerName = "Renew", MaxSeats = 1 })).LicenseKey;

        Assert.Equal(LicenseKeyStatus.Active, (await Validate(svc, key, "A")).Status);
        clock.Advance(LicenseService.ActivationLease - TimeSpan.FromDays(1));
        Assert.Equal(LicenseKeyStatus.Active, (await Validate(svc, key, "A")).Status);
        clock.Advance(LicenseService.ActivationLease - TimeSpan.FromDays(1));

        // Past the first validation's lease, inside the renewed one.
        Assert.Equal(LicenseKeyStatus.SeatLimitReached, (await Validate(svc, key, "B")).Status);
    }

    [Fact(Timeout = 120000)]
    public async Task SeatLease_Ephemeral_machines_validating_once_each_do_not_exhaust_the_key()
    {
        // Every CI runner has a fresh machine id. Counting every activation ever seen used up a CI key in days.
        var clock = new ManualClock(new DateTimeOffset(2026, 11, 1, 0, 0, 0, TimeSpan.Zero));
        var svc = new LicenseService(_db, NullLogger<LicenseService>.Instance, clock);
        var key = (await svc.CreateAsync(new LicenseCreateRequest { CustomerName = "CI", MaxSeats = 2 })).LicenseKey;

        for (var runner = 0; runner < 10; runner++)
        {
            Assert.Equal(LicenseKeyStatus.Active, (await Validate(svc, key, $"runner-{runner}")).Status);
            clock.Advance(LicenseService.ActivationLease / 2 + TimeSpan.FromMinutes(1));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SeatLease_A_rejected_validation_does_not_renew_the_lease()
    {
        // The seat limit used to be re-checked AFTER the renewal was saved, so a machine turned away still
        // extended its lease, and by validating repeatedly kept holding a seat it had been refused.
        var start = new DateTimeOffset(2026, 11, 1, 0, 0, 0, TimeSpan.Zero);
        var clock = new ManualClock(start);
        var svc = new LicenseService(_db, NullLogger<LicenseService>.Instance, clock);
        var created = await svc.CreateAsync(new LicenseCreateRequest { CustomerName = "Shrunk", MaxSeats = 2 });

        Assert.Equal(LicenseKeyStatus.Active, (await Validate(svc, created.LicenseKey, "A")).Status);
        Assert.Equal(LicenseKeyStatus.Active, (await Validate(svc, created.LicenseKey, "B")).Status);

        // An administrator lowers the key to one seat while both machines still hold a lease.
        var license = await _db.LicenseKeys.SingleAsync(k => k.Id == created.Id);
        license.MaxSeats = 1;
        await _db.SaveChangesAsync();

        clock.Advance(TimeSpan.FromDays(6));
        var refused = await Validate(svc, created.LicenseKey, "A");
        Assert.Equal(LicenseKeyStatus.SeatLimitReached, refused.Status);
        Assert.Equal(2, refused.SeatsUsed);

        var lastSeen = await _db.LicenseActivations.AsNoTracking()
            .Where(a => a.LicenseKeyId == created.Id && a.MachineId == "A")
            .Select(a => a.LastSeenAt)
            .SingleAsync();
        Assert.Equal(start, lastSeen);
    }

    [Fact(Timeout = 120000)]
    public async Task SeatLease_Concurrent_admissions_never_exceed_the_seat_limit()
    {
        // Two lapsed or new machines on a one-seat key could both read a free seat and both persist a lease.
        // Separate contexts on one database file stand in for separate requests (or server instances).
        var path = Path.Combine(Path.GetTempPath(), $"license_seats_{Guid.NewGuid():N}.db");
        var connectionString = $"Data Source={path}";
        ServingDbContext NewContext() => new(new DbContextOptionsBuilder<ServingDbContext>()
            .UseSqlite(connectionString)
            .Options);

        try
        {
            string key;
            Guid licenseId;
            using (var setup = NewContext())
            {
                setup.Database.EnsureCreated();
                var created = await new LicenseService(setup, NullLogger<LicenseService>.Instance)
                    .CreateAsync(new LicenseCreateRequest { CustomerName = "Race", MaxSeats = 1 });
                key = created.LicenseKey;
                licenseId = created.Id;
            }

            const int contenders = 8;
            var results = await Task.WhenAll(Enumerable.Range(0, contenders).Select(i => Task.Run(async () =>
            {
                using var context = NewContext();
                var svc = new LicenseService(context, NullLogger<LicenseService>.Instance);
                return await Validate(svc, key, $"machine-{i}");
            })));

            Assert.Equal(1, results.Count(r => r.Status == LicenseKeyStatus.Active));
            Assert.Equal(contenders - 1, results.Count(r => r.Status == LicenseKeyStatus.SeatLimitReached));

            using var check = NewContext();
            Assert.Equal(1, await check.LicenseActivations.CountAsync(a => a.LicenseKeyId == licenseId && a.IsActive));
        }
        finally
        {
            Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
            if (File.Exists(path)) File.Delete(path);
        }
    }

    private static Task<LicenseValidationResponse> Validate(LicenseService svc, string key, string machine) =>
        svc.ValidateAsync(new LicenseValidateRequest { Key = key, MachineId = machine });

    private sealed class ManualClock(DateTimeOffset start) : TimeProvider
    {
        private DateTimeOffset _now = start;

        public override DateTimeOffset GetUtcNow() => _now;

        public void Advance(TimeSpan by) => _now += by;
    }

    [Fact(Timeout = 120000)]
    public async Task EndToEnd_EncryptWithServerToken_DecryptSucceeds()
    {
        ModelTypeRegistry.Register(typeof(StubModelSerializer).Name, typeof(StubModelSerializer));

        // Step 1: Server-side — create license
        var createResponse = await _svc.CreateAsync(new LicenseCreateRequest
        {
            CustomerName = "E2E Test",
            Tier = SubscriptionTier.Enterprise,
            MaxSeats = 5
        });

        // Step 2: Server-side — validate and get decryption token
        var validateResponse = await _svc.ValidateAsync(new LicenseValidateRequest
        {
            Key = createResponse.LicenseKey,
            MachineId = "e2e-machine"
        });

        Assert.Equal(LicenseKeyStatus.Active, validateResponse.Status);
        Assert.False(string.IsNullOrWhiteSpace(validateResponse.DecryptionToken));

        // Step 3: Client-side — encrypt model with the license key
        var tempFile = Path.Combine(Path.GetTempPath(), $"e2e_{Guid.NewGuid():N}.aimf");
        try
        {
            var payload = new byte[512];
            RandomNumberGenerator.Fill(payload);
            var model = new StubModelSerializer
            {
                Payload = payload,
                InputShapeValue = new[] { 512 },
                OutputShapeValue = new[] { 10 }
            };

            // Pass the server decryption token as proof of server-side validation — server-issued keys
            // are escrow-signed and cannot be verified offline, so the persist guard honors the token
            // instead of offline-rejecting the key.
            ModelLoader.SaveEncrypted(model, tempFile, createResponse.LicenseKey,
                model.GetInputShape(), model.GetOutputShape(),
                decryptionToken: System.Text.Encoding.UTF8.GetBytes(validateResponse.DecryptionToken));

            // Step 4: Client-side — decrypt with the same license key
            var loaded = ModelLoader.Load<double>(tempFile, createResponse.LicenseKey,
                decryptionToken: System.Text.Encoding.UTF8.GetBytes(validateResponse.DecryptionToken));
            Assert.NotNull(loaded);
            Assert.IsType<StubModelSerializer>(loaded);
            Assert.Equal(payload, ((StubModelSerializer)loaded).GetDeserializedData());

            // Step 5: Verify the decryption token is stable across validations
            var validateResponse2 = await _svc.ValidateAsync(new LicenseValidateRequest
            {
                Key = createResponse.LicenseKey,
                MachineId = "e2e-machine"
            });
            Assert.Equal(validateResponse.DecryptionToken, validateResponse2.DecryptionToken);
        }
        finally
        {
            if (File.Exists(tempFile)) File.Delete(tempFile);
        }
    }
}
#endif
