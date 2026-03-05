#if !NET471
namespace AiDotNet.Tests.IntegrationTests.Licensing;

using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Threading.Tasks;
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

    [Fact]
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

    [Fact]
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

    [Fact]
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

    [Fact]
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

    [Fact]
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

    [Fact]
    public async Task ValidateAsync_InvalidKey_ReturnsInvalid()
    {
        var validateResponse = await _svc.ValidateAsync(new LicenseValidateRequest
        {
            Key = "aidn.bogus.key123"
        });

        Assert.Equal(LicenseKeyStatus.Invalid, validateResponse.Status);
    }

    [Fact]
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

    [Fact]
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

    [Fact]
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

            ModelLoader.SaveEncrypted(model, tempFile, createResponse.LicenseKey,
                model.GetInputShape(), model.GetOutputShape());

            // Step 4: Client-side — decrypt with the same license key
            var loaded = ModelLoader.Load<double>(tempFile, createResponse.LicenseKey);
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
