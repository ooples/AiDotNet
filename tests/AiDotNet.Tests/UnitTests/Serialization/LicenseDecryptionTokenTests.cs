namespace AiDotNet.Tests.UnitTests.Serialization;

using System;
using AiDotNet.Enums;
using AiDotNet.Models;
using Xunit;

/// <summary>
/// Tests for the DecryptionToken field on LicenseValidationResult and its integration
/// with the three-layer obfuscation system.
/// </summary>
public class LicenseDecryptionTokenTests
{
    [Fact]
    public void LicenseValidationResult_DecryptionToken_IsNull_ByDefault()
    {
        var result = new LicenseValidationResult(LicenseKeyStatus.Active);
        Assert.Null(result.DecryptionToken);
    }

    [Fact]
    public void LicenseValidationResult_DecryptionToken_CanBeSet()
    {
        var token = new byte[] { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
            0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20 };

        var result = new LicenseValidationResult(
            LicenseKeyStatus.Active,
            tier: "Pro",
            decryptionToken: token);

        Assert.NotNull(result.DecryptionToken);
        Assert.Equal(32, result.DecryptionToken.Length);
        Assert.Equal(token, result.DecryptionToken);
    }

    [Fact]
    public void LicenseValidationResult_AllFields_ArePopulated()
    {
        var token = new byte[] { 0x42, 0x43 };
        var expires = DateTimeOffset.UtcNow.AddDays(30);

        var result = new LicenseValidationResult(
            LicenseKeyStatus.Active,
            tier: "Enterprise",
            expiresAt: expires,
            seatsUsed: 5,
            seatsMax: 10,
            validatedAt: DateTimeOffset.UtcNow,
            message: "Valid",
            decryptionToken: token);

        Assert.Equal(LicenseKeyStatus.Active, result.Status);
        Assert.Equal("Enterprise", result.Tier);
        Assert.Equal(expires, result.ExpiresAt);
        Assert.Equal(5, result.SeatsUsed);
        Assert.Equal(10, result.SeatsMax);
        Assert.Equal("Valid", result.Message);
        Assert.Equal(token, result.DecryptionToken);
    }

    [Fact]
    public void LicenseValidationResult_Invalid_Status_HasNoToken()
    {
        var result = new LicenseValidationResult(
            LicenseKeyStatus.Invalid,
            message: "Invalid key");

        Assert.Null(result.DecryptionToken);
        Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
    }

    [Fact]
    public void LicenseValidationResult_Revoked_Status_HasNoToken()
    {
        var result = new LicenseValidationResult(
            LicenseKeyStatus.Revoked,
            message: "Key revoked");

        Assert.Null(result.DecryptionToken);
        Assert.Equal(LicenseKeyStatus.Revoked, result.Status);
    }

    [Fact]
    public void LicenseKeyStatus_HasExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(LicenseKeyStatus), LicenseKeyStatus.Active));
        Assert.True(Enum.IsDefined(typeof(LicenseKeyStatus), LicenseKeyStatus.Invalid));
        Assert.True(Enum.IsDefined(typeof(LicenseKeyStatus), LicenseKeyStatus.Expired));
        Assert.True(Enum.IsDefined(typeof(LicenseKeyStatus), LicenseKeyStatus.Revoked));
        Assert.True(Enum.IsDefined(typeof(LicenseKeyStatus), LicenseKeyStatus.SeatLimitReached));
        Assert.True(Enum.IsDefined(typeof(LicenseKeyStatus), LicenseKeyStatus.ValidationPending));
    }
}
