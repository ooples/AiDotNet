namespace AiDotNet.Tests.UnitTests.Serialization;

using System;
using System.Security.Cryptography;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using Xunit;

public class ObfuscationTests
{
    [Fact]
    public void BuildKeyProvider_ReturnsEmpty_WhenNoEmbeddedResource()
    {
        // In test/dev builds there is no embedded build key
        var key = BuildKeyProvider.GetBuildKey();
        Assert.NotNull(key);
        Assert.Empty(key);
    }

    [Fact]
    public void BuildKeyProvider_IsOfficialBuild_ReturnsFalse_InDevBuild()
    {
        Assert.False(BuildKeyProvider.IsOfficialBuild);
    }

    [Fact]
    public void PayloadEncryptionScheme_AesGcm256Signed_HasCorrectValue()
    {
        Assert.Equal(2, (int)PayloadEncryptionScheme.AesGcm256Signed);
    }

    [Fact]
    public void PayloadEncryptionScheme_None_HasCorrectValue()
    {
        Assert.Equal(0, (int)PayloadEncryptionScheme.None);
    }

    [Fact]
    public void PayloadEncryptionScheme_AesGcm256_HasCorrectValue()
    {
        Assert.Equal(1, (int)PayloadEncryptionScheme.AesGcm256);
    }

    [Fact]
    public void AssemblyIntegrityChecker_NoHash_ReturnsTrue_InDevBuild()
    {
        // Dev builds have no integrity hash, so verification should pass
        Assert.True(AssemblyIntegrityChecker.VerifyIntegrity());
    }

#if !NET471
    [Fact]
    public void EncryptSigned_WithoutBuildKey_ProducesValidCiphertext()
    {
        // In dev builds (no build key), EncryptSigned should still work
        // because AssemblyIntegrityChecker returns true for dev builds
        var plaintext = new byte[] { 0xDE, 0xAD, 0xBE, 0xEF, 0x42, 0x00, 0xFF };
        var licenseKey = "test-license-key-signed-12345";
        var aad = ModelPayloadEncryption.BuildAad("TestModel", new[] { 784 }, new[] { 10 });

        var encrypted = ModelPayloadEncryption.EncryptSigned(plaintext, licenseKey, aad);

        Assert.NotNull(encrypted.Salt);
        Assert.NotNull(encrypted.Nonce);
        Assert.NotNull(encrypted.Tag);
        Assert.NotNull(encrypted.Ciphertext);
        Assert.Equal(16, encrypted.Salt.Length);
        Assert.Equal(12, encrypted.Nonce.Length);
        Assert.Equal(16, encrypted.Tag.Length);
        Assert.Equal(plaintext.Length, encrypted.Ciphertext.Length);

        // Ciphertext should differ from plaintext
        Assert.NotEqual(plaintext, encrypted.Ciphertext);
    }

    [Fact]
    public void DecryptSigned_RoundTrip_ProducesIdenticalBytes()
    {
        var plaintext = new byte[] { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 };
        var licenseKey = "roundtrip-test-license-key";
        var aad = ModelPayloadEncryption.BuildAad("RoundTripModel", new[] { 100 }, new[] { 5 });

        var encrypted = ModelPayloadEncryption.EncryptSigned(plaintext, licenseKey, aad);

        var decrypted = ModelPayloadEncryption.DecryptSigned(
            encrypted.Ciphertext, licenseKey, encrypted.Salt, encrypted.Nonce, encrypted.Tag, aad);

        Assert.Equal(plaintext, decrypted);
    }

    [Fact]
    public void DecryptSigned_WithDecryptionToken_RoundTrip()
    {
        var plaintext = new byte[] { 0xAA, 0xBB, 0xCC, 0xDD };
        var licenseKey = "token-test-license-key";
        var aad = ModelPayloadEncryption.BuildAad("TokenModel", new[] { 50 }, new[] { 2 });
        var decryptionToken = new byte[] { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
            0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20 };

        var encrypted = ModelPayloadEncryption.EncryptSigned(plaintext, licenseKey, aad, decryptionToken);

        var decrypted = ModelPayloadEncryption.DecryptSigned(
            encrypted.Ciphertext, licenseKey, encrypted.Salt, encrypted.Nonce, encrypted.Tag, aad, decryptionToken);

        Assert.Equal(plaintext, decrypted);
    }

    [Fact]
    public void DecryptSigned_WrongToken_Throws()
    {
        var plaintext = new byte[] { 0xAA, 0xBB, 0xCC, 0xDD };
        var licenseKey = "wrong-token-test-key";
        var aad = ModelPayloadEncryption.BuildAad("WrongTokenModel", new[] { 50 }, new[] { 2 });
        var correctToken = new byte[] { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
            0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20 };
        var wrongToken = new byte[] { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };

        var encrypted = ModelPayloadEncryption.EncryptSigned(plaintext, licenseKey, aad, correctToken);

        // Decrypting with the wrong token should throw CryptographicException
        Assert.ThrowsAny<CryptographicException>(() =>
            ModelPayloadEncryption.DecryptSigned(
                encrypted.Ciphertext, licenseKey, encrypted.Salt, encrypted.Nonce, encrypted.Tag, aad, wrongToken));
    }

    [Fact]
    public void DecryptSigned_WrongLicenseKey_Throws()
    {
        var plaintext = new byte[] { 0x11, 0x22, 0x33, 0x44 };
        var correctKey = "correct-license-key";
        var wrongKey = "wrong-license-key";
        var aad = ModelPayloadEncryption.BuildAad("WrongKeyModel", new[] { 10 }, new[] { 1 });

        var encrypted = ModelPayloadEncryption.EncryptSigned(plaintext, correctKey, aad);

        Assert.ThrowsAny<CryptographicException>(() =>
            ModelPayloadEncryption.DecryptSigned(
                encrypted.Ciphertext, wrongKey, encrypted.Salt, encrypted.Nonce, encrypted.Tag, aad));
    }

    [Fact]
    public void EncryptSigned_NullPayload_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelPayloadEncryption.EncryptSigned(null, "key", "aad"));
    }

    [Fact]
    public void EncryptSigned_EmptyLicenseKey_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            ModelPayloadEncryption.EncryptSigned(new byte[] { 0x01 }, "", "aad"));
    }

    [Fact]
    public void Signed_And_Standard_Produce_Different_Ciphertext()
    {
        // Even with the same inputs, signed and standard encryption should produce
        // different results because the key derivation differs
        var plaintext = new byte[] { 0x01, 0x02, 0x03, 0x04, 0x05 };
        var licenseKey = "same-key-different-derivation";
        var aad = ModelPayloadEncryption.BuildAad("DiffTest", new[] { 5 }, new[] { 1 });

        var standard = ModelPayloadEncryption.Encrypt(plaintext, licenseKey, aad);
        var signed = ModelPayloadEncryption.EncryptSigned(plaintext, licenseKey, aad);

        // Different salt/nonce means different ciphertext regardless, but verify
        // we can't cross-decrypt
        Assert.ThrowsAny<CryptographicException>(() =>
            ModelPayloadEncryption.Decrypt(
                signed.Ciphertext, licenseKey, signed.Salt, signed.Nonce, signed.Tag, aad));
    }
#endif
}
