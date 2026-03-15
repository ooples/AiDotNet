using System.Security.Cryptography;
using System.Text;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Tests verifying that fork builds (no embedded build key) cannot decrypt
/// models encrypted by official builds, and that the three-layer security
/// system works correctly.
/// </summary>
[Collection("LicensingTests")]
public class BuildKeyAndEncryptionTests
{
    // ─── BuildKeyProvider tests ───

    [Fact]
    public void BuildKeyProvider_GetBuildKey_ReturnsValidResult()
    {
        // GetBuildKey returns either a 32-byte key (official/local build with key)
        // or an empty array (dev/fork build without key)
        var key = BuildKeyProvider.GetBuildKey();
        Assert.NotNull(key);
        Assert.True(key.Length == 0 || key.Length == 32,
            $"Build key should be 0 bytes (dev) or 32 bytes (official), got {key.Length}");
    }

    [Fact]
    public void BuildKeyProvider_IsOfficialBuild_MatchesKeyPresence()
    {
        var key = BuildKeyProvider.GetBuildKey();
        Assert.Equal(key.Length > 0, BuildKeyProvider.IsOfficialBuild);
    }

    [Fact]
    public void BuildKeyProvider_GetBuildKey_ReturnsConsistentResults()
    {
        var key1 = BuildKeyProvider.GetBuildKey();
        var key2 = BuildKeyProvider.GetBuildKey();

        // Both calls must return identical content (length AND bytes)
        Assert.Equal(key1.Length, key2.Length);
        Assert.Equal(key1, key2);
    }

    [Fact]
    public void BuildKeyProvider_GetBuildKey_ReturnsDefensiveCopy()
    {
        var key1 = BuildKeyProvider.GetBuildKey();
        var key2 = BuildKeyProvider.GetBuildKey();

        if (key1.Length > 0)
        {
            // Must be separate array instances (defensive copy)
            Assert.False(ReferenceEquals(key1, key2));

            // Mutating one must not affect the other
            var originalFirstByte = key1[0];
            key1[0] = (byte)(originalFirstByte ^ 0xFF);
            Assert.Equal(originalFirstByte, key2[0]);
        }
        else
        {
            // Dev build: key is empty, just verify both return empty arrays
            Assert.Empty(key1);
            Assert.Empty(key2);
        }
    }

    // ─── AssemblyIntegrityChecker tests ───

    [Fact]
    public void AssemblyIntegrityChecker_DevBuild_PassesIntegrity()
    {
        // Dev builds with no integrity hash should always pass
        Assert.True(AssemblyIntegrityChecker.VerifyIntegrity());
    }

    [Fact]
    public void AssemblyIntegrityChecker_VerifyIntegrity_IsConsistent()
    {
        var result1 = AssemblyIntegrityChecker.VerifyIntegrity();
        var result2 = AssemblyIntegrityChecker.VerifyIntegrity();
        Assert.Equal(result1, result2);
    }

#if !NET471
    // ─── ModelPayloadEncryption basic tests ───

    [Fact]
    public void Encrypt_Decrypt_RoundTrip_WithLicenseKey()
    {
        var payload = Encoding.UTF8.GetBytes("test model weights payload data here");
        var licenseKey = "aidn.test12345678.abcdefghijklmnop";
        var aad = ModelPayloadEncryption.BuildAad("TestModel", [10], [1]);

        var encrypted = ModelPayloadEncryption.Encrypt(payload, licenseKey, aad);

        Assert.NotNull(encrypted);
        Assert.NotNull(encrypted.Salt);
        Assert.NotNull(encrypted.Nonce);
        Assert.NotNull(encrypted.Tag);
        Assert.NotNull(encrypted.Ciphertext);
        Assert.Equal(16, encrypted.Salt.Length);
        Assert.Equal(12, encrypted.Nonce.Length);
        Assert.Equal(16, encrypted.Tag.Length);
        Assert.Equal(payload.Length, encrypted.Ciphertext.Length);

        // Ciphertext should differ from plaintext
        Assert.False(payload.AsSpan().SequenceEqual(encrypted.Ciphertext));

        var decrypted = ModelPayloadEncryption.Decrypt(
            encrypted.Ciphertext, licenseKey, encrypted.Salt, encrypted.Nonce, encrypted.Tag, aad);

        Assert.Equal(payload, decrypted);
    }

    [Fact]
    public void Decrypt_WithWrongKey_ThrowsCryptographicException()
    {
        var payload = Encoding.UTF8.GetBytes("secret model data");
        var correctKey = "aidn.correctkey1.abcdefghijklmnop";
        var wrongKey = "aidn.wrongkeyval.zyxwvutsrqponmlk";
        var aad = ModelPayloadEncryption.BuildAad("TestModel", [5], [1]);

        var encrypted = ModelPayloadEncryption.Encrypt(payload, correctKey, aad);

        Assert.ThrowsAny<CryptographicException>(() =>
            ModelPayloadEncryption.Decrypt(
                encrypted.Ciphertext, wrongKey, encrypted.Salt, encrypted.Nonce, encrypted.Tag, aad));
    }

    [Fact]
    public void Decrypt_WithWrongAad_ThrowsCryptographicException()
    {
        var payload = Encoding.UTF8.GetBytes("secret model data");
        var licenseKey = "aidn.test12345678.abcdefghijklmnop";
        var correctAad = ModelPayloadEncryption.BuildAad("ModelA", [10], [1]);
        var wrongAad = ModelPayloadEncryption.BuildAad("ModelB", [20], [5]);

        var encrypted = ModelPayloadEncryption.Encrypt(payload, licenseKey, correctAad);

        Assert.ThrowsAny<CryptographicException>(() =>
            ModelPayloadEncryption.Decrypt(
                encrypted.Ciphertext, licenseKey, encrypted.Salt, encrypted.Nonce, encrypted.Tag, wrongAad));
    }

    [Fact]
    public void Encrypt_WithNullPayload_ThrowsArgumentNull()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelPayloadEncryption.Encrypt(null!, "key", "aad"));
    }

    [Fact]
    public void Encrypt_WithEmptyLicenseKey_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            ModelPayloadEncryption.Encrypt(new byte[10], "", "aad"));
    }

    [Fact]
    public void Decrypt_WithNullCiphertext_ThrowsArgumentNull()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelPayloadEncryption.Decrypt(null!, "key", new byte[16], new byte[12], new byte[16], "aad"));
    }

    [Fact]
    public void Decrypt_WithNullSalt_ThrowsArgumentNull()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelPayloadEncryption.Decrypt(new byte[10], "key", null!, new byte[12], new byte[16], "aad"));
    }

    // ─── EncryptSigned fork-incompatibility tests ───

    [Fact]
    public void EncryptSigned_DecryptSigned_RoundTrip_InDevBuild()
    {
        // In dev builds (no build key), EncryptSigned/DecryptSigned should still work
        // because both sides derive the same key from an empty build key
        var payload = Encoding.UTF8.GetBytes("model weights for round trip test");
        var licenseKey = "aidn.testkey12345.abcdefghijklmnop";
        var aad = ModelPayloadEncryption.BuildAad("TestModel", [10], [1]);

        var encrypted = ModelPayloadEncryption.EncryptSigned(payload, licenseKey, aad);
        var decrypted = ModelPayloadEncryption.DecryptSigned(
            encrypted.Ciphertext, licenseKey, encrypted.Salt, encrypted.Nonce, encrypted.Tag, aad);

        Assert.Equal(payload, decrypted);
    }

    [Fact]
    public void EncryptSigned_CannotDecryptWithPlainDecrypt()
    {
        // EncryptSigned incorporates build key into derivation.
        // Plain Decrypt uses a different derivation path (PBKDF2 only, no HMAC layer).
        // Even with empty build key, the HMAC step changes the derived key.
        var payload = Encoding.UTF8.GetBytes("model data that should not cross decrypt methods");
        var licenseKey = "aidn.testkey12345.abcdefghijklmnop";
        var aad = ModelPayloadEncryption.BuildAad("TestModel", [10], [1]);

        var encrypted = ModelPayloadEncryption.EncryptSigned(payload, licenseKey, aad);

        // Attempting to decrypt with the plain method should fail because the key derivation
        // differs (plain = PBKDF2 only; signed = PBKDF2 + HMAC(buildKey))
        Assert.ThrowsAny<CryptographicException>(() =>
            ModelPayloadEncryption.Decrypt(
                encrypted.Ciphertext, licenseKey, encrypted.Salt, encrypted.Nonce, encrypted.Tag, aad));
    }

    [Fact]
    public void PlainEncrypt_CannotDecryptWithSignedDecrypt()
    {
        // Conversely, plain-encrypted data cannot be decrypted with SignedDecrypt
        var payload = Encoding.UTF8.GetBytes("plain encrypted data");
        var licenseKey = "aidn.testkey12345.abcdefghijklmnop";
        var aad = ModelPayloadEncryption.BuildAad("TestModel", [10], [1]);

        var encrypted = ModelPayloadEncryption.Encrypt(payload, licenseKey, aad);

        Assert.ThrowsAny<CryptographicException>(() =>
            ModelPayloadEncryption.DecryptSigned(
                encrypted.Ciphertext, licenseKey, encrypted.Salt, encrypted.Nonce, encrypted.Tag, aad));
    }

    [Fact]
    public void EncryptSigned_WithDecryptionToken_ProducesDifferentCiphertext()
    {
        var payload = Encoding.UTF8.GetBytes("model data with token test");
        var licenseKey = "aidn.testkey12345.abcdefghijklmnop";
        var aad = ModelPayloadEncryption.BuildAad("TestModel", [10], [1]);
        var token = RandomNumberGenerator.GetBytes(32);

        var encryptedNoToken = ModelPayloadEncryption.EncryptSigned(payload, licenseKey, aad);
        var encryptedWithToken = ModelPayloadEncryption.EncryptSigned(payload, licenseKey, aad, token);

        // Different tokens produce different keys, so cross-decryption should fail
        Assert.ThrowsAny<CryptographicException>(() =>
            ModelPayloadEncryption.DecryptSigned(
                encryptedWithToken.Ciphertext, licenseKey,
                encryptedWithToken.Salt, encryptedWithToken.Nonce, encryptedWithToken.Tag,
                aad, decryptionToken: null));

        // But decrypting with the correct token should work
        var decrypted = ModelPayloadEncryption.DecryptSigned(
            encryptedWithToken.Ciphertext, licenseKey,
            encryptedWithToken.Salt, encryptedWithToken.Nonce, encryptedWithToken.Tag,
            aad, token);
        Assert.Equal(payload, decrypted);
    }

    [Fact]
    public void DifferentBuildKeys_ProduceDifferentDerivedKeys()
    {
        // Demonstrate the principle: HMAC-SHA256(baseKey, buildKeyA) != HMAC-SHA256(baseKey, buildKeyB)
        // This is why fork builds can't decrypt official builds
        var baseKey = new byte[32];
        RandomNumberGenerator.Fill(baseKey);

        var buildKeyA = new byte[32];
        RandomNumberGenerator.Fill(buildKeyA);

        var buildKeyB = new byte[32]; // Different "fork" build key
        RandomNumberGenerator.Fill(buildKeyB);

        var emptyBuildKey = Array.Empty<byte>(); // Dev/fork build

        using var hmacA = new HMACSHA256(baseKey);
        using var hmacB = new HMACSHA256(baseKey);
        using var hmacEmpty = new HMACSHA256(baseKey);

        var derivedA = hmacA.ComputeHash(buildKeyA);
        var derivedB = hmacB.ComputeHash(buildKeyB);
        var derivedEmpty = hmacEmpty.ComputeHash(emptyBuildKey);

        // All three derived keys should be different
        Assert.False(derivedA.AsSpan().SequenceEqual(derivedB));
        Assert.False(derivedA.AsSpan().SequenceEqual(derivedEmpty));
        Assert.False(derivedB.AsSpan().SequenceEqual(derivedEmpty));
    }

    [Fact]
    public void IntegrityHash_SelfConsistency_HMAC()
    {
        // Verify the CI/CD integrity hash formula: HMAC-SHA256(buildKey, buildKey)
        var buildKey = new byte[32];
        RandomNumberGenerator.Fill(buildKey);

        using var hmac = new HMACSHA256(buildKey);
        var hash1 = hmac.ComputeHash(buildKey);

        // Should be deterministic
        using var hmac2 = new HMACSHA256(buildKey);
        var hash2 = hmac2.ComputeHash(buildKey);

        Assert.True(hash1.AsSpan().SequenceEqual(hash2));
        Assert.Equal(32, hash1.Length);
    }
#endif

    // ─── BuildAad tests ───

    [Fact]
    public void BuildAad_ProducesDeterministicOutput()
    {
        var aad1 = ModelPayloadEncryption.BuildAad("MyModel", [10, 5], [1]);
        var aad2 = ModelPayloadEncryption.BuildAad("MyModel", [10, 5], [1]);
        Assert.Equal(aad1, aad2);
    }

    [Fact]
    public void BuildAad_DifferentModels_ProduceDifferentOutput()
    {
        var aad1 = ModelPayloadEncryption.BuildAad("ModelA", [10], [1]);
        var aad2 = ModelPayloadEncryption.BuildAad("ModelB", [10], [1]);
        Assert.NotEqual(aad1, aad2);
    }

    [Fact]
    public void BuildAad_DifferentShapes_ProduceDifferentOutput()
    {
        var aad1 = ModelPayloadEncryption.BuildAad("Model", [10], [1]);
        var aad2 = ModelPayloadEncryption.BuildAad("Model", [20], [5]);
        Assert.NotEqual(aad1, aad2);
    }

    [Fact]
    public void BuildAad_NullTypeName_DoesNotThrow()
    {
        var aad = ModelPayloadEncryption.BuildAad(null!, [10], [1]);
        Assert.NotNull(aad);
        Assert.Contains("|", aad);
    }

    [Fact]
    public void BuildAad_EmptyShapes_ProducesValidOutput()
    {
        var aad = ModelPayloadEncryption.BuildAad("Model", [], []);
        Assert.Equal("Model||", aad);
    }
}
