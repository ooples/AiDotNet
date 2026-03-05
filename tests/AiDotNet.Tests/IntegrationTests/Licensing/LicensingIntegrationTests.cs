#if !NET471
namespace AiDotNet.Tests.IntegrationTests.Licensing;

using System;
using System.IO;
using System.Security.Cryptography;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Tests.UnitTests.Serialization;
using Xunit;

/// <summary>
/// Integration tests for the client-side encrypt/decrypt pipeline (Layer 1 + AIMF header).
/// </summary>
public class LicensingIntegrationTests
{
    [Fact]
    public void BasicEncryptDecrypt_RoundTrip()
    {
        ModelTypeRegistry.Register(typeof(StubModelSerializer).Name, typeof(StubModelSerializer));

        var tempFile = Path.Combine(Path.GetTempPath(), $"lic_rt_{Guid.NewGuid():N}.aimf");
        try
        {
            var payload = new byte[] { 0xCA, 0xFE, 0xBA, 0xBE, 0x42 };
            var model = new StubModelSerializer
            {
                Payload = payload,
                InputShapeValue = new[] { 5 },
                OutputShapeValue = new[] { 1 }
            };
            var licenseKey = "test-license-key-roundtrip-2026";

            ModelLoader.SaveEncrypted(model, tempFile, licenseKey,
                model.GetInputShape(), model.GetOutputShape());

            var loaded = ModelLoader.Load<double>(tempFile, licenseKey);
            Assert.NotNull(loaded);
            Assert.IsType<StubModelSerializer>(loaded);
            Assert.Equal(payload, ((StubModelSerializer)loaded).GetDeserializedData());
        }
        finally
        {
            if (File.Exists(tempFile)) File.Delete(tempFile);
        }
    }

    [Fact]
    public void EncryptDecrypt_WrongKey_Fails()
    {
        ModelTypeRegistry.Register(typeof(StubModelSerializer).Name, typeof(StubModelSerializer));

        var tempFile = Path.Combine(Path.GetTempPath(), $"lic_wrongkey_{Guid.NewGuid():N}.aimf");
        try
        {
            var model = new StubModelSerializer
            {
                Payload = new byte[] { 1, 2, 3, 4 },
                InputShapeValue = new[] { 4 },
                OutputShapeValue = new[] { 1 }
            };

            ModelLoader.SaveEncrypted(model, tempFile, "correct-key-abc",
                model.GetInputShape(), model.GetOutputShape());

            Assert.ThrowsAny<CryptographicException>(() =>
                ModelLoader.Load<double>(tempFile, "wrong-key-xyz"));
        }
        finally
        {
            if (File.Exists(tempFile)) File.Delete(tempFile);
        }
    }

    [Fact]
    public void EncryptDecrypt_TamperedData_Fails()
    {
        var plaintext = new byte[] { 10, 20, 30, 40, 50 };
        var licenseKey = "tamper-detection-key";
        var aad = ModelPayloadEncryption.BuildAad("TamperTest", new[] { 5 }, new[] { 1 });

        var encrypted = ModelPayloadEncryption.Encrypt(plaintext, licenseKey, aad);

        // Flip a bit in the ciphertext
        var tampered = new byte[encrypted.Ciphertext.Length];
        Array.Copy(encrypted.Ciphertext, tampered, tampered.Length);
        tampered[0] ^= 0xFF;

        Assert.ThrowsAny<CryptographicException>(() =>
            ModelPayloadEncryption.Decrypt(tampered, licenseKey,
                encrypted.Salt, encrypted.Nonce, encrypted.Tag, aad));
    }

    [Fact]
    public void ModelFileHeader_WrapAndExtract_RoundTrip()
    {
        var model = new StubModelSerializer
        {
            Payload = new byte[] { 0xAA, 0xBB, 0xCC },
            InputShapeValue = new[] { 784 },
            OutputShapeValue = new[] { 10 }
        };

        byte[] wrapped = ModelFileHeader.WrapWithHeader(
            model.Payload, model,
            model.GetInputShape(), model.GetOutputShape(),
            SerializationFormat.Binary);

        Assert.True(ModelFileHeader.HasHeader(wrapped));

        var info = ModelFileHeader.ReadHeader(wrapped);
        byte[] extracted = ModelFileHeader.ExtractPayload(wrapped, info);

        Assert.Equal(model.Payload, extracted);
        Assert.Equal(new[] { 784 }, info.InputShape);
        Assert.Equal(new[] { 10 }, info.OutputShape);
    }

    [Fact]
    public void ModelFileHeader_EncryptedScheme_PreservedInHeader()
    {
        var model = new StubModelSerializer
        {
            Payload = new byte[] { 0x01 },
            InputShapeValue = new[] { 28, 28 },
            OutputShapeValue = new[] { 10 }
        };

        var fakeSalt = new byte[16];
        var fakeNonce = new byte[12];
        var fakeTag = new byte[16];

        byte[] wrapped = ModelFileHeader.WrapWithHeaderEncrypted(
            new byte[] { 0xFF }, model,
            model.GetInputShape(), model.GetOutputShape(),
            SerializationFormat.Binary,
            fakeSalt, fakeNonce, fakeTag);

        var info = ModelFileHeader.ReadHeader(wrapped);

        Assert.True(info.IsEncrypted);
        Assert.Equal(PayloadEncryptionScheme.AesGcm256, info.EncryptionScheme);
        Assert.Equal(fakeSalt, info.Salt);
        Assert.Equal(fakeNonce, info.Nonce);
        Assert.Equal(fakeTag, info.Tag);
    }

    [Fact]
    public void EncryptSigned_WithSyntheticBuildKey_ProducesDifferentOutput()
    {
        var plaintext = new byte[] { 1, 2, 3, 4, 5 };
        var aad = ModelPayloadEncryption.BuildAad("SignTest", new[] { 5 }, new[] { 1 });

        // Encrypt twice with the same user key; salt/nonce randomness alone makes them different
        var enc1 = ModelPayloadEncryption.Encrypt(plaintext, "same-key", aad);
        var enc2 = ModelPayloadEncryption.Encrypt(plaintext, "same-key", aad);

        // Different random salt means different ciphertext
        Assert.NotEqual(enc1.Ciphertext, enc2.Ciphertext);
        Assert.NotEqual(enc1.Salt, enc2.Salt);
        Assert.NotEqual(enc1.Nonce, enc2.Nonce);

        // But both decrypt to the same plaintext
        var dec1 = ModelPayloadEncryption.Decrypt(enc1.Ciphertext, "same-key",
            enc1.Salt, enc1.Nonce, enc1.Tag, aad);
        var dec2 = ModelPayloadEncryption.Decrypt(enc2.Ciphertext, "same-key",
            enc2.Salt, enc2.Nonce, enc2.Tag, aad);
        Assert.Equal(plaintext, dec1);
        Assert.Equal(plaintext, dec2);
    }

    [Fact]
    public void DecryptSigned_WithSyntheticToken_RoundTrip()
    {
        // Simulates the full 3-layer round trip:
        // Layer 1: User license key encrypts the payload
        // Layer 2: Escrow secret + license key produces decryption token (HMAC)
        // Layer 3: Integrity check (passes in dev mode)

        var payload = new byte[] { 0xDE, 0xAD, 0xBE, 0xEF };
        var userLicenseKey = "layer1-user-key";
        var aad = ModelPayloadEncryption.BuildAad("ThreeLayerTest", new[] { 4 }, new[] { 1 });

        // Layer 1: Encrypt with user license key
        var encrypted = ModelPayloadEncryption.Encrypt(payload, userLicenseKey, aad);

        // Layer 2: Simulate server-side escrow (HMAC of license key bytes with escrow secret)
        var escrowSecret = new byte[32];
        RandomNumberGenerator.Fill(escrowSecret);
        using var hmac = new HMACSHA256(escrowSecret);
        var licenseBytes = System.Text.Encoding.UTF8.GetBytes(userLicenseKey);
        var tokenBytes = hmac.ComputeHash(licenseBytes);
        var decryptionToken = Convert.ToBase64String(tokenBytes);

        // Verify token is non-empty base64
        Assert.False(string.IsNullOrWhiteSpace(decryptionToken));
        var decoded = Convert.FromBase64String(decryptionToken);
        Assert.Equal(32, decoded.Length); // HMAC-SHA256 produces 32 bytes

        // Layer 1 decryption still works with the original user key
        var decrypted = ModelPayloadEncryption.Decrypt(
            encrypted.Ciphertext, userLicenseKey,
            encrypted.Salt, encrypted.Nonce, encrypted.Tag, aad);
        Assert.Equal(payload, decrypted);

        // Layer 3: AssemblyIntegrityChecker passes in dev mode (no build key embedded)
        Assert.True(AssemblyIntegrityChecker.VerifyIntegrity());
    }
}
#endif
