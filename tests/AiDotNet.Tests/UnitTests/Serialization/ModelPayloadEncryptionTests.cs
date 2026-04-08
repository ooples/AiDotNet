namespace AiDotNet.Tests.UnitTests.Serialization;

using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using Xunit;

public class ModelPayloadEncryptionTests
{
#if !NET471
    [Fact]
    public void EncryptDecrypt_RoundTrip_ProducesIdenticalBytes()
    {
        var plaintext = new byte[] { 0xDE, 0xAD, 0xBE, 0xEF, 0x42, 0x00, 0xFF };
        var licenseKey = "test-license-key-12345";
        var aad = ModelPayloadEncryption.BuildAad("TestModel", new[] { 784 }, new[] { 10 });

        var encrypted = ModelPayloadEncryption.Encrypt(plaintext, licenseKey, aad);

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

        var decrypted = ModelPayloadEncryption.Decrypt(
            encrypted.Ciphertext, licenseKey, encrypted.Salt, encrypted.Nonce, encrypted.Tag, aad);

        Assert.Equal(plaintext, decrypted);
    }

    [Fact]
    public void EncryptDecrypt_EmptyPayload_RoundTrip()
    {
        var plaintext = Array.Empty<byte>();
        var licenseKey = "my-empty-payload-key";
        var aad = ModelPayloadEncryption.BuildAad("EmptyModel", Array.Empty<int>(), Array.Empty<int>());

        var encrypted = ModelPayloadEncryption.Encrypt(plaintext, licenseKey, aad);

        Assert.Empty(encrypted.Ciphertext);

        var decrypted = ModelPayloadEncryption.Decrypt(
            encrypted.Ciphertext, licenseKey, encrypted.Salt, encrypted.Nonce, encrypted.Tag, aad);

        Assert.Empty(decrypted);
    }

    [Fact]
    public void EncryptDecrypt_LargePayload_RoundTrip()
    {
        var plaintext = new byte[1024 * 1024]; // 1 MB
        new Random(42).NextBytes(plaintext);
        var licenseKey = "large-payload-key-98765";
        var aad = ModelPayloadEncryption.BuildAad("LargeModel", new[] { 3, 224, 224 }, new[] { 1000 });

        var encrypted = ModelPayloadEncryption.Encrypt(plaintext, licenseKey, aad);
        var decrypted = ModelPayloadEncryption.Decrypt(
            encrypted.Ciphertext, licenseKey, encrypted.Salt, encrypted.Nonce, encrypted.Tag, aad);

        Assert.Equal(plaintext, decrypted);
    }

    [Fact]
    public void Decrypt_WrongKey_ThrowsCryptographicException()
    {
        var plaintext = new byte[] { 1, 2, 3, 4, 5 };
        var correctKey = "correct-key-abc";
        var wrongKey = "wrong-key-xyz";
        var aad = ModelPayloadEncryption.BuildAad("TestModel", new[] { 5 }, new[] { 1 });

        var encrypted = ModelPayloadEncryption.Encrypt(plaintext, correctKey, aad);

        Assert.ThrowsAny<CryptographicException>(() =>
            ModelPayloadEncryption.Decrypt(
                encrypted.Ciphertext, wrongKey, encrypted.Salt, encrypted.Nonce, encrypted.Tag, aad));
    }

    [Fact]
    public void Decrypt_WrongAad_ThrowsCryptographicException()
    {
        var plaintext = new byte[] { 10, 20, 30 };
        var licenseKey = "aad-test-key";
        var correctAad = ModelPayloadEncryption.BuildAad("CorrectModel", new[] { 100 }, new[] { 10 });
        var wrongAad = ModelPayloadEncryption.BuildAad("WrongModel", new[] { 200 }, new[] { 20 });

        var encrypted = ModelPayloadEncryption.Encrypt(plaintext, licenseKey, correctAad);

        Assert.ThrowsAny<CryptographicException>(() =>
            ModelPayloadEncryption.Decrypt(
                encrypted.Ciphertext, licenseKey, encrypted.Salt, encrypted.Nonce, encrypted.Tag, wrongAad));
    }

    [Fact]
    public void Encrypt_NullPayload_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelPayloadEncryption.Encrypt(null, "key", "aad"));
    }

    [Fact]
    public void Encrypt_NullKey_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            ModelPayloadEncryption.Encrypt(new byte[] { 1 }, null, "aad"));
    }

    [Fact]
    public void Encrypt_EmptyKey_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            ModelPayloadEncryption.Encrypt(new byte[] { 1 }, "", "aad"));
    }

    [Fact]
    public void Decrypt_NullCiphertext_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelPayloadEncryption.Decrypt(null, "key", new byte[16], new byte[12], new byte[16], "aad"));
    }

    [Fact]
    public void Decrypt_NullKey_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            ModelPayloadEncryption.Decrypt(new byte[] { 1 }, null, new byte[16], new byte[12], new byte[16], "aad"));
    }

    [Fact]
    public void Decrypt_NullSalt_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelPayloadEncryption.Decrypt(new byte[] { 1 }, "key", null, new byte[12], new byte[16], "aad"));
    }

    [Fact]
    public void Decrypt_NullNonce_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelPayloadEncryption.Decrypt(new byte[] { 1 }, "key", new byte[16], null, new byte[16], "aad"));
    }

    [Fact]
    public void Decrypt_NullTag_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelPayloadEncryption.Decrypt(new byte[] { 1 }, "key", new byte[16], new byte[12], null, "aad"));
    }

    [Fact]
    public void Encrypt_ProducesDifferentSaltAndNonce_EachCall()
    {
        var plaintext = new byte[] { 1, 2, 3 };
        var licenseKey = "determinism-test-key";
        var aad = "test-aad";

        var encrypted1 = ModelPayloadEncryption.Encrypt(plaintext, licenseKey, aad);
        var encrypted2 = ModelPayloadEncryption.Encrypt(plaintext, licenseKey, aad);

        // Salt and nonce should be random each time
        Assert.NotEqual(encrypted1.Salt, encrypted2.Salt);
        Assert.NotEqual(encrypted1.Nonce, encrypted2.Nonce);
    }

    [Fact]
    public void BuildAad_ProducesDeterministicString()
    {
        var aad1 = ModelPayloadEncryption.BuildAad("MyModel", new[] { 3, 224, 224 }, new[] { 1000 });
        var aad2 = ModelPayloadEncryption.BuildAad("MyModel", new[] { 3, 224, 224 }, new[] { 1000 });

        Assert.Equal(aad1, aad2);
        Assert.Equal("MyModel|3,224,224|1000", aad1);
    }

    [Fact]
    public void BuildAad_EmptyShapes()
    {
        var aad = ModelPayloadEncryption.BuildAad("Model", Array.Empty<int>(), Array.Empty<int>());
        Assert.Equal("Model||", aad);
    }

    [Fact]
    public void BuildAad_NullInputs()
    {
        var aad = ModelPayloadEncryption.BuildAad(null, null, null);
        Assert.Equal("||", aad);
    }
#endif

    // Header encryption round-trip tests (these work on all targets because
    // they don't use actual AES-GCM, they just test the header format)

    [Fact]
    public void WrapWithHeaderEncrypted_ReadHeader_RoundTrip()
    {
        var model = new StubModelSerializer
        {
            Payload = new byte[] { 0xCA, 0xFE },
            InputShapeValue = new[] { 784 },
            OutputShapeValue = new[] { 10 }
        };

        var fakeSalt = new byte[16];
        var fakeNonce = new byte[12];
        var fakeTag = new byte[16];
        new Random(42).NextBytes(fakeSalt);
        new Random(43).NextBytes(fakeNonce);
        new Random(44).NextBytes(fakeTag);

        var fakeCiphertext = new byte[] { 0xAA, 0xBB };

        byte[] wrapped = ModelFileHeader.WrapWithHeaderEncrypted(
            fakeCiphertext, model,
            model.GetInputShape(), model.GetOutputShape(),
            SerializationFormat.Binary,
            fakeSalt, fakeNonce, fakeTag);

        Assert.True(ModelFileHeader.HasHeader(wrapped));

        var info = ModelFileHeader.ReadHeader(wrapped);

        Assert.Equal(ModelFileHeader.CurrentEnvelopeVersion, info.EnvelopeVersion);
        Assert.Equal(SerializationFormat.Binary, info.Format);
        Assert.True(info.IsEncrypted);
        Assert.Equal(PayloadEncryptionScheme.AesGcm256, info.EncryptionScheme);
        Assert.Equal(fakeSalt, info.Salt);
        Assert.Equal(fakeNonce, info.Nonce);
        Assert.Equal(fakeTag, info.Tag);
        Assert.Equal(new[] { 784 }, info.InputShape);
        Assert.Equal(new[] { 10 }, info.OutputShape);
        Assert.Equal(fakeCiphertext.Length, info.PayloadLength);

        // Verify payload extraction returns the ciphertext
        byte[] extractedPayload = ModelFileHeader.ExtractPayload(wrapped, info);
        Assert.Equal(fakeCiphertext, extractedPayload);
    }

    [Fact]
    public void WrapWithHeader_Unencrypted_HasEncryptionNone()
    {
        var model = new StubModelSerializer { Payload = new byte[] { 1, 2, 3 } };

        byte[] wrapped = ModelFileHeader.WrapWithHeader(
            model.Payload, model, Array.Empty<int>(), Array.Empty<int>(), SerializationFormat.Binary);

        var info = ModelFileHeader.ReadHeader(wrapped);

        Assert.False(info.IsEncrypted);
        Assert.Equal(PayloadEncryptionScheme.None, info.EncryptionScheme);
        Assert.Null(info.Salt);
        Assert.Null(info.Nonce);
        Assert.Null(info.Tag);
    }

    [Fact]
    public void WrapWithHeaderEncrypted_ThrowsOnNullSalt()
    {
        var model = new StubModelSerializer { Payload = new byte[] { 1 } };
        Assert.Throws<ArgumentNullException>(() =>
            ModelFileHeader.WrapWithHeaderEncrypted(
                new byte[] { 1 }, model, Array.Empty<int>(), Array.Empty<int>(),
                SerializationFormat.Binary, null, new byte[12], new byte[16]));
    }

    [Fact]
    public void WrapWithHeaderEncrypted_ThrowsOnNullNonce()
    {
        var model = new StubModelSerializer { Payload = new byte[] { 1 } };
        Assert.Throws<ArgumentNullException>(() =>
            ModelFileHeader.WrapWithHeaderEncrypted(
                new byte[] { 1 }, model, Array.Empty<int>(), Array.Empty<int>(),
                SerializationFormat.Binary, new byte[16], null, new byte[16]));
    }

    [Fact]
    public void WrapWithHeaderEncrypted_ThrowsOnNullTag()
    {
        var model = new StubModelSerializer { Payload = new byte[] { 1 } };
        Assert.Throws<ArgumentNullException>(() =>
            ModelFileHeader.WrapWithHeaderEncrypted(
                new byte[] { 1 }, model, Array.Empty<int>(), Array.Empty<int>(),
                SerializationFormat.Binary, new byte[16], new byte[12], null));
    }

    [Fact]
    public void Inspect_EncryptedFile_ShowsIsEncrypted()
    {
        var model = new StubModelSerializer
        {
            Payload = new byte[] { 0x01 },
            InputShapeValue = new[] { 28, 28 },
            OutputShapeValue = new[] { 10 }
        };

        var tempFile = Path.Combine(Path.GetTempPath(), $"aimf_enc_inspect_{Guid.NewGuid():N}.bin");
        try
        {
            var fakeSalt = new byte[16];
            var fakeNonce = new byte[12];
            var fakeTag = new byte[16];

            byte[] wrapped = ModelFileHeader.WrapWithHeaderEncrypted(
                new byte[] { 0xFF }, model,
                model.GetInputShape(), model.GetOutputShape(),
                SerializationFormat.Binary,
                fakeSalt, fakeNonce, fakeTag);

            File.WriteAllBytes(tempFile, wrapped);

            var info = ModelLoader.Inspect(tempFile);

            Assert.True(info.IsEncrypted);
            Assert.Equal(PayloadEncryptionScheme.AesGcm256, info.EncryptionScheme);
            Assert.Equal(new[] { 28, 28 }, info.InputShape);
            Assert.Equal(new[] { 10 }, info.OutputShape);
        }
        finally
        {
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

#if !NET471
    [Fact]
    public void SaveEncrypted_Load_WithCorrectKey_RoundTrip()
    {
        // Register the stub so the registry can resolve it
        ModelTypeRegistry.Register(typeof(StubModelSerializer).Name, typeof(StubModelSerializer));

        var tempFile = Path.Combine(Path.GetTempPath(), $"aimf_enc_rt_{Guid.NewGuid():N}.bin");
        try
        {
            var payload = new byte[] { 0xCA, 0xFE, 0xBA, 0xBE, 0x42 };
            var model = new StubModelSerializer
            {
                Payload = payload,
                InputShapeValue = new[] { 5 },
                OutputShapeValue = new[] { 1 }
            };
            var licenseKey = "my-super-secret-license-key-2025";

            ModelLoader.SaveEncrypted(
                model, tempFile, licenseKey,
                model.GetInputShape(), model.GetOutputShape());

            // Verify file is encrypted
            var info = ModelLoader.Inspect(tempFile);
            Assert.True(info.IsEncrypted);
            Assert.Equal(typeof(StubModelSerializer).Name, info.TypeName);

            // Load with correct key
            var loaded = ModelLoader.Load<double>(tempFile, licenseKey);
            Assert.NotNull(loaded);
            Assert.IsType<StubModelSerializer>(loaded);

            var loadedStub = (StubModelSerializer)loaded;
            Assert.Equal(payload, loadedStub.GetDeserializedData());
        }
        finally
        {
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    [Fact]
    public void Load_Encrypted_WithoutKey_ThrowsInvalidOperation()
    {
        ModelTypeRegistry.Register(typeof(StubModelSerializer).Name, typeof(StubModelSerializer));

        var tempFile = Path.Combine(Path.GetTempPath(), $"aimf_enc_nokey_{Guid.NewGuid():N}.bin");
        try
        {
            var model = new StubModelSerializer
            {
                Payload = new byte[] { 1, 2, 3 },
                InputShapeValue = new[] { 3 },
                OutputShapeValue = new[] { 1 }
            };

            ModelLoader.SaveEncrypted(
                model, tempFile, "secret-key",
                model.GetInputShape(), model.GetOutputShape());

            var ex = Assert.Throws<InvalidOperationException>(() =>
                ModelLoader.Load<double>(tempFile));

            Assert.Contains("encrypted", ex.Message, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("license key", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    [Fact]
    public void Load_Encrypted_WithWrongKey_ThrowsCryptographicException()
    {
        ModelTypeRegistry.Register(typeof(StubModelSerializer).Name, typeof(StubModelSerializer));

        var tempFile = Path.Combine(Path.GetTempPath(), $"aimf_enc_badkey_{Guid.NewGuid():N}.bin");
        try
        {
            var model = new StubModelSerializer
            {
                Payload = new byte[] { 10, 20, 30, 40 },
                InputShapeValue = new[] { 4 },
                OutputShapeValue = new[] { 1 }
            };

            ModelLoader.SaveEncrypted(
                model, tempFile, "correct-key",
                model.GetInputShape(), model.GetOutputShape());

            Assert.ThrowsAny<CryptographicException>(() =>
                ModelLoader.Load<double>(tempFile, "wrong-key"));
        }
        finally
        {
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    [Fact]
    public void Load_Unencrypted_WithKey_StillWorks()
    {
        // Providing a key for an unencrypted file should just work (key is ignored)
        ModelTypeRegistry.Register(typeof(StubModelSerializer).Name, typeof(StubModelSerializer));

        var payload = new byte[] { 0x11, 0x22 };
        var model = new StubModelSerializer { Payload = payload };

        byte[] wrapped = ModelFileHeader.WrapWithHeader(
            payload, model, Array.Empty<int>(), Array.Empty<int>(), SerializationFormat.Binary);

        var loaded = ModelLoader.LoadFromBytes<double>(wrapped, "unnecessary-key");
        Assert.NotNull(loaded);
        Assert.IsType<StubModelSerializer>(loaded);
        Assert.Equal(payload, ((StubModelSerializer)loaded).GetDeserializedData());
    }

    [Fact]
    public void SaveEncrypted_ThrowsOnNullModel()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelLoader.SaveEncrypted(null, "file.aimf", "key", Array.Empty<int>(), Array.Empty<int>()));
    }

    [Fact]
    public void SaveEncrypted_ThrowsOnEmptyPath()
    {
        var model = new StubModelSerializer { Payload = new byte[] { 1 } };
        Assert.Throws<ArgumentException>(() =>
            ModelLoader.SaveEncrypted(model, "", "key", Array.Empty<int>(), Array.Empty<int>()));
    }

    [Fact]
    public void SaveEncrypted_ThrowsOnEmptyKey()
    {
        var model = new StubModelSerializer { Payload = new byte[] { 1 } };
        Assert.Throws<ArgumentException>(() =>
            ModelLoader.SaveEncrypted(model, "file.aimf", "", Array.Empty<int>(), Array.Empty<int>()));
    }

    [Fact]
    public void Encrypted_File_Does_Not_Contain_Plaintext_Payload()
    {
        // Proves that the raw bytes on disk do NOT contain the original model weights
        var tempFile = Path.Combine(Path.GetTempPath(), $"aimf_no_plaintext_{Guid.NewGuid():N}.bin");
        try
        {
            // Use a known repeating pattern that would be easy to find if leaked
            var payload = new byte[256];
            for (int i = 0; i < payload.Length; i++)
            {
                payload[i] = (byte)(i % 256);
            }

            var model = new StubModelSerializer
            {
                Payload = payload,
                InputShapeValue = new[] { 256 },
                OutputShapeValue = new[] { 1 }
            };

            ModelLoader.SaveEncrypted(model, tempFile, "license-key-xyz",
                model.GetInputShape(), model.GetOutputShape());

            // Read raw file bytes
            byte[] fileBytes = File.ReadAllBytes(tempFile);

            // The file should NOT contain any 32+ byte substring from the payload
            // (Header will contain some small values like shape ints, but not the bulk payload)
            var info = ModelFileHeader.ReadHeader(fileBytes);
            byte[] rawPayloadOnDisk = ModelFileHeader.ExtractPayload(fileBytes, info);

            // The on-disk payload must NOT equal the original plaintext
            Assert.NotEqual(payload, rawPayloadOnDisk);

            // Verify that no contiguous 16-byte window from plaintext appears in the file payload
            for (int start = 0; start <= payload.Length - 16; start++)
            {
                var window = new byte[16];
                Array.Copy(payload, start, window, 0, 16);

                bool found = ContainsSubsequence(rawPayloadOnDisk, window);
                Assert.False(found,
                    $"Encrypted file contains 16-byte plaintext window starting at offset {start}. " +
                    "Encryption is not hiding model weights.");
            }
        }
        finally
        {
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    [Fact]
    public void Encrypted_Payload_Cannot_Be_Deserialized_Directly()
    {
        // Proves that if someone extracts the raw payload from an encrypted file,
        // they cannot directly deserialize it as a model
        var payload = Encoding.UTF8.GetBytes("this-is-secret-model-data-12345");
        var model = new StubModelSerializer
        {
            Payload = payload,
            InputShapeValue = new[] { 30 },
            OutputShapeValue = new[] { 1 }
        };

        var licenseKey = "my-secret-key";
        var aad = ModelPayloadEncryption.BuildAad(
            typeof(StubModelSerializer).Name,
            model.GetInputShape(),
            model.GetOutputShape());

        var encrypted = ModelPayloadEncryption.Encrypt(payload, licenseKey, aad);

        // The encrypted ciphertext should NOT contain the original ASCII payload
        string ciphertextAsString = Encoding.UTF8.GetString(encrypted.Ciphertext);
        Assert.DoesNotContain("this-is-secret-model-data", ciphertextAsString);

        // Attempting to "deserialize" the encrypted bytes directly should yield garbage
        // (the stub just stores bytes, but real models would fail to parse)
        var fakeModel = new StubModelSerializer();
        fakeModel.Deserialize(encrypted.Ciphertext);
        byte[] rawData = fakeModel.GetDeserializedData();

        // The deserialized data should NOT match the original payload
        Assert.NotEqual(payload, rawData);
    }

    [Fact]
    public void Tampering_With_Encrypted_Payload_Causes_Auth_Failure()
    {
        // Proves GCM authentication catches any modification
        var payload = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var licenseKey = "tamper-test-key";
        var aad = "tamper-test-aad";

        var encrypted = ModelPayloadEncryption.Encrypt(payload, licenseKey, aad);

        // Tamper with a single byte of ciphertext
        var tampered = new byte[encrypted.Ciphertext.Length];
        Array.Copy(encrypted.Ciphertext, tampered, tampered.Length);
        tampered[0] ^= 0xFF; // Flip all bits of first byte

        Assert.ThrowsAny<CryptographicException>(() =>
            ModelPayloadEncryption.Decrypt(tampered, licenseKey, encrypted.Salt, encrypted.Nonce, encrypted.Tag, aad));
    }

    [Fact]
    public void Different_Keys_Produce_Different_Ciphertexts()
    {
        // Even with the same plaintext, different keys should produce different ciphertexts.
        // Note: salt and nonce are independently generated per Encrypt call.
        var payload = new byte[] { 42, 42, 42, 42, 42, 42, 42, 42 };
        var aad = "key-difference-test";

        var enc1 = ModelPayloadEncryption.Encrypt(payload, "key-alpha", aad);
        var enc2 = ModelPayloadEncryption.Encrypt(payload, "key-beta", aad);

        // Salt and nonce are random, so ciphertexts will differ anyway,
        // but verify they're actually different
        Assert.NotEqual(enc1.Ciphertext, enc2.Ciphertext);

        // Cross-decryption must fail: key1's ciphertext with key2 should fail
        Assert.ThrowsAny<CryptographicException>(() =>
            ModelPayloadEncryption.Decrypt(enc1.Ciphertext, "key-beta", enc1.Salt, enc1.Nonce, enc1.Tag, aad));
    }

    [Fact]
    public void Full_File_RoundTrip_With_Shape_And_Type_Preservation()
    {
        // Full integration: save encrypted, inspect (no key needed), load (key needed)
        ModelTypeRegistry.Register(typeof(StubModelSerializer).Name, typeof(StubModelSerializer));

        var tempFile = Path.Combine(Path.GetTempPath(), $"aimf_full_rt_{Guid.NewGuid():N}.bin");
        try
        {
            var payload = new byte[1024];
            new Random(99).NextBytes(payload);
            var model = new StubModelSerializer
            {
                Payload = payload,
                InputShapeValue = new[] { 3, 32, 32 },
                OutputShapeValue = new[] { 100 }
            };
            var key = "production-license-key-abc123";

            // Save encrypted
            ModelLoader.SaveEncrypted(model, tempFile, key,
                model.GetInputShape(), model.GetOutputShape(),
                SerializationFormat.Binary);

            // Inspect without key - should work, shows metadata
            var info = ModelLoader.Inspect(tempFile);
            Assert.True(info.IsEncrypted);
            Assert.Equal(typeof(StubModelSerializer).Name, info.TypeName);
            Assert.Equal(new[] { 3, 32, 32 }, info.InputShape);
            Assert.Equal(new[] { 100 }, info.OutputShape);
            Assert.Equal(SerializationFormat.Binary, info.Format);

            // Load without key - should fail with clear message
            var ex = Assert.Throws<InvalidOperationException>(() =>
                ModelLoader.Load<double>(tempFile));
            Assert.Contains("encrypted", ex.Message, StringComparison.OrdinalIgnoreCase);

            // Load with wrong key - should fail with crypto error
            Assert.ThrowsAny<CryptographicException>(() =>
                ModelLoader.Load<double>(tempFile, "wrong-key"));

            // Load with correct key - should succeed
            var loaded = ModelLoader.Load<double>(tempFile, key);
            Assert.IsType<StubModelSerializer>(loaded);
            Assert.Equal(payload, ((StubModelSerializer)loaded).GetDeserializedData());
        }
        finally
        {
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    private static bool ContainsSubsequence(byte[] haystack, byte[] needle)
    {
        if (needle.Length > haystack.Length)
        {
            return false;
        }

        for (int i = 0; i <= haystack.Length - needle.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < needle.Length; j++)
            {
                if (haystack[i + j] != needle[j])
                {
                    match = false;
                    break;
                }
            }

            if (match)
            {
                return true;
            }
        }

        return false;
    }
#endif
}
