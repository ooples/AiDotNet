using AiDotNet.FederatedLearning.TEE;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Comprehensive integration tests for Trusted Execution Environment (#537).
/// </summary>
public class TeeIntegrationTests
{
    // ========== SimulatedTeeProvider Lifecycle Tests ==========

    [Fact]
    public void SimulatedProvider_Initialize_SetsIsInitialized()
    {
        var provider = new SimulatedTeeProvider<double>();
        var options = new TeeOptions { SimulationMode = true };

        provider.Initialize(options);

        Assert.True(provider.IsInitialized);
    }

    [Fact]
    public void SimulatedProvider_NotInitialized_ThrowsOnSealData()
    {
        var provider = new SimulatedTeeProvider<double>();
        var data = new byte[] { 1, 2, 3, 4 };

        Assert.Throws<InvalidOperationException>(() => provider.SealData(data));
    }

    [Fact]
    public void SimulatedProvider_NotInitialized_ThrowsOnUnsealData()
    {
        var provider = new SimulatedTeeProvider<double>();
        var data = new byte[32];

        Assert.Throws<InvalidOperationException>(() => provider.UnsealData(data));
    }

    [Fact]
    public void SimulatedProvider_NotInitialized_ThrowsOnGetMeasurementHash()
    {
        var provider = new SimulatedTeeProvider<double>();

        Assert.Throws<InvalidOperationException>(() => provider.GetMeasurementHash());
    }

    [Fact]
    public void SimulatedProvider_NotInitialized_ThrowsOnGenerateAttestationQuote()
    {
        var provider = new SimulatedTeeProvider<double>();

        Assert.Throws<InvalidOperationException>(() => provider.GenerateAttestationQuote(new byte[] { 1 }));
    }

    [Fact]
    public void SimulatedProvider_Initialize_NullOptions_Throws()
    {
        var provider = new SimulatedTeeProvider<double>();

        Assert.Throws<ArgumentNullException>(() => provider.Initialize(null));
    }

    [Fact]
    public void SimulatedProvider_Destroy_ResetsState()
    {
        var provider = new SimulatedTeeProvider<double>();
        provider.Initialize(new TeeOptions { SimulationMode = true });
        Assert.True(provider.IsInitialized);

        provider.Destroy();

        Assert.False(provider.IsInitialized);
        Assert.Throws<InvalidOperationException>(() => provider.SealData(new byte[] { 1 }));
    }

    [Fact]
    public void SimulatedProvider_ProviderType_IsSimulated()
    {
        var provider = new SimulatedTeeProvider<double>();

        Assert.Equal(TeeProviderType.Simulated, provider.ProviderType);
    }

    // ========== Data Sealing/Unsealing Tests ==========

    [Fact]
    public void SimulatedProvider_SealUnseal_RoundTrips()
    {
        var provider = new SimulatedTeeProvider<double>();
        provider.Initialize(new TeeOptions { SimulationMode = true });
        var plaintext = new byte[] { 10, 20, 30, 40, 50, 60, 70, 80 };

        var sealed_ = provider.SealData(plaintext);
        var unsealed = provider.UnsealData(sealed_);

        Assert.Equal(plaintext, unsealed);
    }

    [Fact]
    public void SimulatedProvider_SealData_ProducesDifferentOutput()
    {
        var provider = new SimulatedTeeProvider<double>();
        provider.Initialize(new TeeOptions { SimulationMode = true });
        var plaintext = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

        var sealed_ = provider.SealData(plaintext);

        // Encrypted data should differ from plaintext
        Assert.NotEqual(plaintext, sealed_);
        // Encrypted data should be longer due to IV/tag
        Assert.True(sealed_.Length > plaintext.Length);
    }

    [Fact]
    public void SimulatedProvider_SealData_NullPlaintext_Throws()
    {
        var provider = new SimulatedTeeProvider<double>();
        provider.Initialize(new TeeOptions { SimulationMode = true });

        Assert.Throws<ArgumentException>(() => provider.SealData(null));
    }

    [Fact]
    public void SimulatedProvider_SealData_EmptyPlaintext_Throws()
    {
        var provider = new SimulatedTeeProvider<double>();
        provider.Initialize(new TeeOptions { SimulationMode = true });

        Assert.Throws<ArgumentException>(() => provider.SealData(Array.Empty<byte>()));
    }

    [Fact]
    public void SimulatedProvider_UnsealData_TooShort_Throws()
    {
        var provider = new SimulatedTeeProvider<double>();
        provider.Initialize(new TeeOptions { SimulationMode = true });

        Assert.Throws<ArgumentException>(() => provider.UnsealData(new byte[] { 1, 2, 3 }));
    }

    [Fact]
    public void SimulatedProvider_SealUnseal_LargePayload()
    {
        var provider = new SimulatedTeeProvider<double>();
        provider.Initialize(new TeeOptions { SimulationMode = true });
        var plaintext = new byte[10000];
        for (int i = 0; i < plaintext.Length; i++)
        {
            plaintext[i] = (byte)(i % 256);
        }

        var sealed_ = provider.SealData(plaintext);
        var unsealed = provider.UnsealData(sealed_);

        Assert.Equal(plaintext, unsealed);
    }

    // ========== Attestation Tests ==========

    [Fact]
    public void SimulatedProvider_GetMeasurementHash_ReturnsNonEmpty()
    {
        var provider = new SimulatedTeeProvider<double>();
        provider.Initialize(new TeeOptions { SimulationMode = true });

        string hash = provider.GetMeasurementHash();

        Assert.False(string.IsNullOrEmpty(hash));
    }

    [Fact]
    public void SimulatedProvider_MeasurementHash_IsConsistent()
    {
        var provider = new SimulatedTeeProvider<double>();
        provider.Initialize(new TeeOptions { SimulationMode = true });

        string hash1 = provider.GetMeasurementHash();
        string hash2 = provider.GetMeasurementHash();

        Assert.Equal(hash1, hash2);
    }

    [Fact]
    public void SimulatedProvider_GenerateAttestationQuote_ReturnsNonEmpty()
    {
        var provider = new SimulatedTeeProvider<double>();
        provider.Initialize(new TeeOptions { SimulationMode = true });
        var reportData = new byte[] { 1, 2, 3, 4 };

        byte[] quote = provider.GenerateAttestationQuote(reportData);

        Assert.NotNull(quote);
        Assert.True(quote.Length > 0);
    }

    [Fact]
    public void SimulatedProvider_GenerateAttestationQuote_NullReportData_Throws()
    {
        var provider = new SimulatedTeeProvider<double>();
        provider.Initialize(new TeeOptions { SimulationMode = true });

        Assert.Throws<ArgumentNullException>(() => provider.GenerateAttestationQuote(null));
    }

    [Fact]
    public void SimulatedProvider_GetMaxEnclaveMemory_ReturnsPositive()
    {
        var provider = new SimulatedTeeProvider<double>();

        long maxMemory = provider.GetMaxEnclaveMemory();

        Assert.True(maxMemory > 0);
    }

    // ========== TeeAesHelper Tests ==========

    [Fact]
    public void TeeAesHelper_EncryptDecrypt_RoundTrips()
    {
        var key = new byte[32]; // AES-256 key
        for (int i = 0; i < 32; i++) key[i] = (byte)i;
        var plaintext = new byte[] { 100, 200, 50, 75, 125, 250, 10, 20 };

        var encrypted = TeeAesHelper.Encrypt(key, plaintext);
        var decrypted = TeeAesHelper.Decrypt(key, encrypted);

        Assert.Equal(plaintext, decrypted);
    }

    [Fact]
    public void TeeAesHelper_Encrypt_DifferentFromPlaintext()
    {
        var key = new byte[32];
        for (int i = 0; i < 32; i++) key[i] = (byte)(i + 1);
        var plaintext = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

        var encrypted = TeeAesHelper.Encrypt(key, plaintext);

        Assert.NotEqual(plaintext, encrypted);
    }

    [Fact]
    public void TeeAesHelper_DecryptWithWrongKey_Fails()
    {
        var key1 = new byte[32];
        var key2 = new byte[32];
        for (int i = 0; i < 32; i++)
        {
            key1[i] = (byte)i;
            key2[i] = (byte)(i + 1);
        }
        var plaintext = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

        var encrypted = TeeAesHelper.Encrypt(key1, plaintext);

        // Decrypting with wrong key should throw or return different data
        Assert.ThrowsAny<Exception>(() => TeeAesHelper.Decrypt(key2, encrypted));
    }

    // ========== TeeSecureAggregation Tests ==========

    [Fact]
    public void TeeSecureAggregation_Constructor_NullProvider_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new TeeSecureAggregation<double>(null, new TeeOptions()));
    }

    [Fact]
    public void TeeSecureAggregation_Constructor_NullOptions_Throws()
    {
        var provider = new SimulatedTeeProvider<double>();

        Assert.Throws<ArgumentNullException>(() =>
            new TeeSecureAggregation<double>(provider, null));
    }

    [Fact]
    public void TeeSecureAggregation_BeginRound_Succeeds()
    {
        var provider = new SimulatedTeeProvider<double>();
        provider.Initialize(new TeeOptions { SimulationMode = true });
        var teeAgg = new TeeSecureAggregation<double>(provider, new TeeOptions { SimulationMode = true });

        // Should not throw
        teeAgg.BeginRound(roundNumber: 1, expectedClients: 3);
    }

    [Fact]
    public void TeeSecureAggregation_GenerateSessionKey_ReturnsNonEmpty()
    {
        var provider = new SimulatedTeeProvider<double>();
        provider.Initialize(new TeeOptions { SimulationMode = true });
        var teeAgg = new TeeSecureAggregation<double>(provider, new TeeOptions { SimulationMode = true });

        byte[] key = teeAgg.GenerateSessionKey();

        Assert.NotNull(key);
        Assert.True(key.Length > 0);
    }

    [Fact]
    public void TeeSecureAggregation_EncryptForSubmission_ReturnsNonNull()
    {
        var provider = new SimulatedTeeProvider<double>();
        provider.Initialize(new TeeOptions { SimulationMode = true });
        var teeAgg = new TeeSecureAggregation<double>(provider, new TeeOptions { SimulationMode = true });
        var gradient = CreateTensor(1.0, 2.0, 3.0, 4.0, 5.0);

        byte[] encrypted = teeAgg.EncryptForSubmission(gradient);

        Assert.NotNull(encrypted);
        Assert.True(encrypted.Length > 0);
    }

    [Fact]
    public void TeeSecureAggregation_SubmitAndAggregate_ProducesResult()
    {
        var provider = new SimulatedTeeProvider<double>();
        provider.Initialize(new TeeOptions { SimulationMode = true });
        var teeAgg = new TeeSecureAggregation<double>(provider, new TeeOptions { SimulationMode = true });

        teeAgg.BeginRound(roundNumber: 1, expectedClients: 2);

        var gradient1 = CreateTensor(1.0, 2.0, 3.0);
        var gradient2 = CreateTensor(3.0, 4.0, 5.0);

        var encrypted1 = teeAgg.EncryptForSubmission(gradient1);
        var encrypted2 = teeAgg.EncryptForSubmission(gradient2);

        teeAgg.SubmitEncryptedUpdate(clientId: 0, encrypted1, weight: 0.5);
        teeAgg.SubmitEncryptedUpdate(clientId: 1, encrypted2, weight: 0.5);

        var aggregated = teeAgg.Aggregate();

        Assert.NotNull(aggregated);
    }

    [Fact]
    public void TeeSecureAggregation_GetAttestationQuote_ReturnsNonEmpty()
    {
        var provider = new SimulatedTeeProvider<double>();
        provider.Initialize(new TeeOptions { SimulationMode = true });
        var teeAgg = new TeeSecureAggregation<double>(provider, new TeeOptions { SimulationMode = true });

        byte[] quote = teeAgg.GetAttestationQuote(new byte[] { 1, 2, 3 });

        Assert.NotNull(quote);
        Assert.True(quote.Length > 0);
    }

    // ========== Hardware Provider Type Tests ==========

    [Fact]
    public void IntelSgxProvider_ProviderType_IsSgx()
    {
        var provider = new IntelSgxTeeProvider<double>();
        Assert.Equal(TeeProviderType.Sgx, provider.ProviderType);
    }

    [Fact]
    public void IntelTdxProvider_ProviderType_IsTdx()
    {
        var provider = new IntelTdxTeeProvider<double>();
        Assert.Equal(TeeProviderType.Tdx, provider.ProviderType);
    }

    [Fact]
    public void AmdSevSnpProvider_ProviderType_IsSevSnp()
    {
        var provider = new AmdSevSnpTeeProvider<double>();
        Assert.Equal(TeeProviderType.SevSnp, provider.ProviderType);
    }

    [Fact]
    public void ArmCcaProvider_ProviderType_IsCca()
    {
        var provider = new ArmCcaTeeProvider<double>();
        Assert.Equal(TeeProviderType.ArmCca, provider.ProviderType);
    }

    [Theory]
    [InlineData(typeof(IntelSgxTeeProvider<double>))]
    [InlineData(typeof(IntelTdxTeeProvider<double>))]
    [InlineData(typeof(AmdSevSnpTeeProvider<double>))]
    [InlineData(typeof(ArmCcaTeeProvider<double>))]
    public void AllProviders_CanInitializeAndSealUnseal(Type providerType)
    {
        var provider = (TeeProviderBase<double>)Activator.CreateInstance(providerType);
        provider.Initialize(new TeeOptions { SimulationMode = true });

        var plaintext = new byte[] { 10, 20, 30, 40, 50, 60, 70, 80 };
        var sealed_ = provider.SealData(plaintext);
        var unsealed = provider.UnsealData(sealed_);

        Assert.Equal(plaintext, unsealed);
    }

    [Theory]
    [InlineData(typeof(IntelSgxTeeProvider<double>))]
    [InlineData(typeof(IntelTdxTeeProvider<double>))]
    [InlineData(typeof(AmdSevSnpTeeProvider<double>))]
    [InlineData(typeof(ArmCcaTeeProvider<double>))]
    public void AllProviders_MaxEnclaveMemory_IsPositive(Type providerType)
    {
        var provider = (TeeProviderBase<double>)Activator.CreateInstance(providerType);

        long maxMem = provider.GetMaxEnclaveMemory();

        Assert.True(maxMem > 0, $"{providerType.Name} should report positive max memory");
    }

    // ========== TeeOptions Defaults Tests ==========

    [Fact]
    public void TeeOptions_DefaultValues()
    {
        var options = new TeeOptions();

        Assert.Equal(TeeProviderType.Simulated, options.Provider);
        Assert.Equal(AttestationPolicy.Strict, options.Policy);
        Assert.NotNull(options.Attestation);
        Assert.Equal(256, options.MaxEnclaveMemoryMb);
        Assert.True(options.SimulationMode);
        Assert.True(options.RequireAttestation);
        Assert.Equal(string.Empty, options.ExpectedMeasurement);
    }

    [Fact]
    public void TeeProviderType_HasAllExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(TeeProviderType), TeeProviderType.Simulated));
        Assert.True(Enum.IsDefined(typeof(TeeProviderType), TeeProviderType.Sgx));
        Assert.True(Enum.IsDefined(typeof(TeeProviderType), TeeProviderType.Tdx));
        Assert.True(Enum.IsDefined(typeof(TeeProviderType), TeeProviderType.SevSnp));
        Assert.True(Enum.IsDefined(typeof(TeeProviderType), TeeProviderType.ArmCca));
    }

    // ========== Helper ==========

    private static Tensor<double> CreateTensor(params double[] values)
    {
        var tensor = new Tensor<double>(new[] { values.Length });
        for (int i = 0; i < values.Length; i++)
        {
            tensor[i] = values[i];
        }

        return tensor;
    }
}
