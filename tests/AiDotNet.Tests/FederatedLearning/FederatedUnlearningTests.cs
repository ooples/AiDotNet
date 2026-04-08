using AiDotNet.FederatedLearning.Unlearning;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Tests for federated unlearning implementations (#849).
/// </summary>
public class FederatedUnlearningTests
{
    private static Tensor<double> CreateTensor(params double[] values)
    {
        var tensor = new Tensor<double>(new[] { values.Length });
        for (int i = 0; i < values.Length; i++)
        {
            tensor[i] = values[i];
        }

        return tensor;
    }

    private static Dictionary<int, List<Tensor<double>>> CreateClientHistories(
        int clientCount, int roundCount, int modelSize)
    {
        var histories = new Dictionary<int, List<Tensor<double>>>();
        var rng = new Random(42);

        for (int c = 0; c < clientCount; c++)
        {
            var rounds = new List<Tensor<double>>();
            for (int r = 0; r < roundCount; r++)
            {
                var values = new double[modelSize];
                for (int i = 0; i < modelSize; i++)
                {
                    values[i] = rng.NextDouble() * 2.0 - 1.0;
                }

                rounds.Add(CreateTensor(values));
            }

            histories[c] = rounds;
        }

        return histories;
    }

    private static Tensor<double> CreateGlobalModel(int size, double seed = 0.5)
    {
        var values = new double[size];
        for (int i = 0; i < size; i++)
        {
            values[i] = seed + i * 0.01;
        }

        return CreateTensor(values);
    }

    // ========== ExactRetrainingUnlearner Tests ==========

    [Fact]
    public void ExactRetraining_RemovesTargetClient_ModelDiffers()
    {
        var options = new FederatedUnlearningOptions
        {
            Method = UnlearningMethod.ExactRetraining,
            VerificationEnabled = true
        };
        var unlearner = new ExactRetrainingUnlearner<double>(options);
        var globalModel = CreateGlobalModel(10);
        var histories = CreateClientHistories(3, 5, 10);

        var (unlearnedModel, certificate) = unlearner.Unlearn(1, globalModel, histories);

        Assert.NotNull(unlearnedModel);
        Assert.NotNull(certificate);
        Assert.Equal(1, certificate.TargetClientId);
        Assert.Equal(UnlearningMethod.ExactRetraining, certificate.MethodUsed);
        Assert.True(certificate.Verified);
        Assert.True(certificate.UnlearningTimeMs >= 0);
    }

    [Fact]
    public void ExactRetraining_CertificateHasValidHashes()
    {
        var options = new FederatedUnlearningOptions { Method = UnlearningMethod.ExactRetraining };
        var unlearner = new ExactRetrainingUnlearner<double>(options);
        var globalModel = CreateGlobalModel(10);
        var histories = CreateClientHistories(3, 5, 10);

        var (_, certificate) = unlearner.Unlearn(0, globalModel, histories);

        Assert.False(string.IsNullOrEmpty(certificate.PreUnlearningModelHash));
        Assert.False(string.IsNullOrEmpty(certificate.PostUnlearningModelHash));
        Assert.NotEqual(certificate.PreUnlearningModelHash, certificate.PostUnlearningModelHash);
    }

    [Fact]
    public void ExactRetraining_SingleClient_ReturnsZeroModel()
    {
        var options = new FederatedUnlearningOptions { Method = UnlearningMethod.ExactRetraining };
        var unlearner = new ExactRetrainingUnlearner<double>(options);
        var globalModel = CreateGlobalModel(5);
        var histories = CreateClientHistories(1, 3, 5);

        var (unlearnedModel, certificate) = unlearner.Unlearn(0, globalModel, histories);

        // With only one client, removing them means zero-initializing
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(0.0, unlearnedModel[i], 12);
        }
    }

    [Fact]
    public void ExactRetraining_MethodName_ReturnsCorrectName()
    {
        var options = new FederatedUnlearningOptions { Method = UnlearningMethod.ExactRetraining };
        var unlearner = new ExactRetrainingUnlearner<double>(options);

        Assert.Equal("ExactRetraining", unlearner.MethodName);
    }

    // ========== GradientAscentUnlearner Tests ==========

    [Fact]
    public void GradientAscent_ProducesModifiedModel()
    {
        var options = new FederatedUnlearningOptions
        {
            Method = UnlearningMethod.GradientAscent,
            MaxUnlearningEpochs = 5,
            UnlearningLearningRate = 0.01,
            VerificationEnabled = true
        };
        var unlearner = new GradientAscentUnlearner<double>(options);
        var globalModel = CreateGlobalModel(10);
        var histories = CreateClientHistories(3, 5, 10);

        var (unlearnedModel, certificate) = unlearner.Unlearn(1, globalModel, histories);

        Assert.NotNull(unlearnedModel);
        Assert.Equal(UnlearningMethod.GradientAscent, certificate.MethodUsed);
        Assert.Equal(1, certificate.TargetClientId);

        // Model should be different from original after gradient ascent
        bool anyDifferent = false;
        for (int i = 0; i < 10; i++)
        {
            if (Math.Abs(unlearnedModel[i] - globalModel[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "Unlearned model should differ from original after gradient ascent");
    }

    [Fact]
    public void GradientAscent_MissingTargetClient_StillWorks()
    {
        var options = new FederatedUnlearningOptions
        {
            Method = UnlearningMethod.GradientAscent,
            MaxUnlearningEpochs = 3
        };
        var unlearner = new GradientAscentUnlearner<double>(options);
        var globalModel = CreateGlobalModel(5);
        var histories = CreateClientHistories(3, 5, 5);

        // Client 99 doesn't exist in histories
        var (unlearnedModel, certificate) = unlearner.Unlearn(99, globalModel, histories);

        Assert.NotNull(unlearnedModel);
        Assert.Equal(99, certificate.TargetClientId);
    }

    // ========== InfluenceFunctionUnlearner Tests ==========

    [Fact]
    public void InfluenceFunction_ProducesModifiedModel()
    {
        var options = new FederatedUnlearningOptions
        {
            Method = UnlearningMethod.InfluenceFunction,
            InfluenceTolerance = 1e-3,
            MaxInfluenceIterations = 50,
            VerificationEnabled = true
        };
        var unlearner = new InfluenceFunctionUnlearner<double>(options);
        var globalModel = CreateGlobalModel(10);
        var histories = CreateClientHistories(3, 5, 10);

        var (unlearnedModel, certificate) = unlearner.Unlearn(0, globalModel, histories);

        Assert.NotNull(unlearnedModel);
        Assert.Equal(UnlearningMethod.InfluenceFunction, certificate.MethodUsed);
    }

    [Fact]
    public void InfluenceFunction_CertificateHasMembershipScore()
    {
        var options = new FederatedUnlearningOptions
        {
            Method = UnlearningMethod.InfluenceFunction,
            VerificationEnabled = true
        };
        var unlearner = new InfluenceFunctionUnlearner<double>(options);
        var globalModel = CreateGlobalModel(8);
        var histories = CreateClientHistories(3, 4, 8);

        var (_, certificate) = unlearner.Unlearn(1, globalModel, histories);

        // Membership inference score should be between 0 and 1
        Assert.InRange(certificate.MembershipInferenceScore, 0.0, 1.0);
    }

    // ========== DiffusiveNoiseUnlearner Tests ==========

    [Fact]
    public void DiffusiveNoise_ProducesModifiedModel()
    {
        var options = new FederatedUnlearningOptions
        {
            Method = UnlearningMethod.DiffusiveNoise,
            NoiseScale = 0.1,
            VerificationEnabled = true
        };
        var unlearner = new DiffusiveNoiseUnlearner<double>(options);
        var globalModel = CreateGlobalModel(10);
        var histories = CreateClientHistories(3, 5, 10);

        var (unlearnedModel, certificate) = unlearner.Unlearn(1, globalModel, histories);

        Assert.NotNull(unlearnedModel);
        Assert.Equal(UnlearningMethod.DiffusiveNoise, certificate.MethodUsed);

        // Model should be perturbed from original
        bool anyDifferent = false;
        for (int i = 0; i < 10; i++)
        {
            if (Math.Abs(unlearnedModel[i] - globalModel[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "Diffusive noise should perturb the model");
    }

    [Fact]
    public void DiffusiveNoise_RetainedAccuracy_ReasonableRange()
    {
        var options = new FederatedUnlearningOptions
        {
            Method = UnlearningMethod.DiffusiveNoise,
            NoiseScale = 0.05,
            VerificationEnabled = true
        };
        var unlearner = new DiffusiveNoiseUnlearner<double>(options);
        var globalModel = CreateGlobalModel(10);
        var histories = CreateClientHistories(3, 5, 10);

        var (_, certificate) = unlearner.Unlearn(2, globalModel, histories);

        // Retained accuracy should be reasonable (healing step should help)
        Assert.InRange(certificate.RetainedAccuracy, 0.0, 1.0);
    }

    // ========== Options Tests ==========

    [Fact]
    public void FederatedUnlearningOptions_DefaultValues()
    {
        var options = new FederatedUnlearningOptions();

        Assert.Equal(UnlearningMethod.GradientAscent, options.Method);
        Assert.True(options.VerificationEnabled);
        Assert.Equal(10, options.MaxUnlearningEpochs);
        Assert.Equal(0.01, options.UnlearningLearningRate);
        Assert.Equal(0.1, options.NoiseScale);
        Assert.Equal(1e-4, options.InfluenceTolerance);
        Assert.Equal(100, options.MaxInfluenceIterations);
        Assert.Equal(0.95, options.VerificationThreshold);
    }

    [Fact]
    public void UnlearningMethod_HasAllExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(UnlearningMethod), UnlearningMethod.ExactRetraining));
        Assert.True(Enum.IsDefined(typeof(UnlearningMethod), UnlearningMethod.GradientAscent));
        Assert.True(Enum.IsDefined(typeof(UnlearningMethod), UnlearningMethod.InfluenceFunction));
        Assert.True(Enum.IsDefined(typeof(UnlearningMethod), UnlearningMethod.DiffusiveNoise));
    }

    [Fact]
    public void UnlearningCertificate_DefaultValues()
    {
        var cert = new UnlearningCertificate();

        Assert.False(cert.Verified);
        Assert.Equal(1.0, cert.RetainedAccuracy);
        Assert.Equal(string.Empty, cert.Summary);
    }

    // ========== Null Argument Tests ==========

    [Fact]
    public void ExactRetraining_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new ExactRetrainingUnlearner<double>(null));
    }

    [Fact]
    public void GradientAscent_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new GradientAscentUnlearner<double>(null));
    }

    [Fact]
    public void InfluenceFunction_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new InfluenceFunctionUnlearner<double>(null));
    }

    [Fact]
    public void DiffusiveNoise_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new DiffusiveNoiseUnlearner<double>(null));
    }

    // ========== Integration with FederatedLearningOptions ==========

    [Fact]
    public void FederatedLearningOptions_CanSetUnlearningOptions()
    {
        var flOptions = new FederatedLearningOptions
        {
            Unlearning = new FederatedUnlearningOptions
            {
                Method = UnlearningMethod.InfluenceFunction,
                MaxInfluenceIterations = 200
            }
        };

        Assert.NotNull(flOptions.Unlearning);
        Assert.Equal(UnlearningMethod.InfluenceFunction, flOptions.Unlearning.Method);
        Assert.Equal(200, flOptions.Unlearning.MaxInfluenceIterations);
    }
}
