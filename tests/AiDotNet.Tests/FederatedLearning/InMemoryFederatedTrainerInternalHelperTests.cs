using System.Reflection;
using AiDotNet.FederatedLearning.Trainers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class InMemoryFederatedTrainerInternalHelperTests
{
    private static readonly Type TrainerType = typeof(InMemoryFederatedTrainer<double, Matrix<double>, Vector<double>>);

    [Fact]
    public void Clamp01_ClampsValues()
    {
        Assert.Equal(0.0, InvokePrivateStatic<double>("Clamp01", -1.0), precision: 12);
        Assert.Equal(0.5, InvokePrivateStatic<double>("Clamp01", 0.5), precision: 12);
        Assert.Equal(1.0, InvokePrivateStatic<double>("Clamp01", 2.0), precision: 12);
    }

    [Fact]
    public void ComputeStalenessWeight_SupportsAllModes()
    {
        Assert.Equal(1.0, InvokePrivateStatic<double>("ComputeStalenessWeight", 0, null), precision: 12);
        Assert.Equal(0.25, InvokePrivateStatic<double>("ComputeStalenessWeight", 3, null), precision: 12);

        var constant = new AsyncFederatedLearningOptions { StalenessWeighting = FederatedStalenessWeighting.Constant };
        Assert.Equal(1.0, InvokePrivateStatic<double>("ComputeStalenessWeight", 5, constant), precision: 12);

        var exponential = new AsyncFederatedLearningOptions { StalenessWeighting = FederatedStalenessWeighting.Exponential, StalenessDecayRate = 2.0 };
        Assert.Equal(Math.Exp(-6.0), InvokePrivateStatic<double>("ComputeStalenessWeight", 3, exponential), precision: 12);

        var polynomial = new AsyncFederatedLearningOptions { StalenessWeighting = FederatedStalenessWeighting.Polynomial, StalenessDecayRate = 2.0 };
        Assert.Equal(1.0 / 16.0, InvokePrivateStatic<double>("ComputeStalenessWeight", 3, polynomial), precision: 12);

        AssertInnerException<InvalidOperationException>(() => InvokePrivateStatic<object>(
            "ComputeStalenessWeight",
            1,
            new AsyncFederatedLearningOptions { StalenessWeighting = (FederatedStalenessWeighting)999 }));
    }

    [Fact]
    public void MixParameters_HandlesAlphaBoundsAndLengthMismatch()
    {
        var trainer = CreateTrainer();
        var current = new Vector<double>(new[] { 1.0, 2.0 });
        var target = new Vector<double>(new[] { 3.0, 6.0 });

        AssertInnerException<ArgumentException>(() => InvokePrivateInstance<object>(trainer, "MixParameters", current, new Vector<double>(3), 0.5));

        var alphaZero = InvokePrivateInstance<Vector<double>>(trainer, "MixParameters", current, target, 0.0);
        Assert.True(ReferenceEquals(current, alphaZero));

        var alphaOne = InvokePrivateInstance<Vector<double>>(trainer, "MixParameters", current, target, 1.0);
        Assert.True(ReferenceEquals(target, alphaOne));

        var mixed = InvokePrivateInstance<Vector<double>>(trainer, "MixParameters", current, target, 0.5);
        Assert.Equal(2.0, mixed[0], precision: 12);
        Assert.Equal(4.0, mixed[1], precision: 12);
    }

    [Fact]
    public void CreateDefaultServerOptimizer_ResolvesKnownOptimizersAndThrowsForUnknown()
    {
        Assert.Null(InvokePrivateStatic<object>("CreateDefaultServerOptimizer", (object?)null));
        Assert.Null(InvokePrivateStatic<object>("CreateDefaultServerOptimizer", new FederatedServerOptimizerOptions { Optimizer = FederatedServerOptimizer.None }));

        var avgm = InvokePrivateStatic<IFederatedServerOptimizer<double>>("CreateDefaultServerOptimizer", new FederatedServerOptimizerOptions
        {
            Optimizer = FederatedServerOptimizer.FedAvgM,
            LearningRate = 1.0,
            Momentum = 0.9
        });
        Assert.Equal("FedAvgM", avgm.GetOptimizerName());

        var adagrad = InvokePrivateStatic<IFederatedServerOptimizer<double>>("CreateDefaultServerOptimizer", new FederatedServerOptimizerOptions
        {
            Optimizer = FederatedServerOptimizer.FedAdagrad,
            LearningRate = 1.0,
            Epsilon = 1e-8
        });
        Assert.Equal("FedAdagrad", adagrad.GetOptimizerName());

        var adam = InvokePrivateStatic<IFederatedServerOptimizer<double>>("CreateDefaultServerOptimizer", new FederatedServerOptimizerOptions
        {
            Optimizer = FederatedServerOptimizer.FedAdam,
            LearningRate = 1.0,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        });
        Assert.Equal("FedAdam", adam.GetOptimizerName());

        var yogi = InvokePrivateStatic<IFederatedServerOptimizer<double>>("CreateDefaultServerOptimizer", new FederatedServerOptimizerOptions
        {
            Optimizer = FederatedServerOptimizer.FedYogi,
            LearningRate = 1.0,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        });
        Assert.Equal("FedYogi", yogi.GetOptimizerName());

        AssertInnerException<InvalidOperationException>(() => InvokePrivateStatic<object>(
            "CreateDefaultServerOptimizer",
            new FederatedServerOptimizerOptions { Optimizer = (FederatedServerOptimizer)999 }));
    }

    [Fact]
    public void CreateDefaultHeterogeneityCorrection_ResolvesKnownCorrectionsAndThrowsForUnknown()
    {
        Assert.Null(InvokePrivateStatic<object>("CreateDefaultHeterogeneityCorrection", (object?)null));
        Assert.Null(InvokePrivateStatic<object>("CreateDefaultHeterogeneityCorrection", new FederatedHeterogeneityCorrectionOptions { Algorithm = FederatedHeterogeneityCorrection.None }));

        var scaffold = InvokePrivateStatic<IFederatedHeterogeneityCorrection<double>>("CreateDefaultHeterogeneityCorrection", new FederatedHeterogeneityCorrectionOptions
        {
            Algorithm = FederatedHeterogeneityCorrection.Scaffold,
            ClientLearningRate = 1.0
        });
        Assert.Equal("SCAFFOLD", scaffold.GetCorrectionName());

        var fedNova = InvokePrivateStatic<IFederatedHeterogeneityCorrection<double>>("CreateDefaultHeterogeneityCorrection", new FederatedHeterogeneityCorrectionOptions
        {
            Algorithm = FederatedHeterogeneityCorrection.FedNova
        });
        Assert.Equal("FedNova", fedNova.GetCorrectionName());

        var fedDyn = InvokePrivateStatic<IFederatedHeterogeneityCorrection<double>>("CreateDefaultHeterogeneityCorrection", new FederatedHeterogeneityCorrectionOptions
        {
            Algorithm = FederatedHeterogeneityCorrection.FedDyn,
            FedDynAlpha = 0.01
        });
        Assert.Equal("FedDyn", fedDyn.GetCorrectionName());

        AssertInnerException<InvalidOperationException>(() => InvokePrivateStatic<object>(
            "CreateDefaultHeterogeneityCorrection",
            new FederatedHeterogeneityCorrectionOptions { Algorithm = (FederatedHeterogeneityCorrection)999 }));
    }

    [Fact]
    public void CreateDefaultPrivacyAccountant_ResolvesKnownAccountantsAndThrowsForUnknown()
    {
        var basic = InvokePrivateStatic<IPrivacyAccountant>("CreateDefaultPrivacyAccountant", FederatedPrivacyAccountant.Basic, 1.0);
        Assert.Equal("Basic", basic.GetAccountantName());

        var rdp = InvokePrivateStatic<IPrivacyAccountant>("CreateDefaultPrivacyAccountant", FederatedPrivacyAccountant.Rdp, 1.0);
        Assert.Equal("RDP", rdp.GetAccountantName());

        AssertInnerException<InvalidOperationException>(() => InvokePrivateStatic<object>("CreateDefaultPrivacyAccountant", (FederatedPrivacyAccountant)999, 1.0));
    }

    [Fact]
    public void CreateBuiltInSelectionStrategy_ResolvesAllKnownStrategiesAndThrowsForUnknown()
    {
        var selectionOptions = new ClientSelectionOptions
        {
            AvailabilityThreshold = 0.4,
            ExplorationRate = 0.2,
            ClusterCount = 2,
            KMeansIterations = 3
        };

        Assert.Equal("UniformRandom", InvokePrivateStatic<IClientSelectionStrategy>("CreateBuiltInSelectionStrategy", FederatedClientSelectionStrategy.UniformRandom, selectionOptions).GetStrategyName());
        Assert.Equal("WeightedRandom", InvokePrivateStatic<IClientSelectionStrategy>("CreateBuiltInSelectionStrategy", FederatedClientSelectionStrategy.WeightedRandom, selectionOptions).GetStrategyName());
        Assert.Equal("Stratified", InvokePrivateStatic<IClientSelectionStrategy>("CreateBuiltInSelectionStrategy", FederatedClientSelectionStrategy.Stratified, selectionOptions).GetStrategyName());
        Assert.Equal("AvailabilityAware", InvokePrivateStatic<IClientSelectionStrategy>("CreateBuiltInSelectionStrategy", FederatedClientSelectionStrategy.AvailabilityAware, selectionOptions).GetStrategyName());
        Assert.Equal("PerformanceAware", InvokePrivateStatic<IClientSelectionStrategy>("CreateBuiltInSelectionStrategy", FederatedClientSelectionStrategy.PerformanceAware, selectionOptions).GetStrategyName());
        Assert.Equal("Clustered", InvokePrivateStatic<IClientSelectionStrategy>("CreateBuiltInSelectionStrategy", FederatedClientSelectionStrategy.Clustered, selectionOptions).GetStrategyName());

        AssertInnerException<InvalidOperationException>(() => InvokePrivateStatic<object>("CreateBuiltInSelectionStrategy", (FederatedClientSelectionStrategy)999, selectionOptions));
    }

    [Fact]
    public void EstimateRoundCommunicationMB_HandlesBoundsAndClampsRatio()
    {
        Assert.Equal(0.0, InvokePrivateStatic<double>("EstimateRoundCommunicationMB", 0, 10, 1.0), precision: 12);
        Assert.Equal(0.0, InvokePrivateStatic<double>("EstimateRoundCommunicationMB", 2, 0, 1.0), precision: 12);

        var half = InvokePrivateStatic<double>("EstimateRoundCommunicationMB", 2, 10, 0.5);
        Assert.Equal(0.00024, half, precision: 12);

        var negative = InvokePrivateStatic<double>("EstimateRoundCommunicationMB", 2, 10, -5.0);
        Assert.Equal(InvokePrivateStatic<double>("EstimateRoundCommunicationMB", 2, 10, 0.0), negative, precision: 12);

        var aboveOne = InvokePrivateStatic<double>("EstimateRoundCommunicationMB", 2, 10, 5.0);
        Assert.Equal(InvokePrivateStatic<double>("EstimateRoundCommunicationMB", 2, 10, 1.0), aboveOne, precision: 12);
    }

    [Fact]
    public void ResolveCompressionOptions_SupportsLegacyUseCompression()
    {
        Assert.Null(InvokePrivateStatic<object>("ResolveCompressionOptions", (object?)null));

        var explicitCompression = new FederatedCompressionOptions { Strategy = FederatedCompressionStrategy.Threshold, Threshold = 0.1 };
        var explicitOptions = new FederatedLearningOptions { Compression = explicitCompression };
        var resolvedExplicit = InvokePrivateStatic<FederatedCompressionOptions>("ResolveCompressionOptions", explicitOptions);
        Assert.True(ReferenceEquals(explicitCompression, resolvedExplicit));

        var legacy = new FederatedLearningOptions
        {
            UseCompression = true,
            CompressionRatio = 0.25
        };
        var resolvedLegacy = InvokePrivateStatic<FederatedCompressionOptions>("ResolveCompressionOptions", legacy);
        Assert.Equal(FederatedCompressionStrategy.TopK, resolvedLegacy.Strategy);
        Assert.Equal(0.25, resolvedLegacy.Ratio, precision: 12);
        Assert.True(resolvedLegacy.UseErrorFeedback);
    }

    [Fact]
    public void ResolvePersonalizationOptions_SupportsLegacyEnablePersonalization()
    {
        Assert.Null(InvokePrivateStatic<object>("ResolvePersonalizationOptions", (object?)null));

        var explicitPersonalization = new FederatedPersonalizationOptions { Enabled = true, Strategy = FederatedPersonalizationStrategy.FedPer, PersonalizedParameterFraction = 0.5 };
        var explicitOptions = new FederatedLearningOptions { Personalization = explicitPersonalization };
        var resolvedExplicit = InvokePrivateStatic<FederatedPersonalizationOptions>("ResolvePersonalizationOptions", explicitOptions);
        Assert.True(ReferenceEquals(explicitPersonalization, resolvedExplicit));

        var legacy = new FederatedLearningOptions
        {
            EnablePersonalization = true,
            PersonalizationLayerFraction = 0.4
        };
        var resolvedLegacy = InvokePrivateStatic<FederatedPersonalizationOptions>("ResolvePersonalizationOptions", legacy);
        Assert.True(resolvedLegacy.Enabled);
        Assert.Equal(FederatedPersonalizationStrategy.FedPer, resolvedLegacy.Strategy);
        Assert.Equal(0.4, resolvedLegacy.PersonalizedParameterFraction, precision: 12);
    }

    [Fact]
    public void ResolvePersonalizedIndices_ComputesTailIndices()
    {
        Assert.Empty(InvokePrivateStatic<int[]>("ResolvePersonalizedIndices", 0.25, 0));
        Assert.Empty(InvokePrivateStatic<int[]>("ResolvePersonalizedIndices", double.NaN, 10));
        Assert.Empty(InvokePrivateStatic<int[]>("ResolvePersonalizedIndices", 0.0, 10));

        var all = InvokePrivateStatic<int[]>("ResolvePersonalizedIndices", 2.0, 3);
        Assert.Equal(new[] { 0, 1, 2 }, all);

        var half = InvokePrivateStatic<int[]>("ResolvePersonalizedIndices", 0.5, 10);
        Assert.Equal(new[] { 5, 6, 7, 8, 9 }, half);
    }

    [Fact]
    public void GetClusterId_NormalizesNegativeClientIds()
    {
        Assert.Equal(0, InvokePrivateStatic<int>("GetClusterId", 5, 0));
        Assert.Equal(2, InvokePrivateStatic<int>("GetClusterId", -1, 3));
    }

    [Fact]
    public void MaskIndices_MasksOnlyValidIndicesAndThrowsOnLengthMismatch()
    {
        var parameters = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var baseline = new Vector<double>(new[] { 10.0, 20.0, 30.0 });

        var masked = InvokePrivateStatic<Vector<double>>("MaskIndices", parameters, baseline, new[] { -1, 1, 99 });
        Assert.Equal(1.0, masked[0], precision: 12);
        Assert.Equal(20.0, masked[1], precision: 12);
        Assert.Equal(3.0, masked[2], precision: 12);

        AssertInnerException<ArgumentException>(() => InvokePrivateStatic<object>("MaskIndices", new Vector<double>(2), baseline, new[] { 0 }));
    }

    [Fact]
    public void ResolveEncryptedIndices_BuildsAllOrRanges()
    {
        AssertInnerException<ArgumentNullException>(() => InvokePrivateStatic<object>("ResolveEncryptedIndices", null!, 5, HomomorphicEncryptionMode.Hybrid));

        Assert.Empty(InvokePrivateStatic<int[]>("ResolveEncryptedIndices", new HomomorphicEncryptionOptions(), 0, HomomorphicEncryptionMode.Hybrid));

        var heOnly = InvokePrivateStatic<int[]>("ResolveEncryptedIndices", new HomomorphicEncryptionOptions(), 4, HomomorphicEncryptionMode.HeOnly);
        Assert.Equal(new[] { 0, 1, 2, 3 }, heOnly);

        var ranges = new HomomorphicEncryptionOptions
        {
            EncryptedRanges = new List<ParameterIndexRange>
            {
                new ParameterIndexRange { Start = -2, Length = 2 },
                new ParameterIndexRange { Start = 1, Length = 3 },
                new ParameterIndexRange { Start = 10, Length = 1 },
                new ParameterIndexRange { Start = 0, Length = 0 }
            }
        };

        var hybrid = InvokePrivateStatic<int[]>("ResolveEncryptedIndices", ranges, 5, HomomorphicEncryptionMode.Hybrid);
        Assert.Equal(new[] { 0, 1, 2, 3 }, hybrid);
    }

    [Fact]
    public void ComputeQuantizationRatio_ClampsBits()
    {
        Assert.Equal(0.0, InvokePrivateStatic<double>("ComputeQuantizationRatio", -1), precision: 12);
        Assert.Equal(0.0, InvokePrivateStatic<double>("ComputeQuantizationRatio", 0), precision: 12);
        Assert.Equal(0.0625, InvokePrivateStatic<double>("ComputeQuantizationRatio", 4), precision: 12);
        Assert.Equal(1.0, InvokePrivateStatic<double>("ComputeQuantizationRatio", 1000), precision: 12);
    }

    [Fact]
    public void EstimateBytesPerNumericType_FallsBackForUnsupportedTypes()
    {
        Assert.Equal(8, InvokePrivateStatic<int>("EstimateBytesPerNumericType", typeof(double)));
        Assert.Equal(8, InvokePrivateStatic<int>("EstimateBytesPerNumericType", typeof(string)));
    }

    [Fact]
    public void PersonalizationHelpers_HandleAllStrategies()
    {
        var trainer = CreateTrainer();
        var globalModel = new MockFullModel(_ => new Vector<double>(6), parameterCount: 6);
        var globalParams = globalModel.GetParameters();
        var personalizedIndices = new[] { 4, 5 };

        var clientState = new Dictionary<int, Vector<double>>();
        var clusterState = new Dictionary<int, Vector<double>>();

        // Head-split: use client state values for the personalized indices.
        var clientId = 7;
        var clientHead = globalParams.Clone();
        clientHead[4] = 99.0;
        clientHead[5] = 100.0;
        clientState[clientId] = clientHead;

        var headSplitStart = InvokePrivateInstance<IFullModel<double, Matrix<double>, Vector<double>>>(
            trainer,
            "CreatePersonalizedStartModel",
            FederatedPersonalizationStrategy.FedPer,
            new FederatedPersonalizationOptions { Enabled = true, Strategy = FederatedPersonalizationStrategy.FedPer },
            clientId,
            globalModel,
            globalParams,
            personalizedIndices,
            clientState,
            clusterState);

        var headSplitParams = headSplitStart.GetParameters();
        Assert.Equal(99.0, headSplitParams[4], precision: 12);
        Assert.Equal(100.0, headSplitParams[5], precision: 12);

        // Clustered: use cluster head values when available.
        var clusteredOptions = new FederatedPersonalizationOptions { Enabled = true, Strategy = FederatedPersonalizationStrategy.Clustered, ClusterCount = 2 };
        var clusterId = InvokePrivateStatic<int>("GetClusterId", 3, clusteredOptions.ClusterCount);
        var clusterHead = globalParams.Clone();
        clusterHead[4] = -5.0;
        clusterHead[5] = -6.0;
        clusterState[clusterId] = clusterHead;

        var clusteredStart = InvokePrivateInstance<IFullModel<double, Matrix<double>, Vector<double>>>(
            trainer,
            "CreatePersonalizedStartModel",
            FederatedPersonalizationStrategy.Clustered,
            clusteredOptions,
            3,
            globalModel,
            globalParams,
            personalizedIndices,
            clientState,
            clusterState);

        var clusteredParams = clusteredStart.GetParameters();
        Assert.Equal(-5.0, clusteredParams[4], precision: 12);
        Assert.Equal(-6.0, clusteredParams[5], precision: 12);

        // PFedMe initializes per-client theta.
        var pFedMeClientState = new Dictionary<int, Vector<double>>();
        var pFedMeStart = InvokePrivateInstance<IFullModel<double, Matrix<double>, Vector<double>>>(
            trainer,
            "CreatePersonalizedStartModel",
            FederatedPersonalizationStrategy.PFedMe,
            new FederatedPersonalizationOptions { Enabled = true, Strategy = FederatedPersonalizationStrategy.PFedMe },
            1,
            globalModel,
            globalParams,
            personalizedIndices,
            pFedMeClientState,
            null);
        Assert.True(pFedMeClientState.ContainsKey(1));
        Assert.Equal(globalParams[0], pFedMeStart.GetParameters()[0], precision: 12);

        // Unknown/Ditto: returns the global model reference.
        var dittoStart = InvokePrivateInstance<IFullModel<double, Matrix<double>, Vector<double>>>(
            trainer,
            "CreatePersonalizedStartModel",
            FederatedPersonalizationStrategy.Ditto,
            new FederatedPersonalizationOptions { Enabled = true, Strategy = FederatedPersonalizationStrategy.Ditto },
            1,
            globalModel,
            globalParams,
            personalizedIndices,
            clientState,
            clusterState);
        Assert.True(ReferenceEquals(globalModel, dittoStart));

        // Post-local: Head-split masks personalized indices.
        var baseline = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 });
        var trained = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0, 2.0, 3.0 });
        var perClientState = new Dictionary<int, Vector<double>>();
        var fedPerMasked = InvokePrivateInstance<Vector<double>>(
            trainer,
            "ApplyPersonalizationAfterLocalTraining",
            FederatedPersonalizationStrategy.FedPer,
            new FederatedPersonalizationOptions { Enabled = true, Strategy = FederatedPersonalizationStrategy.FedPer },
            0,
            baseline,
            trained,
            personalizedIndices,
            perClientState);
        Assert.Equal(1.0, fedPerMasked[0], precision: 12);
        Assert.Equal(0.0, fedPerMasked[4], precision: 12);
        Assert.Equal(0.0, fedPerMasked[5], precision: 12);

        // Ditto stores the trained parameters but returns them unchanged for global aggregation.
        var dittoState = new Dictionary<int, Vector<double>>();
        var dittoReturned = InvokePrivateInstance<Vector<double>>(
            trainer,
            "ApplyPersonalizationAfterLocalTraining",
            FederatedPersonalizationStrategy.Ditto,
            new FederatedPersonalizationOptions { Enabled = true, Strategy = FederatedPersonalizationStrategy.Ditto, DittoLambda = 0.1 },
            5,
            baseline,
            trained,
            personalizedIndices,
            dittoState);
        Assert.True(dittoState.ContainsKey(5));
        Assert.Equal(trained[5], dittoReturned[5], precision: 12);

        // PFedMe updates theta via repeated meta updates.
        var pFedMeState = new Dictionary<int, Vector<double>>();
        var pFedMeUpdated = InvokePrivateInstance<Vector<double>>(
            trainer,
            "ApplyPersonalizationAfterLocalTraining",
            FederatedPersonalizationStrategy.PFedMe,
            new FederatedPersonalizationOptions { Enabled = true, Strategy = FederatedPersonalizationStrategy.PFedMe, PFedMeMu = 0.0, PFedMeInnerSteps = 2 },
            9,
            baseline,
            trained,
            personalizedIndices,
            pFedMeState);
        Assert.True(pFedMeState.ContainsKey(9));
        Assert.Equal(trained[0], pFedMeUpdated[0], precision: 12);
    }

    private static InMemoryFederatedTrainer<double, Matrix<double>, Vector<double>> CreateTrainer()
    {
        var model = new MockFullModel(_ => new Vector<double>(2), parameterCount: 2);
        var optimizer = new FederatedNoOpOptimizer(model);
        return new InMemoryFederatedTrainer<double, Matrix<double>, Vector<double>>(optimizer, randomSeed: 123);
    }

    private static T InvokePrivateStatic<T>(string methodName, params object?[] args)
    {
        return (T)InvokePrivateStatic(methodName, args)!;
    }

    private static object? InvokePrivateStatic(string methodName, params object?[] args)
    {
        var method = TrainerType.GetMethod(methodName, BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);
        return method!.Invoke(null, args);
    }

    private static T InvokePrivateInstance<T>(object instance, string methodName, params object?[] args)
    {
        return (T)InvokePrivateInstance(instance, methodName, args)!;
    }

    private static object? InvokePrivateInstance(object instance, string methodName, params object?[] args)
    {
        var method = TrainerType.GetMethod(methodName, BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(method);
        return method!.Invoke(instance, args);
    }

    private static void AssertInnerException<TException>(Action action)
        where TException : Exception
    {
        var tie = Assert.Throws<TargetInvocationException>(action);
        Assert.IsType<TException>(tie.InnerException);
    }
}
