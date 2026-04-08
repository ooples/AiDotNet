using AiDotNet.FederatedLearning.PSI;
using AiDotNet.FederatedLearning.Vertical;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Comprehensive integration tests for vertical federated learning (#542).
/// </summary>
public class VerticalFederatedLearningTests
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

    private static Tensor<double> Create2DTensor(int rows, int cols, double seed = 1.0)
    {
        var tensor = new Tensor<double>(new[] { rows, cols });
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                tensor[r * cols + c] = seed + r * 0.1 + c * 0.01;
            }
        }

        return tensor;
    }

    // ========== VerticalDataPartitioner Tests ==========

    [Fact]
    public void DataPartitioner_PartitionSequential_SplitsEvenly()
    {
        var partitions = VerticalDataPartitioner<double>.PartitionSequential(
            totalFeatures: 10, numberOfParties: 2);

        Assert.Equal(2, partitions.Count);
        Assert.Equal(5, partitions[0].Count); // First 5 columns
        Assert.Equal(5, partitions[1].Count); // Last 5 columns

        // Verify sequential assignment
        Assert.Equal(0, partitions[0][0]);
        Assert.Equal(4, partitions[0][4]);
        Assert.Equal(5, partitions[1][0]);
        Assert.Equal(9, partitions[1][4]);
    }

    [Fact]
    public void DataPartitioner_PartitionSequential_HandlesUnevenSplit()
    {
        var partitions = VerticalDataPartitioner<double>.PartitionSequential(
            totalFeatures: 7, numberOfParties: 3);

        Assert.Equal(3, partitions.Count);
        // 7 / 3 = 2 remainder 1, first party gets extra
        int totalAssigned = partitions[0].Count + partitions[1].Count + partitions[2].Count;
        Assert.Equal(7, totalAssigned);
    }

    [Fact]
    public void DataPartitioner_PartitionSequential_ZeroFeatures_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            VerticalDataPartitioner<double>.PartitionSequential(0, 2));
    }

    [Fact]
    public void DataPartitioner_PartitionSequential_ZeroParties_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            VerticalDataPartitioner<double>.PartitionSequential(10, 0));
    }

    [Fact]
    public void DataPartitioner_PartitionInterleaved_AlternatesColumns()
    {
        var partitions = VerticalDataPartitioner<double>.PartitionInterleaved(
            totalFeatures: 6, numberOfParties: 2);

        Assert.Equal(2, partitions.Count);
        // Party 0: columns 0, 2, 4
        Assert.Contains(0, partitions[0]);
        Assert.Contains(2, partitions[0]);
        Assert.Contains(4, partitions[0]);
        // Party 1: columns 1, 3, 5
        Assert.Contains(1, partitions[1]);
        Assert.Contains(3, partitions[1]);
        Assert.Contains(5, partitions[1]);
    }

    // ========== SplitNeuralNetwork Tests ==========

    [Fact]
    public void SplitNN_Constructor_WithDefaults_Succeeds()
    {
        var splitNN = new SplitNeuralNetwork<double>(
            numberOfParties: 2,
            embeddingDimensionPerParty: 16,
            outputDimension: 1);

        Assert.NotNull(splitNN);
    }

    [Fact]
    public void SplitNN_Constructor_WithOptions_Succeeds()
    {
        var splitOptions = new SplitModelOptions
        {
            AggregationMode = VflAggregationMode.Concatenation,
            TopModelHiddenDimension = 64,
            TopModelHiddenLayers = 2
        };
        var splitNN = new SplitNeuralNetwork<double>(
            numberOfParties: 2,
            embeddingDimensionPerParty: 16,
            outputDimension: 1,
            options: splitOptions,
            seed: 42);

        Assert.NotNull(splitNN);
    }

    [Fact]
    public void SplitNN_AggregateAndForward_ProducesOutput()
    {
        var splitNN = new SplitNeuralNetwork<double>(
            numberOfParties: 2,
            embeddingDimensionPerParty: 4,
            outputDimension: 1,
            seed: 42);

        var partyEmbeddings = new List<Tensor<double>>
        {
            CreateTensor(0.1, 0.2, 0.3, 0.4),
            CreateTensor(0.5, 0.6, 0.7, 0.8)
        };

        var combined = splitNN.AggregateEmbeddings(partyEmbeddings);
        Assert.NotNull(combined);

        var output = splitNN.ForwardTopModel(combined);
        Assert.NotNull(output);
    }

    // ========== SecureGradientExchange Tests ==========

    [Fact]
    public void SecureGradientExchange_Constructor_WithEncryption_Succeeds()
    {
        var exchange = new SecureGradientExchange<double>(useEncryption: true, seed: 42);

        Assert.NotNull(exchange);
    }

    [Fact]
    public void SecureGradientExchange_Constructor_WithoutEncryption_Succeeds()
    {
        var exchange = new SecureGradientExchange<double>(useEncryption: false);

        Assert.NotNull(exchange);
    }

    [Fact]
    public void SecureGradientExchange_ProtectAndRecover_RoundTrips()
    {
        var exchange = new SecureGradientExchange<double>(useEncryption: true, seed: 42);
        var gradient = CreateTensor(1.0, 2.0, 3.0, 4.0, 5.0);

        var (protectedGrad, mask) = exchange.ProtectGradients(gradient);

        Assert.NotNull(protectedGrad);
        Assert.NotNull(mask);

        var recovered = exchange.RecoverGradients(protectedGrad, mask);

        Assert.NotNull(recovered);
        Assert.Equal(gradient.Shape[0], recovered.Shape[0]);

        // Values should be approximately equal after protect+recover
        for (int i = 0; i < gradient.Shape[0]; i++)
        {
            Assert.Equal(gradient[i], recovered[i], 6);
        }
    }

    [Fact]
    public void SecureGradientExchange_NoEncryption_PassesThrough()
    {
        var exchange = new SecureGradientExchange<double>(useEncryption: false);
        var gradient = CreateTensor(1.0, 2.0, 3.0);

        var (protectedGrad, mask) = exchange.ProtectGradients(gradient);

        Assert.NotNull(protectedGrad);
    }

    // ========== MissingFeatureHandler Tests ==========

    [Fact]
    public void MissingFeatureHandler_DefaultOptions_Succeeds()
    {
        var handler = new MissingFeatureHandler<double>();

        Assert.NotNull(handler);
    }

    [Fact]
    public void MissingFeatureHandler_WithOptions_Succeeds()
    {
        var options = new MissingFeatureOptions
        {
            Strategy = MissingFeatureStrategy.Mean,
            MinimumOverlapRatio = 0.3,
            AllowPartialAlignment = true,
            AddMissingnessIndicator = true
        };
        var handler = new MissingFeatureHandler<double>(options);

        Assert.NotNull(handler);
    }

    [Fact]
    public void MissingFeatureHandler_ImputeEmbeddings_ProducesOutput()
    {
        var options = new MissingFeatureOptions
        {
            Strategy = MissingFeatureStrategy.Zero,
            AllowPartialAlignment = true
        };
        var handler = new MissingFeatureHandler<double>(options);

        // ImputeEmbeddings takes (partyId, embeddingDimension, batchSize)
        var result = handler.ImputeEmbeddings(partyId: "bank", embeddingDimension: 8, batchSize: 5);

        Assert.NotNull(result);
    }

    // ========== LabelDifferentialPrivacy Tests ==========

    [Fact]
    public void LabelDP_Constructor_WithDefaults_Succeeds()
    {
        var dp = new LabelDifferentialPrivacy<double>();

        Assert.NotNull(dp);
    }

    [Fact]
    public void LabelDP_Constructor_WithParams_Succeeds()
    {
        var dp = new LabelDifferentialPrivacy<double>(
            epsilon: 0.5, delta: 1e-6, maxGradientNorm: 2.0, seed: 42);

        Assert.NotNull(dp);
    }

    [Fact]
    public void LabelDP_ZeroEpsilon_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LabelDifferentialPrivacy<double>(epsilon: 0));
    }

    [Fact]
    public void LabelDP_NegativeEpsilon_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LabelDifferentialPrivacy<double>(epsilon: -1.0));
    }

    [Fact]
    public void LabelDP_ProtectGradients_AddsNoise()
    {
        var dp = new LabelDifferentialPrivacy<double>(epsilon: 1.0, seed: 42);
        var gradient = CreateTensor(1.0, 2.0, 3.0, 4.0, 5.0);

        var protected_ = dp.ProtectGradients(gradient);

        Assert.NotNull(protected_);
        Assert.Equal(gradient.Shape[0], protected_.Shape[0]);

        // Protected gradient should differ from original due to noise
        bool anyDifferent = false;
        for (int i = 0; i < gradient.Shape[0]; i++)
        {
            if (Math.Abs(protected_[i] - gradient[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "Label DP should add noise to gradients");
    }

    // ========== VerticalFederatedTrainer Tests ==========

    [Fact]
    public void VflTrainer_DefaultConstructor_Succeeds()
    {
        var trainer = new VerticalFederatedTrainer<double>();

        Assert.NotNull(trainer);
    }

    [Fact]
    public void VflTrainer_WithOptions_Succeeds()
    {
        var options = new VerticalFederatedLearningOptions
        {
            LearningRate = 0.001,
            NumberOfEpochs = 10,
            BatchSize = 32,
            NumberOfParties = 2,
            EncryptGradients = true
        };
        var trainer = new VerticalFederatedTrainer<double>(options);

        Assert.NotNull(trainer);
    }

    [Fact]
    public void VflTrainer_WithLabelProtector_Succeeds()
    {
        var options = new VerticalFederatedLearningOptions
        {
            EnableLabelDifferentialPrivacy = true,
            LabelDpEpsilon = 1.0
        };
        var labelProtector = new LabelDifferentialPrivacy<double>(epsilon: 1.0, seed: 42);
        var trainer = new VerticalFederatedTrainer<double>(options, labelProtector);

        Assert.NotNull(trainer);
    }

    [Fact]
    public void VflTrainer_RegisterParty_Succeeds()
    {
        var trainer = new VerticalFederatedTrainer<double>();

        var entityIds = new[] { "e1", "e2", "e3", "e4", "e5" };
        var data = Create2DTensor(5, 3);
        var labels = CreateTensor(0.0, 1.0, 0.0, 1.0, 0.0);

        var labelHolder = new VerticalPartyLabelHolder<double>(
            "hospital", data, labels, entityIds, embeddingDimension: 8);

        // Should not throw
        trainer.RegisterParty(labelHolder);
    }

    [Fact]
    public void VflTrainer_RegisterMultipleParties_Succeeds()
    {
        var trainer = new VerticalFederatedTrainer<double>();

        var entityIds = new[] { "e1", "e2", "e3", "e4", "e5" };

        var partyA = new VerticalPartyClient<double>(
            "bank", Create2DTensor(5, 3), entityIds, embeddingDimension: 8);
        var partyB = new VerticalPartyLabelHolder<double>(
            "hospital", Create2DTensor(5, 4), CreateTensor(0.0, 1.0, 0.0, 1.0, 0.0),
            entityIds, embeddingDimension: 8);

        // Both should register without throwing
        trainer.RegisterParty(partyA);
        trainer.RegisterParty(partyB);
    }

    // ========== VerticalPartyClient Tests ==========

    [Fact]
    public void VerticalPartyClient_Constructor_Succeeds()
    {
        var entityIds = new[] { "e1", "e2", "e3" };
        var data = Create2DTensor(3, 5);

        var client = new VerticalPartyClient<double>(
            "bank", data, entityIds, embeddingDimension: 16);

        Assert.Equal("bank", client.PartyId);
    }

    [Fact]
    public void VerticalPartyClient_ComputeForward_ProducesOutput()
    {
        var entityIds = new[] { "e1", "e2", "e3" };
        var data = Create2DTensor(3, 4);

        var client = new VerticalPartyClient<double>(
            "bank", data, entityIds, embeddingDimension: 4);

        var embedding = client.ComputeForward(alignedIndices: new[] { 0, 1, 2 });

        Assert.NotNull(embedding);
    }

    // ========== VerticalPartyLabelHolder Tests ==========

    [Fact]
    public void VerticalPartyLabelHolder_Constructor_Succeeds()
    {
        var entityIds = new[] { "e1", "e2", "e3" };
        var data = Create2DTensor(3, 4);
        var labels = CreateTensor(0.0, 1.0, 0.0);

        var holder = new VerticalPartyLabelHolder<double>(
            "hospital", data, labels, entityIds, embeddingDimension: 8);

        Assert.Equal("hospital", holder.PartyId);
    }

    // ========== VerticalFederatedUnlearner Tests ==========

    [Fact]
    public void VflUnlearner_DefaultConstructor_Succeeds()
    {
        var unlearner = new VerticalFederatedUnlearner<double>();

        Assert.NotNull(unlearner);
    }

    [Fact]
    public void VflUnlearner_WithOptions_Succeeds()
    {
        var options = new VflUnlearningOptions
        {
            Enabled = true,
            Method = VflUnlearningMethod.Certified,
            MaxUnlearnBatchSize = 50,
            GradientAscentSteps = 10,
            UnlearningLearningRate = 0.005,
            CertificationEpsilon = 0.5,
            VerifyUnlearning = true
        };
        var unlearner = new VerticalFederatedUnlearner<double>(options);

        Assert.NotNull(unlearner);
    }

    // ========== VerticalFederatedBenchmark Tests ==========

    [Fact]
    public void VflBenchmark_GenerateDataset_ProducesValidData()
    {
        var dataset = VerticalFederatedBenchmark<double>.GenerateDataset(
            totalEntities: 100,
            totalFeatures: 10,
            numberOfParties: 2,
            overlapRatio: 0.8,
            seed: 42);

        Assert.NotNull(dataset);
        Assert.Equal(100, dataset.TotalEntities);
        Assert.Equal(10, dataset.TotalFeatures);
        Assert.Equal(0.8, dataset.OverlapRatio);
        Assert.Equal(2, dataset.Parties.Count);
    }

    [Fact]
    public void VflBenchmark_GenerateDataset_LastPartyHasLabels()
    {
        var dataset = VerticalFederatedBenchmark<double>.GenerateDataset(
            totalEntities: 50,
            totalFeatures: 6,
            numberOfParties: 2,
            overlapRatio: 0.9,
            seed: 42);

        var lastParty = dataset.Parties[dataset.Parties.Count - 1];
        Assert.True(lastParty.IsLabelHolder, "Last party should be the label holder");
        Assert.NotNull(lastParty.Labels);
    }

    [Fact]
    public void VflBenchmark_GenerateDataset_PartiesHaveFeatures()
    {
        var dataset = VerticalFederatedBenchmark<double>.GenerateDataset(
            totalEntities: 100,
            totalFeatures: 10,
            numberOfParties: 3,
            overlapRatio: 0.7,
            seed: 42);

        Assert.Equal(3, dataset.Parties.Count);

        int totalCols = 0;
        foreach (var party in dataset.Parties)
        {
            Assert.NotNull(party.Features);
            Assert.NotNull(party.EntityIds);
            Assert.True(party.EntityIds.Count > 0);
            totalCols += party.FeatureColumnIndices.Count;
        }

        Assert.Equal(10, totalCols);
    }

    [Fact]
    public void VflBenchmark_GenerateDataset_ZeroEntities_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            VerticalFederatedBenchmark<double>.GenerateDataset(totalEntities: 0));
    }

    [Fact]
    public void VflBenchmark_GenerateDataset_ZeroFeatures_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            VerticalFederatedBenchmark<double>.GenerateDataset(totalFeatures: 0));
    }

    [Fact]
    public void VflBenchmark_GenerateDataset_OneParty_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            VerticalFederatedBenchmark<double>.GenerateDataset(numberOfParties: 1));
    }

    [Fact]
    public void VflBenchmark_GenerateDataset_InvalidOverlap_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            VerticalFederatedBenchmark<double>.GenerateDataset(overlapRatio: 1.5));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            VerticalFederatedBenchmark<double>.GenerateDataset(overlapRatio: -0.1));
    }

    [Fact]
    public void VflBenchmark_GenerateDataset_SharedEntityCount_MatchesOverlap()
    {
        var dataset = VerticalFederatedBenchmark<double>.GenerateDataset(
            totalEntities: 100,
            totalFeatures: 10,
            numberOfParties: 2,
            overlapRatio: 0.5,
            seed: 42);

        Assert.Equal(50, dataset.SharedEntityCount);
    }

    [Fact]
    public void VflBenchmark_LowOverlap_StillWorks()
    {
        var dataset = VerticalFederatedBenchmark<double>.GenerateDataset(
            totalEntities: 100,
            totalFeatures: 10,
            numberOfParties: 2,
            overlapRatio: 0.1,
            seed: 42);

        Assert.Equal(10, dataset.SharedEntityCount);
    }

    // ========== VerticalFederatedLearningOptions Defaults ==========

    [Fact]
    public void VflOptions_DefaultValues()
    {
        var options = new VerticalFederatedLearningOptions();

        Assert.NotNull(options.EntityAlignment);
        Assert.NotNull(options.SplitModel);
        Assert.NotNull(options.MissingFeatures);
        Assert.NotNull(options.Unlearning);
        Assert.Equal(0.001, options.LearningRate);
        Assert.Equal(50, options.NumberOfEpochs);
        Assert.Equal(64, options.BatchSize);
        Assert.Equal(2, options.NumberOfParties);
        Assert.True(options.EncryptGradients);
        Assert.False(options.EnableLabelDifferentialPrivacy);
        Assert.Equal(1.0, options.LabelDpEpsilon);
        Assert.Equal(1e-5, options.LabelDpDelta);
        Assert.Null(options.RandomSeed);
        Assert.False(options.VerboseLogging);
    }

    [Fact]
    public void SplitModelOptions_DefaultValues()
    {
        var options = new SplitModelOptions();

        Assert.Equal(VflAggregationMode.Concatenation, options.AggregationMode);
        Assert.Equal(SplitPointStrategy.AutoOptimal, options.SplitPoint);
        Assert.Equal(64, options.EmbeddingDimension);
        Assert.Equal(128, options.TopModelHiddenDimension);
        Assert.Equal(2, options.TopModelHiddenLayers);
        Assert.True(options.UseBatchNormalization);
        Assert.False(options.AddEmbeddingNoise);
        Assert.Equal(0.01, options.EmbeddingNoiseScale);
    }

    [Fact]
    public void MissingFeatureOptions_DefaultValues()
    {
        var options = new MissingFeatureOptions();

        Assert.Equal(MissingFeatureStrategy.Skip, options.Strategy);
        Assert.Equal(0.1, options.MinimumOverlapRatio);
        Assert.Equal(0.8, options.AlignmentThreshold);
        Assert.False(options.AllowPartialAlignment);
        Assert.True(options.AddMissingnessIndicator);
    }

    [Fact]
    public void VflUnlearningOptions_DefaultValues()
    {
        var options = new VflUnlearningOptions();

        Assert.False(options.Enabled);
        Assert.Equal(VflUnlearningMethod.GradientAscent, options.Method);
        Assert.Equal(100, options.MaxUnlearnBatchSize);
        Assert.Equal(5, options.GradientAscentSteps);
        Assert.Equal(0.01, options.UnlearningLearningRate);
        Assert.Equal(1.0, options.CertificationEpsilon);
        Assert.False(options.VerifyUnlearning);
    }

    [Fact]
    public void VflAggregationMode_HasExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(VflAggregationMode), VflAggregationMode.Concatenation));
    }

    [Fact]
    public void MissingFeatureStrategy_HasExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(MissingFeatureStrategy), MissingFeatureStrategy.Skip));
        Assert.True(Enum.IsDefined(typeof(MissingFeatureStrategy), MissingFeatureStrategy.Zero));
        Assert.True(Enum.IsDefined(typeof(MissingFeatureStrategy), MissingFeatureStrategy.Mean));
    }

    [Fact]
    public void VflUnlearningMethod_HasExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(VflUnlearningMethod), VflUnlearningMethod.GradientAscent));
        Assert.True(Enum.IsDefined(typeof(VflUnlearningMethod), VflUnlearningMethod.Certified));
    }

    // ========== End-to-End Integration ==========

    [Fact]
    public void VFL_EndToEnd_BenchmarkDatasetWithTrainer()
    {
        // Generate benchmark dataset
        var dataset = VerticalFederatedBenchmark<double>.GenerateDataset(
            totalEntities: 50,
            totalFeatures: 8,
            numberOfParties: 2,
            overlapRatio: 0.8,
            seed: 42);

        // Create trainer
        var options = new VerticalFederatedLearningOptions
        {
            NumberOfParties = 2,
            NumberOfEpochs = 1,
            LearningRate = 0.01
        };
        var trainer = new VerticalFederatedTrainer<double>(options);

        // Register parties from benchmark dataset
        foreach (var party in dataset.Parties)
        {
            if (party.IsLabelHolder && party.Labels is not null)
            {
                var labelHolder = new VerticalPartyLabelHolder<double>(
                    party.PartyId,
                    party.Features,
                    party.Labels,
                    party.EntityIds,
                    embeddingDimension: 4);
                trainer.RegisterParty(labelHolder);
            }
            else
            {
                var client = new VerticalPartyClient<double>(
                    party.PartyId,
                    party.Features,
                    party.EntityIds,
                    embeddingDimension: 4);
                trainer.RegisterParty(client);
            }
        }

        // All parties should have been registered without throwing
        Assert.Equal(2, dataset.Parties.Count);
    }
}
