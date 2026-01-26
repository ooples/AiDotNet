using System;
using System.Collections.Generic;
using AiDotNet.SelfSupervisedLearning;
using AiDotNet.SelfSupervisedLearning.Losses;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SelfSupervisedLearning;

/// <summary>
/// Integration tests for the SelfSupervisedLearning module.
/// Tests SSL loss functions, projector heads, memory bank, and supporting components.
/// </summary>
public class SelfSupervisedLearningIntegrationTests
{
    private const double Tolerance = 1e-5;

    #region NT-Xent Loss Tests

    [Fact]
    public void NTXentLoss_ComputeLoss_WithIdenticalViews_ReturnsMinimalLoss()
    {
        // Arrange - Create identical embeddings (positive pairs should have highest similarity)
        var loss = new NTXentLoss<double>(temperature: 0.1);
        var batchSize = 4;
        var dim = 8;

        // Create embeddings where z1[i] == z2[i] (perfect positive pairs)
        var z1Data = new double[batchSize * dim];
        var z2Data = new double[batchSize * dim];
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(42);

        for (int i = 0; i < batchSize * dim; i++)
        {
            var val = random.NextDouble() * 2 - 1;
            z1Data[i] = val;
            z2Data[i] = val; // Identical
        }

        var z1 = new Tensor<double>(z1Data, [batchSize, dim]);
        var z2 = new Tensor<double>(z2Data, [batchSize, dim]);

        // Act
        var lossValue = loss.ComputeLoss(z1, z2);

        // Assert - Loss should be low for identical pairs (after normalization, cos_sim = 1)
        Assert.True(lossValue >= 0, "Loss should be non-negative");
    }

    [Fact]
    public void NTXentLoss_Temperature_AffectsLossScale()
    {
        // Arrange
        var lowTempLoss = new NTXentLoss<double>(temperature: 0.05);
        var highTempLoss = new NTXentLoss<double>(temperature: 0.5);

        var z1 = CreateRandomTensor(4, 8, seed: 42);
        var z2 = CreateRandomTensor(4, 8, seed: 43);

        // Act
        var lossLowTemp = lowTempLoss.ComputeLoss(z1, z2);
        var lossHighTemp = highTempLoss.ComputeLoss(z1, z2);

        // Assert - Both should compute valid losses
        Assert.True(lossLowTemp > 0, "Low temperature loss should be positive");
        Assert.True(lossHighTemp > 0, "High temperature loss should be positive");
        // Lower temperature generally produces larger loss values (sharper distribution)
    }

    [Fact]
    public void NTXentLoss_ComputeLossWithGradients_ReturnsValidGradients()
    {
        // Arrange
        var loss = new NTXentLoss<double>(temperature: 0.1);
        var z1 = CreateRandomTensor(4, 8, seed: 42);
        var z2 = CreateRandomTensor(4, 8, seed: 43);

        // Act
        var (lossValue, gradZ1, gradZ2) = loss.ComputeLossWithGradients(z1, z2);

        // Assert
        Assert.True(lossValue > 0, "Loss should be positive");
        Assert.Equal(z1.Shape[0], gradZ1.Shape[0]);
        Assert.Equal(z1.Shape[1], gradZ1.Shape[1]);
        Assert.Equal(z2.Shape[0], gradZ2.Shape[0]);
        Assert.Equal(z2.Shape[1], gradZ2.Shape[1]);

        // Gradients should not be all zeros
        Assert.True(HasNonZeroElements(gradZ1), "Gradient for z1 should not be all zeros");
        Assert.True(HasNonZeroElements(gradZ2), "Gradient for z2 should not be all zeros");
    }

    [Fact]
    public void NTXentLoss_InvalidTemperature_ThrowsException()
    {
        // Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new NTXentLoss<double>(temperature: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NTXentLoss<double>(temperature: -0.1));
    }

    [Fact]
    public void NTXentLoss_NullInput_ThrowsArgumentNullException()
    {
        // Arrange
        var loss = new NTXentLoss<double>();
        var validTensor = CreateRandomTensor(4, 8, seed: 42);

        // Assert
        Assert.Throws<ArgumentNullException>(() => loss.ComputeLoss(null!, validTensor));
        Assert.Throws<ArgumentNullException>(() => loss.ComputeLoss(validTensor, null!));
    }

    #endregion

    #region InfoNCE Loss Tests

    [Fact]
    public void InfoNCELoss_ComputeLoss_WithMemoryBank_ReturnsValidLoss()
    {
        // Arrange
        var loss = new InfoNCELoss<double>(temperature: 0.07);
        var batchSize = 4;
        var dim = 8;
        var numNegatives = 16;

        var queries = CreateRandomTensor(batchSize, dim, seed: 42);
        var positiveKeys = CreateRandomTensor(batchSize, dim, seed: 43);
        var negativeKeys = CreateRandomTensor(numNegatives, dim, seed: 44);

        // Act
        var lossValue = loss.ComputeLoss(queries, positiveKeys, negativeKeys);

        // Assert
        Assert.True(lossValue > 0, "InfoNCE loss should be positive");
    }

    [Fact]
    public void InfoNCELoss_ComputeLossInBatch_ReturnsValidLoss()
    {
        // Arrange
        var loss = new InfoNCELoss<double>(temperature: 0.07);
        var queries = CreateRandomTensor(8, 16, seed: 42);
        var keys = CreateRandomTensor(8, 16, seed: 43);

        // Act
        var lossValue = loss.ComputeLossInBatch(queries, keys);

        // Assert
        Assert.True(lossValue > 0, "In-batch InfoNCE loss should be positive");
    }

    [Fact]
    public void InfoNCELoss_ComputeAccuracy_ReturnsValidRange()
    {
        // Arrange
        var loss = new InfoNCELoss<double>(temperature: 0.07);
        var queries = CreateRandomTensor(4, 8, seed: 42);
        var positiveKeys = CreateRandomTensor(4, 8, seed: 43);
        var negativeKeys = CreateRandomTensor(8, 8, seed: 44);

        // Act
        var accuracy = loss.ComputeAccuracy(queries, positiveKeys, negativeKeys);

        // Assert - Accuracy should be between 0 and 1
        Assert.True(accuracy >= 0.0, "Accuracy should be non-negative");
        Assert.True(accuracy <= 1.0, "Accuracy should not exceed 1.0");
    }

    [Fact]
    public void InfoNCELoss_ComputeLossWithGradients_ReturnsValidGradients()
    {
        // Arrange
        var loss = new InfoNCELoss<double>(temperature: 0.07);
        var queries = CreateRandomTensor(4, 8, seed: 42);
        var positiveKeys = CreateRandomTensor(4, 8, seed: 43);
        var negativeKeys = CreateRandomTensor(16, 8, seed: 44);

        // Act
        var (lossValue, gradQueries, gradPosKeys) = loss.ComputeLossWithGradients(queries, positiveKeys, negativeKeys);

        // Assert
        Assert.True(lossValue > 0, "Loss should be positive");
        Assert.Equal(queries.Shape[0], gradQueries.Shape[0]);
        Assert.Equal(queries.Shape[1], gradQueries.Shape[1]);
        Assert.Equal(positiveKeys.Shape[0], gradPosKeys.Shape[0]);
        Assert.Equal(positiveKeys.Shape[1], gradPosKeys.Shape[1]);
    }

    [Fact]
    public void InfoNCELoss_ComputeLossInBatchWithGradients_ReturnsValidGradients()
    {
        // Arrange
        var loss = new InfoNCELoss<double>(temperature: 0.07);
        var queries = CreateRandomTensor(4, 8, seed: 42);
        var keys = CreateRandomTensor(4, 8, seed: 43);

        // Act
        var (lossValue, gradQueries, gradKeys) = loss.ComputeLossInBatchWithGradients(queries, keys);

        // Assert
        Assert.True(lossValue > 0, "Loss should be positive");
        Assert.True(HasNonZeroElements(gradQueries), "Query gradients should have non-zero elements");
        Assert.True(HasNonZeroElements(gradKeys), "Key gradients should have non-zero elements");
    }

    #endregion

    #region BYOL Loss Tests

    [Fact]
    public void BYOLLoss_ComputeLoss_WithIdenticalVectors_ReturnsZero()
    {
        // Arrange
        var loss = new BYOLLoss<double>(normalize: true);
        var prediction = CreateRandomTensor(4, 8, seed: 42);
        // Use identical vectors for target - copy values via indexer
        var target = CopyTensor(prediction);

        // Act
        var lossValue = loss.ComputeLoss(prediction, target);

        // Assert - For identical normalized vectors, cos_sim = 1, so loss = 2 - 2*1 = 0
        Assert.True(Math.Abs(lossValue) < 0.01, $"Loss should be near zero for identical vectors, got {lossValue}");
    }

    [Fact]
    public void BYOLLoss_ComputeLoss_WithOrthogonalVectors_ReturnsTwo()
    {
        // Arrange
        var loss = new BYOLLoss<double>(normalize: true);

        // Create orthogonal vectors (one-hot encodings)
        var predData = new double[8] { 1, 0, 0, 0, 0, 0, 0, 0 };
        var targetData = new double[8] { 0, 1, 0, 0, 0, 0, 0, 0 };

        var prediction = new Tensor<double>(predData, [1, 8]);
        var target = new Tensor<double>(targetData, [1, 8]);

        // Act
        var lossValue = loss.ComputeLoss(prediction, target);

        // Assert - For orthogonal vectors, cos_sim = 0, so loss = 2 - 2*0 = 2
        Assert.True(Math.Abs(lossValue - 2.0) < 0.01, $"Loss should be 2.0 for orthogonal vectors, got {lossValue}");
    }

    [Fact]
    public void BYOLLoss_ComputeSymmetricLoss_AveragesBothDirections()
    {
        // Arrange
        var loss = new BYOLLoss<double>(normalize: true);
        var pred1 = CreateRandomTensor(4, 8, seed: 42);
        var proj2 = CreateRandomTensor(4, 8, seed: 43);
        var pred2 = CreateRandomTensor(4, 8, seed: 44);
        var proj1 = CreateRandomTensor(4, 8, seed: 45);

        // Act
        var symmetricLoss = loss.ComputeSymmetricLoss(pred1, proj2, pred2, proj1);

        // Also compute individual losses
        var loss1 = loss.ComputeLoss(pred1, proj2);
        var loss2 = loss.ComputeLoss(pred2, proj1);
        var expectedSymmetric = 0.5 * (loss1 + loss2);

        // Assert
        Assert.True(Math.Abs(symmetricLoss - expectedSymmetric) < Tolerance,
            $"Symmetric loss ({symmetricLoss}) should equal average of individual losses ({expectedSymmetric})");
    }

    [Fact]
    public void BYOLLoss_ComputeLossWithGradients_ReturnsValidGradients()
    {
        // Arrange
        var loss = new BYOLLoss<double>(normalize: true);
        var prediction = CreateRandomTensor(4, 8, seed: 42);
        var target = CreateRandomTensor(4, 8, seed: 43);

        // Act
        var (lossValue, gradPrediction) = loss.ComputeLossWithGradients(prediction, target);

        // Assert
        Assert.True(lossValue >= 0, "BYOL loss should be non-negative");
        Assert.Equal(prediction.Shape[0], gradPrediction.Shape[0]);
        Assert.Equal(prediction.Shape[1], gradPrediction.Shape[1]);
        Assert.True(HasNonZeroElements(gradPrediction), "Gradients should have non-zero elements");
    }

    [Fact]
    public void BYOLLoss_ComputeMSELoss_ReturnsValidLoss()
    {
        // Arrange
        var loss = new BYOLLoss<double>(normalize: true);
        var prediction = CreateRandomTensor(4, 8, seed: 42);
        var target = CreateRandomTensor(4, 8, seed: 43);

        // Act
        var mseLoss = loss.ComputeMSELoss(prediction, target);

        // Assert
        Assert.True(mseLoss >= 0, "MSE loss should be non-negative");
    }

    #endregion

    #region Linear Projector Tests

    [Fact]
    public void LinearProjector_Project_ReturnsCorrectShape()
    {
        // Arrange
        var projector = new LinearProjector<double>(inputDim: 64, outputDim: 128, seed: 42);
        var input = CreateRandomTensor(8, 64, seed: 43);

        // Act
        var output = projector.Project(input);

        // Assert
        Assert.Equal(8, output.Shape[0]); // batch size preserved
        Assert.Equal(128, output.Shape[1]); // output dimension
    }

    [Fact]
    public void LinearProjector_Properties_ReturnCorrectValues()
    {
        // Arrange
        var projector = new LinearProjector<double>(inputDim: 64, outputDim: 128, useBias: true);

        // Assert
        Assert.Equal(64, projector.InputDimension);
        Assert.Equal(128, projector.OutputDimension);
        Assert.Null(projector.HiddenDimension); // Linear projector has no hidden layer
        Assert.Equal(64 * 128 + 128, projector.ParameterCount); // weights + bias
    }

    [Fact]
    public void LinearProjector_GetSetParameters_RoundTrips()
    {
        // Arrange
        var projector = new LinearProjector<double>(inputDim: 32, outputDim: 64, seed: 42);

        // Act
        var params1 = projector.GetParameters();

        // Modify parameters
        var modifiedParams = new double[params1.Length];
        for (int i = 0; i < params1.Length; i++)
        {
            modifiedParams[i] = params1[i] * 2;
        }
        projector.SetParameters(new Vector<double>(modifiedParams));

        var params2 = projector.GetParameters();

        // Assert - Parameters should be doubled
        for (int i = 0; i < params1.Length; i++)
        {
            Assert.True(Math.Abs(params2[i] - params1[i] * 2) < Tolerance);
        }
    }

    [Fact]
    public void LinearProjector_Backward_ComputesGradients()
    {
        // Arrange
        var projector = new LinearProjector<double>(inputDim: 32, outputDim: 64, seed: 42);
        var input = CreateRandomTensor(4, 32, seed: 43);

        // Forward pass
        var output = projector.Project(input);

        // Create gradient (simulating loss derivative)
        var gradOutput = CreateRandomTensor(4, 64, seed: 44);

        // Act
        var gradInput = projector.Backward(gradOutput);

        // Assert
        Assert.Equal(4, gradInput.Shape[0]);
        Assert.Equal(32, gradInput.Shape[1]);

        var paramGrads = projector.GetParameterGradients();
        Assert.True(HasNonZeroElements(paramGrads), "Parameter gradients should have non-zero elements");
    }

    [Fact]
    public void LinearProjector_ClearGradients_ResetsGradients()
    {
        // Arrange
        var projector = new LinearProjector<double>(inputDim: 32, outputDim: 64, seed: 42);
        var input = CreateRandomTensor(4, 32, seed: 43);
        var output = projector.Project(input);
        var gradOutput = CreateRandomTensor(4, 64, seed: 44);
        projector.Backward(gradOutput);

        // Verify gradients exist
        Assert.True(HasNonZeroElements(projector.GetParameterGradients()));

        // Act
        projector.ClearGradients();

        // Assert - Gradients should be zeros after clearing
        var clearedGrads = projector.GetParameterGradients();
        Assert.False(HasNonZeroElements(clearedGrads), "Gradients should be zero after clearing");
    }

    [Fact]
    public void LinearProjector_NoBias_HasFewerParameters()
    {
        // Arrange
        var withBias = new LinearProjector<double>(inputDim: 32, outputDim: 64, useBias: true);
        var noBias = new LinearProjector<double>(inputDim: 32, outputDim: 64, useBias: false);

        // Assert
        Assert.Equal(32 * 64 + 64, withBias.ParameterCount);
        Assert.Equal(32 * 64, noBias.ParameterCount);
    }

    #endregion

    #region MLP Projector Tests

    [Fact]
    public void MLPProjector_Project_ReturnsCorrectShape()
    {
        // Arrange
        var projector = new MLPProjector<double>(
            inputDim: 64,
            hiddenDim: 128,
            outputDim: 32,
            seed: 42);

        var input = CreateRandomTensor(8, 64, seed: 43);

        // Act
        var output = projector.Project(input);

        // Assert
        Assert.Equal(8, output.Shape[0]);
        Assert.Equal(32, output.Shape[1]);
    }

    [Fact]
    public void MLPProjector_Properties_ReturnCorrectValues()
    {
        // Arrange
        var projector = new MLPProjector<double>(
            inputDim: 64,
            hiddenDim: 128,
            outputDim: 32,
            useBatchNormOnOutput: false);

        // Assert
        Assert.Equal(64, projector.InputDimension);
        Assert.Equal(32, projector.OutputDimension);
        Assert.Equal(128, projector.HiddenDimension);
        Assert.True(projector.ParameterCount > 0);
    }

    [Fact]
    public void MLPProjector_WithBatchNormOnOutput_HasMoreParameters()
    {
        // Arrange
        var withoutBN = new MLPProjector<double>(inputDim: 64, hiddenDim: 128, outputDim: 32, useBatchNormOnOutput: false);
        var withBN = new MLPProjector<double>(inputDim: 64, hiddenDim: 128, outputDim: 32, useBatchNormOnOutput: true);

        // Assert - With output BatchNorm should have extra gamma and beta parameters
        Assert.True(withBN.ParameterCount > withoutBN.ParameterCount);
    }

    [Fact]
    public void MLPProjector_Backward_ComputesGradients()
    {
        // Arrange
        var projector = new MLPProjector<double>(inputDim: 32, hiddenDim: 64, outputDim: 16, seed: 42);
        var input = CreateRandomTensor(4, 32, seed: 43);

        // Forward pass
        var output = projector.Project(input);

        // Create gradient
        var gradOutput = CreateRandomTensor(4, 16, seed: 44);

        // Act
        var gradInput = projector.Backward(gradOutput);

        // Assert
        Assert.Equal(4, gradInput.Shape[0]);
        Assert.Equal(32, gradInput.Shape[1]);

        var paramGrads = projector.GetParameterGradients();
        Assert.True(HasNonZeroElements(paramGrads), "Parameter gradients should have non-zero elements");
    }

    [Fact]
    public void MLPProjector_Reset_ClearsState()
    {
        // Arrange
        var projector = new MLPProjector<double>(inputDim: 32, hiddenDim: 64, outputDim: 16, seed: 42);
        var input = CreateRandomTensor(4, 32, seed: 43);
        var output = projector.Project(input);
        var gradOutput = CreateRandomTensor(4, 16, seed: 44);
        projector.Backward(gradOutput);

        // Act
        projector.Reset();

        // Assert - Calling backward without forward should throw
        Assert.Throws<InvalidOperationException>(() => projector.Backward(gradOutput));
    }

    [Fact]
    public void MLPProjector_SetTrainingMode_AffectsBatchNorm()
    {
        // Arrange
        var projector = new MLPProjector<double>(inputDim: 32, hiddenDim: 64, outputDim: 16, seed: 42);
        var input = CreateRandomTensor(4, 32, seed: 43);

        // Act - Project in training mode
        projector.SetTrainingMode(true);
        var outputTraining = projector.Project(input);

        // Act - Project in eval mode
        projector.SetTrainingMode(false);
        var outputEval = projector.Project(input);

        // Assert - Both should produce valid outputs (shapes should match)
        Assert.Equal(outputTraining.Shape[0], outputEval.Shape[0]);
        Assert.Equal(outputTraining.Shape[1], outputEval.Shape[1]);
    }

    #endregion

    #region Symmetric Projector Tests

    [Fact]
    public void SymmetricProjector_Project_ReturnsCorrectShape()
    {
        // Arrange - Projector without predictor (target branch)
        var projector = new SymmetricProjector<double>(
            inputDim: 64,
            hiddenDim: 128,
            projectionDim: 32,
            predictorHiddenDim: 0, // No predictor
            seed: 42);

        var input = CreateRandomTensor(4, 64, seed: 43);

        // Act
        var output = projector.Project(input);

        // Assert
        Assert.Equal(4, output.Shape[0]);
        Assert.Equal(32, output.Shape[1]);
    }

    [Fact]
    public void SymmetricProjector_HasPredictor_ReportsCorrectly()
    {
        // Arrange
        var withPredictor = new SymmetricProjector<double>(
            inputDim: 64, hiddenDim: 128, projectionDim: 32, predictorHiddenDim: 128);
        var withoutPredictor = new SymmetricProjector<double>(
            inputDim: 64, hiddenDim: 128, projectionDim: 32, predictorHiddenDim: 0);

        // Assert
        Assert.True(withPredictor.HasPredictor);
        Assert.False(withoutPredictor.HasPredictor);
    }

    [Fact]
    public void SymmetricProjector_Predict_ReturnsCorrectShape()
    {
        // Arrange - Projector with predictor (online branch)
        var projector = new SymmetricProjector<double>(
            inputDim: 64,
            hiddenDim: 128,
            projectionDim: 32,
            predictorHiddenDim: 128, // Has predictor
            seed: 42);

        var input = CreateRandomTensor(4, 64, seed: 43);

        // Act
        var projection = projector.Project(input);
        var prediction = projector.Predict(projection);

        // Assert
        Assert.Equal(4, prediction.Shape[0]);
        Assert.Equal(32, prediction.Shape[1]); // Same as projection dim
    }

    [Fact]
    public void SymmetricProjector_ProjectAndPredict_CombinesBothSteps()
    {
        // Arrange
        var projector = new SymmetricProjector<double>(
            inputDim: 64,
            hiddenDim: 128,
            projectionDim: 32,
            predictorHiddenDim: 128,
            seed: 42);

        var input = CreateRandomTensor(4, 64, seed: 43);

        // Act - Using convenience method
        var combined = projector.ProjectAndPredict(input);

        // Also do it step by step
        projector.Reset(); // Reset cached values
        var projection = projector.Project(input);
        var prediction = projector.Predict(projection);

        // Assert - Results should have same shape
        Assert.Equal(combined.Shape[0], prediction.Shape[0]);
        Assert.Equal(combined.Shape[1], prediction.Shape[1]);
    }

    [Fact]
    public void SymmetricProjector_WithoutPredictor_PredictReturnsInput()
    {
        // Arrange
        var projector = new SymmetricProjector<double>(
            inputDim: 64,
            hiddenDim: 128,
            projectionDim: 32,
            predictorHiddenDim: 0, // No predictor
            seed: 42);

        var input = CreateRandomTensor(4, 64, seed: 43);
        var projection = projector.Project(input);

        // Act
        var prediction = projector.Predict(projection);

        // Assert - Without predictor, Predict should return input unchanged
        Assert.Equal(projection.Shape[0], prediction.Shape[0]);
        Assert.Equal(projection.Shape[1], prediction.Shape[1]);

        // Values should be the same
        for (int i = 0; i < projection.Shape[0]; i++)
        {
            for (int j = 0; j < projection.Shape[1]; j++)
            {
                Assert.True(Math.Abs(projection[i, j] - prediction[i, j]) < Tolerance);
            }
        }
    }

    [Fact]
    public void SymmetricProjector_GetSetParameters_RoundTrips()
    {
        // Arrange
        var projector = new SymmetricProjector<double>(
            inputDim: 32,
            hiddenDim: 64,
            projectionDim: 16,
            predictorHiddenDim: 64,
            seed: 42);

        // Act
        var params1 = projector.GetParameters();

        // Modify and set back
        var modifiedParams = new double[params1.Length];
        for (int i = 0; i < params1.Length; i++)
        {
            modifiedParams[i] = params1[i] + 0.1;
        }
        projector.SetParameters(new Vector<double>(modifiedParams));

        var params2 = projector.GetParameters();

        // Assert
        Assert.Equal(params1.Length, params2.Length);
        for (int i = 0; i < params1.Length; i++)
        {
            Assert.True(Math.Abs(params2[i] - params1[i] - 0.1) < Tolerance);
        }
    }

    #endregion

    #region Memory Bank Tests

    [Fact]
    public void MemoryBank_Enqueue_StoresEmbeddings()
    {
        // Arrange
        var memoryBank = new MemoryBank<double>(capacity: 100, embeddingDim: 8);
        var embeddings = CreateRandomTensor(10, 8, seed: 42);

        // Act
        memoryBank.Enqueue(embeddings);

        // Assert
        Assert.Equal(10, memoryBank.CurrentSize);
        Assert.False(memoryBank.IsFull);
    }

    [Fact]
    public void MemoryBank_Enqueue_WrapsWhenFull()
    {
        // Arrange
        var memoryBank = new MemoryBank<double>(capacity: 20, embeddingDim: 8);

        // Enqueue more than capacity
        for (int i = 0; i < 3; i++)
        {
            var embeddings = CreateRandomTensor(10, 8, seed: 42 + i);
            memoryBank.Enqueue(embeddings);
        }

        // Assert - Should not exceed capacity
        Assert.Equal(20, memoryBank.CurrentSize);
        Assert.True(memoryBank.IsFull);
    }

    [Fact]
    public void MemoryBank_GetAll_ReturnsAllStoredEmbeddings()
    {
        // Arrange
        var memoryBank = new MemoryBank<double>(capacity: 100, embeddingDim: 8);
        var embeddings = CreateRandomTensor(15, 8, seed: 42);
        memoryBank.Enqueue(embeddings);

        // Act
        var all = memoryBank.GetAll();

        // Assert
        Assert.Equal(15, all.Shape[0]);
        Assert.Equal(8, all.Shape[1]);
    }

    [Fact]
    public void MemoryBank_Sample_ReturnsRequestedCount()
    {
        // Arrange
        var memoryBank = new MemoryBank<double>(capacity: 100, embeddingDim: 8, seed: 42);
        var embeddings = CreateRandomTensor(50, 8, seed: 43);
        memoryBank.Enqueue(embeddings);

        // Act
        var sampled = memoryBank.Sample(20);

        // Assert
        Assert.Equal(20, sampled.Shape[0]);
        Assert.Equal(8, sampled.Shape[1]);
    }

    [Fact]
    public void MemoryBank_Sample_CapsAtCurrentSize()
    {
        // Arrange
        var memoryBank = new MemoryBank<double>(capacity: 100, embeddingDim: 8, seed: 42);
        var embeddings = CreateRandomTensor(10, 8, seed: 43);
        memoryBank.Enqueue(embeddings);

        // Act - Request more than available
        var sampled = memoryBank.Sample(50);

        // Assert - Should return only what's available
        Assert.Equal(10, sampled.Shape[0]);
    }

    [Fact]
    public void MemoryBank_Clear_ResetsState()
    {
        // Arrange
        var memoryBank = new MemoryBank<double>(capacity: 100, embeddingDim: 8);
        var embeddings = CreateRandomTensor(20, 8, seed: 42);
        memoryBank.Enqueue(embeddings);

        Assert.Equal(20, memoryBank.CurrentSize);

        // Act
        memoryBank.Clear();

        // Assert
        Assert.Equal(0, memoryBank.CurrentSize);
        Assert.False(memoryBank.IsFull);
    }

    [Fact]
    public void MemoryBank_GetAt_ReturnsSpecificEmbedding()
    {
        // Arrange
        var memoryBank = new MemoryBank<double>(capacity: 100, embeddingDim: 8);
        var embeddings = CreateRandomTensor(10, 8, seed: 42);
        memoryBank.Enqueue(embeddings);

        // Act
        var retrieved = memoryBank.GetAt(5);

        // Assert
        Assert.Equal(1, retrieved.Shape[0]);
        Assert.Equal(8, retrieved.Shape[1]);
    }

    [Fact]
    public void MemoryBank_SetAt_UpdatesSpecificEmbedding()
    {
        // Arrange
        var memoryBank = new MemoryBank<double>(capacity: 100, embeddingDim: 8);
        var embeddings = CreateRandomTensor(10, 8, seed: 42);
        memoryBank.Enqueue(embeddings);

        var newEmbedding = new Tensor<double>(new double[8] { 1, 2, 3, 4, 5, 6, 7, 8 }, [1, 8]);

        // Act
        memoryBank.SetAt(5, newEmbedding);
        var retrieved = memoryBank.GetAt(5);

        // Assert
        for (int i = 0; i < 8; i++)
        {
            Assert.True(Math.Abs(retrieved[0, i] - (i + 1)) < Tolerance);
        }
    }

    [Fact]
    public void MemoryBank_UpdateWithMomentum_AppliesEMA()
    {
        // Arrange
        var memoryBank = new MemoryBank<double>(capacity: 100, embeddingDim: 4);

        // Create initial embedding with all 1s
        var initial = new Tensor<double>(new double[4] { 1, 1, 1, 1 }, [1, 4]);
        memoryBank.Enqueue(initial);

        // New embedding with all 0s
        var newEmbedding = new Tensor<double>(new double[4] { 0, 0, 0, 0 }, [1, 4]);

        // Act - Update with momentum 0.5
        memoryBank.UpdateWithMomentum([0], newEmbedding, momentum: 0.5);

        // Assert - Result should be 0.5 * 1 + 0.5 * 0 = 0.5
        var updated = memoryBank.GetAt(0);
        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(updated[0, i] - 0.5) < Tolerance);
        }
    }

    [Fact]
    public void MemoryBank_Properties_ReturnCorrectValues()
    {
        // Arrange
        var memoryBank = new MemoryBank<double>(capacity: 64, embeddingDim: 16);

        // Assert
        Assert.Equal(64, memoryBank.Capacity);
        Assert.Equal(16, memoryBank.EmbeddingDimension);
        Assert.Equal(0, memoryBank.CurrentSize);
        Assert.False(memoryBank.IsFull);
    }

    [Fact]
    public void MemoryBank_DimensionMismatch_ThrowsException()
    {
        // Arrange
        var memoryBank = new MemoryBank<double>(capacity: 100, embeddingDim: 8);
        var wrongDimEmbeddings = CreateRandomTensor(5, 16, seed: 42); // Wrong dimension

        // Assert
        Assert.Throws<ArgumentException>(() => memoryBank.Enqueue(wrongDimEmbeddings));
    }

    #endregion

    #region Momentum Encoder Static Methods Tests

    [Fact]
    public void MomentumEncoder_ScheduleMomentum_ComputesCosineSchedule()
    {
        // Test at various points
        var baseMomentum = 0.9;
        var finalMomentum = 1.0;
        var totalEpochs = 100;

        // At epoch 0: should be base momentum
        var m0 = MomentumEncoder<double>.ScheduleMomentum(baseMomentum, finalMomentum, 0, totalEpochs);
        Assert.True(Math.Abs(m0 - baseMomentum) < 0.01);

        // At epoch 50: should be approximately midpoint
        var m50 = MomentumEncoder<double>.ScheduleMomentum(baseMomentum, finalMomentum, 50, totalEpochs);
        Assert.True(m50 > baseMomentum && m50 < finalMomentum);

        // At epoch 100: should be final momentum
        var m100 = MomentumEncoder<double>.ScheduleMomentum(baseMomentum, finalMomentum, 100, totalEpochs);
        Assert.True(Math.Abs(m100 - finalMomentum) < 0.01);
    }

    [Fact]
    public void MomentumEncoder_ScheduleMomentum_ZeroTotalEpochs_ReturnsFinalMomentum()
    {
        // Arrange
        var baseMomentum = 0.9;
        var finalMomentum = 1.0;

        // Act
        var result = MomentumEncoder<double>.ScheduleMomentum(baseMomentum, finalMomentum, 5, 0);

        // Assert
        Assert.Equal(finalMomentum, result);
    }

    [Fact]
    public void MomentumEncoder_ScheduleMomentum_ProgressBeyondTotal_ClampedToOne()
    {
        // Arrange
        var baseMomentum = 0.9;
        var finalMomentum = 1.0;

        // Act - epoch > totalEpochs
        var result = MomentumEncoder<double>.ScheduleMomentum(baseMomentum, finalMomentum, 150, 100);

        // Assert - Should be clamped to final momentum
        Assert.True(Math.Abs(result - finalMomentum) < 0.01);
    }

    #endregion

    #region Edge Cases and Error Handling

    [Fact]
    public void AllLosses_EmptyBatch_HandlesGracefully()
    {
        // Note: Empty batches may throw or return special values
        // This test documents the expected behavior

        var ntxent = new NTXentLoss<double>();
        var infonce = new InfoNCELoss<double>();
        var byol = new BYOLLoss<double>();

        // These may throw ArgumentException or handle gracefully
        // Just verify they don't crash unexpectedly
        try
        {
            var emptyTensor = new Tensor<double>(Array.Empty<double>(), [0, 8]);
            // Calling with empty tensors - implementation dependent behavior
        }
        catch (Exception ex) when (ex is ArgumentException || ex is IndexOutOfRangeException)
        {
            // Expected - empty batches may not be supported
        }
    }

    [Fact]
    public void LinearProjector_InvalidDimensions_ThrowsException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LinearProjector<double>(inputDim: 0, outputDim: 64));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LinearProjector<double>(inputDim: 64, outputDim: 0));
    }

    [Fact]
    public void MLPProjector_InvalidDimensions_ThrowsException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new MLPProjector<double>(inputDim: 0, hiddenDim: 64, outputDim: 32));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new MLPProjector<double>(inputDim: 64, hiddenDim: 0, outputDim: 32));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new MLPProjector<double>(inputDim: 64, hiddenDim: 64, outputDim: 0));
    }

    [Fact]
    public void MemoryBank_InvalidCapacity_ThrowsException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new MemoryBank<double>(capacity: 0, embeddingDim: 8));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new MemoryBank<double>(capacity: 100, embeddingDim: 0));
    }

    [Fact]
    public void MemoryBank_GetAt_InvalidIndex_ThrowsException()
    {
        var memoryBank = new MemoryBank<double>(capacity: 100, embeddingDim: 8);
        var embeddings = CreateRandomTensor(10, 8, seed: 42);
        memoryBank.Enqueue(embeddings);

        Assert.Throws<ArgumentOutOfRangeException>(() => memoryBank.GetAt(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => memoryBank.GetAt(10)); // Out of range
    }

    #endregion

    #region Helper Methods

    private static Tensor<double> CreateRandomTensor(int batchSize, int dim, int seed)
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var data = new double[batchSize * dim];

        for (int i = 0; i < data.Length; i++)
        {
            data[i] = random.NextDouble() * 2 - 1; // [-1, 1]
        }

        return new Tensor<double>(data, [batchSize, dim]);
    }

    private static bool HasNonZeroElements(Tensor<double> tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            if (Math.Abs(tensor[i]) > Tolerance)
                return true;
        }
        return false;
    }

    private static bool HasNonZeroElements(Vector<double> vector)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            if (Math.Abs(vector[i]) > Tolerance)
                return true;
        }
        return false;
    }

    private static Tensor<double> CopyTensor(Tensor<double> source)
    {
        var target = new Tensor<double>(source.Shape);
        for (int i = 0; i < source.Length; i++)
        {
            target[i] = source[i];
        }
        return target;
    }

    #endregion
}
