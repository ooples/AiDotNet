using System;
using System.Collections.Generic;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Regression tests for Issue #1185: OptimizationDataBatcher mutating the source tensor's
/// shape and causing CopySample to throw out-of-range when batchSize &lt; dataset rows.
/// </summary>
/// <remarks>
/// The original bug was a reference-cast on the shape array:
/// <c>var newShape = (int[])tensor._shape;</c> aliased the source's shape, so mutating
/// <c>newShape[0]</c> also mutated the source. The fix clones the array first.
/// </remarks>
public class OptimizationDataBatcherIssue1185Tests
{
    /// <summary>
    /// Reproduces the exact scenario from issue #1185: 629-row dataset, batch size 64.
    /// Before the fix, the second batch's CopySample would throw because the source
    /// tensor's first dim had been silently mutated from 629 → 64.
    /// </summary>
    [Fact]
    public void Issue1185_TensorBatcher_DoesNotMutateSourceShape()
    {
        // Arrange — match the issue's repro: 629 rows × 7 features, batch size 64.
        const int rows = 629;
        const int features = 7;
        const int batchSize = 64;

        var xTrain = new Tensor<double>(new[] { rows, features });
        var yTrain = new Tensor<double>(new[] { rows, 1 });
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < features; j++) xTrain[i, j] = i * features + j;
            yTrain[i, 0] = i;
        }

        var inputData = new OptimizationInputData<double, Tensor<double>, Tensor<double>>
        {
            XTrain = xTrain, YTrain = yTrain,
            XValidation = xTrain, YValidation = yTrain,
            XTest = xTrain, YTest = yTrain
        };

        var batcher = new OptimizationDataBatcher<double, Tensor<double>, Tensor<double>>(
            inputData, batchSize: batchSize, shuffle: true, seed: 42);

        // Act + Assert — drain a full epoch. None of the batches should throw,
        // and the source tensor's shape must be preserved across every call.
        int batchesSeen = 0;
        int totalRowsCovered = 0;
        var coveredIndices = new HashSet<int>();
        foreach (var (xBatch, yBatch, indices) in batcher.GetBatches())
        {
            Assert.Equal(rows, xTrain.Shape[0]);  // source NOT mutated
            Assert.Equal(rows, yTrain.Shape[0]);
            Assert.Equal(indices.Length, xBatch.Shape[0]);
            Assert.Equal(features, xBatch.Shape[1]);
            // Label-side shape: must mirror xBatch's batch dim and preserve
            // the trailing-1 column so loss functions don't silently see a
            // rank-collapsed target.
            Assert.Equal(indices.Length, yBatch.Shape[0]);
            Assert.Equal(1, yBatch.Shape[1]);
            foreach (var idx in indices)
            {
                Assert.InRange(idx, 0, rows - 1);
                coveredIndices.Add(idx);
            }
            batchesSeen++;
            totalRowsCovered += indices.Length;
        }

        Assert.Equal(batcher.NumBatches, batchesSeen);
        Assert.Equal(rows, totalRowsCovered);  // every row sampled exactly once
        Assert.Equal(rows, coveredIndices.Count);
    }

    /// <summary>
    /// Walks two consecutive epochs to make sure the second epoch sees the same source
    /// extent the first did. Before the fix the FIRST batch's mutation would persist into
    /// the next epoch's index generation as well.
    /// </summary>
    [Fact]
    public void Issue1185_TensorBatcher_SecondEpochStillCovers629Rows()
    {
        const int rows = 629;
        const int features = 4;
        const int batchSize = 32;

        var xTrain = new Tensor<double>(new[] { rows, features });
        var yTrain = new Tensor<double>(new[] { rows, 1 });
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < features; j++) xTrain[i, j] = i + j;
            yTrain[i, 0] = i;
        }

        var inputData = new OptimizationInputData<double, Tensor<double>, Tensor<double>>
        {
            XTrain = xTrain, YTrain = yTrain,
            XValidation = xTrain, YValidation = yTrain,
            XTest = xTrain, YTest = yTrain
        };

        var batcher = new OptimizationDataBatcher<double, Tensor<double>, Tensor<double>>(
            inputData, batchSize: batchSize, shuffle: true, seed: 7);

        for (int epoch = 0; epoch < 2; epoch++)
        {
            int total = 0;
            foreach (var (_, _, indices) in batcher.GetBatches())
            {
                total += indices.Length;
                foreach (var idx in indices)
                    Assert.InRange(idx, 0, rows - 1);
            }
            Assert.Equal(rows, total);
            Assert.Equal(rows, xTrain.Shape[0]);
            Assert.Equal(rows, yTrain.Shape[0]);
        }
    }

    /// <summary>
    /// Higher-rank tensors (e.g. image tensors with shape [N, C, H, W]) must also have
    /// their shape preserved.
    /// </summary>
    [Fact]
    public void Issue1185_TensorBatcher_PreservesShapeForRank4Inputs()
    {
        const int rows = 200;
        const int channels = 3;
        const int height = 8;
        const int width = 8;
        const int batchSize = 16;

        var xTrain = new Tensor<double>(new[] { rows, channels, height, width });
        var yTrain = new Tensor<double>(new[] { rows, 1 });
        for (int i = 0; i < rows; i++) yTrain[i, 0] = i;

        var inputData = new OptimizationInputData<double, Tensor<double>, Tensor<double>>
        {
            XTrain = xTrain, YTrain = yTrain,
            XValidation = xTrain, YValidation = yTrain,
            XTest = xTrain, YTest = yTrain
        };

        var batcher = new OptimizationDataBatcher<double, Tensor<double>, Tensor<double>>(
            inputData, batchSize: batchSize, shuffle: true, seed: 1);

        foreach (var (xBatch, yBatch, indices) in batcher.GetBatches())
        {
            Assert.Equal(rows, xTrain.Shape[0]);
            Assert.Equal(channels, xTrain.Shape[1]);
            Assert.Equal(height, xTrain.Shape[2]);
            Assert.Equal(width, xTrain.Shape[3]);
            Assert.Equal(indices.Length, xBatch.Shape[0]);
            Assert.Equal(channels, xBatch.Shape[1]);
            Assert.Equal(height, xBatch.Shape[2]);
            Assert.Equal(width, xBatch.Shape[3]);
            // Label-side shape: matches xBatch's batch dim and keeps the
            // trailing-1 column the source yTrain was constructed with.
            Assert.Equal(indices.Length, yBatch.Shape[0]);
            Assert.Equal(1, yBatch.Shape[1]);
        }
    }
}
