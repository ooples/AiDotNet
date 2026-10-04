using System;
using System.Linq;
using AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// ObjectDetectionOptions.RandomSeed must decide a detector's initial weights. A detector builds a backbone
/// that is itself a network with no seed of its own, and the backbone used to end the detector's seed scope:
/// its weights, and everything built after it, then came from the process-shared generator, so the same
/// options gave different weights depending on what had run before (FasterRCNN's Train_ShouldReduceLoss
/// passed alone and failed with its class).
/// </summary>
[Collection("FusedTrainingSerial")]
public class DetectorInitializationSeedTests
{
    private static double[] InitialWeights(int seed)
    {
        var options = new ObjectDetectionOptions<double>
        {
            Size = ModelSize.Small, NumClasses = 3, InputSize = new[] { 64, 64 }, RandomSeed = seed
        };
        var model = new FasterRCNN<double>(options);
        // The heads resolve their shapes on the first forward; read the weights after it.
        model.Predict(new Tensor<double>(new[] { 1, 3, 64, 64 }));
        return ((IParameterizable<double, Tensor<double>, Tensor<double>>)model).GetParameters().ToArray();
    }

    [Fact]
    public void SameSeed_GivesTheSameWeights_WhateverRanBefore()
    {
        var first = InitialWeights(7);
        // Move the process-shared generator, as earlier tests in a run do.
        for (int i = 0; i < 1000; i++) _ = RandomHelper.ThreadSafeRandom.Next();
        var second = InitialWeights(7);

        Assert.Equal(first.Length, second.Length);
        Assert.True(first.SequenceEqual(second),
            "Two detectors built from the same RandomSeed got different initial weights.");
    }

    [Fact]
    public void DifferentSeeds_GiveDifferentWeights()
    {
        // Positive control: the seed must actually reach the weights, not merely be ignored consistently.
        Assert.False(InitialWeights(7).SequenceEqual(InitialWeights(8)),
            "RandomSeed did not change the initial weights.");
    }
}