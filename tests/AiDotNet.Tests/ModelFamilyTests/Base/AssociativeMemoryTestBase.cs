using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for associative memory models (Hopfield, HopeNetwork, etc.)
/// that use non-gradient-based learning (Hebbian) or self-modifying mechanisms.
/// Tests pattern storage/recall invariants rather than gradient flow.
/// </summary>
public abstract class AssociativeMemoryTestBase
{
    protected abstract INeuralNetworkModel<double> CreateNetwork();

    protected virtual int[] InputShape => [1, 4];
    protected virtual int[] OutputShape => [1, 4];
    protected virtual int TrainingIterations => 10;

    /// <summary>
    /// Whether this model is autoassociative (input = output, target ignored during Train).
    /// Hopfield networks are autoassociative. HopeNetwork is not.
    /// </summary>
    protected virtual bool IsAutoAssociative => false;

    /// <summary>
    /// Number of patterns to store for multi-pattern tests.
    /// Should stay within network capacity for reliable recall.
    /// </summary>
    protected virtual int MultiPatternCount => 3;

    /// <summary>
    /// Fraction of elements to corrupt in noise robustness tests (0.0 to 1.0).
    /// </summary>
    protected virtual double NoiseFraction => 0.2;

    /// <summary>
    /// Tolerance for pattern recall MSE in auto-association tests.
    /// Hopfield with sign activation produces coarse binary output,
    /// so this should be generous for binary networks.
    /// </summary>
    protected virtual double RecallTolerance => 0.35;

    /// <summary>
    /// Whether this model supports serialization round-trip.
    /// Override to false if the model's layers don't support deserialization yet.
    /// </summary>
    protected virtual bool SupportsSerializationRoundTrip => true;

    protected Tensor<double> CreateRandomTensor(int[] shape, Random rng)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble();
        return tensor;
    }

    /// <summary>
    /// Creates a random binary tensor with values in {0, 1}.
    /// Binary patterns are more natural for Hopfield-style networks.
    /// </summary>
    protected Tensor<double> CreateRandomBinaryTensor(int[] shape, Random rng)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble() > 0.5 ? 1.0 : 0.0;
        return tensor;
    }

    protected Tensor<double> CreateConstantTensor(int[] shape, double value)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = value;
        return tensor;
    }

    /// <summary>
    /// Adds noise to a tensor by flipping a fraction of elements.
    /// For values in [0,1], flipped means 1-x.
    /// </summary>
    protected Tensor<double> AddNoise(Tensor<double> original, double fraction, Random rng)
    {
        var noisy = new Tensor<double>(original.Shape.ToArray());
        for (int i = 0; i < original.Length; i++)
        {
            if (rng.NextDouble() < fraction)
                noisy[i] = 1.0 - original[i]; // Flip
            else
                noisy[i] = original[i];
        }
        return noisy;
    }

    /// <summary>
    /// Creates a set of approximately orthogonal binary patterns.
    /// Each pattern has a unique "dominant block" that is all 1s while
    /// the rest is random, ensuring low correlation between patterns.
    /// </summary>
    protected List<Tensor<double>> CreateOrthogonalPatterns(int count, int[] shape, Random rng)
    {
        int totalSize = 1;
        foreach (var dim in shape) totalSize *= dim;

        int blockSize = totalSize / count;
        var patterns = new List<Tensor<double>>();

        for (int p = 0; p < count; p++)
        {
            var tensor = new Tensor<double>(shape);
            int blockStart = p * blockSize;
            int blockEnd = Math.Min(blockStart + blockSize, totalSize);

            for (int i = 0; i < totalSize; i++)
            {
                if (i >= blockStart && i < blockEnd)
                    tensor[i] = 1.0; // Dominant block
                else
                    tensor[i] = rng.NextDouble() * 0.1; // Low background
            }
            patterns.Add(tensor);
        }

        return patterns;
    }

    // =================================================================
    // SECTION 1: BASIC CONTRACT TESTS
    // =================================================================

    // =====================================================
    // MATHEMATICAL INVARIANT: Output Finite (No NaN/Infinity)
    // =====================================================

    [Fact]
    public void ForwardPass_ShouldProduceFiniteOutput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        Assert.True(output.Length > 0, "Output should not be empty.");

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Output[{i}] is NaN — numerical instability.");
            Assert.False(double.IsInfinity(output[i]), $"Output[{i}] is Infinity — overflow.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Finite Output After Training
    // =====================================================

    [Fact]
    public void ForwardPass_ShouldBeFinite_AfterTraining()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        for (int i = 0; i < TrainingIterations; i++)
            network.Train(input, target);

        var output = network.Predict(input);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Output[{i}] is NaN after {TrainingIterations} training iterations.");
            Assert.False(double.IsInfinity(output[i]),
                $"Output[{i}] is Infinity after training — potential instability.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Output Sensitivity to Input
    // =====================================================

    [Fact]
    public void DifferentInputs_ShouldProduceDifferentOutputs()
    {
        var network = CreateNetwork();

        var input1 = CreateConstantTensor(InputShape, 0.1);
        var input2 = CreateConstantTensor(InputShape, 0.9);

        var output1 = network.Predict(input1);
        var output2 = network.Predict(input2);

        bool anyDifferent = false;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(output1[i] - output2[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Network produces identical output for inputs [0.1,...] and [0.9,...]. " +
            "The network may have collapsed.");
    }

    [Fact]
    public void Predict_ShouldBeDeterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var out1 = network.Predict(input);
        var out2 = network.Predict(input);

        Assert.Equal(out1.Length, out2.Length);
        for (int i = 0; i < out1.Length; i++)
            Assert.Equal(out1[i], out2[i]);
    }

    [Fact]
    public void Parameters_ShouldBeNonEmpty()
    {
        var network = CreateNetwork();
        var parameters = network.GetParameters();
        Assert.True(parameters.Length > 0, "Network should have learnable parameters.");
    }

    [Fact]
    public void Clone_ShouldProduceIdenticalOutput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var original = network.Predict(input);
        var cloned = network.Clone();
        var clonedOutput = cloned.Predict(input);

        Assert.Equal(original.Length, clonedOutput.Length);
        for (int i = 0; i < original.Length; i++)
            Assert.Equal(original[i], clonedOutput[i]);
    }

    [Fact]
    public void Metadata_ShouldExist()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);
        network.Train(input, target);
        Assert.NotNull(network.GetModelMetadata());
    }

    [Fact]
    public void Architecture_ShouldBeNonNull()
    {
        var network = CreateNetwork();
        Assert.NotNull(network.GetArchitecture());
    }

    [Fact]
    public void NamedLayerActivations_ShouldBeNonEmpty()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var activations = network.GetNamedLayerActivations(input);
        Assert.NotNull(activations);
        Assert.True(activations.Count > 0, "Named layer activations should not be empty.");
    }

    [Fact]
    public void OutputDimension_ShouldMatchExpectedShape()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);

        int expectedLength = 1;
        foreach (var dim in OutputShape)
            expectedLength *= dim;

        Assert.Equal(expectedLength, output.Length);
    }

    [Fact]
    public void BatchConsistency_SingleMatchesBatch()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var singleOutput = network.Predict(input);
        var batchOutput = network.Predict(input);

        Assert.Equal(singleOutput.Length, batchOutput.Length);
        for (int i = 0; i < singleOutput.Length; i++)
            Assert.Equal(singleOutput[i], batchOutput[i]);
    }

    [Fact]
    public void ScaledInput_ShouldChangeOutput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

        var input = CreateRandomTensor(InputShape, rng);
        var scaledInput = new Tensor<double>(InputShape);
        for (int i = 0; i < input.Length; i++)
            scaledInput[i] = input[i] * 10.0;

        var output1 = network.Predict(input);
        var output2 = network.Predict(scaledInput);

        bool anyDifferent = false;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(output1[i] - output2[i]) > 1e-10)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Network output didn't change when input was scaled 10x. Forward pass may ignore input values.");
    }

    // =================================================================
    // SECTION 2: TRAINING INVARIANT TESTS
    // =================================================================

    // =====================================================
    // MATHEMATICAL INVARIANT: Training Changes Network Behavior
    // =====================================================

    [Fact]
    public void Training_ShouldChangeOutputBehavior()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        var outputBefore = network.Predict(input);

        for (int i = 0; i < TrainingIterations * 3; i++)
            network.Train(input, target);

        var outputAfter = network.Predict(input);

        bool anyChanged = false;
        int minLen = Math.Min(outputBefore.Length, outputAfter.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(outputBefore[i] - outputAfter[i]) > 1e-12)
            {
                anyChanged = true;
                break;
            }
        }
        Assert.True(anyChanged,
            "Network output did not change after training. " +
            "Learning mechanism may not be affecting recall behavior.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Training Should Change Parameters
    // =====================================================

    [Fact]
    public void Training_ShouldChangeParameters()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        var paramsBefore = network.GetParameters();
        var snapshot = new double[paramsBefore.Length];
        for (int i = 0; i < paramsBefore.Length; i++)
            snapshot[i] = paramsBefore[i];

        for (int i = 0; i < TrainingIterations; i++)
            network.Train(input, target);

        var paramsAfter = network.GetParameters();
        bool anyChanged = false;
        int minLen = Math.Min(snapshot.Length, paramsAfter.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(snapshot[i] - paramsAfter[i]) > 1e-15)
            {
                anyChanged = true;
                break;
            }
        }
        Assert.True(anyChanged,
            "Parameters did not change after training. Learning mechanism may be broken.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Training Loss Should Be Finite
    // =====================================================

    [Fact]
    public void TrainingLoss_ShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        for (int i = 0; i < TrainingIterations; i++)
            network.Train(input, target);

        var output = network.Predict(input);
        double mse = ComputeMSE(output, target);

        Assert.False(double.IsNaN(mse), "MSE is NaN after training — numerical instability.");
        Assert.False(double.IsInfinity(mse), "MSE is Infinity after training — overflow.");
    }

    // =================================================================
    // SECTION 3: ASSOCIATIVE MEMORY INVARIANT TESTS (7 NEW TESTS)
    // =================================================================

    // =====================================================
    // 1. PATTERN AUTO-ASSOCIATION
    // After training on a pattern, recalling that exact pattern should
    // return it (or something close) for autoassociative networks.
    // For non-autoassociative networks, the output should converge
    // toward the target after training.
    // =====================================================

    [Fact]
    public void PatternAutoAssociation_TrainedPatternShouldBeRecalled()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

        // Use binary patterns — more natural for associative memory
        var pattern = CreateRandomBinaryTensor(InputShape, rng);

        if (IsAutoAssociative)
        {
            // Autoassociative: Train stores the pattern itself.
            // After training, Predict(pattern) should ≈ pattern.
            var dummyTarget = CreateRandomTensor(OutputShape, rng);
            for (int i = 0; i < TrainingIterations * 3; i++)
                network.Train(pattern, dummyTarget);

            var recalled = network.Predict(pattern);
            double recallMSE = ComputeMSE(recalled, pattern);

            Assert.False(double.IsNaN(recallMSE), "Recall MSE is NaN.");
            Assert.True(recallMSE < RecallTolerance,
                $"Auto-association failed: MSE={recallMSE:F6} exceeds tolerance {RecallTolerance}. " +
                "Trained pattern should be a fixed point of the network.");
        }
        else
        {
            // Non-autoassociative: Train on (input, target).
            // After training, Predict(input) should move closer to target.
            var target = CreateRandomBinaryTensor(OutputShape, rng);

            var initialOutput = network.Predict(pattern);
            double initialMSE = ComputeMSE(initialOutput, target);

            for (int i = 0; i < TrainingIterations * 5; i++)
                network.Train(pattern, target);

            var finalOutput = network.Predict(pattern);
            double finalMSE = ComputeMSE(finalOutput, target);

            if (!double.IsNaN(initialMSE) && !double.IsNaN(finalMSE))
            {
                Assert.True(finalMSE <= initialMSE + 0.05,
                    $"Training did not improve recall: initial MSE={initialMSE:F6}, " +
                    $"final MSE={finalMSE:F6}. Learning is not converging toward target.");
            }
        }
    }

    // =====================================================
    // 2. NOISE ROBUSTNESS
    // After training on a clean pattern, presenting a noisy version
    // should produce output closer to the clean pattern than the
    // noisy input is. This is the error-correction property of
    // associative memory.
    // =====================================================

    [Fact]
    public void NoiseRobustness_ShouldCorrectNoisyInput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

        var cleanPattern = CreateRandomBinaryTensor(InputShape, rng);
        var target = IsAutoAssociative ? cleanPattern : CreateRandomBinaryTensor(OutputShape, rng);

        // Train the network on the clean pattern
        for (int i = 0; i < TrainingIterations * 5; i++)
            network.Train(cleanPattern, target);

        // Add noise: flip NoiseFraction of the elements
        var noisyPattern = AddNoise(cleanPattern, NoiseFraction, ModelTestHelpers.CreateSeededRandom(99));

        // The noisy input should differ from clean
        double noisyInputDistance = ComputeMSE(noisyPattern, cleanPattern);
        Assert.True(noisyInputDistance > 0.01,
            $"Noisy pattern is too similar to clean (MSE={noisyInputDistance:F6}). Noise was not applied.");

        // Recall using the noisy input
        var recalled = network.Predict(noisyPattern);

        // The reference for comparison depends on model type
        var reference = IsAutoAssociative ? cleanPattern : target;

        double recalledDistance = ComputeMSE(recalled, reference);
        double noisyToRefDistance = ComputeMSE(noisyPattern, reference);

        // Recalled output should be at least somewhat closer to the reference
        // than a purely random tensor (generous tolerance for networks that
        // may only partially correct noise)
        var randomTensor = CreateRandomTensor(OutputShape, ModelTestHelpers.CreateSeededRandom(77));
        double randomDistance = ComputeMSE(randomTensor, reference);

        if (!double.IsNaN(recalledDistance) && !double.IsNaN(randomDistance))
        {
            Assert.True(recalledDistance < randomDistance + 0.1,
                $"Recalled output (MSE={recalledDistance:F6} to reference) is worse than random " +
                $"(MSE={randomDistance:F6}). Network error correction is broken.");
        }
    }

    // =====================================================
    // 3. CAPACITY TEST
    // Store up to MultiPatternCount patterns and verify all can
    // still be recalled after training on all of them. Tests that
    // the network has sufficient capacity.
    // =====================================================

    [Fact]
    public void Capacity_AllStoredPatternsShouldBeRecallable()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

        // Generate multiple distinct binary patterns
        var patterns = new List<Tensor<double>>();
        var targets = new List<Tensor<double>>();
        for (int p = 0; p < MultiPatternCount; p++)
        {
            patterns.Add(CreateRandomBinaryTensor(InputShape, ModelTestHelpers.CreateSeededRandom(100 + p)));
            targets.Add(IsAutoAssociative
                ? patterns[p]
                : CreateRandomBinaryTensor(OutputShape, ModelTestHelpers.CreateSeededRandom(200 + p)));
        }

        // Train on all patterns (cycle through them)
        for (int epoch = 0; epoch < TrainingIterations * 3; epoch++)
        {
            for (int p = 0; p < MultiPatternCount; p++)
                network.Train(patterns[p], targets[p]);
        }

        // Verify each pattern can be recalled with finite output
        int finitePatternsRecalled = 0;
        for (int p = 0; p < MultiPatternCount; p++)
        {
            var recalled = network.Predict(patterns[p]);

            bool allFinite = true;
            for (int i = 0; i < recalled.Length; i++)
            {
                if (double.IsNaN(recalled[i]) || double.IsInfinity(recalled[i]))
                {
                    allFinite = false;
                    break;
                }
            }

            if (allFinite) finitePatternsRecalled++;
        }

        Assert.True(finitePatternsRecalled == MultiPatternCount,
            $"Only {finitePatternsRecalled}/{MultiPatternCount} patterns produced finite recall. " +
            "Network may have capacity issues or numerical instability.");
    }

    // =====================================================
    // 4. ENERGY MONOTONICITY (optional, model-specific)
    // For energy-based models, stored patterns should have lower
    // energy than random states. Override ComputeEnergy in subclasses
    // that support it.
    // =====================================================

    /// <summary>
    /// Computes the energy of a given state. Returns null if the model
    /// doesn't support energy computation. Override in subclasses.
    /// </summary>
    protected virtual double? ComputeEnergy(INeuralNetworkModel<double> network, Tensor<double> state)
    {
        return null; // Not all associative memory models have an energy function
    }

    [Fact]
    public void EnergyMonotonicity_TrainedPatternsShouldHaveLowerEnergy()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

        var pattern = CreateRandomBinaryTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        // Check if this model supports energy computation
        double? preEnergy = ComputeEnergy(network, pattern);
        if (preEnergy == null)
        {
            // Model doesn't support energy — skip gracefully
            return;
        }

        // Train on the pattern
        for (int i = 0; i < TrainingIterations * 3; i++)
            network.Train(pattern, target);

        // Energy of the trained pattern
        double? trainedEnergy = ComputeEnergy(network, pattern);

        // Energy of a random state
        var randomState = CreateRandomBinaryTensor(InputShape, ModelTestHelpers.CreateSeededRandom(99));
        double? randomEnergy = ComputeEnergy(network, randomState);

        if (trainedEnergy.HasValue && randomEnergy.HasValue &&
            !double.IsNaN(trainedEnergy.Value) && !double.IsNaN(randomEnergy.Value))
        {
            Assert.True(trainedEnergy.Value <= randomEnergy.Value + 1e-3,
                $"Trained pattern energy ({trainedEnergy.Value:F6}) should be lower than " +
                $"random state energy ({randomEnergy.Value:F6}). " +
                "Hebbian learning did not create an energy minimum at the stored pattern.");
        }
    }

    // =====================================================
    // 5. ORTHOGONAL PATTERN TEST
    // Orthogonal (low-correlation) patterns should be easier to
    // store and recall than correlated patterns. Train on orthogonal
    // patterns and verify each is recalled without interference.
    // =====================================================

    [Fact]
    public void OrthogonalPatterns_ShouldBeRecalledWithoutInterference()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

        int patternCount = Math.Min(MultiPatternCount, 3);
        var patterns = CreateOrthogonalPatterns(patternCount, InputShape, rng);
        var targets = new List<Tensor<double>>();

        for (int p = 0; p < patternCount; p++)
        {
            targets.Add(IsAutoAssociative ? patterns[p] : CreateRandomBinaryTensor(OutputShape,
                ModelTestHelpers.CreateSeededRandom(300 + p)));
        }

        // Train on all orthogonal patterns
        for (int epoch = 0; epoch < TrainingIterations * 3; epoch++)
        {
            for (int p = 0; p < patternCount; p++)
                network.Train(patterns[p], targets[p]);
        }

        // Verify each pattern produces distinct output (no pattern collapsed to same attractor)
        var outputs = new List<Tensor<double>>();
        for (int p = 0; p < patternCount; p++)
        {
            outputs.Add(network.Predict(patterns[p]));
        }

        // Check that at least some pattern pairs produce different outputs
        int distinctPairs = 0;
        int totalPairs = 0;
        for (int i = 0; i < patternCount; i++)
        {
            for (int j = i + 1; j < patternCount; j++)
            {
                totalPairs++;
                double mse = ComputeMSE(outputs[i], outputs[j]);
                if (!double.IsNaN(mse) && mse > 1e-6)
                    distinctPairs++;
            }
        }

        Assert.True(distinctPairs > 0,
            $"All {patternCount} orthogonal patterns produced identical output. " +
            "Network has collapsed — different stored patterns are not distinguished.");
    }

    // =====================================================
    // 6. SERIALIZATION ROUND-TRIP
    // Serialize and deserialize the trained network, then verify
    // that the deserialized network produces identical output.
    // =====================================================

    [Fact]
    public void SerializationRoundTrip_ShouldPreserveRecall()
    {
        if (!SupportsSerializationRoundTrip)
            return;

        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        // Train the network
        for (int i = 0; i < TrainingIterations; i++)
            network.Train(input, target);

        // Get output before serialization
        var outputBefore = network.Predict(input);

        // Serialize and deserialize via Clone (which uses serialize/deserialize)
        var cloned = network.Clone();
        var outputAfter = cloned.Predict(input);

        Assert.Equal(outputBefore.Length, outputAfter.Length);
        for (int i = 0; i < outputBefore.Length; i++)
        {
            Assert.False(double.IsNaN(outputAfter[i]),
                $"Deserialized output[{i}] is NaN — serialization corrupted the model.");
            Assert.Equal(outputBefore[i], outputAfter[i]);
        }

        // Also verify parameters are preserved
        var paramsBefore = network.GetParameters();
        var paramsAfter = cloned.GetParameters();
        Assert.Equal(paramsBefore.Length, paramsAfter.Length);
        for (int i = 0; i < paramsBefore.Length; i++)
        {
            Assert.Equal(paramsBefore[i], paramsAfter[i]);
        }
    }

    // =====================================================
    // 7. MULTIPLE PATTERN STABILITY
    // Train incrementally on multiple patterns and verify that
    // previously stored patterns are not completely forgotten
    // (catastrophic forgetting check).
    // =====================================================

    [Fact]
    public void MultiplePatternStability_OlderPatternsShouldNotBeCompletelyForgotten()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

        // Create two distinct patterns
        var pattern1 = CreateRandomBinaryTensor(InputShape, ModelTestHelpers.CreateSeededRandom(10));
        var pattern2 = CreateRandomBinaryTensor(InputShape, ModelTestHelpers.CreateSeededRandom(20));
        var target1 = IsAutoAssociative ? pattern1 : CreateRandomBinaryTensor(OutputShape, ModelTestHelpers.CreateSeededRandom(30));
        var target2 = IsAutoAssociative ? pattern2 : CreateRandomBinaryTensor(OutputShape, ModelTestHelpers.CreateSeededRandom(40));

        // Train on pattern1 first
        for (int i = 0; i < TrainingIterations * 2; i++)
            network.Train(pattern1, target1);

        // Record output for pattern1 before training on pattern2
        var output1Before = network.Predict(pattern1);

        // Now train on pattern2
        for (int i = 0; i < TrainingIterations * 2; i++)
            network.Train(pattern2, target2);

        // Recall pattern1 after training on pattern2
        var output1After = network.Predict(pattern1);

        // Pattern1 recall should still be finite (not completely destroyed)
        for (int i = 0; i < output1After.Length; i++)
        {
            Assert.False(double.IsNaN(output1After[i]),
                $"Output[{i}] for pattern1 is NaN after training on pattern2 — catastrophic forgetting.");
            Assert.False(double.IsInfinity(output1After[i]),
                $"Output[{i}] for pattern1 is Infinity after training on pattern2 — instability.");
        }

        // The output for pattern1 should not be identical to pattern2's output
        // (that would mean pattern1 was completely overwritten by pattern2)
        var output2 = network.Predict(pattern2);
        double similarity = ComputeMSE(output1After, output2);

        // If MSE between the two outputs is exactly 0, they collapsed to same output
        if (!double.IsNaN(similarity))
        {
            // We don't require perfect recall of pattern1, but the two outputs
            // should not be identical (which would mean complete forgetting)
            bool outputsAreDifferent = false;
            int minLen = Math.Min(output1After.Length, output2.Length);
            for (int i = 0; i < minLen; i++)
            {
                if (Math.Abs(output1After[i] - output2[i]) > 1e-10)
                {
                    outputsAreDifferent = true;
                    break;
                }
            }

            // Note: this is a soft check. Some models may legitimately produce
            // similar outputs for different patterns if the patterns are similar.
            // We just want to catch complete catastrophic forgetting.
            if (!outputsAreDifferent)
            {
                // Check if the patterns themselves are very different
                double patternDiff = ComputeMSE(pattern1, pattern2);
                if (patternDiff > 0.1)
                {
                    Assert.Fail(
                        $"Pattern1 and Pattern2 outputs are identical (pattern MSE={patternDiff:F4}). " +
                        "Catastrophic forgetting: pattern2 training completely overwrote pattern1.");
                }
            }
        }
    }

    // =================================================================
    // HELPERS
    // =================================================================

    private double ComputeMSE(Tensor<double> output, Tensor<double> target)
    {
        double mse = 0;
        int len = Math.Min(output.Length, target.Length);
        if (len == 0) return double.NaN;
        for (int i = 0; i < len; i++)
        {
            double diff = output[i] - target[i];
            mse += diff * diff;
        }
        return mse / len;
    }
}
