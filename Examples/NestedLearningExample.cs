using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.NestedLearning;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using MathNet.Numerics.LinearAlgebra;

namespace AiDotNet.Examples
{
    /// <summary>
    /// Example demonstrating Nested Learning for continual learning across multiple tasks.
    /// Shows how to train on sequential tasks without catastrophic forgetting.
    /// </summary>
    public class NestedLearningExample
    {
        /// <summary>
        /// Demonstrates nested learning on a sequence of classification tasks.
        /// </summary>
        public static void RunContinualLearningExample()
        {
            Console.WriteLine("=== Nested Learning: Continual Learning Example ===\n");

            // Create a simple feedforward network
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 10,
                OutputSize = 3,
                HiddenLayerSizes = new[] { 32, 16 }
            };

            var model = new FeedForwardNeuralNetwork<double>(architecture);

            // Create nested learner with 3 optimization levels
            var learner = new NestedLearner<double, Tensor<double>, Tensor<double>>(
                model: model,
                lossFunction: new CrossEntropyLoss<double>(),
                numLevels: 3,
                memoryDimension: 64);

            Console.WriteLine("Created Nested Learner with 3 optimization levels");
            Console.WriteLine($"Update frequencies: {string.Join(", ", learner.UpdateFrequencies)}");
            Console.WriteLine();

            // Generate synthetic tasks
            var task1Data = GenerateSyntheticTask(numSamples: 100, taskId: 1);
            var task2Data = GenerateSyntheticTask(numSamples: 100, taskId: 2);
            var task3Data = GenerateSyntheticTask(numSamples: 100, taskId: 3);

            // Train on Task 1
            Console.WriteLine("Training on Task 1...");
            var result1 = learner.Train(task1Data, numLevels: 3, maxIterations: 200);
            Console.WriteLine($"  Final Loss: {result1.FinalLoss:F4}");
            Console.WriteLine($"  Converged: {result1.Converged}");
            Console.WriteLine($"  Duration: {result1.Duration.TotalSeconds:F2}s");
            Console.WriteLine();

            // Adapt to Task 2 while preserving Task 1 knowledge
            Console.WriteLine("Adapting to Task 2 (preservation strength: 0.7)...");
            var result2 = learner.AdaptToNewTask(task2Data, preservationStrength: 0.7);
            Console.WriteLine($"  New Task Loss: {result2.NewTaskLoss:F4}");
            Console.WriteLine($"  Forgetting Metric: {result2.ForgettingMetric:F4}");
            Console.WriteLine($"  Adaptation Steps: {result2.AdaptationSteps}");
            Console.WriteLine();

            // Adapt to Task 3
            Console.WriteLine("Adapting to Task 3 (preservation strength: 0.7)...");
            var result3 = learner.AdaptToNewTask(task3Data, preservationStrength: 0.7);
            Console.WriteLine($"  New Task Loss: {result3.NewTaskLoss:F4}");
            Console.WriteLine($"  Forgetting Metric: {result3.ForgettingMetric:F4}");
            Console.WriteLine($"  Adaptation Steps: {result3.AdaptationSteps}");
            Console.WriteLine();

            // Inspect memory system
            var memorySystem = learner.GetMemorySystem();
            Console.WriteLine("Memory System State:");
            for (int i = 0; i < memorySystem.NumberOfFrequencyLevels; i++)
            {
                var memState = memorySystem.MemoryStates[i];
                var memNorm = Vector<double>.Build.DenseOfArray(memState.ToArray()).L2Norm();
                Console.WriteLine($"  Level {i} memory norm: {memNorm:F4} (decay rate: {memorySystem.DecayRates[i]:F3})");
            }
            Console.WriteLine();

            Console.WriteLine("=== Nested Learning Example Complete ===");
        }

        /// <summary>
        /// Demonstrates the Hope architecture for sequence modeling.
        /// </summary>
        public static void RunHopeArchitectureExample()
        {
            Console.WriteLine("=== Hope Architecture: Self-Modifying Recurrent Network ===\n");

            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 20,
                OutputSize = 5
            };

            // Create Hope network
            var hope = new HopeNetwork<double>(
                architecture,
                optimizer: null,
                lossFunction: new MeanSquaredError<double>(),
                hiddenDim: 128,
                numCMSLevels: 3,
                numRecurrentLayers: 2);

            hope.AddOutputLayer(outputDim: 5, ActivationFunction.Softmax);

            Console.WriteLine("Created Hope Network:");
            Console.WriteLine($"  Hidden Dimension: 128");
            Console.WriteLine($"  CMS Levels: 3");
            Console.WriteLine($"  Recurrent Layers: 2");
            Console.WriteLine();

            // Process a sequence
            var sequenceLength = 50;
            var random = new Random(42);

            Console.WriteLine($"Processing sequence of length {sequenceLength}...");

            for (int t = 0; t < sequenceLength; t++)
            {
                // Generate random input
                var inputData = Enumerable.Range(0, 20)
                    .Select(_ => random.NextDouble())
                    .ToArray();
                var input = Tensor<double>.CreateFromArray(inputData, new[] { 20 });

                // Forward pass
                var output = hope.Forward(input);

                // Every 10 steps, consolidate memory
                if ((t + 1) % 10 == 0)
                {
                    hope.ConsolidateMemory();
                    Console.WriteLine($"  Step {t + 1}: Memory consolidated");
                }
            }

            Console.WriteLine();

            // Inspect CMS blocks
            var cmsBlocks = hope.GetCMSBlocks();
            Console.WriteLine("CMS Block States:");
            for (int i = 0; i < cmsBlocks.Length; i++)
            {
                var memStates = cmsBlocks[i].GetMemoryStates();
                Console.WriteLine($"  CMS Block {i}: {memStates.Length} frequency levels");
            }

            Console.WriteLine();
            Console.WriteLine($"Adaptation steps performed: {hope.AdaptationStep}");
            Console.WriteLine();

            Console.WriteLine("=== Hope Architecture Example Complete ===");
        }

        /// <summary>
        /// Demonstrates the Continuum Memory System layer.
        /// </summary>
        public static void RunContinuumMemoryExample()
        {
            Console.WriteLine("=== Continuum Memory System (CMS) Example ===\n");

            var cms = new ContinuumMemorySystem<double>(
                memoryDimension: 64,
                numFrequencyLevels: 4);

            Console.WriteLine("Created CMS with 4 frequency levels");
            Console.WriteLine($"Memory dimension: 64");
            Console.WriteLine($"Decay rates: {string.Join(", ", cms.DecayRates.Select(r => r.ToString("F3")))}");
            Console.WriteLine();

            var random = new Random(42);

            // Store patterns at different frequencies
            Console.WriteLine("Storing patterns at different frequency levels...");
            for (int step = 0; step < 100; step++)
            {
                var pattern = Tensor<double>.CreateFromArray(
                    Enumerable.Range(0, 64).Select(_ => random.NextDouble()).ToArray(),
                    new[] { 64 });

                // Determine which levels to update based on frequency
                var updateMask = new bool[4];
                updateMask[0] = true; // Level 0 updates every step
                updateMask[1] = step % 10 == 0; // Level 1 updates every 10 steps
                updateMask[2] = step % 50 == 0; // Level 2 updates every 50 steps
                updateMask[3] = step % 100 == 0; // Level 3 updates every 100 steps

                cms.Update(pattern, updateMask);

                if (step % 25 == 0 && step > 0)
                {
                    Console.WriteLine($"  Step {step}: Updated levels {string.Join(", ", updateMask.Select((u, i) => u ? i.ToString() : null).Where(s => s != null))}");
                }
            }

            Console.WriteLine();

            // Consolidate memories
            Console.WriteLine("Consolidating memories across frequency levels...");
            cms.Consolidate();

            // Inspect memory states
            Console.WriteLine("\nMemory State Norms:");
            for (int i = 0; i < cms.NumberOfFrequencyLevels; i++)
            {
                var memState = cms.MemoryStates[i];
                var norm = Vector<double>.Build.DenseOfArray(memState.ToArray()).L2Norm();
                Console.WriteLine($"  Level {i}: {norm:F4}");
            }

            Console.WriteLine();
            Console.WriteLine("=== CMS Example Complete ===");
        }

        /// <summary>
        /// Demonstrates context flow mechanism.
        /// </summary>
        public static void RunContextFlowExample()
        {
            Console.WriteLine("=== Context Flow Example ===\n");

            var contextFlow = new ContextFlow<double>(
                contextDimension: 32,
                numLevels: 3,
                seed: 42);

            Console.WriteLine("Created Context Flow with 3 levels");
            Console.WriteLine($"Context dimension: 32");
            Console.WriteLine();

            var random = new Random(42);

            // Propagate context through levels
            Console.WriteLine("Propagating context through optimization levels...");
            for (int step = 0; step < 10; step++)
            {
                var input = Tensor<double>.CreateFromArray(
                    Enumerable.Range(0, 32).Select(_ => random.NextDouble()).ToArray(),
                    new[] { 32 });

                for (int level = 0; level < 3; level++)
                {
                    var context = contextFlow.PropagateContext(input, level);
                    var contextNorm = Vector<double>.Build.DenseOfArray(context.ToArray()).L2Norm();

                    if (step == 0 || step == 9)
                    {
                        Console.WriteLine($"  Step {step}, Level {level}: Context norm = {contextNorm:F4}");
                    }
                }
            }

            Console.WriteLine();

            // Get final context states
            Console.WriteLine("Final Context States:");
            for (int level = 0; level < 3; level++)
            {
                var state = contextFlow.GetContextState(level);
                var norm = Vector<double>.Build.DenseOfArray(state.ToArray()).L2Norm();
                Console.WriteLine($"  Level {level}: {norm:F4}");
            }

            Console.WriteLine();
            Console.WriteLine("=== Context Flow Example Complete ===");
        }

        /// <summary>
        /// Run all nested learning examples.
        /// </summary>
        public static void RunAllExamples()
        {
            RunContinualLearningExample();
            Console.WriteLine("\n" + new string('=', 60) + "\n");

            RunHopeArchitectureExample();
            Console.WriteLine("\n" + new string('=', 60) + "\n");

            RunContinuumMemoryExample();
            Console.WriteLine("\n" + new string('=', 60) + "\n");

            RunContextFlowExample();
        }

        private static List<(Tensor<double> Input, Tensor<double> Output)> GenerateSyntheticTask(
            int numSamples,
            int taskId)
        {
            var random = new Random(taskId * 1000);
            var data = new List<(Tensor<double>, Tensor<double>)>();

            for (int i = 0; i < numSamples; i++)
            {
                // Generate input
                var inputData = Enumerable.Range(0, 10)
                    .Select(_ => random.NextDouble() + taskId * 0.1) // Offset by task
                    .ToArray();
                var input = Tensor<double>.CreateFromArray(inputData, new[] { 10 });

                // Generate output (simple classification based on task)
                var outputData = new double[3];
                outputData[taskId % 3] = 1.0;
                var output = Tensor<double>.CreateFromArray(outputData, new[] { 3 });

                data.Add((input, output));
            }

            return data;
        }
    }
}
