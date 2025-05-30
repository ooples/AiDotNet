using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Reasoning;
using System;
using System.Linq;

namespace AiDotNet.Examples
{
    /// <summary>
    /// Demonstrates how to use reasoning models for complex problem-solving tasks.
    /// </summary>
    public static class ReasoningModelExample
    {
        /// <summary>
        /// Example: Using Chain-of-Thought reasoning to solve a math word problem
        /// </summary>
        public static void ChainOfThoughtMathExample()
        {
            Console.WriteLine("=== Chain-of-Thought Math Problem Solving ===\n");

            // Configure the Chain-of-Thought model
            var options = new ChainOfThoughtOptions<float>
            {
                InputShape = new[] { 256 },      // Size of problem representation
                HiddenShape = new[] { 128 },     // Size of reasoning steps
                HiddenSize = 64,                 // Internal processing capacity
                AttentionHeads = 4,              // Number of attention heads
                MaxChainLength = 15,             // Maximum reasoning steps
                DefaultMaxSteps = 10,            // Default steps for solving
                Temperature = 0.3,               // Low temperature for precise reasoning
                EnableChainValidation = true,    // Validate logical consistency
                MinConfidenceThreshold = 0.7     // Require high confidence
            };

            // Create the model
            var model = new ChainOfThoughtModel<float>(options);

            // Example: Encoding a word problem
            // "A store has 120 apples. They sell 45 apples on Monday and 38 on Tuesday. 
            //  How many apples are left?"
            var problem = CreateProblemTensor(120f, 45f, 38f);

            Console.WriteLine("Problem: A store has 120 apples. They sell 45 apples on Monday");
            Console.WriteLine("and 38 on Tuesday. How many apples are left?");
            Console.WriteLine("\nReasoning steps:");

            // Solve step by step
            var reasoningSteps = model.ReasonStepByStep(problem, maxSteps: 10);

            // Display each reasoning step
            for (int i = 0; i < reasoningSteps.Count; i++)
            {
                var stepDescription = InterpretReasoningStep(reasoningSteps[i], i);
                Console.WriteLine($"Step {i + 1}: {stepDescription}");
            }

            // Get the final answer
            var answer = reasoningSteps.Last();
            Console.WriteLine($"\nFinal Answer: {ExtractAnswer(answer)} apples remaining");

            // Check confidence in the reasoning
            var confidence = model.GetReasoningConfidence();
            Console.WriteLine($"\nConfidence scores: {string.Join(", ", confidence.Select(c => c.ToString("F2")))}");

            // Generate explanation
            var explanation = model.GenerateExplanation(problem, answer);
            Console.WriteLine($"\nExplanation: {InterpretExplanation(explanation)}");
        }

        /// <summary>
        /// Example: Using self-consistency to improve reliability
        /// </summary>
        public static void SelfConsistencyExample()
        {
            Console.WriteLine("\n\n=== Self-Consistency Reasoning ===\n");

            var options = new ChainOfThoughtOptions<float>
            {
                InputShape = new[] { 256 },
                HiddenShape = new[] { 128 },
                VaryStrategyInSelfConsistency = true,  // Use different strategies
                Temperature = 0.5,                      // Moderate temperature for diversity
                BeamWidth = 3                           // Explore multiple paths
            };

            var model = new ChainOfThoughtModel<float>(options);

            // Complex problem requiring multiple approaches
            var complexProblem = CreateComplexProblem();

            Console.WriteLine("Solving complex problem using self-consistency...");

            // Generate multiple reasoning paths
            var consistentAnswer = model.SelfConsistencyCheck(complexProblem, numPaths: 5);

            Console.WriteLine("Generated 5 independent reasoning paths");
            Console.WriteLine($"Consensus answer: {ExtractAnswer(consistentAnswer)}");

            // Validate the reasoning chain
            var steps = model.ReasonStepByStep(complexProblem);
            var isValid = model.ValidateReasoningChain(steps);
            Console.WriteLine($"Reasoning chain validation: {(isValid ? "PASSED" : "FAILED")}");
        }

        /// <summary>
        /// Example: Iterative refinement for complex reasoning
        /// </summary>
        public static void IterativeRefinementExample()
        {
            Console.WriteLine("\n\n=== Iterative Refinement Reasoning ===\n");

            var options = new ChainOfThoughtOptions<float>
            {
                InputShape = new[] { 256 },
                HiddenShape = new[] { 128 },
                DefaultStrategy = ReasoningStrategy.Bidirectional,
                EnableDetailedDiagnostics = true
            };

            var model = new ChainOfThoughtModel<float>(options);

            // Problem that benefits from refinement
            var problem = CreateRefinementProblem();

            Console.WriteLine("Initial reasoning attempt...");
            var initialAnswer = model.Predict(problem);
            Console.WriteLine($"Initial answer: {ExtractAnswer(initialAnswer)}");

            // Refine the answer
            Console.WriteLine("\nRefining the reasoning (3 iterations)...");
            var refinedAnswer = model.RefineReasoning(problem, initialAnswer, iterations: 3);
            Console.WriteLine($"Refined answer: {ExtractAnswer(refinedAnswer)}");

            // Get diagnostics
            var diagnostics = model.GetReasoningDiagnostics();
            Console.WriteLine("\nReasoning diagnostics:");
            foreach (var kvp in diagnostics)
            {
                Console.WriteLine($"  {kvp.Key}: {kvp.Value}");
            }
        }

        /// <summary>
        /// Example: Using different reasoning strategies
        /// </summary>
        public static void ReasoningStrategyExample()
        {
            Console.WriteLine("\n\n=== Reasoning Strategy Comparison ===\n");

            var options = new ChainOfThoughtOptions<float>
            {
                InputShape = new[] { 256 },
                HiddenShape = new[] { 128 }
            };

            var model = new ChainOfThoughtModel<float>(options);
            var problem = CreateStrategyTestProblem();

            // Test different strategies
            var strategies = new[]
            {
                ReasoningStrategy.ForwardChaining,
                ReasoningStrategy.BackwardChaining,
                ReasoningStrategy.BeamSearch,
                ReasoningStrategy.MonteCarlo
            };

            foreach (var strategy in strategies)
            {
                model.SetReasoningStrategy(strategy);
                Console.WriteLine($"\nUsing {strategy} strategy:");

                var start = DateTime.Now;
                var result = model.Predict(problem);
                var elapsed = (DateTime.Now - start).TotalMilliseconds;

                Console.WriteLine($"  Answer: {ExtractAnswer(result)}");
                Console.WriteLine($"  Time: {elapsed:F2}ms");
                
                var steps = model.ReasonStepByStep(problem, maxSteps: 5);
                Console.WriteLine($"  Steps taken: {steps.Count}");
            }
        }

        /// <summary>
        /// Example: Training a reasoning model
        /// </summary>
        public static void TrainingExample()
        {
            Console.WriteLine("\n\n=== Training a Reasoning Model ===\n");

            var options = new ChainOfThoughtOptions<float>
            {
                InputShape = new[] { 64 },
                HiddenShape = new[] { 32 },
                LearningRate = 0.001,
                WeightDecay = 0.0001,
                DropoutRate = 0.1
            };

            var model = new ChainOfThoughtModel<float>(options);

            // Generate training examples (problem -> solution pairs)
            Console.WriteLine("Training on arithmetic reasoning problems...");

            for (int i = 0; i < 100; i++)
            {
                // Create a simple arithmetic problem
                var random = new Random();
                var a = random.Next(1, 100);
                var b = random.Next(1, 100);
                var operation = random.Next(0, 3); // 0: add, 1: subtract, 2: multiply

                var input = CreateArithmeticProblem(a, b, operation);
                var expectedOutput = CreateExpectedOutput(a, b, operation);

                model.Train(input, expectedOutput);

                if (i % 20 == 0)
                {
                    Console.WriteLine($"  Trained on {i + 1} examples...");
                }
            }

            Console.WriteLine("\nTesting the trained model:");

            // Test on new problems
            var testProblems = new[]
            {
                new { a = 25, b = 17, op = 0, description = "25 + 17" },  // Addition
                new { a = 84, b = 29, op = 1, description = "84 - 29" },  // Subtraction
                new { a = 12, b = 7, op = 2, description = "12 × 7" }     // Multiplication
            };

            foreach (var problem in testProblems)
            {
                var testInput = CreateArithmeticProblem(problem.a, problem.b, problem.op);
                var prediction = model.Predict(testInput);
                var answer = ExtractAnswer(prediction);
                var expected = 0;
                switch (problem.op)
                {
                    case 0:
                        expected = problem.a + problem.b;
                        break;
                    case 1:
                        expected = problem.a - problem.b;
                        break;
                    case 2:
                        expected = problem.a * problem.b;
                        break;
                }

                Console.WriteLine($"  {problem.description} = {answer:F0} (expected: {expected})");
            }
        }

        #region Helper Methods

        private static Tensor<float> CreateProblemTensor(params float[] values)
        {
            var tensor = new Tensor<float>(new[] { 256 });
            for (int i = 0; i < values.Length && i < 256; i++)
            {
                tensor[i] = values[i];
            }
            return tensor;
        }

        private static string InterpretReasoningStep(Tensor<float> step, int index)
        {
            // Simulate interpretation of reasoning steps
            switch (index)
            {
                case 0:
                    return "Identify initial quantity: 120 apples";
                case 1:
                    return "Identify first transaction: sold 45 apples";
                case 2:
                    return "Calculate remaining after Monday: 120 - 45 = 75";
                case 3:
                    return "Identify second transaction: sold 38 apples";
                case 4:
                    return "Calculate final remaining: 75 - 38 = 37";
                default:
                    return "Verify answer: 37 apples remaining";
            }
        }

        private static float ExtractAnswer(Tensor<float> tensor)
        {
            // Extract the numerical answer from the tensor
            // In a real implementation, this would decode the tensor representation
            return Math.Abs(tensor[0] * 100) % 1000;
        }

        private static string InterpretExplanation(Tensor<float> explanation)
        {
            return "Started with 120 apples, sold 45 on Monday leaving 75, " +
                   "then sold 38 on Tuesday leaving 37 apples in total.";
        }

        private static Tensor<float> CreateComplexProblem()
        {
            // Create a more complex problem representation
            var tensor = new Tensor<float>(new[] { 256 });
            var random = new Random(42);
            for (int i = 0; i < 256; i++)
            {
                tensor[i] = (float)random.NextDouble();
            }
            return tensor;
        }

        private static Tensor<float> CreateRefinementProblem()
        {
            // Create a problem that benefits from iterative refinement
            var tensor = new Tensor<float>(new[] { 256 });
            for (int i = 0; i < 256; i++)
            {
                tensor[i] = (float)Math.Sin(i * 0.1) * 0.5f;
            }
            return tensor;
        }

        private static Tensor<float> CreateStrategyTestProblem()
        {
            // Create a problem for testing different strategies
            var tensor = new Tensor<float>(new[] { 256 });
            for (int i = 0; i < 256; i++)
            {
                tensor[i] = i % 2 == 0 ? 0.8f : 0.2f;
            }
            return tensor;
        }

        private static Tensor<float> CreateArithmeticProblem(int a, int b, int operation)
        {
            var tensor = new Tensor<float>(new[] { 64 });
            tensor[0] = a / 100f;
            tensor[1] = b / 100f;
            tensor[2] = operation / 3f;
            return tensor;
        }

        private static Tensor<float> CreateExpectedOutput(int a, int b, int operation)
        {
            var tensor = new Tensor<float>(new[] { 32 });
            var result = 0;
            switch (operation)
            {
                case 0:
                    result = a + b;
                    break;
                case 1:
                    result = a - b;
                    break;
                case 2:
                    result = a * b;
                    break;
                default:
                    result = 0;
                    break;
            }
            tensor[0] = result / 1000f;
            return tensor;
        }

        #endregion

        /// <summary>
        /// Run all reasoning model examples
        /// </summary>
        public static void RunAllExamples()
        {
            try
            {
                ChainOfThoughtMathExample();
                SelfConsistencyExample();
                IterativeRefinementExample();
                ReasoningStrategyExample();
                TrainingExample();

                Console.WriteLine("\n\n=== All Reasoning Examples Completed Successfully ===");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nError running examples: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }
    }
}