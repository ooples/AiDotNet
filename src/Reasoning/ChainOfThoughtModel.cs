using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AiDotNet.Reasoning
{
    /// <summary>
    /// Implements a Chain-of-Thought (CoT) reasoning model that breaks down complex problems
    /// into sequential reasoning steps.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <remarks>
    /// <para>
    /// Chain-of-Thought reasoning is inspired by how humans solve complex problems by thinking
    /// step-by-step. Instead of jumping directly to an answer, the model generates intermediate
    /// reasoning steps that lead to the final solution.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Imagine solving a word problem in math class. You don't just write
    /// the answer - you show your work step by step. That's what Chain-of-Thought does. It makes
    /// the AI "think out loud" which helps it solve harder problems and lets you see how it
    /// reached its conclusion.
    /// </para>
    /// </remarks>
    public class ChainOfThoughtModel<T> : ReasoningModelBase<T>
    {
        private readonly NeuralNetwork<T> _reasoningNetwork = default!;
        private readonly NeuralNetwork<T> _validationNetwork = default!;
        private readonly Dictionary<string, List<Tensor<T>>> _reasoningCache = default!;
        private readonly ChainOfThoughtOptions<T> _cotOptions = default!;

        /// <summary>
        /// Gets the maximum reasoning depth this model can handle effectively.
        /// </summary>
        public override int MaxReasoningDepth => _cotOptions.MaxChainLength;

        /// <summary>
        /// Gets whether this model supports iterative refinement.
        /// </summary>
        public override bool SupportsIterativeRefinement => true;

        /// <summary>
        /// Initializes a new instance of the ChainOfThoughtModel class.
        /// </summary>
        /// <param name="options">Configuration options for the model</param>
        public ChainOfThoughtModel(ChainOfThoughtOptions<T> options)
            : base(options)
        {
            _cotOptions = options;
            _reasoningCache = new Dictionary<string, List<Tensor<T>>>();

            // Build the reasoning network architecture
            var reasoningArchitecture = new NeuralNetworkArchitecture<T>(NetworkComplexity.Medium);
            _reasoningNetwork = new NeuralNetwork<T>(reasoningArchitecture);

            // Build the validation network for checking reasoning consistency
            var validationArchitecture = new NeuralNetworkArchitecture<T>(NetworkComplexity.Simple);
            _validationNetwork = new NeuralNetwork<T>(validationArchitecture);
        }

        /// <summary>
        /// Trains the Chain-of-Thought model.
        /// </summary>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            // Train the reasoning network to generate appropriate intermediate steps
            var reasoningSteps = GenerateTrainingSteps(input, expectedOutput);
            
            for (int i = 0; i < reasoningSteps.Count - 1; i++)
            {
                _reasoningNetwork.Train(reasoningSteps[i], reasoningSteps[i + 1]);
            }

            // Train the validation network to recognize valid reasoning chains
            var validChain = new Tensor<T>(new[] { 1 });
            validChain[0] = NumOps.One;
            
            foreach (var step in reasoningSteps)
            {
                _validationNetwork.Train(step, validChain);
            }
        }

        /// <summary>
        /// Performs multi-step reasoning on the input.
        /// </summary>
        public override List<Tensor<T>> ReasonStepByStep(Tensor<T> input, int maxSteps = 10)
        {
            var steps = new List<Tensor<T>> { input };
            var currentStep = input;
            var stepCount = 0;

            // Check cache if enabled
            if (Options.EnableReasoningCache)
            {
                var cacheKey = GetCacheKey(input);
                if (_reasoningCache.TryGetValue(cacheKey, out var cachedSteps))
                {
                    LastDiagnostics["CacheHit"] = true;
                    return new List<Tensor<T>>(cachedSteps);
                }
            }

            while (stepCount < maxSteps && !IsTerminalStep(currentStep))
            {
                var nextStep = GenerateNextStep(currentStep, steps);
                
                if (Options.EnableChainValidation && !IsValidTransition(currentStep, nextStep))
                {
                    // Try alternative reasoning path
                    nextStep = RegenerateStep(currentStep, steps);
                }

                steps.Add(nextStep);
                currentStep = nextStep;
                stepCount++;

                // Check for convergence
                if (HasConverged(steps))
                {
                    break;
                }
            }

            // Cache the result if enabled
            if (Options.EnableReasoningCache)
            {
                var cacheKey = GetCacheKey(input);
                _reasoningCache[cacheKey] = new List<Tensor<T>>(steps);
            }

            return steps;
        }

        /// <summary>
        /// Generates an explanation for the reasoning process.
        /// </summary>
        public override Tensor<T> GenerateExplanation(Tensor<T> input, Tensor<T> prediction)
        {
            // Generate a summary of the reasoning chain
            var explanationSteps = new List<Tensor<T>>();

            foreach (var step in LastReasoningSteps)
            {
                var importance = CalculateStepImportance(step, prediction);
                if (NumOps.GreaterThan(importance, NumOps.FromDouble(_cotOptions.MinConfidenceThreshold)))
                {
                    explanationSteps.Add(step);
                }
            }

            // Combine important steps into explanation
            return CombineExplanationSteps(explanationSteps);
        }

        /// <summary>
        /// Validates the logical consistency of a reasoning chain.
        /// </summary>
        public override bool ValidateReasoningChain(List<Tensor<T>> reasoningSteps)
        {
            if (reasoningSteps.Count < 2)
            {
                return true; // Single step chains are trivially valid
            }

            for (int i = 0; i < reasoningSteps.Count - 1; i++)
            {
                if (!IsValidTransition(reasoningSteps[i], reasoningSteps[i + 1]))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Calculates confidence scores for each reasoning step.
        /// </summary>
        protected override Vector<T> CalculateConfidenceScores(List<Tensor<T>> reasoningSteps)
        {
            var scores = new T[reasoningSteps.Count];

            for (int i = 0; i < reasoningSteps.Count; i++)
            {
                var validation = _validationNetwork.Predict(reasoningSteps[i]);
                scores[i] = validation[0];
            }

            return new Vector<T>(scores);
        }

        /// <summary>
        /// Performs a single refinement step.
        /// </summary>
        protected override Tensor<T> PerformRefinementStep(Tensor<T> input, Tensor<T> currentReasoning, int iteration)
        {
            // Identify weak points in the current reasoning
            var weakPoints = IdentifyWeakPoints(currentReasoning);

            // Generate alternative reasoning for weak points
            var refinedReasoning = new Tensor<T>(currentReasoning.Shape);
            for (int i = 0; i < currentReasoning.Length; i++)
                refinedReasoning[i] = currentReasoning[i];
            
            foreach (var weakPoint in weakPoints)
            {
                var alternative = GenerateAlternativeReasoning(input, currentReasoning, weakPoint);
                refinedReasoning = IntegrateRefinement(refinedReasoning, alternative, weakPoint);
            }

            return refinedReasoning;
        }

        /// <summary>
        /// Gets the model type.
        /// </summary>
        protected override ModelType GetModelType()
        {
            return ModelType.ChainOfThoughtModel;
        }

        /// <summary>
        /// Gets a description of the model.
        /// </summary>
        protected override string GetModelDescription()
        {
            return $"Chain-of-Thought reasoning model with {MaxReasoningDepth} max depth and {_cotOptions.AttentionHeads} attention heads";
        }

        /// <summary>
        /// Estimates the complexity of the model.
        /// </summary>
        protected override double EstimateComplexity()
        {
            var reasoningParams = _reasoningNetwork.GetParameters();
            var validationParams = _validationNetwork.GetParameters();
            return (reasoningParams.Length + validationParams.Length) * MaxReasoningDepth;
        }

        #region Private Helper Methods

        private List<Tensor<T>> GenerateTrainingSteps(Tensor<T> input, Tensor<T> output)
        {
            // Generate synthetic intermediate steps for training
            var steps = new List<Tensor<T>> { input };
            var stepCount = Math.Min(_cotOptions.DefaultMaxSteps, _cotOptions.MaxChainLength);
            
            for (int i = 1; i < stepCount; i++)
            {
                var progress = (double)i / stepCount;
                var interpolated = InterpolateTensors(input, output, progress);
                steps.Add(interpolated);
            }
            
            steps.Add(output);
            return steps;
        }

        private Tensor<T> InterpolateTensors(Tensor<T> start, Tensor<T> end, double progress)
        {
            var result = new Tensor<T>(start.Shape);
            var progressT = NumOps.FromDouble(progress);
            var oneMinusProgress = NumOps.Subtract(NumOps.One, progressT);

            for (int i = 0; i < start.Length; i++)
            {
                var startVal = start[i];
                var endVal = end[i];
                var interpolated = NumOps.Add(
                    NumOps.Multiply(startVal, oneMinusProgress),
                    NumOps.Multiply(endVal, progressT)
                );
                result[i] = interpolated;
            }

            return result;
        }

        private Tensor<T> GenerateNextStep(Tensor<T> currentStep, List<Tensor<T>> history)
        {
            var nextStep = _reasoningNetwork.Predict(currentStep);

            // Apply temperature-based randomness if using stochastic strategies
            if (CurrentStrategy == ReasoningStrategy.MonteCarlo || 
                CurrentStrategy == ReasoningStrategy.HeuristicGuided)
            {
                nextStep = ApplyTemperature(nextStep, _cotOptions.Temperature);
            }

            return nextStep;
        }

        private Tensor<T> ApplyTemperature(Tensor<T> tensor, double temperature)
        {
            var result = new Tensor<T>(tensor.Shape);
            for (int i = 0; i < tensor.Length; i++)
                result[i] = tensor[i];
            var tempT = NumOps.FromDouble(temperature);

            for (int i = 0; i < result.Length; i++)
            {
                var value = result[i];
                var noise = NumOps.FromDouble((Random.NextDouble() - 0.5) * 2.0);
                var scaled = NumOps.Multiply(noise, tempT);
                result[i] = NumOps.Add(value, scaled);
            }

            return result;
        }

        private bool IsValidTransition(Tensor<T> from, Tensor<T> to)
        {
            var combined = ConcatenateTensors(from, to);
            var validation = _validationNetwork.Predict(combined);
            var score = validation[0];
            return NumOps.GreaterThan(score, NumOps.FromDouble(0.5));
        }

        private Tensor<T> ConcatenateTensors(Tensor<T> a, Tensor<T> b)
        {
            // Simple concatenation for validation
            var combinedSize = Math.Min(a.Length, b.Length);
            var result = new Tensor<T>(new[] { combinedSize });

            for (int i = 0; i < combinedSize; i++)
            {
                var avg = NumOps.Divide(
                    NumOps.Add(a[i % a.Length], b[i % b.Length]),
                    NumOps.FromDouble(2.0)
                );
                result[i] = avg;
            }

            return result;
        }

        private bool IsTerminalStep(Tensor<T> step)
        {
            // Check if the step represents a final answer
            var norm = CalculateNorm(step);
            return NumOps.LessThan(norm, NumOps.FromDouble(_cotOptions.TerminalThreshold));
        }

        private T CalculateNorm(Tensor<T> tensor)
        {
            var sum = NumOps.Zero;
            for (int i = 0; i < tensor.Length; i++)
            {
                var value = tensor[i];
                sum = NumOps.Add(sum, NumOps.Multiply(value, value));
            }
            return NumOps.Sqrt(sum);
        }

        private bool HasConverged(List<Tensor<T>> steps)
        {
            if (steps.Count < 3) return false;

            var recent = steps.Skip(Math.Max(0, steps.Count - 3)).ToList();
            var diff1 = TensorDifference(recent[0], recent[1]);
            var diff2 = TensorDifference(recent[1], recent[2]);

            return NumOps.LessThan(diff2, NumOps.Multiply(diff1, NumOps.FromDouble(0.1)));
        }

        private T TensorDifference(Tensor<T> a, Tensor<T> b)
        {
            var sum = NumOps.Zero;
            for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
            {
                var diff = NumOps.Subtract(a[i], b[i]);
                sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
            }
            return NumOps.Sqrt(sum);
        }

        private Tensor<T> RegenerateStep(Tensor<T> currentStep, List<Tensor<T>> history)
        {
            // Add noise and regenerate
            var noisyStep = ApplyTemperature(currentStep, _cotOptions.Temperature * 2);
            return GenerateNextStep(noisyStep, history);
        }

        private string GetCacheKey(Tensor<T> input)
        {
            // Simple hash-based cache key
            var hash = 0;
            for (int i = 0; i < Math.Min(10, input.Length); i++)
            {
                if (input[i] != null)
                    hash = hash * 31 + input[i]!.GetHashCode();
            }
            return $"{CurrentStrategy}_{hash}";
        }

        private T CalculateStepImportance(Tensor<T> step, Tensor<T> finalPrediction)
        {
            // Calculate how much this step contributed to the final prediction
            var similarity = NumOps.Zero;
            var normStep = CalculateNorm(step);
            var normPred = CalculateNorm(finalPrediction);

            if (NumOps.GreaterThan(normStep, NumOps.Zero) && NumOps.GreaterThan(normPred, NumOps.Zero))
            {
                for (int i = 0; i < Math.Min(step.Length, finalPrediction.Length); i++)
                {
                    similarity = NumOps.Add(similarity,
                        NumOps.Multiply(step[i], finalPrediction[i]));
                }
                similarity = NumOps.Divide(similarity, NumOps.Multiply(normStep, normPred));
            }

            return similarity;
        }

        private Tensor<T> CombineExplanationSteps(List<Tensor<T>> steps)
        {
            if (steps.Count == 0)
            {
                return new Tensor<T>(_cotOptions.HiddenShape);
            }

            // Average the important steps
            var result = new Tensor<T>(steps[0].Shape);
            foreach (var step in steps)
            {
                result = result.Add(step);
            }

            var count = NumOps.FromDouble(steps.Count);
            var divisor = new Tensor<T>(result.Shape);
            for (int i = 0; i < divisor.Length; i++)
                divisor[i] = count;
            return result.Divide(divisor);
        }

        private List<int> IdentifyWeakPoints(Tensor<T> reasoning)
        {
            var weakPoints = new List<int>();
            var validation = _validationNetwork.Predict(reasoning);
            var score = validation[0];

            if (NumOps.LessThan(score, NumOps.FromDouble(_cotOptions.MinConfidenceThreshold)))
            {
                // Find indices with low activation
                for (int i = 0; i < reasoning.Length; i++)
                {
                    var value = NumOps.Abs(reasoning[i]);
                    if (NumOps.LessThan(value, NumOps.FromDouble(0.1)))
                    {
                        weakPoints.Add(i);
                    }
                }
            }

            return weakPoints.Take(5).ToList(); // Limit to top 5 weak points
        }

        private Tensor<T> GenerateAlternativeReasoning(Tensor<T> input, Tensor<T> current, int weakPoint)
        {
            // Generate alternative reasoning focusing on the weak point
            var alternative = new Tensor<T>(current.Shape);
            for (int i = 0; i < current.Length; i++)
                alternative[i] = current[i];
            
            // Amplify the weak dimension
            var currentValue = alternative[weakPoint];
            var amplified = NumOps.Multiply(currentValue, NumOps.FromDouble(2.0));
            alternative[weakPoint] = amplified;

            // Re-process through reasoning network
            return _reasoningNetwork.Predict(alternative);
        }

        private Tensor<T> IntegrateRefinement(Tensor<T> original, Tensor<T> refinement, int position)
        {
            var result = new Tensor<T>(original.Shape);
            for (int i = 0; i < original.Length; i++)
                result[i] = original[i];
            
            // Blend the refinement at the specified position
            var alpha = NumOps.FromDouble(0.3); // Blending factor
            var oneMinusAlpha = NumOps.Subtract(NumOps.One, alpha);

            var originalValue = result[position];
            var refinementValue = refinement[position];
            var blended = NumOps.Add(
                NumOps.Multiply(originalValue, oneMinusAlpha),
                NumOps.Multiply(refinementValue, alpha)
            );

            result[position] = blended;
            return result;
        }

        #endregion

        #region IParameterizable Implementation

        public override Vector<T> GetParameters()
        {
            // Get parameters from both networks and flatten them into a single vector
            var allParams = new List<T>();
            
            var reasoningParams = _reasoningNetwork.GetParameters();
            var validationParams = _validationNetwork.GetParameters();
            
            for (int i = 0; i < reasoningParams.Length; i++)
                allParams.Add(reasoningParams[i]);
            
            for (int i = 0; i < validationParams.Length; i++)
                allParams.Add(validationParams[i]);
            
            return new Vector<T>(allParams.ToArray());
        }

        public override void SetParameters(Vector<T> parameters)
        {
            // Get current parameter counts for each network
            var reasoningParams = _reasoningNetwork.GetParameters();
            var validationParams = _validationNetwork.GetParameters();
            
            int offset = 0;
            
            // Set parameters for reasoning network
            var reasoningSize = reasoningParams.Length;
            if (offset + reasoningSize > parameters.Length)
                throw new ArgumentException("Parameter vector size mismatch");
            
            var reasoningData = new T[reasoningSize];
            for (int j = 0; j < reasoningSize; j++)
            {
                reasoningData[j] = parameters[offset + j];
            }
            _reasoningNetwork.SetParameters(new Vector<T>(reasoningData));
            offset += reasoningSize;
            
            // Set parameters for validation network
            var validationSize = validationParams.Length;
            if (offset + validationSize > parameters.Length)
                throw new ArgumentException("Parameter vector size mismatch");
            
            var validationData = new T[validationSize];
            for (int j = 0; j < validationSize; j++)
            {
                validationData[j] = parameters[offset + j];
            }
            _validationNetwork.SetParameters(new Vector<T>(validationData));
        }

        public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
        {
            var newModel = new ChainOfThoughtModel<T>(_cotOptions);
            newModel.SetParameters(parameters);
            return newModel;
        }

        #endregion

        #region IModelSerializer Implementation

        public override byte[] Serialize()
        {
            using (var ms = new MemoryStream())
            using (var writer = new BinaryWriter(ms))
            {
                // Write model type identifier
                writer.Write("ChainOfThoughtModel");
                
                // Write options
                writer.Write(_cotOptions.MaxChainLength);
                writer.Write(_cotOptions.MinConfidenceThreshold);
                writer.Write(_cotOptions.ConvergenceThreshold);
                writer.Write(_cotOptions.AttentionHeads);
                writer.Write(_cotOptions.HiddenSize);
                writer.Write(_cotOptions.DefaultMaxSteps);
                writer.Write(_cotOptions.EnableReasoningCache);
                writer.Write(_cotOptions.BeamWidth);
                writer.Write(_cotOptions.ExplorationFactor);
                
                // Save neural networks
                var reasoningBytes = _reasoningNetwork.Serialize();
                var validationBytes = _validationNetwork.Serialize();
                
                writer.Write(reasoningBytes.Length);
                writer.Write(reasoningBytes);
                
                writer.Write(validationBytes.Length);
                writer.Write(validationBytes);
                
                // Save cache if enabled
                writer.Write(Options.EnableReasoningCache && _reasoningCache.Count > 0);
                if (Options.EnableReasoningCache && _reasoningCache.Count > 0)
                {
                    writer.Write(_reasoningCache.Count);
                    foreach (var kvp in _reasoningCache)
                    {
                        writer.Write(kvp.Key);
                        writer.Write(kvp.Value.Count);
                        foreach (var tensor in kvp.Value)
                        {
                            writer.Write(tensor.Shape.Length);
                            foreach (var dim in tensor.Shape)
                                writer.Write(dim);
                            
                            for (int i = 0; i < tensor.Length; i++)
                            {
                                // Write numeric value as double for simplicity
                                writer.Write(Convert.ToDouble(tensor[i]));
                            }
                        }
                    }
                }
                
                return ms.ToArray();
            }
        }

        public override void Deserialize(byte[] data)
        {
            using (var ms = new MemoryStream(data))
            using (var reader = new BinaryReader(ms))
            {
                // Read and verify model type
                var modelType = reader.ReadString();
                if (modelType != "ChainOfThoughtModel")
                    throw new InvalidOperationException($"Invalid model type: {modelType}");
                
                // Read options
                _cotOptions.MaxChainLength = reader.ReadInt32();
                _cotOptions.MinConfidenceThreshold = reader.ReadDouble();
                _cotOptions.ConvergenceThreshold = reader.ReadDouble();
                _cotOptions.AttentionHeads = reader.ReadInt32();
                _cotOptions.HiddenSize = reader.ReadInt32();
                _cotOptions.DefaultMaxSteps = reader.ReadInt32();
                _cotOptions.EnableReasoningCache = reader.ReadBoolean();
                _cotOptions.BeamWidth = reader.ReadInt32();
                _cotOptions.ExplorationFactor = reader.ReadDouble();
                
                // Load neural networks
                var reasoningSize = reader.ReadInt32();
                var reasoningBytes = reader.ReadBytes(reasoningSize);
                _reasoningNetwork.Deserialize(reasoningBytes);
                
                var validationSize = reader.ReadInt32();
                var validationBytes = reader.ReadBytes(validationSize);
                _validationNetwork.Deserialize(validationBytes);
                
                // Load cache if present
                _reasoningCache.Clear();
                var hasCache = reader.ReadBoolean();
                if (hasCache)
                {
                    var cacheCount = reader.ReadInt32();
                    for (int i = 0; i < cacheCount; i++)
                    {
                        var key = reader.ReadString();
                        var tensorCount = reader.ReadInt32();
                        var tensors = new List<Tensor<T>>();
                        
                        for (int j = 0; j < tensorCount; j++)
                        {
                            var shapeLength = reader.ReadInt32();
                            var shape = new int[shapeLength];
                            for (int k = 0; k < shapeLength; k++)
                                shape[k] = reader.ReadInt32();
                            
                            var tensor = new Tensor<T>(shape);
                            for (int k = 0; k < tensor.Length; k++)
                            {
                                var value = reader.ReadDouble();
                                tensor[k] = NumOps.FromDouble(value);
                            }
                            
                            tensors.Add(tensor);
                        }
                        
                        _reasoningCache[key] = tensors;
                    }
                }
            }
        }

        #endregion

        #region ICloneable Implementation

        public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
        {
            var copy = new ChainOfThoughtModel<T>(_cotOptions);
            copy.SetParameters(GetParameters());
            
            if (Options.EnableReasoningCache)
            {
                foreach (var kvp in _reasoningCache)
                {
                    copy._reasoningCache[kvp.Key] = new List<Tensor<T>>(kvp.Value.Select(t => 
                    {
                        var deepCopy = new Tensor<T>(t.Shape);
                        for (int i = 0; i < t.Length; i++)
                            deepCopy[i] = t[i];
                        return deepCopy;
                    }));
                }
            }

            return copy;
        }

        public override IFullModel<T, Tensor<T>, Tensor<T>> Clone()
        {
            return DeepCopy();
        }

        #endregion
    }
}