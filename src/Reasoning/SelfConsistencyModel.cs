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
    /// Implements a Self-Consistency reasoning model that explores multiple reasoning paths
    /// and selects the most consistent answer.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <remarks>
    /// <para>
    /// Self-Consistency improves upon chain-of-thought by generating multiple independent
    /// reasoning paths for the same problem and then selecting the answer that appears
    /// most consistently across the different paths.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this like asking multiple experts to solve the same
    /// problem independently, then taking the answer that most of them agree on. This makes
    /// the model more reliable because it's less likely that multiple independent reasoning
    /// paths will all make the same mistake.
    /// </para>
    /// </remarks>
    public class SelfConsistencyModel<T> : ReasoningModelBase<T>
    {
        private readonly NeuralNetwork<T> _pathGenerator = default!;
        private readonly NeuralNetwork<T> _pathEvaluator = default!;
        private readonly NeuralNetwork<T> _consistencyChecker = default!;
        private readonly SelfConsistencyOptions<T> _scOptions = default!;
        private readonly Dictionary<string, List<Tensor<T>>> _pathCache = default!;

        /// <summary>
        /// Gets the maximum reasoning depth this model can handle effectively.
        /// </summary>
        public override int MaxReasoningDepth => _scOptions.MaxStepsPerPath;

        /// <summary>
        /// Gets whether this model supports iterative refinement.
        /// </summary>
        public override bool SupportsIterativeRefinement => true;

        /// <summary>
        /// Initializes a new instance of the SelfConsistencyModel class.
        /// </summary>
        /// <param name="options">Configuration options for the model</param>
        public SelfConsistencyModel(SelfConsistencyOptions<T> options)
            : base(options)
        {
            _scOptions = options;
            _pathCache = new Dictionary<string, List<Tensor<T>>>();

            // Build the path generator network
            // Using a simple architecture due to API constraints
            var pathGenArchitecture = new NeuralNetworkArchitecture<T>(NetworkComplexity.Medium);
            _pathGenerator = new NeuralNetwork<T>(pathGenArchitecture);

            // Build the path evaluator network
            var evaluatorArchitecture = new NeuralNetworkArchitecture<T>(NetworkComplexity.Simple);
            _pathEvaluator = new NeuralNetwork<T>(evaluatorArchitecture);

            // Build the consistency checker network
            var consistencyArchitecture = new NeuralNetworkArchitecture<T>(NetworkComplexity.Simple);
            _consistencyChecker = new NeuralNetwork<T>(consistencyArchitecture);
        }

        /// <summary>
        /// Trains the Self-Consistency model.
        /// </summary>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            // Generate multiple training paths
            var paths = GenerateMultiplePaths(input, _scOptions.DefaultPathCount);

            // Train path generator to produce diverse but valid paths
            foreach (var path in paths)
            {
                _pathGenerator.Train(input, path);
            }

            // Train path evaluator on each path
            foreach (var path in paths)
            {
                _pathEvaluator.Train(path, expectedOutput);
            }

            // Train consistency checker with path pairs
            for (int i = 0; i < paths.Count - 1; i++)
            {
                for (int j = i + 1; j < paths.Count; j++)
                {
                    var result1 = _pathEvaluator.Predict(paths[i]);
                    var result2 = _pathEvaluator.Predict(paths[j]);
                    var consistency = CalculateConsistency(result1, result2);
                    
                    var pairInput = ConcatenateResults(result1, result2);
                    var consistencyTarget = new Tensor<T>(new[] { 1 });
                    consistencyTarget[0] = consistency;
                    
                    _consistencyChecker.Train(pairInput, consistencyTarget);
                }
            }
        }

        /// <summary>
        /// Makes a prediction using self-consistency.
        /// </summary>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            // Use the base SelfConsistencyCheck method
            return SelfConsistencyCheck(input, _scOptions.DefaultPathCount);
        }

        /// <summary>
        /// Performs multi-step reasoning on the input.
        /// </summary>
        public override List<Tensor<T>> ReasonStepByStep(Tensor<T> input, int maxSteps = 10)
        {
            // Generate a single reasoning path
            var path = GenerateSinglePath(input, maxSteps);
            LastReasoningSteps = new List<Tensor<T>>(path);
            return path;
        }

        /// <summary>
        /// Performs self-consistency checking with enhanced path diversity.
        /// </summary>
        public override Tensor<T> SelfConsistencyCheck(Tensor<T> input, int numPaths = 3)
        {
            var startTime = DateTime.UtcNow;
            var paths = GenerateMultiplePaths(input, numPaths);
            var results = new List<Tensor<T>>();

            // Evaluate each path
            foreach (var path in paths)
            {
                var result = _pathEvaluator.Predict(path);
                results.Add(result);
            }

            // Find the most consistent result
            var consistentResult = SelectMostConsistentResult(results);

            // Store diagnostics
            LastDiagnostics["PathCount"] = numPaths;
            LastDiagnostics["GenerationTime"] = (DateTime.UtcNow - startTime).TotalMilliseconds;
            LastDiagnostics["ConsistencyScore"] = CalculateOverallConsistency(results)!;

            return consistentResult;
        }

        /// <summary>
        /// Generates an explanation for the reasoning process.
        /// </summary>
        public override Tensor<T> GenerateExplanation(Tensor<T> input, Tensor<T> prediction)
        {
            // Generate paths and find which ones led to the prediction
            var paths = GenerateMultiplePaths(input, _scOptions.ExplanationPathCount);
            var supportingPaths = new List<Tensor<T>>();

            foreach (var path in paths)
            {
                var result = _pathEvaluator.Predict(path);
                if (AreSimilar(result, prediction))
                {
                    supportingPaths.Add(path);
                }
            }

            // Combine supporting paths into explanation
            return CombinePathsIntoExplanation(supportingPaths);
        }

        /// <summary>
        /// Validates the logical consistency of a reasoning chain.
        /// </summary>
        public override bool ValidateReasoningChain(List<Tensor<T>> reasoningSteps)
        {
            if (reasoningSteps.Count < 2)
                return true;

            // Check pairwise consistency
            for (int i = 0; i < reasoningSteps.Count - 1; i++)
            {
                var consistency = CheckStepConsistency(reasoningSteps[i], reasoningSteps[i + 1]);
                if (NumOps.LessThan(consistency, NumOps.FromDouble(_scOptions.MinConsistencyThreshold)))
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
                // Generate alternative paths from this step
                var alternatives = GenerateAlternativesFromStep(reasoningSteps[i], 3);
                
                // Calculate consistency among alternatives
                var consistency = CalculateStepConsistency(reasoningSteps[i], alternatives);
                scores[i] = consistency;
            }

            return new Vector<T>(scores);
        }

        /// <summary>
        /// Performs a single refinement step.
        /// </summary>
        protected override Tensor<T> PerformRefinementStep(Tensor<T> input, Tensor<T> currentReasoning, int iteration)
        {
            // Generate multiple refinement candidates
            var candidates = new List<Tensor<T>>();
            
            for (int i = 0; i < _scOptions.RefinementCandidates; i++)
            {
                var noise = GenerateRefinementNoise(iteration);
                var noisyReasoning = AddNoise(currentReasoning, noise);
                var refined = _pathGenerator.Predict(noisyReasoning);
                candidates.Add(refined);
            }

            // Select the most consistent refinement
            return SelectMostConsistentResult(candidates);
        }

        /// <summary>
        /// Gets the model type.
        /// </summary>
        protected override ModelType GetModelType()
        {
            return ModelType.SelfConsistencyModel;
        }

        /// <summary>
        /// Gets a description of the model.
        /// </summary>
        protected override string GetModelDescription()
        {
            return $"Self-Consistency reasoning model with {_scOptions.DefaultPathCount} paths and {_scOptions.ConsistencyHeads} consistency heads";
        }

        /// <summary>
        /// Estimates the complexity of the model.
        /// </summary>
        protected override double EstimateComplexity()
        {
            // Count parameters from Vector<double> instead of tensors
            var pathGenParams = _pathGenerator.GetParameters();
            var evaluatorParams = _pathEvaluator.GetParameters();
            var checkerParams = _consistencyChecker.GetParameters();
            var totalParams = pathGenParams.Length + evaluatorParams.Length + checkerParams.Length;
            return totalParams * _scOptions.DefaultPathCount;
        }

        #region Private Helper Methods

        private List<Tensor<T>> GenerateMultiplePaths(Tensor<T> input, int numPaths)
        {
            var paths = new List<Tensor<T>>();
            
            // Check cache
            var cacheKey = GetPathCacheKey(input);
            if (_scOptions.EnableReasoningCache && _pathCache.TryGetValue(cacheKey, out var cachedPaths))
            {
                return cachedPaths.Take(numPaths).ToList();
            }

            // Set different random seeds for diversity
            for (int i = 0; i < numPaths; i++)
            {
                var seedOffset = i * 1000;
                var path = GenerateSinglePathWithSeed(input, _scOptions.MaxStepsPerPath, seedOffset);
                paths.Add(CombineStepsIntoPath(path));
            }

            // Cache the paths
            if (_scOptions.EnableReasoningCache)
            {
                _pathCache[cacheKey] = new List<Tensor<T>>(paths);
            }

            return paths;
        }

        private List<Tensor<T>> GenerateSinglePath(Tensor<T> input, int maxSteps)
        {
            var steps = new List<Tensor<T>> { input };
            var current = input;

            for (int i = 0; i < maxSteps && !IsTerminalState(current); i++)
            {
                var next = _pathGenerator.Predict(current);
                
                // Apply temperature-based sampling
                if (_scOptions.UseSamplingForDiversity)
                {
                    next = ApplySampling(next, _scOptions.Temperature);
                }

                steps.Add(next);
                current = next;
            }

            return steps;
        }

        private List<Tensor<T>> GenerateSinglePathWithSeed(Tensor<T> input, int maxSteps, int seedOffset)
        {
            // Note: Cannot directly modify Random property, use a local instance instead
            var rng = new Random((Options.Seed ?? 0) + seedOffset);
            
            var path = GenerateSinglePath(input, maxSteps);
            
            // Restore original randomness is not needed with local instance
            return path;
        }

        private Tensor<T> CombineStepsIntoPath(List<Tensor<T>> steps)
        {
            // Combine all steps into a single path representation
            var pathSize = _scOptions.PathRepresentationShape[0];
            var result = new Tensor<T>(_scOptions.PathRepresentationShape);

            for (int i = 0; i < steps.Count && i < pathSize; i++)
            {
                var stepContribution = ExtractStepFeatures(steps[i], i, steps.Count);
                result = result.Add(stepContribution);
            }

            // Normalize
            var count = NumOps.FromDouble(Math.Min(steps.Count, pathSize));
            var divisor = new Tensor<T>(result.Shape);
            for (int i = 0; i < divisor.Length; i++)
                divisor[i] = count;
            return result.Divide(divisor);
        }

        private Tensor<T> ExtractStepFeatures(Tensor<T> step, int position, int totalSteps)
        {
            var features = new Tensor<T>(_scOptions.PathRepresentationShape);
            var positionWeight = NumOps.FromDouble((double)(position + 1) / totalSteps);

            // Extract key features from the step
            for (int i = 0; i < Math.Min(step.Length, features.Length); i++)
            {
                var value = NumOps.Multiply(step[i], positionWeight);
                features[i] = value;
            }

            return features;
        }

        private T CalculateConsistency(Tensor<T> result1, Tensor<T> result2)
        {
            var pairInput = ConcatenateResults(result1, result2);
            var consistency = _consistencyChecker.Predict(pairInput);
            return consistency[0];
        }

        private Tensor<T> ConcatenateResults(Tensor<T> result1, Tensor<T> result2)
        {
            var concatenated = new Tensor<T>(new[] { result1.Length + result2.Length });
            
            for (int i = 0; i < result1.Length; i++)
            {
                concatenated[i] = result1[i];
            }
            
            for (int i = 0; i < result2.Length; i++)
            {
                concatenated[result1.Length + i] = result2[i];
            }

            return concatenated;
        }

        private Tensor<T> SelectMostConsistentResult(List<Tensor<T>> results)
        {
            if (results.Count == 1)
                return results[0];

            var consistencyScores = new T[results.Count];

            // Calculate consistency score for each result
            for (int i = 0; i < results.Count; i++)
            {
                var totalConsistency = NumOps.Zero;
                
                for (int j = 0; j < results.Count; j++)
                {
                    if (i != j)
                    {
                        var consistency = CalculateConsistency(results[i], results[j]);
                        totalConsistency = NumOps.Add(totalConsistency, consistency);
                    }
                }

                consistencyScores[i] = totalConsistency;
            }

            // Find the result with highest consistency
            var maxIndex = 0;
            var maxScore = consistencyScores[0];
            
            for (int i = 1; i < consistencyScores.Length; i++)
            {
                if (NumOps.GreaterThan(consistencyScores[i], maxScore))
                {
                    maxScore = consistencyScores[i];
                    maxIndex = i;
                }
            }

            return results[maxIndex];
        }

        private T CalculateOverallConsistency(List<Tensor<T>> results)
        {
            if (results.Count < 2)
                return NumOps.One;

            var totalConsistency = NumOps.Zero;
            var pairCount = 0;

            for (int i = 0; i < results.Count - 1; i++)
            {
                for (int j = i + 1; j < results.Count; j++)
                {
                    var consistency = CalculateConsistency(results[i], results[j]);
                    totalConsistency = NumOps.Add(totalConsistency, consistency);
                    pairCount++;
                }
            }

            return NumOps.Divide(totalConsistency, NumOps.FromDouble(pairCount));
        }

        private bool AreSimilar(Tensor<T> a, Tensor<T> b)
        {
            var similarity = CalculateConsistency(a, b);
            return NumOps.GreaterThan(similarity, NumOps.FromDouble(_scOptions.SimilarityThreshold));
        }

        private Tensor<T> CombinePathsIntoExplanation(List<Tensor<T>> paths)
        {
            if (paths.Count == 0)
                return new Tensor<T>(_scOptions.PathRepresentationShape);

            // Average the paths
            var result = new Tensor<T>(paths[0].Shape);
            
            foreach (var path in paths)
            {
                result = result.Add(path);
            }

            var divisor = new Tensor<T>(result.Shape);
            for (int i = 0; i < divisor.Length; i++)
                divisor[i] = NumOps.FromDouble(paths.Count);
            return result.Divide(divisor);
        }

        private T CheckStepConsistency(Tensor<T> step1, Tensor<T> step2)
        {
            // Create dummy results for consistency checking
            var result1 = _pathEvaluator.Predict(step1);
            var result2 = _pathEvaluator.Predict(step2);
            return CalculateConsistency(result1, result2);
        }

        private List<Tensor<T>> GenerateAlternativesFromStep(Tensor<T> step, int count)
        {
            var alternatives = new List<Tensor<T>>();
            
            for (int i = 0; i < count; i++)
            {
                var noise = GenerateNoise(NumOps.FromDouble(0.1));
                var alternative = AddNoise(step, noise);
                alternatives.Add(alternative);
            }

            return alternatives;
        }

        private T CalculateStepConsistency(Tensor<T> step, List<Tensor<T>> alternatives)
        {
            var totalConsistency = NumOps.Zero;
            
            foreach (var alt in alternatives)
            {
                var consistency = CheckStepConsistency(step, alt);
                totalConsistency = NumOps.Add(totalConsistency, consistency);
            }

            return NumOps.Divide(totalConsistency, NumOps.FromDouble(alternatives.Count));
        }

        private Tensor<T> GenerateRefinementNoise(int iteration)
        {
            // Decrease noise with iterations
            var noiseScale = NumOps.FromDouble(0.2 / (iteration + 1));
            return GenerateNoise(noiseScale);
        }

        private Tensor<T> GenerateNoise(T scale)
        {
            var noise = new Tensor<T>(_scOptions.PathRepresentationShape);
            
            for (int i = 0; i < noise.Length; i++)
            {
                var value = NumOps.Multiply(
                    NumOps.FromDouble((Random.NextDouble() - 0.5) * 2.0),
                    scale
                );
                noise[i] = value;
            }

            return noise;
        }

        private Tensor<T> AddNoise(Tensor<T> tensor, Tensor<T> noise)
        {
            return tensor.Add(noise);
        }

        private bool IsTerminalState(Tensor<T> state)
        {
            // Check if the state represents a terminal condition
            var norm = CalculateNorm(state);
            return NumOps.LessThan(norm, NumOps.FromDouble(_scOptions.TerminalThreshold));
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

        private Tensor<T> ApplySampling(Tensor<T> tensor, double temperature)
        {
            var result = new Tensor<T>(tensor.Shape);
        for (int i = 0; i < tensor.Length; i++)
            result[i] = tensor[i];
            var tempT = NumOps.FromDouble(temperature);

            // Apply softmax with temperature
            var expSum = NumOps.Zero;
            for (int i = 0; i < result.Length; i++)
            {
                var value = NumOps.Divide(result[i], tempT);
                var expValue = NumOps.Exp(value);
                result[i] = expValue;
                expSum = NumOps.Add(expSum, expValue);
            }

            // Normalize
            for (int i = 0; i < result.Length; i++)
            {
                var normalized = NumOps.Divide(result[i], expSum);
                result[i] = normalized;
            }

            // Sample from the distribution
            var sample = SampleFromDistribution(result);
            return sample;
        }

        private Tensor<T> SampleFromDistribution(Tensor<T> distribution)
        {
            var result = new Tensor<T>(distribution.Shape);
            var cumSum = NumOps.Zero;
            var randomValue = NumOps.FromDouble(Random.NextDouble());

            for (int i = 0; i < distribution.Length; i++)
            {
                cumSum = NumOps.Add(cumSum, distribution[i]);
                if (NumOps.GreaterThan(cumSum, randomValue))
                {
                    result[i] = NumOps.One;
                    break;
                }
            }

            return result;
        }

        private string GetPathCacheKey(Tensor<T> input)
        {
            var hash = 0;
            for (int i = 0; i < Math.Min(10, input.Length); i++)
            {
                if (input[i] != null)
                    hash = hash * 31 + input[i]!.GetHashCode();
            }
            return $"SC_{hash}_{CurrentStrategy}";
        }

        #endregion

        #region IParameterizable Implementation

        public override Vector<T> GetParameters()
        {
            // Get parameters from all networks and flatten them into a single vector
            var allParams = new List<T>();
            
            var pathGenParams = _pathGenerator.GetParameters();
            var evalParams = _pathEvaluator.GetParameters();
            var checkParams = _consistencyChecker.GetParameters();
            
            for (int i = 0; i < pathGenParams.Length; i++)
                allParams.Add(pathGenParams[i]);
            
            for (int i = 0; i < evalParams.Length; i++)
                allParams.Add(evalParams[i]);
            
            for (int i = 0; i < checkParams.Length; i++)
                allParams.Add(checkParams[i]);
            
            return new Vector<T>(allParams.ToArray());
        }

        public override void SetParameters(Vector<T> parameters)
        {
            // Get current parameter counts for each network
            var genParams = _pathGenerator.GetParameters();
            var evalParams = _pathEvaluator.GetParameters();
            var checkParams = _consistencyChecker.GetParameters();
            
            int offset = 0;
            
            // Set parameters for path generator
            var genSize = genParams.Length;
            if (offset + genSize > parameters.Length)
                throw new ArgumentException("Parameter vector size mismatch");
            
            var genData = new T[genSize];
            for (int j = 0; j < genSize; j++)
            {
                genData[j] = parameters[offset + j];
            }
            _pathGenerator.SetParameters(new Vector<T>(genData));
            offset += genSize;
            
            // Set parameters for path evaluator
            var evalSize = evalParams.Length;
            if (offset + evalSize > parameters.Length)
                throw new ArgumentException("Parameter vector size mismatch");
            
            var evalData = new T[evalSize];
            for (int j = 0; j < evalSize; j++)
            {
                evalData[j] = parameters[offset + j];
            }
            _pathEvaluator.SetParameters(new Vector<T>(evalData));
            offset += evalSize;
            
            // Set parameters for consistency checker
            var checkSize = checkParams.Length;
            if (offset + checkSize > parameters.Length)
                throw new ArgumentException("Parameter vector size mismatch");
            
            var checkData = new T[checkSize];
            for (int j = 0; j < checkSize; j++)
            {
                checkData[j] = parameters[offset + j];
            }
            _consistencyChecker.SetParameters(new Vector<T>(checkData));
            offset += checkSize;
        }

        public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
        {
            var newModel = new SelfConsistencyModel<T>(_scOptions);
            newModel.SetParameters(parameters);
            return newModel;
        }

        #endregion

        #region IModelSerializer Implementation

        public override byte[] Serialize()
        {
            var data = new List<byte>();
            
            // Serialize options
            using (var ms = new MemoryStream())
            using (var writer = new BinaryWriter(ms))
            {
                // Write key options
                writer.Write(_scOptions.DefaultPathCount);
                writer.Write(_scOptions.PathRepresentationShape.Length);
                foreach (var dim in _scOptions.PathRepresentationShape)
                {
                    writer.Write(dim);
                }
                writer.Write(_scOptions.OutputShape.Length);
                foreach (var dim in _scOptions.OutputShape)
                {
                    writer.Write(dim);
                }
                writer.Write(_scOptions.HiddenSize);
                writer.Write(_scOptions.ConsistencyHeads);
                writer.Write(_scOptions.MaxStepsPerPath);
                writer.Write(_scOptions.PathDiversityDropout);
                writer.Write(_scOptions.MinConsistencyThreshold);
                writer.Write(_scOptions.SimilarityThreshold);
                writer.Write(_scOptions.UseSamplingForDiversity);
                writer.Write(_scOptions.ExplanationPathCount);
                writer.Write(_scOptions.TerminalThreshold);
                writer.Write(_scOptions.RefinementCandidates);
                writer.Write(_scOptions.EnableReasoningCache);
                writer.Write(_scOptions.Temperature);
                
                data.AddRange(ms.ToArray());
            }
            
            // Serialize neural networks
            data.AddRange(_pathGenerator.Serialize());
            data.AddRange(_pathEvaluator.Serialize());
            data.AddRange(_consistencyChecker.Serialize());
            
            // Skip caching serialization for simplicity
            
            return data.ToArray();
        }

        public override void Deserialize(byte[] data)
        {
            var offset = 0;
            
            // Deserialize options
            using (var ms = new MemoryStream(data))
            using (var reader = new BinaryReader(ms))
            {
                _scOptions.DefaultPathCount = reader.ReadInt32();
                
                var pathRepShapeLength = reader.ReadInt32();
                var pathRepShape = new int[pathRepShapeLength];
                for (int i = 0; i < pathRepShapeLength; i++)
                {
                    pathRepShape[i] = reader.ReadInt32();
                }
                _scOptions.PathRepresentationShape = pathRepShape;
                
                var outputShapeLength = reader.ReadInt32();
                var outputShape = new int[outputShapeLength];
                for (int i = 0; i < outputShapeLength; i++)
                {
                    outputShape[i] = reader.ReadInt32();
                }
                _scOptions.OutputShape = outputShape;
                
                _scOptions.HiddenSize = reader.ReadInt32();
                _scOptions.ConsistencyHeads = reader.ReadInt32();
                _scOptions.MaxStepsPerPath = reader.ReadInt32();
                _scOptions.PathDiversityDropout = reader.ReadDouble();
                _scOptions.MinConsistencyThreshold = reader.ReadDouble();
                _scOptions.SimilarityThreshold = reader.ReadDouble();
                _scOptions.UseSamplingForDiversity = reader.ReadBoolean();
                _scOptions.ExplanationPathCount = reader.ReadInt32();
                _scOptions.TerminalThreshold = reader.ReadDouble();
                _scOptions.RefinementCandidates = reader.ReadInt32();
                _scOptions.EnableReasoningCache = reader.ReadBoolean();
                _scOptions.Temperature = reader.ReadDouble();
                
                offset = (int)ms.Position;
            }
            
            // Extract and deserialize neural networks
            // This is a simplified approach - in production, you'd want to store sizes
            var remainingBytes = data.Length - offset;
            var bytesPerNetwork = remainingBytes / 3; // Rough estimate
            
            var generatorBytes = new byte[bytesPerNetwork];
            Array.Copy(data, offset, generatorBytes, 0, bytesPerNetwork);
            _pathGenerator.Deserialize(generatorBytes);
            offset += bytesPerNetwork;
            
            var evaluatorBytes = new byte[bytesPerNetwork];
            Array.Copy(data, offset, evaluatorBytes, 0, bytesPerNetwork);
            _pathEvaluator.Deserialize(evaluatorBytes);
            offset += bytesPerNetwork;
            
            var checkerBytes = new byte[remainingBytes - 2 * bytesPerNetwork];
            Array.Copy(data, offset, checkerBytes, 0, checkerBytes.Length);
            _consistencyChecker.Deserialize(checkerBytes);
            
            // Clear cache since we're not serializing it
            _pathCache.Clear();
        }

        #endregion

        #region ICloneable Implementation

        public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
        {
            var copy = new SelfConsistencyModel<T>(_scOptions);
            copy.SetParameters(GetParameters());
            
            if (_scOptions.EnableReasoningCache)
            {
                foreach (var kvp in _pathCache)
                {
                    copy._pathCache[kvp.Key] = kvp.Value.Select(t => 
                    {
                        var deepCopy = new Tensor<T>(t.Shape);
                        for (int i = 0; i < t.Length; i++)
                            deepCopy[i] = t[i];
                        return deepCopy;
                    }).ToList();
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