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
    /// Implements an Iterative Refinement reasoning model that improves answers through
    /// multiple rounds of self-reflection and correction.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <remarks>
    /// <para>
    /// Iterative Refinement models improve their answers through multiple rounds of
    /// self-reflection and correction. After generating an initial answer, the model
    /// critically examines its own reasoning, identifies potential errors or gaps,
    /// and produces an improved version.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this like writing an essay where you create a
    /// first draft, then revise it multiple times to improve clarity, fix errors, and
    /// strengthen arguments. Each revision builds on insights from previous ones, gradually
    /// converging on a high-quality answer.
    /// </para>
    /// </remarks>
    public class IterativeRefinementModel<T> : ReasoningModelBase<T>
    {
        private readonly NeuralNetwork<T> _initialReasoner = default!;
        private readonly NeuralNetwork<T> _critic = default!;
        private readonly NeuralNetwork<T> _refiner = default!;
        private readonly NeuralNetwork<T> _convergenceDetector = default!;
        private readonly IterativeRefinementOptions<T> _irOptions = default!;
        private readonly List<RefinementHistory> _refinementHistories = default!;

        /// <summary>
        /// Tracks the history of refinements for analysis.
        /// </summary>
        private class RefinementHistory
        {
            public List<Tensor<T>> Iterations { get; set; }
            public List<T> QualityScores { get; set; }
            public List<Tensor<T>> Critiques { get; set; }
            public bool Converged { get; set; }
            public int ConvergenceIteration { get; set; }

            public RefinementHistory()
            {
                Iterations = new List<Tensor<T>>();
                QualityScores = new List<T>();
                Critiques = new List<Tensor<T>>();
                Converged = false;
                ConvergenceIteration = -1;
            }
        }

        /// <summary>
        /// Gets the maximum reasoning depth this model can handle effectively.
        /// </summary>
        public override int MaxReasoningDepth => _irOptions.MaxRefinementIterations;

        /// <summary>
        /// Gets whether this model supports iterative refinement.
        /// </summary>
        public override bool SupportsIterativeRefinement => true;

        /// <summary>
        /// Initializes a new instance of the IterativeRefinementModel class.
        /// </summary>
        /// <param name="options">Configuration options for the model</param>
        public IterativeRefinementModel(IterativeRefinementOptions<T> options)
            : base(options)
        {
            _irOptions = options;
            _refinementHistories = new List<RefinementHistory>();

            // Build the initial reasoning network
            var reasonerArchitecture = new NeuralNetworkArchitecture<T>(NetworkComplexity.Medium);
            _initialReasoner = new NeuralNetwork<T>(reasonerArchitecture);

            // Build the critic network
            var criticArchitecture = new NeuralNetworkArchitecture<T>(NetworkComplexity.Medium);
            _critic = new NeuralNetwork<T>(criticArchitecture);

            // Build the refiner network
            var refinerArchitecture = new NeuralNetworkArchitecture<T>(NetworkComplexity.Medium);
            _refiner = new NeuralNetwork<T>(refinerArchitecture);

            // Build the convergence detector network
            var convergenceArchitecture = new NeuralNetworkArchitecture<T>(NetworkComplexity.Simple);
            _convergenceDetector = new NeuralNetwork<T>(convergenceArchitecture);
        }

        /// <summary>
        /// Trains the Iterative Refinement model.
        /// </summary>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            // Generate initial reasoning
            var initial = _initialReasoner.Predict(input);
            _initialReasoner.Train(input, expectedOutput);

            // Create refinement sequence leading to expected output
            var refinementSequence = GenerateRefinementSequence(initial, expectedOutput);

            // Train critic to identify areas for improvement
            for (int i = 0; i < refinementSequence.Count - 1; i++)
            {
                var current = refinementSequence[i];
                var improved = refinementSequence[i + 1];
                var critique = GenerateCritique(current, improved);
                
                _critic.Train(current, critique);
            }

            // Train refiner on the improvement steps
            for (int i = 0; i < refinementSequence.Count - 1; i++)
            {
                var current = refinementSequence[i];
                var critique = _critic.Predict(current);
                var combined = CombineTensors(current, critique);
                var improved = refinementSequence[i + 1];
                
                _refiner.Train(combined, improved);
            }

            // Train convergence detector
            for (int i = 1; i < refinementSequence.Count; i++)
            {
                var previous = refinementSequence[i - 1];
                var current = refinementSequence[i];
                var combined = CombineTensors(previous, current);
                
                var hasConverged = i == refinementSequence.Count - 1 ? NumOps.One : NumOps.Zero;
                var target = new Tensor<T>(new[] { 1 });
                target[0] = hasConverged;
                
                _convergenceDetector.Train(combined, target);
            }
        }

        /// <summary>
        /// Makes a prediction using iterative refinement.
        /// </summary>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            var startTime = DateTime.UtcNow;

            // Generate initial reasoning
            var current = _initialReasoner.Predict(input);
            
            // Perform iterative refinement
            var refined = RefineReasoning(input, current, _irOptions.DefaultRefinementIterations);

            // Store diagnostics
            LastDiagnostics["RefinementTime"] = (DateTime.UtcNow - startTime).TotalMilliseconds;
            LastDiagnostics["InitialQuality"] = EvaluateQuality(current)!;
            LastDiagnostics["FinalQuality"] = EvaluateQuality(refined)!;

            return refined;
        }

        /// <summary>
        /// Performs multi-step reasoning with refinement at each step.
        /// </summary>
        public override List<Tensor<T>> ReasonStepByStep(Tensor<T> input, int maxSteps = 10)
        {
            var steps = new List<Tensor<T>>();
            var current = input;

            for (int i = 0; i < maxSteps; i++)
            {
                // Generate next reasoning step
                var nextStep = _initialReasoner.Predict(current);
                
                // Apply quick refinement
                var refined = PerformQuickRefinement(nextStep);
                steps.Add(refined);

                // Check if we've reached a terminal state
                if (IsTerminalReasoning(refined))
                    break;

                current = refined;
            }

            LastReasoningSteps = steps;
            return steps;
        }

        /// <summary>
        /// Refines reasoning through iterative improvement.
        /// </summary>
        public override Tensor<T> RefineReasoning(Tensor<T> input, Tensor<T> initialReasoning, int iterations = 3)
        {
            var history = new RefinementHistory();
            var current = new Tensor<T>(initialReasoning.Shape);
            for (int i = 0; i < initialReasoning.Length; i++)
                current[i] = initialReasoning[i];
            var previous = new Tensor<T>(current.Shape);
            for (int i = 0; i < current.Length; i++)
                previous[i] = current[i];

            var currentCopy = new Tensor<T>(current.Shape);
            for (int i = 0; i < current.Length; i++)
                currentCopy[i] = current[i];
            history.Iterations.Add(currentCopy);
            history.QualityScores.Add(EvaluateQuality(current));

            for (int i = 0; i < iterations; i++)
            {
                // Generate critique
                var critique = _critic.Predict(current);
                var critiqueCopy = new Tensor<T>(critique.Shape);
                for (int j = 0; j < critique.Length; j++)
                    critiqueCopy[j] = critique[j];
                history.Critiques.Add(critiqueCopy);

                // Apply refinement
                var combined = CombineTensors(current, critique);
                var refined = _refiner.Predict(combined);

                // Apply residual connection if enabled
                if (_irOptions.UseResidualConnections)
                {
                    refined = ApplyResidualConnection(current, refined);
                }

                // Check for convergence
                var convergenceInput = CombineTensors(previous, refined);
                var convergence = _convergenceDetector.Predict(convergenceInput);
                
                if (NumOps.GreaterThan(convergence[0], NumOps.FromDouble(_irOptions.ConvergenceThreshold)))
                {
                    history.Converged = true;
                    history.ConvergenceIteration = i;
                    current = refined;
                    break;
                }

                // Update for next iteration
                previous = current;
                current = refined;
                
                var currentCopy2 = new Tensor<T>(current.Shape);
                for (int k = 0; k < current.Length; k++)
                    currentCopy2[k] = current[k];
                history.Iterations.Add(currentCopy2);
                history.QualityScores.Add(EvaluateQuality(current));
            }

            _refinementHistories.Add(history);
            return current;
        }

        /// <summary>
        /// Generates an explanation showing the refinement process.
        /// </summary>
        public override Tensor<T> GenerateExplanation(Tensor<T> input, Tensor<T> prediction)
        {
            // Find the refinement history that led to this prediction
            var relevantHistory = FindRelevantHistory(prediction);
            
            if (relevantHistory == null)
            {
                // Generate a new refinement sequence for explanation
                var initial = _initialReasoner.Predict(input);
                var refined = RefineReasoning(input, initial, _irOptions.ExplanationIterations);
                relevantHistory = _refinementHistories.LastOrDefault();
            }

            // Convert refinement history to explanation
            return ConvertHistoryToExplanation(relevantHistory);
        }

        /// <summary>
        /// Validates the logical consistency of a reasoning chain.
        /// </summary>
        public override bool ValidateReasoningChain(List<Tensor<T>> reasoningSteps)
        {
            if (reasoningSteps.Count < 2)
                return true;

            // Check if each step improves upon the previous
            var previousQuality = EvaluateQuality(reasoningSteps[0]);
            
            for (int i = 1; i < reasoningSteps.Count; i++)
            {
                var currentQuality = EvaluateQuality(reasoningSteps[i]);
                
                // Allow small decreases due to exploration
                if (NumOps.LessThan(currentQuality, NumOps.Multiply(previousQuality, NumOps.FromDouble(0.95))))
                {
                    return false;
                }

                previousQuality = currentQuality;
            }

            return true;
        }

        /// <summary>
        /// Calculates confidence scores based on refinement quality.
        /// </summary>
        protected override Vector<T> CalculateConfidenceScores(List<Tensor<T>> reasoningSteps)
        {
            var scores = new T[reasoningSteps.Count];

            for (int i = 0; i < reasoningSteps.Count; i++)
            {
                var quality = EvaluateQuality(reasoningSteps[i]);
                
                // Boost confidence for later iterations
                var iterationBoost = NumOps.FromDouble(1.0 + i * 0.1);
                scores[i] = NumOps.Multiply(quality, iterationBoost);
            }

            return new Vector<T>(scores);
        }

        /// <summary>
        /// Performs a single refinement step.
        /// </summary>
        protected override Tensor<T> PerformRefinementStep(Tensor<T> input, Tensor<T> currentReasoning, int iteration)
        {
            // Generate critique
            var critique = _critic.Predict(currentReasoning);
            
            // Apply adaptive refinement based on iteration
            var adaptiveCritique = ApplyAdaptiveCritique(critique, iteration);
            
            // Refine reasoning
            var combined = CombineTensors(currentReasoning, adaptiveCritique);
            var refined = _refiner.Predict(combined);

            // Apply momentum if enabled
            if (_irOptions.UseMomentum && iteration > 0)
            {
                refined = ApplyMomentum(currentReasoning, refined, iteration);
            }

            return refined;
        }

        /// <summary>
        /// Gets the model type.
        /// </summary>
        protected override ModelType GetModelType()
        {
            return ModelType.IterativeRefinementModel;
        }

        /// <summary>
        /// Gets a description of the model.
        /// </summary>
        protected override string GetModelDescription()
        {
            return $"Iterative Refinement reasoning model with {_irOptions.MaxRefinementIterations} max iterations and {_irOptions.AttentionHeads} attention heads";
        }

        /// <summary>
        /// Estimates the complexity of the model.
        /// </summary>
        protected override double EstimateComplexity()
        {
            var reasonerParams = _initialReasoner.GetParameters();
            var criticParams = _critic.GetParameters();
            var refinerParams = _refiner.GetParameters();
            var convergenceParams = _convergenceDetector.GetParameters();
            
            return (reasonerParams.Length + criticParams.Length + refinerParams.Length + convergenceParams.Length) * _irOptions.MaxRefinementIterations;
        }

        #region Private Helper Methods

        private List<Tensor<T>> GenerateRefinementSequence(Tensor<T> start, Tensor<T> target)
        {
            var sequence = new List<Tensor<T>> { start };
            var steps = _irOptions.TrainingRefinementSteps;
            
            for (int i = 1; i <= steps; i++)
            {
                var progress = (double)i / steps;
                var interpolated = InterpolateTensors(start, target, progress);
                
                // Add controlled noise for diversity
                if (i < steps)
                {
                    var noise = GenerateRefinementNoise(NumOps.FromDouble(0.1 * (1 - progress)));
                    interpolated = interpolated.Add(noise);
                }

                sequence.Add(interpolated);
            }

            return sequence;
        }

        private Tensor<T> InterpolateTensors(Tensor<T> start, Tensor<T> end, double progress)
        {
            var result = new Tensor<T>(start.Shape);
            var progressT = NumOps.FromDouble(progress);
            var oneMinusProgress = NumOps.Subtract(NumOps.One, progressT);

            for (int i = 0; i < Math.Min(start.Length, end.Length); i++)
            {
                var interpolated = NumOps.Add(
                    NumOps.Multiply(start[i], oneMinusProgress),
                    NumOps.Multiply(end[i], progressT)
                );
                result[i] = interpolated;
            }

            return result;
        }

        private Tensor<T> GenerateCritique(Tensor<T> current, Tensor<T> improved)
        {
            var critique = new Tensor<T>(_irOptions.CritiqueShape);
            
            // Identify differences
            for (int i = 0; i < Math.Min(current.Length, improved.Length) && i < critique.Length; i++)
            {
                var diff = NumOps.Subtract(improved[i], current[i]);
                var absDiff = NumOps.Abs(diff);
                critique[i] = absDiff;
            }

            return critique;
        }

        private Tensor<T> CombineTensors(Tensor<T> a, Tensor<T> b)
        {
            var combined = new Tensor<T>(new[] { a.Length + b.Length });
            
            for (int i = 0; i < a.Length; i++)
            {
                combined[i] = a[i];
            }
            
            for (int i = 0; i < b.Length; i++)
            {
                combined[a.Length + i] = b[i];
            }

            return combined;
        }

        private T EvaluateQuality(Tensor<T> reasoning)
        {
            // Use critic to evaluate quality
            var critique = _critic.Predict(reasoning);
            
            // Lower critique values indicate higher quality
            var totalCritique = NumOps.Zero;
            for (int i = 0; i < critique.Length; i++)
            {
                totalCritique = NumOps.Add(totalCritique, critique[i]);
            }
            
            var avgCritique = NumOps.Divide(totalCritique, NumOps.FromDouble(critique.Length));
            return NumOps.Subtract(NumOps.One, avgCritique);
        }

        private Tensor<T> PerformQuickRefinement(Tensor<T> reasoning)
        {
            // Single refinement step for speed
            var critique = _critic.Predict(reasoning);
            var combined = CombineTensors(reasoning, critique);
            return _refiner.Predict(combined);
        }

        private bool IsTerminalReasoning(Tensor<T> reasoning)
        {
            var quality = EvaluateQuality(reasoning);
            return NumOps.GreaterThan(quality, NumOps.FromDouble(_irOptions.TerminalQualityThreshold));
        }

        private Tensor<T> ApplyResidualConnection(Tensor<T> original, Tensor<T> refined)
        {
            var alpha = NumOps.FromDouble(_irOptions.ResidualWeight);
            var oneMinusAlpha = NumOps.Subtract(NumOps.One, alpha);
            
            var result = new Tensor<T>(original.Shape);
            
            for (int i = 0; i < original.Length; i++)
            {
                var value = NumOps.Add(
                    NumOps.Multiply(original[i], oneMinusAlpha),
                    NumOps.Multiply(refined[i], alpha)
                );
                result[i] = value;
            }

            return result;
        }

        private RefinementHistory? FindRelevantHistory(Tensor<T> prediction)
        {
            // Find history with final iteration closest to prediction
            RefinementHistory? bestMatch = null;
            var bestDistance = NumOps.FromDouble(double.MaxValue);

            foreach (var history in _refinementHistories)
            {
                if (history.Iterations.Count == 0)
                    continue;

                var lastIteration = history.Iterations.Last();
                var distance = TensorDistance(lastIteration, prediction);
                
                if (NumOps.LessThan(distance, bestDistance))
                {
                    bestDistance = distance;
                    bestMatch = history;
                }
            }

            return bestMatch;
        }

        private T TensorDistance(Tensor<T> a, Tensor<T> b)
        {
            var sum = NumOps.Zero;
            
            for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
            {
                var diff = NumOps.Subtract(a[i], b[i]);
                sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
            }

            return NumOps.Sqrt(sum);
        }

        private Tensor<T> ConvertHistoryToExplanation(RefinementHistory? history)
        {
            var explanation = new Tensor<T>(_irOptions.ReasoningShape);
            
            if (history == null || history.Iterations.Count == 0)
                return explanation;

            // Weighted average of iterations, with more weight on later iterations
            var totalWeight = NumOps.Zero;
            
            for (int i = 0; i < history.Iterations.Count; i++)
            {
                var weight = NumOps.FromDouble(i + 1);
                totalWeight = NumOps.Add(totalWeight, weight);
                
                var iteration = history.Iterations[i];
                for (int j = 0; j < Math.Min(iteration.Length, explanation.Length); j++)
                {
                    var weighted = NumOps.Multiply(iteration[j], weight);
                    var current = explanation[j];
                    explanation[j] = NumOps.Add(current, weighted);
                }
            }

            // Normalize
            for (int i = 0; i < explanation.Length; i++)
            {
                var normalized = NumOps.Divide(explanation[i], totalWeight);
                explanation[i] = normalized;
            }

            return explanation;
        }

        private Tensor<T> ApplyAdaptiveCritique(Tensor<T> critique, int iteration)
        {
            // Reduce critique strength over iterations
            var scale = NumOps.FromDouble(1.0 / (1.0 + iteration * _irOptions.CritiqueDecay));
            
            var adapted = new Tensor<T>(critique.Shape);
            for (int i = 0; i < critique.Length; i++)
            {
                adapted[i] = NumOps.Multiply(critique[i], scale);
            }

            return adapted;
        }

        private Tensor<T> ApplyMomentum(Tensor<T> previous, Tensor<T> current, int iteration)
        {
            var momentum = NumOps.FromDouble(_irOptions.MomentumCoefficient);
            var oneMinusMomentum = NumOps.Subtract(NumOps.One, momentum);
            
            var result = new Tensor<T>(current.Shape);
            
            for (int i = 0; i < current.Length; i++)
            {
                var value = NumOps.Add(
                    NumOps.Multiply(previous[i], momentum),
                    NumOps.Multiply(current[i], oneMinusMomentum)
                );
                result[i] = value;
            }

            return result;
        }

        private Tensor<T> GenerateRefinementNoise(T scale)
        {
            var noise = new Tensor<T>(_irOptions.ReasoningShape);
            
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

        #endregion

        #region IParameterizable Implementation

        public override Vector<T> GetParameters()
        {
            // Get parameters from all networks and flatten them into a single vector
            var allParams = new List<T>();
            
            var reasonerParams = _initialReasoner.GetParameters();
            var criticParams = _critic.GetParameters();
            var refinerParams = _refiner.GetParameters();
            var convergenceParams = _convergenceDetector.GetParameters();
            
            for (int i = 0; i < reasonerParams.Length; i++)
                allParams.Add(reasonerParams[i]);
            
            for (int i = 0; i < criticParams.Length; i++)
                allParams.Add(criticParams[i]);
            
            for (int i = 0; i < refinerParams.Length; i++)
                allParams.Add(refinerParams[i]);
            
            for (int i = 0; i < convergenceParams.Length; i++)
                allParams.Add(convergenceParams[i]);
            
            return new Vector<T>(allParams.ToArray());
        }

        public override void SetParameters(Vector<T> parameters)
        {
            // Get current parameter counts for each network
            var reasonerParams = _initialReasoner.GetParameters();
            var criticParams = _critic.GetParameters();
            var refinerParams = _refiner.GetParameters();
            var convergeParams = _convergenceDetector.GetParameters();
            
            int offset = 0;
            
            // Set parameters for initial reasoner
            var reasonerSize = reasonerParams.Length;
            if (offset + reasonerSize > parameters.Length)
                throw new ArgumentException("Parameter vector size mismatch");
            
            var reasonerData = new T[reasonerSize];
            for (int j = 0; j < reasonerSize; j++)
            {
                reasonerData[j] = parameters[offset + j];
            }
            _initialReasoner.SetParameters(new Vector<T>(reasonerData));
            offset += reasonerSize;
            
            // Set parameters for critic
            var criticSize = criticParams.Length;
            if (offset + criticSize > parameters.Length)
                throw new ArgumentException("Parameter vector size mismatch");
            
            var criticData = new T[criticSize];
            for (int j = 0; j < criticSize; j++)
            {
                criticData[j] = parameters[offset + j];
            }
            _critic.SetParameters(new Vector<T>(criticData));
            offset += criticSize;
            
            // Set parameters for refiner
            var refinerSize = refinerParams.Length;
            if (offset + refinerSize > parameters.Length)
                throw new ArgumentException("Parameter vector size mismatch");
            
            var refinerData = new T[refinerSize];
            for (int j = 0; j < refinerSize; j++)
            {
                refinerData[j] = parameters[offset + j];
            }
            _refiner.SetParameters(new Vector<T>(refinerData));
            offset += refinerSize;
            
            // Set parameters for convergence detector
            var convergeSize = convergeParams.Length;
            if (offset + convergeSize > parameters.Length)
                throw new ArgumentException("Parameter vector size mismatch");
            
            var convergeData = new T[convergeSize];
            for (int j = 0; j < convergeSize; j++)
            {
                convergeData[j] = parameters[offset + j];
            }
            _convergenceDetector.SetParameters(new Vector<T>(convergeData));
        }

        public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
        {
            var newModel = new IterativeRefinementModel<T>(_irOptions);
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
                writer.Write("IterativeRefinementModel");
                
                // Write options
                writer.Write(_irOptions.MaxRefinementIterations);
                writer.Write(_irOptions.DefaultRefinementIterations);
                writer.Write(_irOptions.ConvergenceThreshold);
                writer.Write(_irOptions.CritiqueDecay);
                writer.Write(_irOptions.UseResidualConnections);
                writer.Write(_irOptions.ResidualWeight);
                writer.Write(_irOptions.UseMomentum);
                writer.Write(_irOptions.MomentumCoefficient);
                writer.Write(_irOptions.TerminalQualityThreshold);
                writer.Write(_irOptions.StoreRefinementHistory);
                writer.Write(_irOptions.HiddenSize);
                writer.Write(_irOptions.AttentionHeads);
                writer.Write(_irOptions.BeamWidth);
                
                // Save neural networks
                var reasonerBytes = _initialReasoner.Serialize();
                var criticBytes = _critic.Serialize();
                var refinerBytes = _refiner.Serialize();
                var convergenceBytes = _convergenceDetector.Serialize();
                
                writer.Write(reasonerBytes.Length);
                writer.Write(reasonerBytes);
                
                writer.Write(criticBytes.Length);
                writer.Write(criticBytes);
                
                writer.Write(refinerBytes.Length);
                writer.Write(refinerBytes);
                
                writer.Write(convergenceBytes.Length);
                writer.Write(convergenceBytes);
                
                // Save refinement histories if enabled
                writer.Write(_irOptions.StoreRefinementHistory && _refinementHistories.Count > 0);
                if (_irOptions.StoreRefinementHistory && _refinementHistories.Count > 0)
                {
                    writer.Write(_refinementHistories.Count);
                    foreach (var history in _refinementHistories)
                    {
                        writer.Write(history.Iterations.Count);
                        foreach (var iteration in history.Iterations)
                        {
                            writer.Write(iteration.Shape.Length);
                            foreach (var dim in iteration.Shape)
                                writer.Write(dim);
                            
                            for (int i = 0; i < iteration.Length; i++)
                                writer.Write(Convert.ToDouble(iteration[i]));
                        }
                        
                        writer.Write(history.QualityScores.Count);
                        foreach (var score in history.QualityScores)
                            writer.Write(Convert.ToDouble(score));
                        
                        writer.Write(history.Converged);
                        writer.Write(history.ConvergenceIteration);
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
                if (modelType != "IterativeRefinementModel")
                    throw new InvalidOperationException($"Invalid model type: {modelType}");
                
                // Read options
                _irOptions.MaxRefinementIterations = reader.ReadInt32();
                _irOptions.DefaultRefinementIterations = reader.ReadInt32();
                _irOptions.ConvergenceThreshold = reader.ReadDouble();
                _irOptions.CritiqueDecay = reader.ReadDouble();
                _irOptions.UseResidualConnections = reader.ReadBoolean();
                _irOptions.ResidualWeight = reader.ReadDouble();
                _irOptions.UseMomentum = reader.ReadBoolean();
                _irOptions.MomentumCoefficient = reader.ReadDouble();
                _irOptions.TerminalQualityThreshold = reader.ReadDouble();
                _irOptions.StoreRefinementHistory = reader.ReadBoolean();
                _irOptions.HiddenSize = reader.ReadInt32();
                _irOptions.AttentionHeads = reader.ReadInt32();
                _irOptions.BeamWidth = reader.ReadInt32();
                
                // Load neural networks
                var reasonerSize = reader.ReadInt32();
                var reasonerBytes = reader.ReadBytes(reasonerSize);
                _initialReasoner.Deserialize(reasonerBytes);
                
                var criticSize = reader.ReadInt32();
                var criticBytes = reader.ReadBytes(criticSize);
                _critic.Deserialize(criticBytes);
                
                var refinerSize = reader.ReadInt32();
                var refinerBytes = reader.ReadBytes(refinerSize);
                _refiner.Deserialize(refinerBytes);
                
                var convergenceSize = reader.ReadInt32();
                var convergenceBytes = reader.ReadBytes(convergenceSize);
                _convergenceDetector.Deserialize(convergenceBytes);
                
                // Load refinement histories if present
                _refinementHistories.Clear();
                var hasHistories = reader.ReadBoolean();
                if (hasHistories)
                {
                    var historyCount = reader.ReadInt32();
                    for (int h = 0; h < historyCount; h++)
                    {
                        var history = new RefinementHistory();
                        
                        var iterationCount = reader.ReadInt32();
                        for (int i = 0; i < iterationCount; i++)
                        {
                            var shapeLength = reader.ReadInt32();
                            var shape = new int[shapeLength];
                            for (int j = 0; j < shapeLength; j++)
                                shape[j] = reader.ReadInt32();
                            
                            var tensor = new Tensor<T>(shape);
                            for (int j = 0; j < tensor.Length; j++)
                            {
                                var value = reader.ReadDouble();
                                tensor[j] = NumOps.FromDouble(value);
                            }
                            
                            history.Iterations.Add(tensor);
                        }
                        
                        var scoreCount = reader.ReadInt32();
                        for (int i = 0; i < scoreCount; i++)
                        {
                            var score = reader.ReadDouble();
                            history.QualityScores.Add(NumOps.FromDouble(score));
                        }
                        
                        history.Converged = reader.ReadBoolean();
                        history.ConvergenceIteration = reader.ReadInt32();
                        
                        _refinementHistories.Add(history);
                    }
                }
            }
        }

        #endregion

        #region ICloneable Implementation

        public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
        {
            var copy = new IterativeRefinementModel<T>(_irOptions);
            copy.SetParameters(GetParameters());
            
            if (_irOptions.StoreRefinementHistory)
            {
                // Deep copy refinement histories
                foreach (var history in _refinementHistories)
                {
                    var historyCopy = new RefinementHistory
                    {
                        Iterations = history.Iterations.Select(t => 
                        {
                            var deepCopy = new Tensor<T>(t.Shape);
                            for (int i = 0; i < t.Length; i++)
                                deepCopy[i] = t[i];
                            return deepCopy;
                        }).ToList(),
                        QualityScores = new List<T>(history.QualityScores),
                        Critiques = history.Critiques.Select(t => 
                        {
                            var deepCopy = new Tensor<T>(t.Shape);
                            for (int i = 0; i < t.Length; i++)
                                deepCopy[i] = t[i];
                            return deepCopy;
                        }).ToList(),
                        Converged = history.Converged,
                        ConvergenceIteration = history.ConvergenceIteration
                    };
                    copy._refinementHistories.Add(historyCopy);
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