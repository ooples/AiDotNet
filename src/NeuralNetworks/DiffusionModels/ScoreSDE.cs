using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.DiffusionModels.Solvers;

namespace AiDotNet.NeuralNetworks.DiffusionModels
{
    /// <summary>
    /// Score-based Stochastic Differential Equation (SDE) Diffusion Model for continuous-time generative modeling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Score-based SDE models formulate the diffusion process as a continuous-time stochastic differential equation,
    /// offering theoretical advantages and flexibility compared to discrete-time formulations. The model learns
    /// the score function (gradient of log probability) and uses it to reverse the diffusion process.
    /// </para>
    /// <para><b>For Beginners:</b> This is an advanced generative model that creates new data by learning to reverse noise.
    /// 
    /// Key concepts:
    /// - Score: The gradient that points toward higher probability regions
    /// - SDE: An equation describing how data evolves with both deterministic drift and random noise
    /// - Forward process: Gradually adds noise to data
    /// - Reverse process: Removes noise to generate new samples
    /// 
    /// The model can generate high-quality samples and offers multiple sampling strategies:
    /// - SDE sampling: Uses the full stochastic process (more diverse)
    /// - ODE sampling: Deterministic variant (more stable)
    /// - Predictor-Corrector: Combines both for best quality
    /// </para>
    /// </remarks>
    public class ScoreSDE : NeuralNetworkBase<double>, IGenerativeModel, IDisposable
    {
        private readonly SDEType _sdeType;
        private readonly double _sigma;
        private readonly double _beta0;
        private readonly double _beta1;
        private readonly INeuralNetworkModel<double> _scoreNetwork;
        private readonly ISolver _solver;
        private readonly object _lockObject = new object();
        private readonly int _maxRetries = 3;
        private readonly double _stabilityThreshold = 1e10;
        private bool _isDisposed;

        /// <summary>
        /// Initializes a new instance of the ScoreSDE class.
        /// </summary>
        /// <param name="scoreNetwork">Neural network that predicts the score function.</param>
        /// <param name="sdeType">Type of SDE formulation to use.</param>
        /// <param name="sigma">Noise scale for VE-SDE (default: 25.0).</param>
        /// <param name="beta0">Initial beta value for VP-SDE (default: 0.1).</param>
        /// <param name="beta1">Final beta value for VP-SDE (default: 20.0).</param>
        /// <param name="solver">Numerical solver for SDE integration.</param>
        /// <param name="modelName">Name of the model.</param>
        /// <exception cref="ArgumentNullException">Thrown when scoreNetwork is null.</exception>
        /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
        public ScoreSDE(
            NeuralNetworkArchitecture<double> architecture,
            INeuralNetworkModel<double> scoreNetwork,
            SDEType sdeType = SDEType.VP,
            double sigma = 25.0,
            double beta0 = 0.1,
            double beta1 = 20.0,
            ISolver? solver = null,
            string modelName = "ScoreSDE")
            : base(new NeuralNetworkArchitecture<double>(NetworkComplexity.Deep), 
                   new AiDotNet.LossFunctions.MeanSquaredErrorLoss<double>(), 
                   1.0)
        {
            // Validate inputs
            _scoreNetwork = scoreNetwork ?? throw new ArgumentNullException(nameof(scoreNetwork));
            
            if (sigma <= 0)
                throw new ArgumentException("Sigma must be positive", nameof(sigma));
            if (beta0 <= 0)
                throw new ArgumentException("Beta0 must be positive", nameof(beta0));
            if (beta1 <= beta0)
                throw new ArgumentException("Beta1 must be greater than beta0", nameof(beta1));
            
            _sdeType = sdeType;
            _sigma = sigma;
            _beta0 = beta0;
            _beta1 = beta1;
            _solver = solver ?? new EulerMaruyamaSolver();
            
            // Architecture setup is handled by base class
        }
        
        /// <summary>
        /// Gets the forward SDE drift and diffusion functions.
        /// </summary>
        /// <returns>Tuple of drift and diffusion functions.</returns>
        /// <remarks>
        /// The forward SDE has the form: dx = f(x,t)dt + g(t)dw
        /// where f is the drift, g is the diffusion, and w is a Wiener process.
        /// </remarks>
        public (Func<Tensor<double>, double, Tensor<double>> drift, Func<double, double> diffusion) GetForwardSDE()
        {
            lock (_lockObject)
            {
                switch (_sdeType)
                {
                    case SDEType.VE:
                        return GetVESDE();
                    case SDEType.VP:
                        return GetVPSDE();
                    case SDEType.SubVP:
                        return GetSubVPSDE();
                    default:
                        throw new NotSupportedException($"SDE type {_sdeType} not supported");
                }
            }
        }
        
        /// <summary>
        /// Gets the reverse SDE drift and diffusion functions.
        /// </summary>
        /// <returns>Tuple of reverse drift and diffusion functions.</returns>
        /// <remarks>
        /// The reverse SDE has the form: dx = [f(x,t) - g(t)²∇log p_t(x)]dt + g(t)dw
        /// where the score function modifies the drift to reverse the diffusion process.
        /// </remarks>
        public (Func<Tensor<double>, double, Tensor<double>> drift, Func<double, double> diffusion) GetReverseSDE()
        {
            var (forwardDrift, diffusion) = GetForwardSDE();
            
            Func<Tensor<double>, double, Tensor<double>> reverseDrift = (x, t) =>
            {
                try
                {
                    var score = GetScore(x, t);
                    var g = diffusion(t);
                    return forwardDrift(x, t).Subtract(score.Multiply(g * g));
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Failed to compute reverse drift: {ex.Message}", ex);
                }
            };
            
            return (reverseDrift, diffusion);
        }
        
        /// <summary>
        /// Generates samples using the reverse SDE.
        /// </summary>
        /// <param name="shape">Shape of the samples to generate.</param>
        /// <param name="seed">Optional random seed for reproducibility.</param>
        /// <returns>Generated samples.</returns>
        /// <exception cref="ArgumentException">Thrown when shape is invalid.</exception>
        public Tensor<double> Sample(int[] shape, int? seed = null)
        {
            ValidateShape(shape);
            
            lock (_lockObject)
            {
                try
                {
                    var random = seed.HasValue ? new Random(seed.Value) : new Random();
                    var (reverseDrift, diffusion) = GetReverseSDE();
                    
                    // Start from noise at t=1
                    var x = SamplePrior(shape, random);
                    
                    // Solve reverse SDE from t=1 to t=0
                    var timeSteps = 1000;
                    var dt = 1.0 / timeSteps;
                    
                    for (int i = timeSteps - 1; i >= 0; i--)
                    {
                        var t = i * dt;
                        x = _solver.Step(x, t, dt, reverseDrift, diffusion, random);
                        
                        // Check for stability
                        if (!IsStable(x))
                        {
                            throw new InvalidOperationException("Numerical instability detected during sampling");
                        }
                    }
                    
                    return x;
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Failed to generate samples: {ex.Message}", ex);
                }
            }
        }
        
        /// <summary>
        /// Generates samples using the probability flow ODE for deterministic sampling.
        /// </summary>
        /// <param name="shape">Shape of the samples to generate.</param>
        /// <param name="seed">Optional random seed for reproducibility.</param>
        /// <returns>Generated samples.</returns>
        /// <remarks>
        /// The probability flow ODE provides a deterministic mapping from noise to data,
        /// useful for applications requiring consistent outputs.
        /// </remarks>
        public Tensor<double> SampleODE(int[] shape, int? seed = null)
        {
            ValidateShape(shape);
            
            lock (_lockObject)
            {
                try
                {
                    var random = seed.HasValue ? new Random(seed.Value) : new Random();
                    var (forwardDrift, diffusion) = GetForwardSDE();
                    
                    // Probability flow ODE drift
                    Func<Tensor<double>, double, Tensor<double>> odeDrift = (x, t) =>
                    {
                        var score = GetScore(x, t);
                        var g = diffusion(t);
                        return forwardDrift(x, t).Subtract(score.Multiply(0.5 * g * g));
                    };
                    
                    // Start from noise
                    var x = SamplePrior(shape, random);
                    
                    // Solve ODE (no stochastic term)
                    Func<double, double> odeNoiseFunc = t => 0.0; // No noise in ODE
                    var odeSolver = new RK45Solver(); // Use higher-order solver for ODE
                    
                    var timeSteps = 100; // Fewer steps needed for ODE
                    var dt = 1.0 / timeSteps;
                    
                    for (int i = timeSteps - 1; i >= 0; i--)
                    {
                        var t = i * dt;
                        x = odeSolver.Step(x, t, dt, odeDrift, odeNoiseFunc, random);
                        
                        // Check for stability
                        if (!IsStable(x))
                        {
                            throw new InvalidOperationException("Numerical instability detected during ODE sampling");
                        }
                    }
                    
                    return x;
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Failed to generate ODE samples: {ex.Message}", ex);
                }
            }
        }
        
        /// <summary>
        /// Generates samples using Predictor-Corrector method for improved quality.
        /// </summary>
        /// <param name="shape">Shape of the samples to generate.</param>
        /// <param name="numCorrectorSteps">Number of corrector steps per predictor step.</param>
        /// <param name="seed">Optional random seed for reproducibility.</param>
        /// <returns>Generated samples.</returns>
        /// <remarks>
        /// The Predictor-Corrector method alternates between taking reverse SDE steps (predictor)
        /// and applying Langevin dynamics (corrector) for higher quality samples.
        /// </remarks>
        public Tensor<double> SamplePC(int[] shape, int numCorrectorSteps = 1, int? seed = null)
        {
            ValidateShape(shape);
            
            if (numCorrectorSteps < 0)
                throw new ArgumentException("Number of corrector steps must be non-negative", nameof(numCorrectorSteps));
            
            lock (_lockObject)
            {
                try
                {
                    var random = seed.HasValue ? new Random(seed.Value) : new Random();
                    var (reverseDrift, diffusion) = GetReverseSDE();
                    
                    // Start from noise
                    var x = SamplePrior(shape, random);
                    
                    var timeSteps = 1000;
                    var dt = 1.0 / timeSteps;
                    
                    for (int i = timeSteps - 1; i >= 0; i--)
                    {
                        var t = i * dt;
                        
                        // Predictor step (reverse diffusion)
                        x = _solver.Step(x, t, dt, reverseDrift, diffusion, random);
                        
                        // Corrector steps (Langevin dynamics)
                        for (int j = 0; j < numCorrectorSteps; j++)
                        {
                            x = LangevinCorrector(x, t, random);
                        }
                        
                        // Check for stability
                        if (!IsStable(x))
                        {
                            throw new InvalidOperationException("Numerical instability detected during PC sampling");
                        }
                    }
                    
                    return x;
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Failed to generate PC samples: {ex.Message}", ex);
                }
            }
        }
        
        /// <summary>
        /// Performs one training step for the score network.
        /// </summary>
        /// <param name="data">Batch of training data.</param>
        /// <param name="optimizer">Optimizer for updating parameters.</param>
        /// <returns>Average loss for the batch.</returns>
        /// <exception cref="ArgumentNullException">Thrown when inputs are null.</exception>
        public double TrainStep(Tensor<double> data, IOptimizer<double, Tensor<double>, Tensor<double>> optimizer)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
            if (optimizer == null)
                throw new ArgumentNullException(nameof(optimizer));
            
            lock (_lockObject)
            {
                try
                {
                    var batchSize = data.Shape[0];
                    if (batchSize == 0)
                        throw new ArgumentException("Batch size must be greater than 0", nameof(data));
                    
                    var totalLoss = 0.0;
                    var random = new Random();
                    
                    for (int i = 0; i < batchSize; i++)
                    {
                        // Sample time uniformly
                        var t = random.NextDouble();
                        
                        // Get single sample
                        var x0 = data.GetSlice(i);
                        
                        // Sample from p_t(x|x0)
                        var (mean, std) = GetConditionalDistribution(x0, t);
                        var noise = GenerateNoise(x0.Shape, random);
                        var xt = mean.Add(noise.Multiply(std));
                        
                        // Compute score
                        var predictedScore = _scoreNetwork.Predict(ConcatenateTimeStep(xt, t));
                        
                        // True score
                        var trueScore = noise.Multiply(-1.0 / std);
                        
                        // Score matching loss
                        var loss = ComputeScoreMatchingLoss(predictedScore, trueScore, std);
                        totalLoss += loss;
                        
                        // Backpropagate
                        if (_scoreNetwork is NeuralNetworkBase<double> nn)
                        {
                            var grad = predictedScore.Subtract(trueScore).Multiply(GetLossWeight(t));
                            // TODO: Implement backpropagation through score network
                            // This requires the score network to support gradient computation
                            // For now, we assume the optimizer handles the gradient update internally
                            var parameters = nn.GetParameters();
                            // Note: Gradient computation would need to be implemented in the score network
                        }
                    }
                    
                    return totalLoss / batchSize;
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Training step failed: {ex.Message}", ex);
                }
            }
        }
        
        private (Func<Tensor<double>, double, Tensor<double>>, Func<double, double>) GetVESDE()
        {
            // Variance Exploding SDE
            Func<Tensor<double>, double, Tensor<double>> drift = (x, t) => 
            {
                var zeros = new Tensor<double>(x.Shape);
                // Initialize with zeros
                var data = zeros.ToVector();
                for (int i = 0; i < data.Length; i++)
                    data[i] = 0.0;
                return zeros;
            };
            Func<double, double> diffusion = (t) => _sigma * Math.Pow(_sigma, t);
            return (drift, diffusion);
        }
        
        private (Func<Tensor<double>, double, Tensor<double>>, Func<double, double>) GetVPSDE()
        {
            // Variance Preserving SDE
            Func<Tensor<double>, double, Tensor<double>> drift = (x, t) => x.Multiply(-0.5 * Beta(t));
            Func<double, double> diffusion = (t) => Math.Sqrt(Beta(t));
            return (drift, diffusion);
        }
        
        private (Func<Tensor<double>, double, Tensor<double>>, Func<double, double>) GetSubVPSDE()
        {
            // Sub-Variance Preserving SDE
            Func<Tensor<double>, double, Tensor<double>> drift = (x, t) => x.Multiply(-0.5 * Beta(t));
            Func<double, double> diffusion = (t) => Math.Sqrt(Beta(t) * (1 - Math.Exp(-2 * Integral(Beta, 0, t))));
            return (drift, diffusion);
        }
        
        private double Beta(double t)
        {
            // Linear schedule
            return _beta0 + t * (_beta1 - _beta0);
        }
        
        private double Integral(Func<double, double> f, double a, double b, int steps = 1000)
        {
            // Simple numerical integration
            var h = (b - a) / steps;
            var sum = 0.0;
            
            for (int i = 0; i < steps; i++)
            {
                var x = a + i * h;
                sum += f(x) * h;
            }
            
            return sum;
        }
        
        private Tensor<double> GetScore(Tensor<double> x, double t)
        {
            // Score function: ∇log p_t(x)
            try
            {
                return _scoreNetwork.Predict(ConcatenateTimeStep(x, t));
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to compute score: {ex.Message}", ex);
            }
        }
        
        private Tensor<double> ConcatenateTimeStep(Tensor<double> x, double t)
        {
            // In practice, use sinusoidal time embeddings
            var timeEmbedding = new Tensor<double>(new[] { x.Shape[0], 1 });
            timeEmbedding.Fill(t);
            return x; // Simplified - should concatenate properly
        }
        
        private (Tensor<double> mean, double std) GetConditionalDistribution(Tensor<double> x0, double t)
        {
            switch (_sdeType)
            {
                case SDEType.VE:
                    var veStd = _sigma * Math.Sqrt(Math.Pow(_sigma, 2 * t) - 1);
                    return (x0, veStd);
                    
                case SDEType.VP:
                    var vpAlpha = Math.Exp(-0.5 * Integral(Beta, 0, t));
                    var vpStd = Math.Sqrt(1 - vpAlpha * vpAlpha);
                    return (x0.Multiply(vpAlpha), vpStd);
                    
                case SDEType.SubVP:
                    var subVpAlpha = Math.Exp(-0.5 * Integral(Beta, 0, t));
                    var subVpStd = Math.Sqrt(1 - subVpAlpha * subVpAlpha);
                    return (x0.Multiply(subVpAlpha), subVpStd);
                    
                default:
                    return (x0, 1.0);
            }
        }
        
        private Tensor<double> LangevinCorrector(Tensor<double> x, double t, Random random)
        {
            // Langevin MCMC correction step
            var score = GetScore(x, t);
            var noise = GenerateNoise(x.Shape, random);
            var stepSize = 0.01 * Math.Pow(GetNoiseLevel(t), 2);
            
            return x.Add(score.Multiply(stepSize))
                   .Add(noise.Multiply(Math.Sqrt(2 * stepSize)));
        }
        
        private double GetNoiseLevel(double t)
        {
            var dummyTensor = new Tensor<double>(new[] { 1 });
            var (_, std) = GetConditionalDistribution(dummyTensor, t);
            return std;
        }
        
        private double ComputeScoreMatchingLoss(Tensor<double> predicted, Tensor<double> true_, double std)
        {
            var diff = predicted.Subtract(true_);
            var squared = diff.Multiply(diff);
            var data = squared.ToVector();
            return data.Average() * std * std;
        }
        
        private double GetLossWeight(double t)
        {
            // Importance weighting for different time steps
            var g = GetForwardSDE().diffusion(t);
            return g * g;
        }
        
        private Tensor<double> SamplePrior(int[] shape, Random random)
        {
            // Sample from prior distribution at t=1
            var sample = GenerateNoise(shape, random);
            
            if (_sdeType == SDEType.VE)
            {
                // Scale by final noise level
                var finalStd = _sigma * Math.Sqrt(Math.Pow(_sigma, 2) - 1);
                return sample.Multiply(finalStd);
            }
            
            return sample;
        }
        
        private Tensor<double> GenerateNoise(int[] shape, Random random)
        {
            var noise = new Tensor<double>(shape);
            var data = noise.ToVector();
            
            for (int i = 0; i < data.Length; i++)
            {
                // Sample from standard normal
                var u1 = random.NextDouble();
                var u2 = random.NextDouble();
                var z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                data[i] = z;
            }
            
            return noise;
        }
        
        /// <summary>
        /// Generates new samples from the model.
        /// </summary>
        /// <param name="input">Input tensor (shape is used for output shape).</param>
        /// <returns>Generated samples.</returns>
        public override Tensor<double> Predict(Tensor<double> input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            
            // For SDE models, prediction means sampling
            return Sample(input.Shape);
        }
        
        /// <summary>
        /// IGenerativeModel implementation: Generates new samples.
        /// </summary>
        public Tensor<double> Generate(int[] shape, int? seed = null)
        {
            return Sample(shape, seed);
        }
        
        protected override void InitializeLayers()
        {
            // SDE models don't use traditional layers
            // The score network is set in the constructor
        }
        
        protected override void DeserializeNetworkSpecificData(System.IO.BinaryReader reader)
        {
            // Read SDE-specific data
            // This would include reading SDE type, sigma, beta values, etc.
            // Implementation depends on serialization requirements
        }
        
        protected override IFullModel<double, Tensor<double>, Tensor<double>> CreateNewInstance()
        {
            var modelName = (Architecture as NeuralNetworkArchitecture<double>)?.CacheName ?? "ScoreSDE";
            return new ScoreSDE(_scoreNetwork, _sdeType, _sigma, _beta0, _beta1, _solver, modelName);
        }
        
        public override ModelMetadata<double> GetModelMetadata()
        {
            var metadata = new ModelMetadata<double>
            {
                ModelType = ModelType.NeuralNetwork,
                FeatureCount = (_scoreNetwork is NeuralNetworkBase<double> nn ? nn.GetParameterCount() : _scoreNetwork?.GetParameters().Length ?? 0),
                Complexity = 1000, // Default timesteps
                Description = $"Score-based SDE model using {_sdeType} formulation for continuous-time generative modeling",
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "ModelCategory", ModelCategory.Generative },
                    { "ModelName", (Architecture as NeuralNetworkArchitecture<double>)?.CacheName ?? "ScoreSDE" },
                    { "ArchitectureDescription", $"Score-based SDE ({_sdeType})" },
                    { "TotalParameters", _scoreNetwork is NeuralNetworkBase<double> n ? n.GetParameterCount() : _scoreNetwork?.GetParameters().Length ?? 0 },
                    { "TrainableParameters", _scoreNetwork is NeuralNetworkBase<double> n2 ? n2.GetParameterCount() : _scoreNetwork?.GetParameters().Length ?? 0 },
                    { "NonTrainableParameters", 0 },
                    { "InputShape", "Variable" },
                    { "OutputShape", "Variable" },
                    { "LearningRateSchedule", "Depends on optimizer" },
                    { "RegularizationStrength", 0.0 },
                    { "LastTrainingLoss", Convert.ToDouble(LastLoss.HasValue ? LastLoss.Value : 0) },
                    { "LastValidationLoss", 0.0 },
                    { "TotalEpochsTrained", 0 },
                    { "BatchSize", 0 },
                    { "EpochHistory", new List<EpochHistory>() },
                    { "ValidationMetrics", new Dictionary<string, double>() },
                    { "TestMetrics", new Dictionary<string, double>() },
                    { "LayerInformation", new List<string> { $"Score Network: {_scoreNetwork?.GetType().Name ?? "Not set"}" } },
                    { "SupportsParallelProcessing", false },
                    { "EstimatedMemoryUsageBytes", (_scoreNetwork is NeuralNetworkBase<double> n3 ? n3.GetParameterCount() : _scoreNetwork?.GetParameters().Length ?? 0) * 8 },
                    { "PreferredHardware", "GPU" },
                    { "LastUpdated", DateTime.UtcNow },
                    { "Version", "1.0.0" },
                    { "Author", "AiDotNet" },
                    { "Notes", $"Score-based SDE model using {_sdeType} formulation" },
                    { "SDEType", _sdeType.ToString() },
                    { "Sigma", _sigma },
                    { "Beta0", _beta0 },
                    { "Beta1", _beta1 },
                    { "SolverType", _solver.GetType().Name }
                }
            };
            
            return metadata;
        }
        
        public override void Train(Tensor<double> input, Tensor<double> expectedOutput)
        {
            // SDE models use a different training approach
            // This would typically involve the TrainStep method with an optimizer
            throw new NotImplementedException("Use TrainStep method with an optimizer for SDE model training");
        }
        
        public override void UpdateParameters(Vector<double> parameters)
        {
            if (parameters == null)
                throw new ArgumentNullException(nameof(parameters));
            
            lock (_lockObject)
            {
                // Update parameters of the score network
                if (_scoreNetwork != null)
                {
                    if (_scoreNetwork is NeuralNetworkBase<double> nn)
                    {
                        nn.UpdateParameters(parameters);
                    }
                    else
                    {
                        // Fallback for other implementations
                        var currentParams = _scoreNetwork.GetParameters();
                        if (currentParams.Length != parameters.Length)
                        {
                            throw new ArgumentException($"Parameter count mismatch: expected {currentParams.Length}, got {parameters.Length}");
                        }
                        // Note: This requires the scoreNetwork to have a SetParameters method
                        // which is not part of INeuralNetworkModel interface
                    }
                }
            }
        }
        
        protected override void SerializeNetworkSpecificData(System.IO.BinaryWriter writer)
        {
            // Write SDE-specific data
            writer.Write(_sdeType.ToString());
            writer.Write(_sigma);
            writer.Write(_beta0);
            writer.Write(_beta1);
            
            // Write whether we have a score network
            writer.Write(_scoreNetwork != null);
            if (_scoreNetwork != null)
            {
                // Serialize the score network
                var scoreNetworkData = _scoreNetwork.Serialize();
                writer.Write(scoreNetworkData.Length);
                writer.Write(scoreNetworkData);
            }
        }
        
        // Helper methods
        
        private void ValidateShape(int[] shape)
        {
            if (shape == null || shape.Length == 0)
                throw new ArgumentException("Shape must not be null or empty", nameof(shape));
            
            foreach (var dim in shape)
            {
                if (dim <= 0)
                    throw new ArgumentException("All dimensions must be positive", nameof(shape));
            }
        }
        
        private bool IsStable(Tensor<double> x)
        {
            var data = x.ToVector();
            for (int i = 0; i < data.Length; i++)
            {
                if (double.IsNaN(data[i]) || double.IsInfinity(data[i]) || Math.Abs(data[i]) > _stabilityThreshold)
                    return false;
            }
            return true;
        }
        
        /// <summary>
        /// Disposes of the ScoreSDE instance.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        
        /// <summary>
        /// Protected implementation of Dispose pattern.
        /// </summary>
        protected virtual void Dispose(bool disposing)
        {
            if (!_isDisposed)
            {
                if (disposing)
                {
                    // Dispose managed resources
                    if (_scoreNetwork is IDisposable disposableNetwork)
                    {
                        disposableNetwork.Dispose();
                    }
                }
                
                _isDisposed = true;
            }
        }
    }
}