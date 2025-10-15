using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks.DiffusionModels
{
    /// <summary>
    /// Denoising Diffusion Probabilistic Model (DDPM) for high-quality data generation
    /// </summary>
    /// <remarks>
    /// <para>
    /// Diffusion models are a class of generative models that learn to generate data by gradually
    /// denoising random noise. The model learns two processes: a forward process that adds noise
    /// to data, and a reverse process that removes noise to generate new samples.
    /// </para>
    /// <para><b>For Beginners:</b> A diffusion model is like learning to clean a dirty painting.
    /// 
    /// The process works in two stages:
    /// 1. Forward process (training): Take a clean image and gradually add noise until it's pure static
    /// 2. Reverse process (generation): Learn to remove noise step by step to recover clean images
    /// 
    /// Think of it like this:
    /// - Imagine taking a photograph and gradually adding more and more TV static to it
    /// - The model learns how to reverse this process
    /// - Starting from pure static, it can gradually remove noise to create a new photograph
    /// 
    /// This approach has proven very effective for:
    /// - High-quality image generation
    /// - Audio synthesis
    /// - 3D shape generation
    /// - Text-to-image generation (when combined with text conditioning)
    /// </para>
    /// </remarks>
    [Serializable]
    public class DiffusionModel : NeuralNetworkBase<double>, IGenerativeModel, IDisposable
    {
        private readonly int _timesteps;
        private readonly double _betaStart;
        private readonly double _betaEnd;
        private readonly List<double> _betas;
        private readonly List<double> _alphas;
        private readonly List<double> _alphasCumprod;
        private readonly List<double> _alphasCumprodPrev;
        private readonly List<double> _sqrtAlphasCumprod;
        private readonly List<double> _sqrtOneMinusAlphasCumprod;
        private readonly List<double> _posteriorMeanCoef1;
        private readonly List<double> _posteriorMeanCoef2;
        private readonly List<double> _posteriorVariance;
        private readonly List<double> _posteriorLogVarianceClipped;
        
        private INeuralNetworkModel<double>? _noisePredictor;
        private readonly object _lockObject = new object();
        private bool _disposed;
        
        // Performance tracking
        private readonly List<double> _trainingLosses = new List<double>();
        private readonly List<double> _validationLosses = new List<double>();
        
        /// <summary>
        /// Gets the number of diffusion timesteps
        /// </summary>
        public int Timesteps => _timesteps;
        
        /// <summary>
        /// Gets the noise predictor model
        /// </summary>
        public INeuralNetworkModel<double>? NoisePredictor => _noisePredictor;
        
        /// <summary>
        /// Gets the training loss history
        /// </summary>
        public IReadOnlyList<double> TrainingLosses => _trainingLosses.AsReadOnly();
        
        /// <summary>
        /// Gets the validation loss history
        /// </summary>
        public IReadOnlyList<double> ValidationLosses => _validationLosses.AsReadOnly();
        
        /// <summary>
        /// Initializes a new instance of the DiffusionModel class
        /// </summary>
        /// <param name="timesteps">Number of diffusion timesteps</param>
        /// <param name="betaStart">Starting value for the noise schedule</param>
        /// <param name="betaEnd">Ending value for the noise schedule</param>
        /// <param name="modelName">Name of the model</param>
        /// <exception cref="ArgumentException">Thrown when parameters are invalid</exception>
        public DiffusionModel(
            int timesteps = 1000,
            double betaStart = 0.0001,
            double betaEnd = 0.02,
            string modelName = "DiffusionModel") : base(
                new NeuralNetworkArchitecture<double>(
                    complexity: NetworkComplexity.Deep,
                    isPlaceholder: true,
                    cacheName: modelName),
                new MeanSquaredErrorLoss<double>(),
                1.0)
        {
            if (timesteps <= 0)
                throw new ArgumentException("Timesteps must be positive", nameof(timesteps));
            if (betaStart <= 0 || betaStart >= 1)
                throw new ArgumentException("Beta start must be between 0 and 1", nameof(betaStart));
            if (betaEnd <= betaStart || betaEnd >= 1)
                throw new ArgumentException("Beta end must be greater than beta start and less than 1", nameof(betaEnd));
            if (string.IsNullOrWhiteSpace(modelName))
                throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));
            
            _timesteps = timesteps;
            _betaStart = betaStart;
            _betaEnd = betaEnd;
            
            // Initialize noise schedule
            _betas = GenerateLinearSchedule(betaStart, betaEnd, timesteps);
            _alphas = _betas.Select(b => 1.0 - b).ToList();
            _alphasCumprod = new List<double>();
            double prod = 1.0;
            foreach (var alpha in _alphas)
            {
                prod *= alpha;
                _alphasCumprod.Add(prod);
            }
            
            _alphasCumprodPrev = new List<double> { 1.0 };
            _alphasCumprodPrev.AddRange(_alphasCumprod.Take(timesteps - 1));
            
            _sqrtAlphasCumprod = _alphasCumprod.Select(Math.Sqrt).ToList();
            _sqrtOneMinusAlphasCumprod = _alphasCumprod.Select(a => Math.Sqrt(1.0 - a)).ToList();
            
            // Pre-compute posterior parameters
            _posteriorMeanCoef1 = new List<double>();
            _posteriorMeanCoef2 = new List<double>();
            _posteriorVariance = new List<double>();
            _posteriorLogVarianceClipped = new List<double>();
            
            for (int t = 0; t < timesteps; t++)
            {
                var beta_t = _betas[t];
                var alpha_t = _alphas[t];
                var alpha_cumprod_t = _alphasCumprod[t];
                var alpha_cumprod_prev_t = _alphasCumprodPrev[t];
                
                _posteriorMeanCoef1.Add(beta_t * Math.Sqrt(alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t));
                _posteriorMeanCoef2.Add((1.0 - alpha_cumprod_prev_t) * Math.Sqrt(alpha_t) / (1.0 - alpha_cumprod_t));
                
                var posterior_var = beta_t * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t);
                _posteriorVariance.Add(posterior_var);
                _posteriorLogVarianceClipped.Add(Math.Log(Math.Max(posterior_var, 1e-20)));
            }
            
            // Architecture.CacheName is set in the constructor via the base class
        }
        
        /// <summary>
        /// Sets the noise predictor model
        /// </summary>
        /// <param name="predictor">The neural network model that predicts noise</param>
        /// <exception cref="ArgumentNullException">Thrown when predictor is null</exception>
        /// <exception cref="ObjectDisposedException">Thrown when the model has been disposed</exception>
        public void SetNoisePredictor(INeuralNetworkModel<double> predictor)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DiffusionModel));
            if (predictor == null)
                throw new ArgumentNullException(nameof(predictor));
            
            lock (_lockObject)
            {
                _noisePredictor = predictor;
            }
        }
        
        /// <summary>
        /// Forward diffusion process - adds noise to data
        /// </summary>
        /// <param name="data">Clean data to add noise to</param>
        /// <param name="t">Timestep indicating noise level</param>
        /// <returns>Tuple of noisy data and the noise that was added</returns>
        /// <exception cref="ArgumentNullException">Thrown when data is null</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when timestep is invalid</exception>
        public (Tensor<double> noisyData, Tensor<double> noise) ForwardDiffusion(Tensor<double> data, int t)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
            if (t < 0 || t >= _timesteps)
                throw new ArgumentOutOfRangeException(nameof(t), $"Timestep must be between 0 and {_timesteps - 1}");
            
            var noise = GenerateNoise(data.Shape, Random);
            var sqrtAlpha = _sqrtAlphasCumprod[t];
            var sqrtOneMinusAlpha = _sqrtOneMinusAlphasCumprod[t];
            
            var noisyData = data.Multiply(sqrtAlpha).Add(noise.Multiply(sqrtOneMinusAlpha));
            
            return (noisyData, noise);
        }
        
        /// <summary>
        /// Forward diffusion process with custom random source
        /// </summary>
        /// <param name="data">Clean data to add noise to</param>
        /// <param name="t">Timestep indicating noise level</param>
        /// <param name="random">Random number generator</param>
        /// <returns>Tuple of noisy data and the noise that was added</returns>
        public (Tensor<double> noisyData, Tensor<double> noise) ForwardDiffusion(Tensor<double> data, int t, Random random)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
            if (t < 0 || t >= _timesteps)
                throw new ArgumentOutOfRangeException(nameof(t), $"Timestep must be between 0 and {_timesteps - 1}");
            if (random == null)
                throw new ArgumentNullException(nameof(random));
            
            var noise = GenerateNoise(data.Shape, random);
            var sqrtAlpha = _sqrtAlphasCumprod[t];
            var sqrtOneMinusAlpha = _sqrtOneMinusAlphasCumprod[t];
            
            var noisyData = data.Multiply(sqrtAlpha).Add(noise.Multiply(sqrtOneMinusAlpha));
            
            return (noisyData, noise);
        }
        
        /// <summary>
        /// Reverse diffusion process - removes noise from data
        /// </summary>
        /// <param name="noisyData">Noisy data to denoise</param>
        /// <param name="t">Current timestep</param>
        /// <returns>Denoised data (mean of the posterior distribution)</returns>
        /// <exception cref="InvalidOperationException">Thrown when noise predictor is not set</exception>
        /// <exception cref="ArgumentNullException">Thrown when noisyData is null</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when timestep is invalid</exception>
        public Tensor<double> ReverseDiffusion(Tensor<double> noisyData, int t)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DiffusionModel));
            if (noisyData == null)
                throw new ArgumentNullException(nameof(noisyData));
            if (t < 0 || t >= _timesteps)
                throw new ArgumentOutOfRangeException(nameof(t), $"Timestep must be between 0 and {_timesteps - 1}");
            
            lock (_lockObject)
            {
                if (_noisePredictor == null)
                    throw new InvalidOperationException("Noise predictor model not set. Call SetNoisePredictor first.");
                
                // Predict noise
                var timestepTensor = CreateTimestepTensor(t, noisyData.Shape[0]);
                var predictedNoise = PredictNoise(noisyData, timestepTensor);
                
                // Compute mean
                var meanCoef1 = _posteriorMeanCoef1[t];
                var meanCoef2 = _posteriorMeanCoef2[t];
                var sqrtOneMinusAlpha = _sqrtOneMinusAlphasCumprod[t];
                var sqrtAlpha = _sqrtAlphasCumprod[t];
                
                var mean = noisyData.Subtract(predictedNoise.Multiply(sqrtOneMinusAlpha))
                                   .Divide(sqrtAlpha);
                
                return mean;
            }
        }
        
        /// <summary>
        /// Generates new samples from random noise
        /// </summary>
        /// <param name="shape">Shape of the samples to generate</param>
        /// <param name="seed">Optional random seed for reproducibility</param>
        /// <returns>Generated samples</returns>
        /// <exception cref="InvalidOperationException">Thrown when noise predictor is not set</exception>
        /// <exception cref="ArgumentException">Thrown when shape is invalid</exception>
        public virtual Tensor<double> Generate(int[] shape, int? seed = null)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DiffusionModel));
            if (shape == null || shape.Length == 0)
                throw new ArgumentException("Shape cannot be null or empty", nameof(shape));
            if (shape.Any(dim => dim <= 0))
                throw new ArgumentException("All shape dimensions must be positive", nameof(shape));
            
            lock (_lockObject)
            {
                if (_noisePredictor == null)
                    throw new InvalidOperationException("Noise predictor model not set. Call SetNoisePredictor first.");
                
                var random = seed.HasValue ? new Random(seed.Value) : new Random();
                
                // Start from pure noise
                var sample = GenerateNoise(shape, random);
                
                // Gradually denoise
                for (int t = _timesteps - 1; t >= 0; t--)
                {
                    sample = SampleTimestep(sample, t, random);
                }
                
                return sample;
            }
        }
        
        /// <summary>
        /// Generates multiple samples in parallel
        /// </summary>
        /// <param name="numSamples">Number of samples to generate</param>
        /// <param name="sampleShape">Shape of each individual sample</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Array of generated samples</returns>
        public async Task<Tensor<double>[]> GenerateMultipleAsync(int numSamples, int[] sampleShape, CancellationToken cancellationToken = default)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DiffusionModel));
            if (numSamples <= 0)
                throw new ArgumentException("Number of samples must be positive", nameof(numSamples));
            
            var tasks = new Task<Tensor<double>>[numSamples];
            
            for (int i = 0; i < numSamples; i++)
            {
                int sampleIndex = i;
                tasks[i] = Task.Run(() => Generate(sampleShape, seed: sampleIndex), cancellationToken);
            }
            
            return await Task.WhenAll(tasks);
        }
        
        /// <summary>
        /// Samples one timestep of the reverse process
        /// </summary>
        private Tensor<double> SampleTimestep(Tensor<double> x, int t, Random random)
        {
            // Get denoised mean
            var mean = ReverseDiffusion(x, t);
            
            if (t > 0)
            {
                // Add noise (except for t=0)
                var variance = _posteriorVariance[t];
                var noise = GenerateNoise(x.Shape, random);
                return mean.Add(noise.Multiply(Math.Sqrt(variance)));
            }
            
            return mean;
        }
        
        /// <summary>
        /// Training step for the diffusion model
        /// </summary>
        /// <param name="data">Batch of training data</param>
        /// <param name="optimizer">Optimizer for updating model parameters</param>
        /// <returns>Average loss for the batch</returns>
        /// <exception cref="InvalidOperationException">Thrown when noise predictor is not set</exception>
        /// <exception cref="ArgumentNullException">Thrown when arguments are null</exception>
        public virtual double TrainStep(Tensor<double> data, IOptimizer<double, Tensor<double>, Tensor<double>> optimizer)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DiffusionModel));
            if (data == null)
                throw new ArgumentNullException(nameof(data));
            if (optimizer == null)
                throw new ArgumentNullException(nameof(optimizer));
            
            lock (_lockObject)
            {
                if (_noisePredictor == null)
                    throw new InvalidOperationException("Noise predictor model not set");
                
                var random = new Random();
                var batchSize = data.Shape[0];
                var totalLoss = 0.0;
                
                // Process samples in parallel for better performance
                var losses = new double[batchSize];
                Parallel.For(0, batchSize, i =>
                {
                    // Sample random timestep
                    var t = random.Next(_timesteps);
                    
                    // Get single sample
                    var sample = GetSample(data, i);
                    
                    // Add noise
                    var (noisyData, noise) = ForwardDiffusion(sample, t, random);
                    
                    // Predict noise
                    var timestepTensor = CreateTimestepTensor(t, 1);
                    var predictedNoise = PredictNoise(noisyData, timestepTensor);
                    
                    // Compute loss (MSE between actual and predicted noise)
                    losses[i] = ComputeMSELoss(noise, predictedNoise);
                });
                
                totalLoss = losses.Sum() / losses.Length;
                
                // Backpropagate through noise predictor
                if (_noisePredictor is NeuralNetworkBase<double> nn)
                {
                    // Note: In practice, you'd accumulate gradients from all samples
                    // This is simplified for illustration
                    var avgGradient = ComputeAverageGradient(data);
                    nn.Backpropagate(avgGradient);
                    optimizer.Step(nn.GetParameters(), nn.GetParameterGradients());
                }
                
                // Track loss
                _trainingLosses.Add(totalLoss);
                LastLoss = NumOps.FromDouble(totalLoss);
                
                return totalLoss;
            }
        }
        
        /// <summary>
        /// Validates the model on a validation dataset
        /// </summary>
        /// <param name="validationData">Validation data</param>
        /// <returns>Average validation loss</returns>
        public double Validate(Tensor<double> validationData)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DiffusionModel));
            if (validationData == null)
                throw new ArgumentNullException(nameof(validationData));
            
            lock (_lockObject)
            {
                if (_noisePredictor == null)
                    throw new InvalidOperationException("Noise predictor model not set");
                
                var random = new Random(42); // Fixed seed for consistent validation
                var batchSize = validationData.Shape[0];
                var totalLoss = 0.0;
                
                for (int i = 0; i < batchSize; i++)
                {
                    var t = random.Next(_timesteps);
                    var sample = GetSample(validationData, i);
                    var (noisyData, noise) = ForwardDiffusion(sample, t, random);
                    var timestepTensor = CreateTimestepTensor(t, 1);
                    var predictedNoise = PredictNoise(noisyData, timestepTensor);
                    totalLoss += ComputeMSELoss(noise, predictedNoise);
                }
                
                var avgLoss = totalLoss / batchSize;
                _validationLosses.Add(avgLoss);
                
                return avgLoss;
            }
        }
        
        private Tensor<double> PredictNoise(Tensor<double> noisyData, Tensor<double> timestep)
        {
            // Combine noisy data and timestep
            var input = ConcatenateInputs(noisyData, timestep);
            return _noisePredictor!.Predict(input);
        }
        
        private Tensor<double> ConcatenateInputs(Tensor<double> data, Tensor<double> timestep)
        {
            // In practice, you'd properly concatenate or condition the data with timestep
            // This is a simplified version - real implementations use more sophisticated methods
            // like sinusoidal positional encoding for the timestep
            return data;
        }
        
        private Tensor<double> CreateTimestepTensor(int t, int batchSize)
        {
            var timestepData = new double[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                timestepData[i] = t;
            }
            return new Tensor<double>(new[] { batchSize }, new Vector<double>(timestepData));
        }
        
        private Tensor<double> GetSample(Tensor<double> data, int index)
        {
            // Extract a single sample from the batch
            // This would be more sophisticated in a real implementation
            return data.SubTensor(index);
        }
        
        private Tensor<double> ComputeAverageGradient(Tensor<double> batch)
        {
            // Simplified gradient computation
            // In practice, this would accumulate gradients across the batch
            return new Tensor<double>(batch.Shape);
        }
        
        private Tensor<double> GenerateNoise(int[] shape, Random random)
        {
            var totalElements = shape.Aggregate(1, (acc, dim) => acc * dim);
            var noiseData = new double[totalElements];
            
            // Generate samples from standard normal distribution using Box-Muller transform
            for (int i = 0; i < totalElements; i += 2)
            {
                var u1 = random.NextDouble();
                var u2 = random.NextDouble();
                
                var radius = Math.Sqrt(-2.0 * Math.Log(u1));
                var theta = 2.0 * Math.PI * u2;
                
                noiseData[i] = radius * Math.Cos(theta);
                if (i + 1 < totalElements)
                {
                    noiseData[i + 1] = radius * Math.Sin(theta);
                }
            }
            
            return new Tensor<double>(shape, new Vector<double>(noiseData));
        }
        
        private double ComputeMSELoss(Tensor<double> actual, Tensor<double> predicted)
        {
            var diff = actual.Subtract(predicted);
            var squared = diff.Multiply(diff);
            var squaredVector = squared.ToVector();
            
            // Compute average manually
            double sum = 0.0;
            for (int i = 0; i < squaredVector.Length; i++)
            {
                sum += squaredVector[i];
            }
            return sum / squaredVector.Length;
        }
        
        private List<double> GenerateLinearSchedule(double start, double end, int steps)
        {
            var schedule = new List<double>();
            var stepSize = (end - start) / (steps - 1);
            
            for (int i = 0; i < steps; i++)
            {
                schedule.Add(start + i * stepSize);
            }
            
            return schedule;
        }
        
        // Implement abstract methods from NeuralNetworkBase
        
        public override Tensor<double> Predict(Tensor<double> input)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DiffusionModel));
            
            // For diffusion models, prediction means generating new samples
            // Input could contain conditioning information
            return Generate(input.Shape);
        }
        
        protected override void InitializeLayers()
        {
            // Diffusion models don't use traditional layers
            // The noise predictor model is set separately via SetNoisePredictor
        }
        
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            // Read timesteps (already set in constructor)
            var savedTimesteps = reader.ReadInt32();
            var savedBetaStart = reader.ReadDouble();
            var savedBetaEnd = reader.ReadDouble();
            
            // Verify consistency
            if (savedTimesteps != _timesteps || 
                Math.Abs(savedBetaStart - _betaStart) > 1e-10 ||
                Math.Abs(savedBetaEnd - _betaEnd) > 1e-10)
            {
                throw new InvalidOperationException("Serialized model has different configuration");
            }
            
            // Read beta schedule
            var betaCount = reader.ReadInt32();
            for (int i = 0; i < betaCount; i++)
            {
                reader.ReadDouble(); // Skip, as we've already computed these
            }
            
            // Read noise predictor
            var hasNoisePredictor = reader.ReadBoolean();
            if (hasNoisePredictor)
            {
                var noisePredictorDataLength = reader.ReadInt32();
                var noisePredictorData = reader.ReadBytes(noisePredictorDataLength);
                // Note: Deserialization of noise predictor would require a factory pattern
                // to create the correct type of model
            }
            
            // Read loss history
            var trainingLossCount = reader.ReadInt32();
            _trainingLosses.Clear();
            for (int i = 0; i < trainingLossCount; i++)
            {
                _trainingLosses.Add(reader.ReadDouble());
            }
            
            var validationLossCount = reader.ReadInt32();
            _validationLosses.Clear();
            for (int i = 0; i < validationLossCount; i++)
            {
                _validationLosses.Add(reader.ReadDouble());
            }
        }
        
        protected override IFullModel<double, Tensor<double>, Tensor<double>> CreateNewInstance()
        {
            return new DiffusionModel(_timesteps, _betaStart, _betaEnd, Architecture.CacheName);
        }
        
        public override ModelMetadata<double> GetModelMetadata()
        {
            var metadata = new ModelMetadata<double>
            {
                ModelType = ModelType.NeuralNetwork,
                FeatureCount = (_noisePredictor is NeuralNetworkBase<double> nn ? nn.GetParameterCount() : _noisePredictor?.GetParameters().Length ?? 0),
                Complexity = _timesteps,
                Description = $"Denoising Diffusion Probabilistic Model with {_timesteps} timesteps for generative tasks",
                AdditionalInfo = new Dictionary<string, object>
                {
                    ["ModelCategory"] = "Generative",
                    ["ModelName"] = Architecture.CacheName,
                    ["ArchitectureDescription"] = $"Diffusion Model with {_timesteps} timesteps",
                    ["TotalParameters"] = (_noisePredictor is NeuralNetworkBase<double> nn1 ? nn1.GetParameterCount() : _noisePredictor?.GetParameters().Length ?? 0),
                    ["TrainableParameters"] = (_noisePredictor is NeuralNetworkBase<double> nn2 ? nn2.GetParameterCount() : _noisePredictor?.GetParameters().Length ?? 0),
                    ["NonTrainableParameters"] = 0,
                    ["InputShape"] = _noisePredictor != null ? "Variable" : "Not set",
                    ["OutputShape"] = _noisePredictor != null ? "Variable" : "Not set",
                    ["LearningRateSchedule"] = "Depends on optimizer",
                    ["RegularizationStrength"] = 0.0,
                    ["LastTrainingLoss"] = Convert.ToDouble(LastLoss),
                    ["LastValidationLoss"] = _validationLosses.Count > 0 ? _validationLosses.Last() : 0.0,
                    ["TotalEpochsTrained"] = _trainingLosses.Count,
                    ["BatchSize"] = 0, // Variable
                    ["ValidationMetrics"] = new Dictionary<string, double>
                    {
                        ["Average Loss"] = _validationLosses.Count > 0 ? _validationLosses.Sum() / _validationLosses.Count : 0.0
                    },
                    ["TestMetrics"] = new Dictionary<string, double>(),
                    ["LayerInformation"] = GetLayerInformation(),
                    ["SupportsParallelProcessing"] = true,
                    ["EstimatedMemoryUsageBytes"] = EstimateMemoryUsage(),
                    ["PreferredHardware"] = "GPU",
                    ["LastUpdated"] = DateTime.UtcNow,
                    ["Version"] = "1.0.0",
                    ["Author"] = "AiDotNet",
                    ["BetaStart"] = _betaStart,
                    ["BetaEnd"] = _betaEnd,
                    ["Timesteps"] = _timesteps,
                    ["NoiseSchedule"] = "Linear",
                    ["HasNoisePredictor"] = _noisePredictor != null
                },
                ModelData = Array.Empty<byte>(), // Could serialize the model here if needed
                FeatureImportance = new Dictionary<string, double>()
            };

            // Add epoch history if available
            if (_trainingLosses.Count > 0)
            {
                var epochHistory = new List<Dictionary<string, object>>();
                for (int i = 0; i < _trainingLosses.Count; i++)
                {
                    epochHistory.Add(new Dictionary<string, object>
                    {
                        ["Epoch"] = i + 1,
                        ["TrainingLoss"] = _trainingLosses[i],
                        ["ValidationLoss"] = i < _validationLosses.Count ? _validationLosses[i] : 0.0
                    });
                }
                metadata.AdditionalInfo["EpochHistory"] = epochHistory;
            }

            return metadata;
        }
        
        public override void Train(Tensor<double> input, Tensor<double> expectedOutput)
        {
            // Diffusion models use a different training approach
            // This would typically involve the TrainStep method with an optimizer
            throw new NotImplementedException("Use TrainStep method with an optimizer for diffusion model training");
        }
        
        public override void UpdateParameters(Vector<double> parameters)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DiffusionModel));
            
            // Update parameters of the noise predictor model
            lock (_lockObject)
            {
                if (_noisePredictor != null)
                {
                    _noisePredictor.UpdateParameters(parameters);
                }
            }
        }
        
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            // Write diffusion-specific data
            writer.Write(_timesteps);
            writer.Write(_betaStart);
            writer.Write(_betaEnd);
            
            // Write beta schedule
            writer.Write(_betas.Count);
            foreach (var beta in _betas)
            {
                writer.Write(beta);
            }
            
            // Write whether we have a noise predictor
            writer.Write(_noisePredictor != null);
            if (_noisePredictor != null)
            {
                // Serialize the noise predictor model
                var noisePredictorData = _noisePredictor.Serialize();
                writer.Write(noisePredictorData.Length);
                writer.Write(noisePredictorData);
            }
            
            // Write loss history
            writer.Write(_trainingLosses.Count);
            foreach (var loss in _trainingLosses)
            {
                writer.Write(loss);
            }
            
            writer.Write(_validationLosses.Count);
            foreach (var loss in _validationLosses)
            {
                writer.Write(loss);
            }
        }
        
        private List<string> GetLayerInformation()
        {
            var info = new List<string>
            {
                $"Timesteps: {_timesteps}",
                $"Beta Schedule: {_betaStart} to {_betaEnd}",
                $"Noise Predictor: {(_noisePredictor != null ? "Set" : "Not set")}"
            };
            
            if (_noisePredictor != null && _noisePredictor is NeuralNetworkBase<double> nn)
            {
                info.Add($"Noise Predictor Parameters: {nn.GetParameterCount()}");
            }
            
            info.Add($"Training Samples Processed: {_trainingLosses.Count}");
            info.Add($"Validation Samples Processed: {_validationLosses.Count}");
            
            return info;
        }
        
        private long EstimateMemoryUsage()
        {
            // Base memory for schedules and parameters
            long baseMemory = _timesteps * 8 * 10; // Approximate memory for schedules
            
            if (_noisePredictor != null)
            {
                // Add memory for noise predictor parameters
                int paramCount = _noisePredictor is NeuralNetworkBase<double> nn ? nn.GetParameterCount() : _noisePredictor.GetParameters().Length;
                baseMemory += paramCount * 8;
            }
            
            // Add memory for loss history
            baseMemory += (_trainingLosses.Count + _validationLosses.Count) * 8;
            
            return baseMemory;
        }
        
        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        
        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        /// <param name="disposing">true to release both managed and unmanaged resources; false to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    lock (_lockObject)
                    {
                        // Dispose noise predictor if it implements IDisposable
                        if (_noisePredictor is IDisposable disposablePredictor)
                        {
                            disposablePredictor.Dispose();
                        }
                        
                        // Clear collections
                        _betas.Clear();
                        _alphas.Clear();
                        _alphasCumprod.Clear();
                        _alphasCumprodPrev.Clear();
                        _sqrtAlphasCumprod.Clear();
                        _sqrtOneMinusAlphasCumprod.Clear();
                        _posteriorMeanCoef1.Clear();
                        _posteriorMeanCoef2.Clear();
                        _posteriorVariance.Clear();
                        _posteriorLogVarianceClipped.Clear();
                        _trainingLosses.Clear();
                        _validationLosses.Clear();
                    }
                }
                
                _disposed = true;
            }
        }
        
        /// <summary>
        /// Finalizer
        /// </summary>
        ~DiffusionModel()
        {
            Dispose(false);
        }
    }
}