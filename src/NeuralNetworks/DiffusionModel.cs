using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;

namespace AiDotNet.NeuralNetworks
{
    /// <summary>
    /// Denoising Diffusion Probabilistic Model (DDPM) for high-quality image generation
    /// </summary>
    public class DiffusionModel : NeuralNetworkBase<double>, IGenerativeModel
    {
        protected readonly int timesteps = default!;
        protected readonly double betaStart = default!;
        protected readonly double betaEnd = default!;
        protected readonly List<double> betas = default!;
        protected readonly List<double> alphas = default!;
        protected readonly List<double> alphasCumprod = default!;
        protected readonly List<double> alphasCumprodPrev = default!;
        protected readonly List<double> sqrtAlphasCumprod = default!;
        protected readonly List<double> sqrtOneMinusAlphasCumprod = default!;
        protected readonly List<double> posteriorMeanCoef1 = default!;
        protected readonly List<double> posteriorMeanCoef2 = default!;
        protected readonly List<double> posteriorVariance = default!;
        protected readonly List<double> posteriorLogVarianceClipped = default!;

        protected INeuralNetworkModel<double> noisePredictor = default!;

        public DiffusionModel(
            NeuralNetworkArchitecture<double> architecture,
            int timesteps = 1000,
            double betaStart = 0.0001,
            double betaEnd = 0.02,
            ILossFunction<double>? lossFunction = null,
            double maxGradNorm = 1.0)
            : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<double>(), maxGradNorm)
        {
            this.timesteps = timesteps;
            this.betaStart = betaStart;
            this.betaEnd = betaEnd;
            
            // Initialize noise schedule
            betas = GenerateLinearSchedule(betaStart, betaEnd, timesteps);
            alphas = betas.Select(b => 1.0 - b).ToList();
            alphasCumprod = new List<double>();
            double prod = 1.0;
            foreach (var alpha in alphas)
            {
                prod *= alpha;
                alphasCumprod.Add(prod);
            }
            
            alphasCumprodPrev = new List<double> { 1.0 };
            alphasCumprodPrev.AddRange(alphasCumprod.Take(timesteps - 1));
            
            sqrtAlphasCumprod = alphasCumprod.Select(Math.Sqrt).ToList();
            sqrtOneMinusAlphasCumprod = alphasCumprod.Select(a => Math.Sqrt(1.0 - a)).ToList();
            
            // Pre-compute posterior parameters
            posteriorMeanCoef1 = new List<double>();
            posteriorMeanCoef2 = new List<double>();
            posteriorVariance = new List<double>();
            posteriorLogVarianceClipped = new List<double>();
            
            for (int t = 0; t < timesteps; t++)
            {
                var beta_t = betas[t];
                var alpha_t = alphas[t];
                var alpha_cumprod_t = alphasCumprod[t];
                var alpha_cumprod_prev_t = alphasCumprodPrev[t];
                
                posteriorMeanCoef1.Add(beta_t * Math.Sqrt(alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t));
                posteriorMeanCoef2.Add((1.0 - alpha_cumprod_prev_t) * Math.Sqrt(alpha_t) / (1.0 - alpha_cumprod_t));
                
                var posterior_var = beta_t * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t);
                posteriorVariance.Add(posterior_var);
                posteriorLogVarianceClipped.Add(Math.Log(Math.Max(posterior_var, 1e-20)));
            }
        }

        /// <summary>
        /// Initializes the layers of the Diffusion Model.
        /// </summary>
        protected override void InitializeLayers()
        {
            // Diffusion models typically don't have traditional layers
            // The noise predictor is set externally via SetNoisePredictor
        }
        
        public void SetNoisePredictor(INeuralNetworkModel<double> predictor)
        {
            noisePredictor = predictor;
        }
        
        /// <summary>
        /// Forward diffusion process - add noise to data
        /// </summary>
        public (Tensor<double> noisyData, Tensor<double> noise) ForwardDiffusion(Tensor<double> data, int t, Random random = null!)
        {
            random ??= new Random();
            
            var noise = GenerateNoise(data.Shape, random);
            var sqrtAlpha = sqrtAlphasCumprod[t];
            var sqrtOneMinusAlpha = sqrtOneMinusAlphasCumprod[t];
            
            var noisyData = data.Multiply(sqrtAlpha).Add(noise.Multiply(sqrtOneMinusAlpha));
            
            return (noisyData, noise);
        }
        
        /// <summary>
        /// Reverse diffusion process - denoise data
        /// </summary>
        public Tensor<double> ReverseDiffusion(Tensor<double> noisyData, int t)
        {
            if (noisePredictor == null)
                throw new InvalidOperationException("Noise predictor model not set");
            
            // Predict noise
            var timestepTensor = new Tensor<double>(new[] { t });
            var predictedNoise = PredictNoise(noisyData, timestepTensor);
            
            // Compute mean
            var meanCoef1 = posteriorMeanCoef1[t];
            var meanCoef2 = posteriorMeanCoef2[t];
            var sqrtOneMinusAlpha = sqrtOneMinusAlphasCumprod[t];
            var sqrtAlpha = Math.Sqrt(alphasCumprod[t]);

            var mean = noisyData.Subtract(predictedNoise.Multiply(sqrtOneMinusAlpha))
                               .Multiply(1.0 / sqrtAlpha);
            
            return mean;
        }
        
        /// <summary>
        /// Generate new samples
        /// </summary>
        public virtual Tensor<double> Generate(int[] shape, int? seed = null)
        {
            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            
            // Start from pure noise
            var sample = GenerateNoise(shape, random);
            
            // Gradually denoise
            for (int t = timesteps - 1; t >= 0; t--)
            {
                sample = SampleTimestep(sample, t, random);
            }
            
            return sample;
        }
        
        /// <summary>
        /// Sample one timestep of the reverse process
        /// </summary>
        private Tensor<double> SampleTimestep(Tensor<double> x, int t, Random random)
        {
            // Get denoised mean
            var mean = ReverseDiffusion(x, t);
            
            if (t > 0)
            {
                // Add noise (except for t=0)
                var variance = posteriorVariance[t];
                var noise = GenerateNoise(x.Shape, random);
                return mean.Add(noise.Multiply(Math.Sqrt(variance)));
            }
            
            return mean;
        }
        
        /// <summary>
        /// Training step for the diffusion model
        /// </summary>
        public virtual double TrainStep(Tensor<double> data, IOptimizer<double, Tensor<double>, Tensor<double>> optimizer)
        {
            if (noisePredictor == null)
                throw new InvalidOperationException("Noise predictor model not set");
            
            var random = new Random();
            var batchSize = data.Shape[0];
            var totalLoss = 0.0;
            
            for (int i = 0; i < batchSize; i++)
            {
                // Sample random timestep
                var t = random.Next(timesteps);
                
                // Get single sample
                var sample = data.GetSlice(i);
                
                // Add noise
                var (noisyData, noise) = ForwardDiffusion(sample, t, random);
                
                // Predict noise
                var timestepTensor = new Tensor<double>(new[] { t });
                var predictedNoise = PredictNoise(noisyData, timestepTensor);
                
                // Compute loss (MSE between actual and predicted noise)
                var loss = ComputeMSELoss(noise, predictedNoise);
                totalLoss += loss;
                
                // Backpropagate through noise predictor
                if (noisePredictor is NeuralNetworkBase<double> nn)
                {
                    nn.Backpropagate(predictedNoise.Subtract(noise));
                    optimizer.Step();
                }
            }
            
            return totalLoss / batchSize;
        }
        
        protected Tensor<double> PredictNoise(Tensor<double> noisyData, Tensor<double> timestep)
        {
            // Combine noisy data and timestep
            var input = ConcatenateInputs(noisyData, timestep);
            return noisePredictor.Predict(input);
        }
        
        protected Tensor<double> ConcatenateInputs(Tensor<double> data, Tensor<double> timestep)
        {
            // Simple concatenation - in practice, you'd use more sophisticated conditioning
            var timestepExpanded = new Tensor<double>(data.Shape.Select(_ => 1).ToArray());
            timestepExpanded.Fill(timestep[0]);
            return data; // Simplified - should properly concatenate
        }
        
        protected Tensor<double> GenerateNoise(int[] shape, Random random)
        {
            var noise = new Tensor<double>(shape);
            var data = noise.Data;
            
            for (int i = 0; i < data.Length; i++)
            {
                // Sample from standard normal distribution
                var u1 = random.NextDouble();
                var u2 = random.NextDouble();
                var z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                data[i] = z;
            }
            
            return noise;
        }
        
        protected double ComputeMSELoss(Tensor<double> actual, Tensor<double> predicted)
        {
            var diff = actual.Subtract(predicted);
            var squared = diff.Multiply(diff);
            return squared.Data.Average();
        }
        
        protected List<double> GenerateLinearSchedule(double start, double end, int steps)
        {
            var schedule = new List<double>();
            var stepSize = (end - start) / (steps - 1);
            
            for (int i = 0; i < steps; i++)
            {
                schedule.Add(start + i * stepSize);
            }
            
            return schedule;
        }

        public Tensor<double> Forward(Tensor<double> input)
        {
            // For compatibility - use Generate method instead
            return Generate(input.Shape);
        }

        public void Backward(Tensor<double> gradOutput)
        {
            // Backward pass is handled in TrainStep
        }

        /// <summary>
        /// Updates the parameters of the Diffusion Model.
        /// </summary>
        /// <param name="parameters">The new parameter values.</param>
        public override void UpdateParameters(Vector<double> parameters)
        {
            // Delegate to noise predictor if it's a neural network
            if (noisePredictor is NeuralNetworkBase<double> nn)
            {
                nn.UpdateParameters(parameters);
            }
        }

        /// <summary>
        /// Creates a new instance of the Diffusion Model.
        /// </summary>
        /// <returns>A new instance of DiffusionModel.</returns>
        protected override IFullModel<double, Tensor<double>, Tensor<double>> CreateNewInstance()
        {
            var newModel = new DiffusionModel(
                Architecture,
                timesteps,
                betaStart,
                betaEnd,
                LossFunction,
                Convert.ToDouble(MaxGradNorm));

            if (noisePredictor != null)
            {
                newModel.SetNoisePredictor(noisePredictor);
            }

            return newModel;
        }

        /// <summary>
        /// Makes a prediction (generates samples) using the Diffusion Model.
        /// </summary>
        /// <param name="input">The input tensor (shape used for generation).</param>
        /// <returns>The generated tensor.</returns>
        public override Tensor<double> Predict(Tensor<double> input)
        {
            return Generate(input.Shape);
        }

        /// <summary>
        /// Trains the Diffusion Model on a single input-output pair.
        /// </summary>
        /// <param name="input">The input data.</param>
        /// <param name="expectedOutput">The expected output (not used for diffusion models).</param>
        public override void Train(Tensor<double> input, Tensor<double> expectedOutput)
        {
            if (noisePredictor == null)
                throw new InvalidOperationException("Noise predictor model not set");

            var random = new Random();

            // Sample random timestep
            var t = random.Next(timesteps);

            // Add noise
            var (noisyData, noise) = ForwardDiffusion(input, t, random);

            // Predict noise
            var timestepTensor = new Tensor<double>(new[] { t });
            var predictedNoise = PredictNoise(noisyData, timestepTensor);

            // Compute loss (MSE between actual and predicted noise)
            LastLoss = NumOps.FromDouble(ComputeMSELoss(noise, predictedNoise));

            // Backpropagate through noise predictor
            if (noisePredictor is NeuralNetworkBase<double> nn)
            {
                var lossGrad = predictedNoise.Subtract(noise);
                nn.Backpropagate(lossGrad);
            }
        }

        /// <summary>
        /// Gets the metadata for this Diffusion Model.
        /// </summary>
        /// <returns>A ModelMetaData object containing information about the model.</returns>
        public override ModelMetaData<double> GetModelMetaData()
        {
            return new ModelMetaData<double>
            {
                ModelType = ModelType.DiffusionModel,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "Timesteps", timesteps },
                    { "BetaStart", betaStart },
                    { "BetaEnd", betaEnd },
                    { "HasNoisePredictor", noisePredictor != null }
                },
                ModelData = this.Serialize()
            };
        }

        /// <summary>
        /// Serializes Diffusion Model-specific data.
        /// </summary>
        /// <param name="writer">The BinaryWriter to write the data to.</param>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(timesteps);
            writer.Write(betaStart);
            writer.Write(betaEnd);

            // Serialize noise schedule parameters
            writer.Write(betas.Count);
            foreach (var beta in betas)
            {
                writer.Write(beta);
            }

            writer.Write(alphas.Count);
            foreach (var alpha in alphas)
            {
                writer.Write(alpha);
            }
        }

        /// <summary>
        /// Deserializes Diffusion Model-specific data.
        /// </summary>
        /// <param name="reader">The BinaryReader to read the data from.</param>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            // Read configuration values (already set in constructor)
            int savedTimesteps = reader.ReadInt32();
            double savedBetaStart = reader.ReadDouble();
            double savedBetaEnd = reader.ReadDouble();

            // Read noise schedule parameters
            int betasCount = reader.ReadInt32();
            for (int i = 0; i < betasCount; i++)
            {
                reader.ReadDouble(); // Skip, already computed in constructor
            }

            int alphasCount = reader.ReadInt32();
            for (int i = 0; i < alphasCount; i++)
            {
                reader.ReadDouble(); // Skip, already computed in constructor
            }
        }

        protected virtual void SaveModelSpecificData(IDictionary<string, object> data)
        {
            data["timesteps"] = timesteps;
            data["betaStart"] = betaStart;
            data["betaEnd"] = betaEnd;
            data["betas"] = betas;
            data["alphas"] = alphas;
        }

        protected void LoadModelSpecificData(IDictionary<string, object> data)
        {
            // Load saved parameters
        }
    }
    
    /// <summary>
    /// Interface for generative models
    /// </summary>
    public interface IGenerativeModel
    {
        Tensor<double> Generate(int[] shape, int? seed = null);
    }
}