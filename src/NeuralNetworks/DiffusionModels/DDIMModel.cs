using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.DiffusionModels
{
    /// <summary>
    /// Denoising Diffusion Implicit Models (DDIM) - Faster sampling than DDPM
    /// </summary>
    public class DDIMModel : DiffusionModel
    {
        private readonly double eta; // Controls stochasticity (0 = deterministic)
        private readonly int samplingSteps; // Can be much less than training timesteps
        
        public DDIMModel(
            NeuralNetworkArchitecture<double> architecture,
            int timesteps = 1000,
            int samplingSteps = 50,
            double eta = 0.0,
            double betaStart = 0.0001,
            double betaEnd = 0.02,
            ILossFunction<double>? lossFunction = null,
            double maxGradNorm = 1.0)
            : base(architecture, timesteps, betaStart, betaEnd, lossFunction, maxGradNorm)
        {
            this.samplingSteps = samplingSteps;
            this.eta = eta;
        }
        
        /// <summary>
        /// DDIM sampling - can skip timesteps for faster generation
        /// </summary>
        public override Tensor<double> Generate(int[] shape, int? seed = null)
        {
            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            
            // Start from pure noise
            var sample = GenerateNoise(shape, random);
            
            // Create subsequence of timesteps for faster sampling
            var timestepSeq = CreateTimestepSequence(timesteps, samplingSteps);
            
            // DDIM sampling loop
            for (int i = timestepSeq.Length - 1; i >= 0; i--)
            {
                int t = timestepSeq[i];
                int tPrev = i > 0 ? timestepSeq[i - 1] : 0;
                
                sample = DDIMSampleStep(sample, t, tPrev, random);
            }
            
            return sample;
        }
        
        /// <summary>
        /// DDIM sampling step
        /// </summary>
        private Tensor<double> DDIMSampleStep(Tensor<double> x, int t, int tPrev, Random random)
        {
            // Predict noise
            var timestepTensor = new Tensor<double>(new[] { t });
            var predictedNoise = PredictNoise(x, timestepTensor);

            // Get alpha values
            var alphaCumprod = alphasCumprod[t];
            var alphaCumprodPrev = tPrev > 0 ? alphasCumprod[tPrev] : 1.0;

            // Compute x0 prediction
            var x0Pred = x.Subtract(predictedNoise.Multiply(Math.Sqrt(1 - alphaCumprod)))
                         .Multiply(1.0 / Math.Sqrt(alphaCumprod));
            
            // Clip x0 prediction
            x0Pred = ClipTensor(x0Pred, -1.0, 1.0);
            
            // Compute variance
            var variance = eta * Math.Sqrt((1 - alphaCumprodPrev) / (1 - alphaCumprod)) * 
                          Math.Sqrt(1 - alphaCumprod / alphaCumprodPrev);
            
            // Direction pointing to x_t
            var direction = predictedNoise.Multiply(Math.Sqrt(1 - alphaCumprodPrev - variance * variance));
            
            // Compute x_{t-1}
            var mean = x0Pred.Multiply(Math.Sqrt(alphaCumprodPrev)).Add(direction);
            
            if (eta > 0 && t > 0)
            {
                var noise = GenerateNoise(x.Shape, random);
                return mean.Add(noise.Multiply(variance));
            }
            
            return mean;
        }
        
        /// <summary>
        /// Create subsequence of timesteps for DDIM sampling
        /// </summary>
        private int[] CreateTimestepSequence(int totalSteps, int numSteps)
        {
            var sequence = new int[numSteps];
            var stepSize = totalSteps / numSteps;
            
            for (int i = 0; i < numSteps; i++)
            {
                sequence[i] = i * stepSize;
            }
            
            // Ensure last step is included
            sequence[numSteps - 1] = totalSteps - 1;
            
            return sequence;
        }
        
        private Tensor<double> ClipTensor(Tensor<double> tensor, double min, double max)
        {
            var clipped = tensor.Clone();
            var data = clipped.Data;
            
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = Math.Max(min, Math.Min(max, data[i]));
            }
            
            return clipped;
        }
        
    }
}