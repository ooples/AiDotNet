using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.DiffusionModels
{
    /// <summary>
    /// Consistency Model - Direct mapping from noise to data in a single step
    /// Based on "Consistency Models" paper
    /// </summary>
    public class ConsistencyModel : NeuralNetworkBase<double>
    {
        private readonly INeuralNetworkModel<double> consistencyFunction;
        private readonly double sigmaMin;
        private readonly double sigmaMax;
        private readonly int numSteps;
        private readonly double rho;
        private readonly ScheduleType scheduleType;
        private readonly bool useDistillation;
        private INeuralNetworkModel<double> teacherModel;
        
        public enum ScheduleType
        {
            Karras,
            Linear,
            Quadratic
        }
        
        public ConsistencyModel(
            INeuralNetworkModel<double> consistencyFunction,
            double sigmaMin = 0.002,
            double sigmaMax = 80.0,
            int numSteps = 18,
            double rho = 7.0,
            ScheduleType scheduleType = ScheduleType.Karras,
            bool useDistillation = false,
            string modelName = "ConsistencyModel")
            : base(modelName)
        {
            this.consistencyFunction = consistencyFunction ?? throw new ArgumentNullException(nameof(consistencyFunction));
            this.sigmaMin = sigmaMin;
            this.sigmaMax = sigmaMax;
            this.numSteps = numSteps;
            this.rho = rho;
            this.scheduleType = scheduleType;
            this.useDistillation = useDistillation;
            
            ModelCategory = ModelCategory.Generative;
        }
        
        /// <summary>
        /// Set teacher model for consistency distillation
        /// </summary>
        public void SetTeacherModel(INeuralNetworkModel<double> teacher)
        {
            if (!useDistillation)
                throw new InvalidOperationException("Model not configured for distillation");
            
            teacherModel = teacher;
        }
        
        /// <summary>
        /// Single-step generation
        /// </summary>
        public Tensor<double> Generate(int[] shape, int? seed = null)
        {
            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            
            // Sample noise at maximum sigma
            var noise = SampleNoise(shape, sigmaMax, random);
            
            // Single forward pass through consistency function
            return consistencyFunction.Predict(ConcatenateSigma(noise, sigmaMax));
        }
        
        /// <summary>
        /// Multi-step generation for better quality
        /// </summary>
        public Tensor<double> GenerateMultiStep(int[] shape, int steps = 0, int? seed = null)
        {
            if (steps == 0) steps = numSteps;
            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            
            // Get sigma schedule
            var sigmas = GetSigmaSchedule(steps);
            
            // Start from noise
            var x = SampleNoise(shape, sigmas[0], random);
            
            // Iterative refinement
            for (int i = 0; i < steps - 1; i++)
            {
                var sigma = sigmas[i];
                var nextSigma = sigmas[i + 1];
                
                // Denoise to sigma_min
                var denoised = consistencyFunction.Predict(ConcatenateSigma(x, sigma));
                
                if (i < steps - 2)
                {
                    // Add noise to go to next sigma level
                    x = AddNoise(denoised, nextSigma, random);
                }
                else
                {
                    x = denoised;
                }
            }
            
            return x;
        }
        
        /// <summary>
        /// Train with consistency training (from scratch)
        /// </summary>
        public double TrainConsistency(Tensor<double> data, IOptimizer<double, Tensor<double>, Tensor<double>> optimizer)
        {
            var batchSize = data.Shape[0];
            var totalLoss = 0.0;
            var random = new Random();
            
            for (int i = 0; i < batchSize; i++)
            {
                var sample = data.GetSlice(new[] { i });
                
                // Sample time step
                var n = random.Next(1, numSteps);
                var sigmas = GetSigmaSchedule(numSteps);
                var sigma = sigmas[n];
                var prevSigma = sigmas[n - 1];
                
                // Add noise
                var noise = GenerateNoise(sample.Shape, random);
                var noisySample = sample.Add(noise.Multiply(sigma));
                
                // Target: one-step denoised version
                var target = GetConsistencyTarget(sample, noisySample, sigma, prevSigma);
                
                // Predict with consistency function
                var predicted = consistencyFunction.Predict(ConcatenateSigma(noisySample, sigma));
                
                // Consistency loss
                var loss = ComputeConsistencyLoss(predicted, target, sigma);
                totalLoss += loss;
                
                // Backpropagate
                if (consistencyFunction is NeuralNetworkBase<double> nn)
                {
                    var grad = predicted.Subtract(target).Multiply(GetLossWeight(sigma));
                    nn.Backward(grad);
                    optimizer.Step(nn.GetParameters(), nn.GetGradients());
                }
            }
            
            return totalLoss / batchSize;
        }
        
        /// <summary>
        /// Train with consistency distillation (from pre-trained diffusion model)
        /// </summary>
        public double TrainDistillation(Tensor<double> data, IOptimizer<double, Tensor<double>, Tensor<double>> optimizer)
        {
            if (!useDistillation || teacherModel == null)
                throw new InvalidOperationException("Teacher model not set for distillation");
            
            var batchSize = data.Shape[0];
            var totalLoss = 0.0;
            var random = new Random();
            
            for (int i = 0; i < batchSize; i++)
            {
                var sample = data.GetSlice(new[] { i });
                
                // Sample adjacent time steps
                var n = random.Next(1, numSteps);
                var sigmas = GetSigmaSchedule(numSteps);
                var sigma = sigmas[n];
                var prevSigma = sigmas[n - 1];
                
                // Add noise
                var noise = GenerateNoise(sample.Shape, random);
                var noisySample = sample.Add(noise.Multiply(sigma));
                
                // Get teacher's one-step denoised output
                var teacherOutput = GetTeacherPrediction(noisySample, sigma, prevSigma);
                
                // Student prediction at current noise level
                var studentCurrent = consistencyFunction.Predict(ConcatenateSigma(noisySample, sigma));
                
                // Student prediction at previous noise level (should be consistent)
                var studentPrev = consistencyFunction.Predict(ConcatenateSigma(teacherOutput, prevSigma));
                
                // Distillation loss: enforce consistency
                var loss = ComputeDistillationLoss(studentCurrent, studentPrev);
                totalLoss += loss;
                
                // Backpropagate
                if (consistencyFunction is NeuralNetworkBase<double> nn)
                {
                    var grad = studentCurrent.Subtract(studentPrev.Detach());
                    nn.Backward(grad);
                    optimizer.Step(nn.GetParameters(), nn.GetGradients());
                }
            }
            
            return totalLoss / batchSize;
        }
        
        /// <summary>
        /// Get sigma schedule
        /// </summary>
        private double[] GetSigmaSchedule(int steps)
        {
            var sigmas = new double[steps];
            
            switch (scheduleType)
            {
                case ScheduleType.Karras:
                    // Karras et al. schedule
                    var minInvRho = Math.Pow(sigmaMin, 1.0 / rho);
                    var maxInvRho = Math.Pow(sigmaMax, 1.0 / rho);
                    
                    for (int i = 0; i < steps; i++)
                    {
                        var t = (double)i / (steps - 1);
                        var sigmaInvRho = maxInvRho + t * (minInvRho - maxInvRho);
                        sigmas[i] = Math.Pow(sigmaInvRho, rho);
                    }
                    break;
                    
                case ScheduleType.Linear:
                    for (int i = 0; i < steps; i++)
                    {
                        var t = (double)i / (steps - 1);
                        sigmas[i] = sigmaMax + t * (sigmaMin - sigmaMax);
                    }
                    break;
                    
                case ScheduleType.Quadratic:
                    for (int i = 0; i < steps; i++)
                    {
                        var t = (double)i / (steps - 1);
                        var sqrtSigma = Math.Sqrt(sigmaMax) + t * (Math.Sqrt(sigmaMin) - Math.Sqrt(sigmaMax));
                        sigmas[i] = sqrtSigma * sqrtSigma;
                    }
                    break;
            }
            
            return sigmas;
        }
        
        /// <summary>
        /// Get consistency target for training
        /// </summary>
        private Tensor<double> GetConsistencyTarget(Tensor<double> clean, Tensor<double> noisy, double sigma, double prevSigma)
        {
            if (sigma == sigmaMin)
            {
                // Boundary condition: f(x, sigma_min) = x
                return noisy;
            }
            
            // For other noise levels, use EMA or stopgrad version of network
            var targetNetwork = GetTargetNetwork();
            return targetNetwork.Predict(ConcatenateSigma(noisy, sigma));
        }
        
        /// <summary>
        /// Get teacher prediction for distillation
        /// </summary>
        private Tensor<double> GetTeacherPrediction(Tensor<double> noisy, double sigma, double targetSigma)
        {
            // Use teacher model to denoise from sigma to targetSigma
            // This is simplified - in practice would use ODE solver
            var predicted = teacherModel.Predict(ConcatenateSigma(noisy, sigma));
            
            // One-step denoising formula
            var alpha = targetSigma / sigma;
            return noisy.Multiply(alpha).Add(predicted.Multiply(1 - alpha));
        }
        
        /// <summary>
        /// Sample noise at given sigma level
        /// </summary>
        private Tensor<double> SampleNoise(int[] shape, double sigma, Random random)
        {
            var noise = GenerateNoise(shape, random);
            return noise.Multiply(sigma);
        }
        
        /// <summary>
        /// Add noise to reach target sigma level
        /// </summary>
        private Tensor<double> AddNoise(Tensor<double> x, double targetSigma, Random random)
        {
            var noise = GenerateNoise(x.Shape, random);
            return x.Add(noise.Multiply(targetSigma));
        }
        
        /// <summary>
        /// Concatenate sigma information to input
        /// </summary>
        private Tensor<double> ConcatenateSigma(Tensor<double> x, double sigma)
        {
            // In practice, use Fourier features or other conditioning
            // Simplified version - should properly concatenate
            return x;
        }
        
        /// <summary>
        /// Get target network (EMA or stop-gradient)
        /// </summary>
        private INeuralNetworkModel<double> GetTargetNetwork()
        {
            // In practice, use exponential moving average (EMA) of parameters
            // For now, return the same network with stop-gradient
            return consistencyFunction;
        }
        
        private double ComputeConsistencyLoss(Tensor<double> predicted, Tensor<double> target, double sigma)
        {
            // Huber loss for robustness
            var diff = predicted.Subtract(target);
            var threshold = 1.0;
            var loss = 0.0;
            
            foreach (var d in diff.Data)
            {
                if (Math.Abs(d) <= threshold)
                {
                    loss += 0.5 * d * d;
                }
                else
                {
                    loss += threshold * (Math.Abs(d) - 0.5 * threshold);
                }
            }
            
            return loss / diff.Data.Length;
        }
        
        private double ComputeDistillationLoss(Tensor<double> student, Tensor<double> teacher)
        {
            // L2 loss for distillation
            var diff = student.Subtract(teacher);
            var squared = diff.Multiply(diff);
            return squared.Data.Average();
        }
        
        private double GetLossWeight(double sigma)
        {
            // Weight loss by 1/sigma^2 for proper scaling
            return 1.0 / (sigma * sigma);
        }
        
        private Tensor<double> GenerateNoise(int[] shape, Random random)
        {
            var noise = new Tensor<double>(shape);
            var data = noise.Data;
            
            for (int i = 0; i < data.Length; i++)
            {
                var u1 = random.NextDouble();
                var u2 = random.NextDouble();
                data[i] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            }
            
            return noise;
        }
        
        public override Tensor<double> Forward(Tensor<double> input)
        {
            // For compatibility
            return Generate(input.Shape);
        }
        
        public override void Backward(Tensor<double> gradOutput)
        {
            // Backward is handled in training methods
        }
        
        protected override void SaveModelSpecificData(IDictionary<string, object> data)
        {
            data["sigmaMin"] = sigmaMin;
            data["sigmaMax"] = sigmaMax;
            data["numSteps"] = numSteps;
            data["rho"] = rho;
            data["scheduleType"] = scheduleType.ToString();
            data["useDistillation"] = useDistillation;
        }
        
        protected override void LoadModelSpecificData(IDictionary<string, object> data)
        {
            // Load model parameters
        }
    }
}