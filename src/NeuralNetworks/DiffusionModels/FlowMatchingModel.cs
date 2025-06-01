using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.DiffusionModels
{
    /// <summary>
    /// Flow Matching / Rectified Flow Model
    /// Learns straight trajectories between noise and data
    /// </summary>
    public class FlowMatchingModel : NeuralNetworkBase<double>
    {
        private readonly INeuralNetworkModel<double> velocityNetwork;
        private readonly FlowType flowType;
        private readonly double sigma;
        private readonly bool useOptimalTransport;
        private readonly int rectificationSteps;
        
        public enum FlowType
        {
            Linear,           // Simple linear interpolation
            OptimalTransport, // OT-based flow
            Rectified,       // Rectified flow (straightened trajectories)
            Conditional      // Conditional flow matching
        }
        
        public FlowMatchingModel(
            INeuralNetworkModel<double> velocityNetwork,
            FlowType flowType = FlowType.Rectified,
            double sigma = 0.01,
            bool useOptimalTransport = false,
            int rectificationSteps = 1,
            string modelName = "FlowMatchingModel")
            : base(modelName)
        {
            this.velocityNetwork = velocityNetwork ?? throw new ArgumentNullException(nameof(velocityNetwork));
            this.flowType = flowType;
            this.sigma = sigma;
            this.useOptimalTransport = useOptimalTransport;
            this.rectificationSteps = rectificationSteps;
            
            ModelCategory = ModelCategory.Generative;
        }
        
        /// <summary>
        /// Generate samples using learned flow
        /// </summary>
        public Tensor<double> Generate(int[] shape, int steps = 100, int? seed = null)
        {
            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            
            // Start from noise at t=0
            var x = SampleNoise(shape, random);
            
            // Integrate velocity field from t=0 to t=1
            var dt = 1.0 / steps;
            
            for (int i = 0; i < steps; i++)
            {
                var t = i * dt;
                var velocity = GetVelocity(x, t);
                x = x.Add(velocity.Multiply(dt));
            }
            
            return x;
        }
        
        /// <summary>
        /// Train the flow matching model
        /// </summary>
        public double TrainStep(Tensor<double> data, IOptimizer<double, Tensor<double>, Tensor<double>> optimizer)
        {
            var batchSize = data.Shape[0];
            var totalLoss = 0.0;
            var random = new Random();
            
            for (int i = 0; i < batchSize; i++)
            {
                var x1 = data.GetSlice(new[] { i });
                
                // Sample noise x0
                var x0 = SampleNoise(x1.Shape, random);
                
                // Sample time uniformly
                var t = random.NextDouble();
                
                // Get interpolated point and target velocity
                var (xt, targetVelocity) = GetTrainingPair(x0, x1, t);
                
                // Predict velocity
                var predictedVelocity = velocityNetwork.Predict(ConcatenateTime(xt, t));
                
                // Flow matching loss
                var loss = ComputeFlowMatchingLoss(predictedVelocity, targetVelocity);
                totalLoss += loss;
                
                // Backpropagate
                if (velocityNetwork is NeuralNetworkBase<double> nn)
                {
                    var grad = predictedVelocity.Subtract(targetVelocity);
                    nn.Backward(grad);
                    optimizer.Step(nn.GetParameters(), nn.GetGradients());
                }
            }
            
            return totalLoss / batchSize;
        }
        
        /// <summary>
        /// Rectify flow for straighter trajectories
        /// </summary>
        public void RectifyFlow(Tensor<double> data, IOptimizer<double, Tensor<double>, Tensor<double>> optimizer, int iterations)
        {
            for (int iter = 0; iter < iterations; iter++)
            {
                Console.WriteLine($"Rectification iteration {iter + 1}/{iterations}");
                
                // Generate paired data using current flow
                var pairedData = GeneratePairedData(data);
                
                // Train on straightened trajectories
                for (int epoch = 0; epoch < 100; epoch++)
                {
                    var loss = TrainOnPairedData(pairedData, optimizer);
                    
                    if (epoch % 10 == 0)
                    {
                        Console.WriteLine($"  Epoch {epoch}: Loss = {loss:F4}");
                    }
                }
            }
        }
        
        /// <summary>
        /// Conditional flow matching for controlled generation
        /// </summary>
        public Tensor<double> GenerateConditional(int[] shape, Tensor<double> condition, int steps = 100, int? seed = null)
        {
            if (flowType != FlowType.Conditional)
                throw new InvalidOperationException("Model not configured for conditional generation");
            
            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            
            // Start from noise
            var x = SampleNoise(shape, random);
            
            // Integrate conditional velocity field
            var dt = 1.0 / steps;
            
            for (int i = 0; i < steps; i++)
            {
                var t = i * dt;
                var velocity = GetConditionalVelocity(x, t, condition);
                x = x.Add(velocity.Multiply(dt));
            }
            
            return x;
        }
        
        /// <summary>
        /// Get training pair for flow matching
        /// </summary>
        private (Tensor<double> xt, Tensor<double> velocity) GetTrainingPair(Tensor<double> x0, Tensor<double> x1, double t)
        {
            switch (flowType)
            {
                case FlowType.Linear:
                    return GetLinearFlow(x0, x1, t);
                    
                case FlowType.OptimalTransport:
                    return GetOptimalTransportFlow(x0, x1, t);
                    
                case FlowType.Rectified:
                    return GetRectifiedFlow(x0, x1, t);
                    
                case FlowType.Conditional:
                    return GetConditionalFlow(x0, x1, t);
                    
                default:
                    return GetLinearFlow(x0, x1, t);
            }
        }
        
        /// <summary>
        /// Linear interpolation flow
        /// </summary>
        private (Tensor<double> xt, Tensor<double> velocity) GetLinearFlow(Tensor<double> x0, Tensor<double> x1, double t)
        {
            // Linear interpolation: x_t = (1-t)x_0 + t*x_1
            var xt = x0.Multiply(1 - t).Add(x1.Multiply(t));
            
            // Constant velocity: v = x_1 - x_0
            var velocity = x1.Subtract(x0);
            
            // Add small noise for regularization
            if (sigma > 0)
            {
                var noise = GenerateNoise(xt.Shape, new Random());
                xt = xt.Add(noise.Multiply(sigma));
            }
            
            return (xt, velocity);
        }
        
        /// <summary>
        /// Optimal transport flow
        /// </summary>
        private (Tensor<double> xt, Tensor<double> velocity) GetOptimalTransportFlow(Tensor<double> x0, Tensor<double> x1, double t)
        {
            if (!useOptimalTransport)
                return GetLinearFlow(x0, x1, t);
            
            // In practice, would use Sinkhorn algorithm or other OT solver
            // For now, use linear flow as approximation
            return GetLinearFlow(x0, x1, t);
        }
        
        /// <summary>
        /// Rectified flow (straightened trajectories)
        /// </summary>
        private (Tensor<double> xt, Tensor<double> velocity) GetRectifiedFlow(Tensor<double> x0, Tensor<double> x1, double t)
        {
            // Start with linear flow
            var (xt, velocity) = GetLinearFlow(x0, x1, t);
            
            // Apply rectification if trained
            if (rectificationSteps > 0)
            {
                // Use learned velocity field for straighter paths
                velocity = GetVelocity(xt, t);
            }
            
            return (xt, velocity);
        }
        
        /// <summary>
        /// Conditional flow
        /// </summary>
        private (Tensor<double> xt, Tensor<double> velocity) GetConditionalFlow(Tensor<double> x0, Tensor<double> x1, double t)
        {
            // Similar to linear flow but velocity depends on condition
            var xt = x0.Multiply(1 - t).Add(x1.Multiply(t));
            var velocity = x1.Subtract(x0);
            
            return (xt, velocity);
        }
        
        /// <summary>
        /// Get velocity at given point and time
        /// </summary>
        private Tensor<double> GetVelocity(Tensor<double> x, double t)
        {
            return velocityNetwork.Predict(ConcatenateTime(x, t));
        }
        
        /// <summary>
        /// Get conditional velocity
        /// </summary>
        private Tensor<double> GetConditionalVelocity(Tensor<double> x, double t, Tensor<double> condition)
        {
            // Concatenate x, t, and condition
            var input = ConcatenateTimeAndCondition(x, t, condition);
            return velocityNetwork.Predict(input);
        }
        
        /// <summary>
        /// Generate paired data for rectification
        /// </summary>
        private List<(Tensor<double> x0, Tensor<double> x1)> GeneratePairedData(Tensor<double> data)
        {
            var pairs = new List<(Tensor<double>, Tensor<double>)>();
            var batchSize = data.Shape[0];
            
            for (int i = 0; i < batchSize; i++)
            {
                var x1 = data.GetSlice(new[] { i });
                
                // Generate x0 by integrating backward
                var x0 = IntegrateBackward(x1);
                
                pairs.Add((x0, x1));
            }
            
            return pairs;
        }
        
        /// <summary>
        /// Integrate velocity field backward to get noise
        /// </summary>
        private Tensor<double> IntegrateBackward(Tensor<double> x1)
        {
            var x = x1.Clone();
            var steps = 100;
            var dt = 1.0 / steps;
            
            // Integrate from t=1 to t=0
            for (int i = steps - 1; i >= 0; i--)
            {
                var t = i * dt;
                var velocity = GetVelocity(x, t);
                x = x.Subtract(velocity.Multiply(dt));
            }
            
            return x;
        }
        
        /// <summary>
        /// Train on paired data for rectification
        /// </summary>
        private double TrainOnPairedData(List<(Tensor<double> x0, Tensor<double> x1)> pairedData, IOptimizer<double, Tensor<double>, Tensor<double>> optimizer)
        {
            var totalLoss = 0.0;
            var random = new Random();
            
            foreach (var (x0, x1) in pairedData)
            {
                var t = random.NextDouble();
                var (xt, targetVelocity) = GetLinearFlow(x0, x1, t);
                
                var predictedVelocity = velocityNetwork.Predict(ConcatenateTime(xt, t));
                var loss = ComputeFlowMatchingLoss(predictedVelocity, targetVelocity);
                totalLoss += loss;
                
                if (velocityNetwork is NeuralNetworkBase<double> nn)
                {
                    var grad = predictedVelocity.Subtract(targetVelocity);
                    nn.Backward(grad);
                    optimizer.Step(nn.GetParameters(), nn.GetGradients());
                }
            }
            
            return totalLoss / pairedData.Count;
        }
        
        private double ComputeFlowMatchingLoss(Tensor<double> predicted, Tensor<double> target)
        {
            var diff = predicted.Subtract(target);
            var squared = diff.Multiply(diff);
            return squared.Data.Average();
        }
        
        private Tensor<double> SampleNoise(int[] shape, Random random)
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
        
        private Tensor<double> GenerateNoise(int[] shape, Random random)
        {
            return SampleNoise(shape, random);
        }
        
        private Tensor<double> ConcatenateTime(Tensor<double> x, double t)
        {
            // In practice, use time embeddings
            // Simplified - should properly concatenate
            return x;
        }
        
        private Tensor<double> ConcatenateTimeAndCondition(Tensor<double> x, double t, Tensor<double> condition)
        {
            // In practice, properly concatenate x, time embedding, and condition
            return x;
        }
        
        public override Tensor<double> Forward(Tensor<double> input)
        {
            return Generate(input.Shape);
        }
        
        public override void Backward(Tensor<double> gradOutput)
        {
            // Backward is handled in TrainStep
        }
        
        protected override void SaveModelSpecificData(IDictionary<string, object> data)
        {
            data["flowType"] = flowType.ToString();
            data["sigma"] = sigma;
            data["useOptimalTransport"] = useOptimalTransport;
            data["rectificationSteps"] = rectificationSteps;
        }
        
        protected override void LoadModelSpecificData(IDictionary<string, object> data)
        {
            // Load model parameters
        }
    }
}