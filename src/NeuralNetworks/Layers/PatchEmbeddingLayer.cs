using System;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers
{
    /// <summary>
    /// Patch embedding layer for Vision Transformer
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class PatchEmbeddingLayer<T> : LayerBase<T>
    {
        private readonly int patchSize;
        private readonly int embedDim;
        private Tensor<T> weight;
        private Tensor<T> bias;
        private Tensor<T>? weightGradient;
        private Tensor<T>? biasGradient;
        private Tensor<T>? lastInput;
        
        public override int ParameterCount => weight.Length + bias.Length;
        public override bool SupportsTraining => true;
        
        public PatchEmbeddingLayer(int patchSize, int embedDim) 
            : base([3, 224, 224], [embedDim]) // Default input shape, will be adjusted
        {
            this.patchSize = patchSize;
            this.embedDim = embedDim;
            
            // Initialize as a linear projection of flattened patches
            var patchDim = patchSize * patchSize * 3; // 3 channels for RGB
            weight = new Tensor<T>(new[] { patchDim, embedDim });
            bias = new Tensor<T>(new[] { embedDim });
            
            InitializeWeights();
        }
        
        public override Tensor<T> Forward(Tensor<T> input)
        {
            lastInput = input;
            
            // Convert image to patches and flatten them
            var patches = ImageToPatches(input);
            
            // Linear projection
            var embedded = patches.MatrixMultiply(weight).Add(bias);
            
            return embedded;
        }
        
        public override Tensor<T> Backward(Tensor<T> gradOutput)
        {
            // Calculate weight and bias gradients
            if (lastInput != null)
            {
                var patches = ImageToPatches(lastInput);
                weightGradient = patches.Transpose().MatrixMultiply(gradOutput);
                biasGradient = gradOutput.Sum(new[] { 0 }); // Sum over batch dimension
            }
            
            // Backward through linear projection
            var gradPatches = gradOutput.MatrixMultiply(weight.Transpose());
            
            return PatchesToImage(gradPatches);
        }
        
        public override Vector<T> GetParameters()
        {
            var parameters = new Vector<T>(ParameterCount);
            int index = 0;
            
            // Copy weights
            for (int i = 0; i < weight.Length; i++)
            {
                parameters[index + i] = weight[i];
            }
            index += weight.Length;
            
            // Copy biases
            for (int i = 0; i < bias.Length; i++)
            {
                parameters[index + i] = bias[i];
            }
            
            return parameters;
        }
        
        public override void UpdateParameters(Vector<T> parameters)
        {
            int index = 0;
            
            // Update weights
            for (int i = 0; i < weight.Length; i++)
            {
                weight[i] = parameters[index + i];
            }
            index += weight.Length;
            
            // Update biases
            for (int i = 0; i < bias.Length; i++)
            {
                bias[i] = parameters[index + i];
            }
        }
        
        public override Vector<T> GetParameterGradients()
        {
            var gradients = new Vector<T>(ParameterCount);
            int index = 0;
            
            if (weightGradient != null)
            {
                for (int i = 0; i < weightGradient.Length; i++)
                {
                    gradients[index + i] = weightGradient[i];
                }
            }
            index += weight.Length;
            
            if (biasGradient != null)
            {
                for (int i = 0; i < biasGradient.Length; i++)
                {
                    gradients[index + i] = biasGradient[i];
                }
            }
            
            return gradients;
        }
        
        private Tensor<T> ImageToPatches(Tensor<T> image)
        {
            // Extract patches from image
            // Simplified implementation
            return image;
        }
        
        private Tensor<T> PatchesToImage(Tensor<T> patches)
        {
            // Reconstruct image from patches
            // Simplified implementation
            return patches;
        }
        
        private void InitializeWeights()
        {
            // Xavier/He initialization
            var scale = Math.Sqrt(2.0 / (weight.Shape[0] + weight.Shape[1]));
            var random = new Random();
            
            for (int i = 0; i < weight.Length; i++)
            {
                weight[i] = NumOps.FromDouble(random.NextDouble() * 2 * scale - scale);
            }
            
            // Initialize bias to zero
            for (int i = 0; i < bias.Length; i++)
            {
                bias[i] = NumOps.Zero;
            }
        }
        
        public override void UpdateParameters(T learningRate)
        {
            if (weightGradient != null && biasGradient != null)
            {
                // Update weights
                for (int i = 0; i < weight.Length; i++)
                {
                    weight[i] = NumOps.Subtract(weight[i], NumOps.Multiply(learningRate, weightGradient[i]));
                }
                
                // Update biases
                for (int i = 0; i < bias.Length; i++)
                {
                    bias[i] = NumOps.Subtract(bias[i], NumOps.Multiply(learningRate, biasGradient[i]));
                }
            }
        }
        
        public override void ResetState()
        {
            // This layer doesn't have internal state to reset
        }
        
        public override void ClearGradients()
        {
            weightGradient = null;
            biasGradient = null;
        }
    }
}