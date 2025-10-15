using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers
{
    /// <summary>
    /// Transformer block for Vision Transformer
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class TransformerBlock<T> : LayerBase<T>
    {
        private readonly MultiHeadAttention<T> attention;
        private readonly LayerNormalizationLayer<T> norm1;
        private readonly LayerNormalizationLayer<T> norm2;
        private readonly MLPBlock<T> mlp;
        private readonly double dropoutRate;
        
        public override int ParameterCount => 
            attention.ParameterCount + norm1.ParameterCount + norm2.ParameterCount + mlp.ParameterCount;
        public override bool SupportsTraining => true;
        
        public TransformerBlock(int embedDim, int numHeads, int mlpDim, double dropoutRate)
            : base([embedDim], [embedDim])
        {
            this.dropoutRate = dropoutRate;
            
            attention = new MultiHeadAttention<T>(embedDim, numHeads, dropoutRate);
            norm1 = new LayerNormalizationLayer<T>(embedDim);
            norm2 = new LayerNormalizationLayer<T>(embedDim);
            mlp = new MLPBlock<T>(embedDim, mlpDim, dropoutRate);
        }
        
        public override Tensor<T> Forward(Tensor<T> input)
        {
            // Self-attention with residual connection
            var attnOutput = attention.Forward(input, input, input);
            var x = norm1.Forward(input.Add(Dropout(attnOutput, dropoutRate)));
            
            // MLP with residual connection
            var mlpOutput = mlp.Forward(x);
            var output = norm2.Forward(x.Add(Dropout(mlpOutput, dropoutRate)));
            
            return output;
        }
        
        public override Tensor<T> Backward(Tensor<T> gradOutput)
        {
            // Backward through second residual block
            var gradNorm2 = norm2.Backward(gradOutput);
            var gradMlp = mlp.Backward(gradNorm2);
            var gradX = gradNorm2.Add(gradMlp);
            
            // Backward through first residual block
            var gradNorm1 = norm1.Backward(gradX);
            var gradAttn = attention.Backward(gradNorm1);
            var gradInput = gradNorm1.Add(gradAttn);
            
            return gradInput;
        }
        
        public override Vector<T> GetParameters()
        {
            var parameters = new List<Vector<T>>
            {
                attention.GetParameters(),
                norm1.GetParameters(),
                norm2.GetParameters(),
                mlp.GetParameters()
            };
            
            return Vector<T>.Concatenate(parameters.ToArray());
        }
        
        public override void UpdateParameters(Vector<T> parameters)
        {
            int index = 0;
            
            // Update attention parameters
            var attnParams = new Vector<T>(attention.ParameterCount);
            for (int i = 0; i < attention.ParameterCount; i++)
            {
                attnParams[i] = parameters[index + i];
            }
            attention.UpdateParameters(attnParams);
            index += attention.ParameterCount;
            
            // Update norm1 parameters
            var norm1Params = new Vector<T>(norm1.ParameterCount);
            for (int i = 0; i < norm1.ParameterCount; i++)
            {
                norm1Params[i] = parameters[index + i];
            }
            norm1.UpdateParameters(norm1Params);
            index += norm1.ParameterCount;
            
            // Update norm2 parameters
            var norm2Params = new Vector<T>(norm2.ParameterCount);
            for (int i = 0; i < norm2.ParameterCount; i++)
            {
                norm2Params[i] = parameters[index + i];
            }
            norm2.UpdateParameters(norm2Params);
            index += norm2.ParameterCount;
            
            // Update mlp parameters
            var mlpParams = new Vector<T>(mlp.ParameterCount);
            for (int i = 0; i < mlp.ParameterCount; i++)
            {
                mlpParams[i] = parameters[index + i];
            }
            mlp.UpdateParameters(mlpParams);
        }
        
        public override Vector<T> GetParameterGradients()
        {
            var gradients = new List<Vector<T>>
            {
                attention.GetParameterGradients(),
                norm1.GetParameterGradients(),
                norm2.GetParameterGradients(),
                mlp.GetParameterGradients()
            };
            
            return Vector<T>.Concatenate(gradients.ToArray());
        }
        
        private Tensor<T> Dropout(Tensor<T> input, double rate)
        {
            // Apply dropout during training
            // Simplified - should check training mode
            return input;
        }
        
        public override void UpdateParameters(T learningRate)
        {
            attention.UpdateParameters(learningRate);
            norm1.UpdateParameters(learningRate);
            norm2.UpdateParameters(learningRate);
            mlp.UpdateParameters(learningRate);
        }
        
        public override void ResetState()
        {
            attention.ResetState();
            norm1.ResetState();
            norm2.ResetState();
            mlp.ResetState();
        }
        
        public override void ClearGradients()
        {
            attention.ClearGradients();
            norm1.ClearGradients();
            norm2.ClearGradients();
            mlp.ClearGradients();
        }
    }
}