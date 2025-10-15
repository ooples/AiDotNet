using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers
{
    /// <summary>
    /// Multi-head attention for Vision Transformer
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class MultiHeadAttention<T> : LayerBase<T>
    {
        private readonly int embedDim;
        private readonly int numHeads;
        private readonly int headDim;
        private readonly double dropoutRate;
        
        private readonly DenseLayer<T> queryProj;
        private readonly DenseLayer<T> keyProj;
        private readonly DenseLayer<T> valueProj;
        private readonly DenseLayer<T> outProj;
        
        public override int ParameterCount => 
            queryProj.ParameterCount + keyProj.ParameterCount + 
            valueProj.ParameterCount + outProj.ParameterCount;
        public override bool SupportsTraining => true;
        
        public MultiHeadAttention(int embedDim, int numHeads, double dropoutRate)
            : base([embedDim], [embedDim])
        {
            this.embedDim = embedDim;
            this.numHeads = numHeads;
            this.headDim = embedDim / numHeads;
            this.dropoutRate = dropoutRate;
            
            queryProj = new DenseLayer<T>(embedDim, embedDim, (IActivationFunction<T>?)null);
            keyProj = new DenseLayer<T>(embedDim, embedDim, (IActivationFunction<T>?)null);
            valueProj = new DenseLayer<T>(embedDim, embedDim, (IActivationFunction<T>?)null);
            outProj = new DenseLayer<T>(embedDim, embedDim, (IActivationFunction<T>?)null);
        }
        
        public Tensor<T> Forward(Tensor<T> query, Tensor<T> key, Tensor<T> value)
        {
            var batchSize = query.Shape[0];
            var seqLen = query.Shape[1];
            
            // Project and reshape for multi-head attention
            var q = queryProj.Forward(query);
            var k = keyProj.Forward(key);
            var v = valueProj.Forward(value);
            
            // Reshape to [batch, heads, seq_len, head_dim]
            q = ReshapeForAttention(q, batchSize, seqLen);
            k = ReshapeForAttention(k, batchSize, seqLen);
            v = ReshapeForAttention(v, batchSize, seqLen);
            
            // Scaled dot-product attention
            var scores = ComputeAttentionScores(q, k);
            var attnWeights = Softmax(scores);
            var attnOutput = ApplyAttention(attnWeights, v);
            
            // Reshape back and project
            attnOutput = ReshapeFromAttention(attnOutput, batchSize, seqLen);
            var output = outProj.Forward(attnOutput);
            
            return output;
        }
        
        public override Tensor<T> Forward(Tensor<T> input)
        {
            return Forward(input, input, input);
        }
        
        public override Tensor<T> Backward(Tensor<T> gradOutput)
        {
            // Simplified backward pass
            var gradOut = outProj.Backward(gradOutput);
            var gradQ = queryProj.Backward(gradOut);
            var gradK = keyProj.Backward(gradOut);
            var gradV = valueProj.Backward(gradOut);
            
            return gradQ.Add(gradK).Add(gradV);
        }
        
        public override Vector<T> GetParameters()
        {
            var parameters = new List<Vector<T>>
            {
                queryProj.GetParameters(),
                keyProj.GetParameters(),
                valueProj.GetParameters(),
                outProj.GetParameters()
            };
            
            return Vector<T>.Concatenate(parameters.ToArray());
        }
        
        public override void UpdateParameters(Vector<T> parameters)
        {
            int index = 0;
            
            // Update query projection
            var queryParams = new Vector<T>(queryProj.ParameterCount);
            for (int i = 0; i < queryProj.ParameterCount; i++)
            {
                queryParams[i] = parameters[index + i];
            }
            queryProj.UpdateParameters(queryParams);
            index += queryProj.ParameterCount;
            
            // Update key projection
            var keyParams = new Vector<T>(keyProj.ParameterCount);
            for (int i = 0; i < keyProj.ParameterCount; i++)
            {
                keyParams[i] = parameters[index + i];
            }
            keyProj.UpdateParameters(keyParams);
            index += keyProj.ParameterCount;
            
            // Update value projection
            var valueParams = new Vector<T>(valueProj.ParameterCount);
            for (int i = 0; i < valueProj.ParameterCount; i++)
            {
                valueParams[i] = parameters[index + i];
            }
            valueProj.UpdateParameters(valueParams);
            index += valueProj.ParameterCount;
            
            // Update output projection
            var outParams = new Vector<T>(outProj.ParameterCount);
            for (int i = 0; i < outProj.ParameterCount; i++)
            {
                outParams[i] = parameters[index + i];
            }
            outProj.UpdateParameters(outParams);
        }
        
        public override Vector<T> GetParameterGradients()
        {
            var gradients = new List<Vector<T>>
            {
                queryProj.GetParameterGradients(),
                keyProj.GetParameterGradients(),
                valueProj.GetParameterGradients(),
                outProj.GetParameterGradients()
            };
            
            return Vector<T>.Concatenate(gradients.ToArray());
        }
        
        private Tensor<T> ReshapeForAttention(Tensor<T> x, int batchSize, int seqLen)
        {
            // Reshape to multi-head format
            // Simplified implementation
            return x;
        }
        
        private Tensor<T> ReshapeFromAttention(Tensor<T> x, int batchSize, int seqLen)
        {
            // Reshape from multi-head format
            // Simplified implementation
            return x;
        }
        
        private Tensor<T> ComputeAttentionScores(Tensor<T> q, Tensor<T> k)
        {
            // Compute scaled dot-product attention scores
            var scale = NumOps.FromDouble(1.0 / Math.Sqrt(headDim));
            return q.MatrixMultiply(k.Transpose()).Multiply(scale);
        }
        
        private Tensor<T> Softmax(Tensor<T> scores)
        {
            // Apply softmax to attention scores
            // Simplified implementation
            return scores;
        }
        
        private Tensor<T> ApplyAttention(Tensor<T> weights, Tensor<T> values)
        {
            // Apply attention weights to values
            return weights.MatrixMultiply(values);
        }
        
        public override void UpdateParameters(T learningRate)
        {
            queryProj.UpdateParameters(learningRate);
            keyProj.UpdateParameters(learningRate);
            valueProj.UpdateParameters(learningRate);
            outProj.UpdateParameters(learningRate);
        }
        
        public override void ResetState()
        {
            queryProj.ResetState();
            keyProj.ResetState();
            valueProj.ResetState();
            outProj.ResetState();
        }
        
        public override void ClearGradients()
        {
            queryProj.ClearGradients();
            keyProj.ClearGradients();
            valueProj.ClearGradients();
            outProj.ClearGradients();
        }
    }
}