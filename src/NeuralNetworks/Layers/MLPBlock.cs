using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers
{
    /// <summary>
    /// MLP block for transformer
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class MLPBlock<T> : LayerBase<T>
    {
        private readonly DenseLayer<T> fc1;
        private readonly DenseLayer<T> fc2;
        private readonly double dropoutRate;
        
        public override int ParameterCount => fc1.ParameterCount + fc2.ParameterCount;
        public override bool SupportsTraining => true;
        
        public MLPBlock(int embedDim, int mlpDim, double dropoutRate)
            : base([embedDim], [embedDim])
        {
            this.dropoutRate = dropoutRate;
            fc1 = new DenseLayer<T>(embedDim, mlpDim, (IActivationFunction<T>)new GELUActivation<T>());
            fc2 = new DenseLayer<T>(mlpDim, embedDim, (IActivationFunction<T>?)null);
        }
        
        public override Tensor<T> Forward(Tensor<T> input)
        {
            var x = fc1.Forward(input);
            x = Dropout(x, dropoutRate);
            x = fc2.Forward(x);
            x = Dropout(x, dropoutRate);
            return x;
        }
        
        public override Tensor<T> Backward(Tensor<T> gradOutput)
        {
            var grad = fc2.Backward(gradOutput);
            grad = fc1.Backward(grad);
            return grad;
        }
        
        public override Vector<T> GetParameters()
        {
            return Vector<T>.Concatenate(fc1.GetParameters(), fc2.GetParameters());
        }
        
        public override void UpdateParameters(Vector<T> parameters)
        {
            int index = 0;
            
            // Update fc1 parameters
            var fc1Params = new Vector<T>(fc1.ParameterCount);
            for (int i = 0; i < fc1.ParameterCount; i++)
            {
                fc1Params[i] = parameters[index + i];
            }
            fc1.UpdateParameters(fc1Params);
            index += fc1.ParameterCount;
            
            // Update fc2 parameters
            var fc2Params = new Vector<T>(fc2.ParameterCount);
            for (int i = 0; i < fc2.ParameterCount; i++)
            {
                fc2Params[i] = parameters[index + i];
            }
            fc2.UpdateParameters(fc2Params);
        }
        
        public override Vector<T> GetParameterGradients()
        {
            return Vector<T>.Concatenate(fc1.GetParameterGradients(), fc2.GetParameterGradients());
        }
        
        private Tensor<T> Dropout(Tensor<T> input, double rate)
        {
            // Apply dropout during training
            // Simplified - should check training mode
            return input;
        }
        
        public override void UpdateParameters(T learningRate)
        {
            fc1.UpdateParameters(learningRate);
            fc2.UpdateParameters(learningRate);
        }
        
        public override void ResetState()
        {
            fc1.ResetState();
            fc2.ResetState();
        }
        
        public override void ClearGradients()
        {
            fc1.ClearGradients();
            fc2.ClearGradients();
        }
    }
}