using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Builds the policy network a policy's options class describes: <c>StateSize</c> inputs, one ReLU dense layer per
    /// <c>HiddenLayers</c> entry (the [256, 256] default is SAC's and TD3's actor), and a linear head of the size the
    /// policy reads, trained with the options' loss function.
    /// </summary>
    internal static class PolicyNetworks
    {
        internal static NeuralNetwork<T> Build<T>(int stateSize, int[] hiddenLayers, int outputSize,
            ILossFunction<T> lossFunction, string optionsName)
        {
            if (stateSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(stateSize), stateSize, optionsName + ".StateSize must be positive.");
            if (outputSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(outputSize), outputSize,
                    optionsName + " describes no action: its action sizes must be positive.");
            if (hiddenLayers is null) throw new ArgumentNullException(nameof(hiddenLayers), optionsName + ".HiddenLayers is null.");
            if (lossFunction is null) throw new ArgumentNullException(nameof(lossFunction), optionsName + ".LossFunction is null.");

            var layers = new List<ILayer<T>>(hiddenLayers.Length + 1);
            foreach (int width in hiddenLayers)
            {
                if (width <= 0)
                    throw new ArgumentOutOfRangeException(nameof(hiddenLayers), width,
                        optionsName + ".HiddenLayers holds a non-positive width.");
                layers.Add(new DenseLayer<T>(width, (IActivationFunction<T>)new ReLUActivation<T>()));
            }
            layers.Add(new DenseLayer<T>(outputSize, (IActivationFunction<T>)new IdentityActivation<T>()));

            var architecture = new NeuralNetworkArchitecture<T>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                complexity: NetworkComplexity.Medium,
                inputSize: stateSize,
                outputSize: outputSize,
                layers: layers);
            return new NeuralNetwork<T>(architecture, lossFunction: lossFunction);
        }

        /// <summary>The seeded generator when the options carry a seed, else null (the policy's secure default).</summary>
        internal static Random? RandomFor(int? seed) => seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : null;
    }
}
