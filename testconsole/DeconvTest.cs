using System;
using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;

namespace AiDotNetTestConsole
{
    public static class DeconvTest
    {
        public static void Run()
        {
            Console.WriteLine("Testing DeconvolutionalLayer...");

            // Input: Batch=1, Depth=2, H=4, W=4
            int batchSize = 1;
            int inputDepth = 2;
            int inputH = 4;
            int inputW = 4;
            var inputShape = new int[] { batchSize, inputDepth, inputH, inputW };
            var input = Tensor<double>.CreateRandom(inputShape);

            // Layer config
            int outputDepth = 2;
            int kernelSize = 3;
            int stride = 2;
            int padding = 1;

            var layer = new DeconvolutionalLayer<double>(new int[] { batchSize, inputDepth, inputH, inputW }, outputDepth, kernelSize, stride, padding, (IActivationFunction<double>)new AiDotNet.ActivationFunctions.ReLUActivation<double>());

            Console.WriteLine($"Input Shape: {string.Join(", ", input.Shape)}");
            Console.WriteLine($"Kernel Size: {kernelSize}, Stride: {stride}, Padding: {padding}");
            Console.WriteLine($"Expected Output H: {(inputH - 1) * stride - 2 * padding + kernelSize}");

            // Forward
            var output = layer.Forward(input);
            Console.WriteLine($"Output Shape: {string.Join(", ", output.Shape.ToArray())}");

            // Update
            layer.UpdateParameters(0.01);
            Console.WriteLine("Parameters updated successfully.");
        }
    }
}
