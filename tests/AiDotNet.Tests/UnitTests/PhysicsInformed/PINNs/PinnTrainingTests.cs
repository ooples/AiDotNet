using System;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.PINNs;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed.PINNs;

public class PinnTrainingTests
{



    private static double EnergyFunctional(double[] x, double[] u, double[,] gradU)
    {
        double diff = u[0] - x[0];
        return diff * diff;
    }

    private static double WeakFormResidual(double[] x, double[] u, double[,] gradU, double[] v, double[,] gradV)
    {
        double diff = u[0] - x[0];
        return diff * v[0];
    }

    private static NeuralNetworkArchitecture<double> CreateLinearArchitecture(int inputSize, int outputSize)
    {
        var layers = new ILayer<double>[]
        {
            new DenseLayer<double>(outputSize, (IActivationFunction<double>)new IdentityActivation<double>())
        };

        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers.ToList());
    }

    private sealed class LinearResidualPde : IPDESpecification<double>, IPDEResidualGradient<double>
    {
        public double ComputeResidual(Vector<double> inputs, Vector<double> outputs, PDEDerivatives<double> derivatives)
        {
            return outputs[0] - inputs[0];
        }

        public int InputDimension => 1;
        public int OutputDimension => 1;
        public string Name => "LinearResidualTest";

        public PDEResidualGradient<double> ComputeResidualGradient(
            Vector<double> inputs,
            Vector<double> outputs,
            PDEDerivatives<double> derivatives)
        {
            var gradient = new PDEResidualGradient<double>(OutputDimension, InputDimension);
            gradient.OutputGradients[0] = 1.0;
            return gradient;
        }
    }
}
