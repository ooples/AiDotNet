using AiDotNet.NeuralNetworks;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Autodiff.Transforms;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.PhysicsInformed;

/// <summary>
/// Exact first and second derivatives of a network's outputs with respect to its inputs, for physics-informed
/// losses (gradients, Laplacians, Hessians of the PDE residual).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Derivatives come from the AiDotNet.Tensors gradient tape (<see cref="TensorFunc{T}"/>): the gradient is
/// reverse-mode, and the Hessian is the Jacobian of a differentiable gradient (double backward), as
/// <c>torch.func.hessian</c> computes it. That works for any network whose forward runs through engine ops, not only
/// dense and activation stacks, and is exact rather than a finite-difference estimate.
/// </para>
/// <para><b>For Beginners:</b> A physics-informed network is trained so that its output satisfies an equation
/// involving its derivatives (for example the heat equation needs the second derivative in space). This class
/// computes those derivatives of the network itself, exactly.
/// </para>
/// </remarks>
public static class NeuralNetworkDerivatives<T>
{
    /// <summary>
    /// Computes the first derivatives <c>[outputDim, inputDim]</c> and second derivatives
    /// <c>[outputDim, inputDim, inputDim]</c> of the network at a single input point.
    /// </summary>
    /// <param name="network">The network.</param>
    /// <param name="inputs">The input point.</param>
    /// <param name="outputDim">How many network outputs to differentiate (the first <paramref name="outputDim"/>).</param>
    public static PDEDerivatives<T> ComputeDerivatives(NeuralNetworkBase<T> network, T[] inputs, int outputDim)
    {
        if (network is null) throw new ArgumentNullException(nameof(network));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (outputDim <= 0) throw new ArgumentOutOfRangeException(nameof(outputDim), "Output dimension must be positive.");

        int n = inputs.Length;
        var point = new Tensor<T>((T[])inputs.Clone(), new[] { n });
        var first = new T[outputDim, n];
        var second = new T[outputDim, n, n];
        for (int k = 0; k < outputDim; k++)
        {
            var component = Component(network, n, k);
            var gradient = TensorFunc<T>.Grad(component)(point).ToArray();
            var hessian = TensorFunc<T>.Hessian(component)(point).ToArray();
            for (int i = 0; i < n; i++)
            {
                first[k, i] = gradient[i];
                for (int j = 0; j < n; j++) second[k, i, j] = hessian[i * n + j];
            }
        }
        return new PDEDerivatives<T> { FirstDerivatives = first, SecondDerivatives = second };
    }

    /// <summary>Computes the Hessian <c>[inputDim, inputDim]</c> of one network output at a single input point.</summary>
    /// <param name="network">The network.</param>
    /// <param name="inputs">The input point.</param>
    /// <param name="outputIndex">Which output to differentiate.</param>
    public static T[,] ComputeHessian(NeuralNetworkBase<T> network, T[] inputs, int outputIndex = 0)
    {
        if (network is null) throw new ArgumentNullException(nameof(network));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        int n = inputs.Length;
        int outputs = OutputDimension(network, inputs);
        if (outputIndex < 0 || outputIndex >= outputs)
            throw new ArgumentOutOfRangeException(nameof(outputIndex),
                $"Output index must be in [0, {outputs - 1}] for a network with {outputs} outputs.");

        var hessian = TensorFunc<T>.Hessian(Component(network, n, outputIndex))(new Tensor<T>((T[])inputs.Clone(), new[] { n })).ToArray();
        var result = new T[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                result[i, j] = hessian[i * n + j];
        return result;
    }

    // x [n] -> network output k, as a one-element tensor (the scalar function the tape differentiates).
    private static Func<Tensor<T>, Tensor<T>> Component(NeuralNetworkBase<T> network, int n, int k) => x =>
    {
        var engine = Tensors.Engines.AiDotNetEngine.Current;
        var output = network.PredictWithGradients(engine.Reshape(x, new[] { 1, n }));
        int total = output.Length;
        if (k >= total)
            throw new ArgumentOutOfRangeException(nameof(k), $"The network has {total} outputs; output {k} was requested.");
        return engine.TensorNarrow(engine.Reshape(output, new[] { total }), 0, k, 1);
    };

    private static int OutputDimension(NeuralNetworkBase<T> network, T[] inputs)
    {
        var output = network.Predict(new Tensor<T>((T[])inputs.Clone(), new[] { 1, inputs.Length }));
        if (output.Rank == 2 && output.Shape[0] == 1) return output.Shape[1];
        if (output.Rank == 1) return output.Shape[0];
        throw new InvalidOperationException("Unexpected output tensor shape for derivative evaluation.");
    }
}
