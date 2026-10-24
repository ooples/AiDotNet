using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// One invertible step of a normalizing flow over a <c>[batch, channels, time]</c> sequence.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IInvertibleFlowStep<T>
{
    /// <summary>
    /// Applies the step (<paramref name="reverse"/> false: data → latent) or its inverse (true: latent → data).
    /// </summary>
    /// <returns>The output, and in the forward direction the log-determinant of the Jacobian summed over the batch
    /// (a scalar tensor on the gradient tape); null in reverse.</returns>
    (Tensor<T> Output, Tensor<T>? LogDeterminant) Transform(Tensor<T> input, bool reverse);
}
