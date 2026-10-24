using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The trained values of one GAN vocoder network, for tests that check which network a step updated.
/// </summary>
/// <remarks>
/// Reads each layer's trainable tensors rather than slicing <c>GetParameters()</c>: that vector also carries buffers
/// such as a normalized convolution's power-iteration vectors <c>u</c> and <c>v</c>, which no optimizer step moves.
/// The discriminators' convolutions put those buffers last, so the model vector's tail is not a discriminator weight:
/// a check on it failed for a discriminator that trained, and passed for one that did not.
/// </remarks>
internal static class GanVocoderWeights
{
    public static double[] Of(IReadOnlyList<LayerBase<double>> layers)
        => layers.SelectMany(layer => layer.GetTrainableParameters()).SelectMany(tensor => tensor.ToArray()).ToArray();

    public static bool AnyChanged(double[] before, double[] after)
        => before.Length == after.Length && before.Zip(after, (a, b) => a != b).Any(changed => changed);
}
