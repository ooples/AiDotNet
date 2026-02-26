using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.SourceSeparation;

/// <summary>
/// Result of music source separation containing individual stems.
/// </summary>
/// <typeparam name="T">The numeric type used for audio samples.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> SeparationResult provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class SeparationResult<T>
{
    /// <summary>Isolated vocal track.</summary>
    public required Tensor<T> Vocals { get; init; }

    /// <summary>Isolated drums/percussion track.</summary>
    public required Tensor<T> Drums { get; init; }

    /// <summary>Isolated bass track.</summary>
    public required Tensor<T> Bass { get; init; }

    /// <summary>Other instruments (guitar, piano, etc.).</summary>
    public required Tensor<T> Other { get; init; }

    /// <summary>Sample rate of output stems.</summary>
    public int SampleRate { get; init; }

    /// <summary>Gets all stems as a dictionary.</summary>
    public Dictionary<string, Tensor<T>> ToDictionary()
    {
        return new Dictionary<string, Tensor<T>>
        {
            ["vocals"] = Vocals,
            ["drums"] = Drums,
            ["bass"] = Bass,
            ["other"] = Other
        };
    }
}
