namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Internal hook for providing constructor metadata needed to reliably round-trip layers through
/// NeuralNetworkBase serialization (used by Clone/DeepCopy).
/// </summary>
/// <remarks>
/// This is intentionally internal to avoid expanding the user-facing surface area.
/// </remarks>
internal interface ILayerSerializationMetadata
{
    Dictionary<string, string> GetSerializationMetadata();
}

