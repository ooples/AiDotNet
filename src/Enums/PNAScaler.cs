namespace AiDotNet.Enums;

/// <summary>
/// Scaler function types for Principal Neighbourhood Aggregation (PNA).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Scalers normalize aggregated features by node degree:
///
/// - **Identity**: No scaling (use raw aggregated values)
/// - **Amplification**: Scale up by degree/avgDegree (high-degree nodes get amplified)
/// - **Attenuation**: Scale down by avgDegree/degree (high-degree nodes get attenuated)
/// </para>
/// </remarks>
public enum PNAScaler
{
    /// <summary>Identity scaler - no scaling applied.</summary>
    Identity,

    /// <summary>Amplification scaler - amplifies signal from high-degree nodes.</summary>
    Amplification,

    /// <summary>Attenuation scaler - attenuates signal from high-degree nodes.</summary>
    Attenuation
}
