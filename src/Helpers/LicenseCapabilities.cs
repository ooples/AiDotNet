namespace AiDotNet.Helpers;

/// <summary>
/// Canonical capability identifiers used by v2 license gating. These strings are the source of truth
/// shared by the server (returned in the online <c>capabilities[]</c> response), the signed offline
/// token <c>caps</c> claim, and the client-side guards that enforce them.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A capability is a named permission a license grants — e.g. "you may save
/// models". The paid tiers grant more capabilities than the free/community tier; the guards check for the
/// specific capability an operation needs. Loading is deliberately NOT gated (free tiers can load); the
/// paid value is in saving/persisting and protecting (encrypting) trained models.</para>
/// </remarks>
internal static class LicenseCapabilities
{
    /// <summary>No specific capability required — any <c>Active</c> license (or the trial) suffices.
    /// Used for load, which is not a paid gate.</summary>
    internal const string None = "";

    /// <summary>Permission to save/persist a model (paid gate).</summary>
    internal const string ModelSave = "model:save";

    /// <summary>Permission to load a model.</summary>
    internal const string ModelLoad = "model:load";

    /// <summary>Permission to save tensors (paid gate; tensor layer).</summary>
    internal const string TensorsSave = "tensors:save";

    /// <summary>Permission to load tensors.</summary>
    internal const string TensorsLoad = "tensors:load";

    /// <summary>Permission to write encrypted model files / IP-protected artifacts (paid gate).</summary>
    internal const string ModelEncrypt = "model:encrypt";

    /// <summary>Permission for air-gapped / fully-offline operation (enterprise gate).</summary>
    internal const string Offline = "offline";
}
