namespace AiDotNet.Deployment.Mobile.Android;

/// <summary>
/// NNAPI execution preferences.
/// </summary>
public enum NNAPIExecutionPreference
{
    /// <summary>Default preference (let the system decide)</summary>
    Default,

    /// <summary>Prefer fast single answer</summary>
    FastSingleAnswer,

    /// <summary>Prefer sustained speed</summary>
    SustainedSpeed,

    /// <summary>Prefer low power consumption</summary>
    LowPower
}
