namespace AiDotNet.Enums;

/// <summary>Controls how descriptor values outside configured bounds are binned.</summary>
public enum EvolutionOutOfRangePolicy
{
    /// <summary>Reject the candidate when a descriptor is outside its configured range.</summary>
    Reject = 0,
    /// <summary>Clamp below-range and above-range values into the first and last bins.</summary>
    Clamp = 1,
    /// <summary>Reserve explicit bins below and above the configured range.</summary>
    OverflowBins = 2
}
