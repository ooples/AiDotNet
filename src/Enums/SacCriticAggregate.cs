namespace AiDotNet.Enums;

/// <summary>
/// How a SAC policy update combines its two soft Q-functions.
/// </summary>
public enum SacCriticAggregate
{
    /// <summary>The smaller of the two, as both SAC papers specify for the policy gradient (2018a §5, 2018b §6).</summary>
    Minimum = 0,

    /// <summary>The first Q-function alone, as the 2018a authors' code (haarnoja/sac) computes the reparameterized loss.</summary>
    FirstCritic = 1,

    /// <summary>The average of the two, as softlearning (the 2018b authors' code) computes the policy loss.</summary>
    Mean = 2,
}
