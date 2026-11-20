namespace AiDotNet.Enums;

/// <summary>
/// The attack ensemble that AutoAttack runs (Croce and Hein 2020).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The standard version runs four complementary attacks once each. The plus
/// version adds untargeted variants and more random restarts; it is slower but slightly stronger.</para>
/// </remarks>
public enum AutoAttackVersion
{
    /// <summary>
    /// APGD-CE, APGD-T (targeted DLR), FAB-T and Square, one restart each (the paper's standard AutoAttack).
    /// </summary>
    Standard,

    /// <summary>
    /// APGD-CE and APGD-DLR with 5 restarts, FAB with 5 restarts, Square, APGD-T and FAB-T (AutoAttack+).
    /// </summary>
    Plus
}
