namespace AiDotNet.Enums;

/// <summary>
/// ABME's training stage (Park et al. 2021, Sec. 4.1): the motion networks train in turn, then
/// everything end to end on the synthesized frame.
/// </summary>
public enum AbmeTrainingStage
{
    /// <summary>The symmetric bilateral motion estimator (SBMENet), on the photometric loss of its warped frames.</summary>
    SymmetricMotion,

    /// <summary>The asymmetric bilateral motion refiner (ABMRNet), with the estimator fixed.</summary>
    AsymmetricMotion,

    /// <summary>The synthesis network trained end to end with both motion networks, on the synthesized frame.</summary>
    Synthesis,
}
