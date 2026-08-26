namespace AiDotNet.Enums;

/// <summary>
/// Identifies the mathematical objective a trainable model actually optimizes.
/// </summary>
public enum TrainingObjectiveKind
{
    /// <summary>A caller-supplied target is optimized by a supervised learner.</summary>
    Supervised,

    /// <summary>The input is reconstructed by an unsupervised learner.</summary>
    Reconstruction,

    /// <summary>A contrastive-divergence energy objective is optimized.</summary>
    ContrastiveDivergence,

    /// <summary>Population fitness is optimized by an evolutionary learner.</summary>
    EvolutionaryFitness,

    /// <summary>A parameterized quantum-circuit energy is optimized.</summary>
    VariationalEnergy,

    /// <summary>A diffusion denoising or distillation objective is optimized.</summary>
    DiffusionDenoising,

    /// <summary>A zero-shot partition or energy objective is evaluated without supervised fitting.</summary>
    ZeroShotEnergy,

    /// <summary>
    /// A scalar Hamiltonian is differentiated with respect to phase-space coordinates and its
    /// symplectic derivative field is fitted to observed time derivatives.
    /// </summary>
    HamiltonianDynamics,

    /// <summary>Several explicitly structured task objectives are optimized together.</summary>
    MultiTask,
}
