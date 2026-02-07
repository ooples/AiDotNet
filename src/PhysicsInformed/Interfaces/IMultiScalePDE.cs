using System;

namespace AiDotNet.PhysicsInformed.Interfaces
{
    /// <summary>
    /// Defines the interface for multi-scale Partial Differential Equations.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.) used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Multi-scale PDEs describe phenomena that occur across multiple length or time scales.
    /// Examples include:
    /// - Turbulent flows (large eddies to small vortices)
    /// - Materials science (atomic to macroscopic behavior)
    /// - Climate modeling (local weather to global patterns)
    /// - Biological systems (molecular to tissue level)
    ///
    /// Why Multi-scale is Challenging:
    /// Traditional single-scale methods struggle because:
    /// 1. Fine scale requires tiny mesh elements (expensive)
    /// 2. Coarse scale misses important details
    /// 3. Different scales have different time dynamics
    ///
    /// Multi-scale Solution Strategy:
    /// 1. Decompose solution into scale components: u = u_coarse + u_fine
    /// 2. Learn each scale with appropriate resolution
    /// 3. Couple scales through cross-scale interactions
    /// 4. Use appropriate loss weights for each scale
    ///
    /// Key Concepts:
    /// - Characteristic Length Scale: The typical size of features at each scale
    /// - Scale Separation: When scales are well-separated (e.g., 10x ratio)
    /// - Cross-scale Coupling: How different scales interact
    /// - Homogenization: Averaging fine-scale behavior for coarse-scale equations
    /// </remarks>
    public interface IMultiScalePDE<T> : IPDESpecification<T>
    {
        /// <summary>
        /// Gets the number of scales in the problem.
        /// </summary>
        /// <remarks>
        /// For Beginners:
        /// This indicates how many distinct length/time scales are present.
        /// For example:
        /// - 2 scales: macro + micro (common in composite materials)
        /// - 3 scales: macro + meso + micro (turbulence modeling)
        /// </remarks>
        int NumberOfScales { get; }

        /// <summary>
        /// Gets the characteristic length scales of the problem.
        /// </summary>
        /// <remarks>
        /// For Beginners:
        /// Characteristic lengths define the "size" of each scale.
        /// Example for a composite material:
        /// - ScaleCharacteristics[0] = 1.0 (macroscopic, meters)
        /// - ScaleCharacteristics[1] = 0.001 (fiber scale, millimeters)
        /// - ScaleCharacteristics[2] = 0.000001 (molecular scale, micrometers)
        /// </remarks>
        T[] ScaleCharacteristicLengths { get; }

        /// <summary>
        /// Computes the PDE residual at a specific scale.
        /// </summary>
        /// <param name="scaleIndex">The index of the scale (0 = coarsest, increasing = finer).</param>
        /// <param name="inputs">The spatial and temporal coordinates.</param>
        /// <param name="outputs">The predicted solution values at this scale.</param>
        /// <param name="derivatives">The derivatives of the solution at this scale.</param>
        /// <returns>The PDE residual at this scale.</returns>
        T ComputeScaleResidual(int scaleIndex, T[] inputs, T[] outputs, PDEDerivatives<T> derivatives);

        /// <summary>
        /// Computes the coupling term between two scales.
        /// </summary>
        /// <param name="coarseIndex">Index of the coarser scale.</param>
        /// <param name="fineIndex">Index of the finer scale.</param>
        /// <param name="inputs">The coordinates.</param>
        /// <param name="coarseOutputs">Solution at the coarse scale.</param>
        /// <param name="fineOutputs">Solution at the fine scale.</param>
        /// <param name="coarseDerivatives">Derivatives at the coarse scale.</param>
        /// <param name="fineDerivatives">Derivatives at the fine scale.</param>
        /// <returns>The coupling residual (should be zero when scales are properly coupled).</returns>
        /// <remarks>
        /// For Beginners:
        /// Coupling terms ensure consistency between scales.
        /// For example, in homogenization:
        /// - Coarse scale "sees" averaged effect of fine scale
        /// - Fine scale is modulated by coarse scale gradient
        ///
        /// Mathematically:
        /// - Upscaling: Average fine-scale solution matches coarse-scale solution
        /// - Downscaling: Fine-scale boundary conditions come from coarse-scale solution
        /// </remarks>
        T ComputeScaleCoupling(
            int coarseIndex,
            int fineIndex,
            T[] inputs,
            T[] coarseOutputs,
            T[] fineOutputs,
            PDEDerivatives<T> coarseDerivatives,
            PDEDerivatives<T> fineDerivatives);

        /// <summary>
        /// Gets the recommended loss weight for each scale.
        /// </summary>
        /// <param name="scaleIndex">The scale index.</param>
        /// <returns>Recommended weight for this scale's loss contribution.</returns>
        /// <remarks>
        /// For Beginners:
        /// Different scales may need different loss weights because:
        /// - Coarse scale dominates in magnitude
        /// - Fine scale captures important details
        /// - Imbalanced gradients can cause training difficulties
        ///
        /// Common strategies:
        /// - Weight proportional to 1/scale_length (finer scales get higher weights)
        /// - Adaptive weighting based on gradient magnitudes
        /// </remarks>
        T GetScaleLossWeight(int scaleIndex);

        /// <summary>
        /// Gets the output dimension for a specific scale.
        /// </summary>
        /// <param name="scaleIndex">The scale index.</param>
        /// <returns>Number of output components at this scale.</returns>
        /// <remarks>
        /// Different scales may have different output dimensions.
        /// For example, in turbulence:
        /// - Coarse scale: mean velocity (3 components)
        /// - Fine scale: velocity fluctuations (3 components) + Reynolds stresses (6 components)
        /// </remarks>
        int GetScaleOutputDimension(int scaleIndex);
    }

    /// <summary>
    /// Provides gradient information for multi-scale PDE residuals.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    public interface IMultiScalePDEGradient<T> : IMultiScalePDE<T>
    {
        /// <summary>
        /// Computes gradients of the scale residual with respect to outputs and derivatives.
        /// </summary>
        /// <param name="scaleIndex">The scale index.</param>
        /// <param name="inputs">The coordinates.</param>
        /// <param name="outputs">The solution at this scale.</param>
        /// <param name="derivatives">The derivatives at this scale.</param>
        /// <returns>Gradients for backpropagation.</returns>
        PDEResidualGradient<T> ComputeScaleResidualGradient(
            int scaleIndex,
            T[] inputs,
            T[] outputs,
            PDEDerivatives<T> derivatives);

        /// <summary>
        /// Computes gradients of the coupling term.
        /// </summary>
        /// <param name="coarseIndex">The coarser scale index.</param>
        /// <param name="fineIndex">The finer scale index.</param>
        /// <param name="inputs">The coordinates.</param>
        /// <param name="coarseOutputs">Solution at coarse scale.</param>
        /// <param name="fineOutputs">Solution at fine scale.</param>
        /// <param name="coarseDerivatives">Derivatives at coarse scale.</param>
        /// <param name="fineDerivatives">Derivatives at fine scale.</param>
        /// <returns>Tuple of gradients for (coarse scale, fine scale).</returns>
        (PDEResidualGradient<T> coarseGradient, PDEResidualGradient<T> fineGradient) ComputeScaleCouplingGradient(
            int coarseIndex,
            int fineIndex,
            T[] inputs,
            T[] coarseOutputs,
            T[] fineOutputs,
            PDEDerivatives<T> coarseDerivatives,
            PDEDerivatives<T> fineDerivatives);
    }

    /// <summary>
    /// Configuration options for multi-scale PINN training.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    public class MultiScaleTrainingOptions<T> : AiDotNet.Models.Options.ModelOptions
    {
        /// <summary>
        /// Whether to use adaptive scale weighting during training.
        /// </summary>
        public bool UseAdaptiveScaleWeighting { get; set; } = true;

        /// <summary>
        /// Whether to train scales sequentially (coarse to fine) or simultaneously.
        /// </summary>
        /// <remarks>
        /// Sequential training can be more stable:
        /// 1. First train coarse scale until convergence
        /// 2. Then add fine scale and continue training
        /// This is called "progressive training" or "curriculum learning".
        /// </remarks>
        public bool UseSequentialScaleTraining { get; set; } = false;

        /// <summary>
        /// Number of epochs to pre-train each scale before adding the next.
        /// </summary>
        public int ScalePretrainingEpochs { get; set; } = 100;

        /// <summary>
        /// Coupling loss weight (balances scale coupling vs individual scale losses).
        /// </summary>
        public T? CouplingWeight { get; set; }

        /// <summary>
        /// Individual weights for each scale (overrides automatic weighting).
        /// </summary>
        public T[]? ManualScaleWeights { get; set; }
    }
}
