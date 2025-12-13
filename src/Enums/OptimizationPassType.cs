namespace AiDotNet.Enums;

/// <summary>
/// Represents the type of optimization pass applied to the computation graph.
/// </summary>
public enum OptimizationPassType
{
    // Operator Fusion Passes
    OperatorFusion,
    ConvBatchNormFusion,
    ConvBatchNormReLUFusion,
    MatMulBiasFusion,
    MatMulBiasActivationFusion,
    ElementwiseFusion,
    AttentionFusion,

    // Graph Structure Optimization
    ConstantFolding,
    DeadCodeElimination,
    CommonSubexpressionElimination,
    LayoutOptimization,

    // Memory Optimization
    InPlaceOptimization,
    MemoryReuseOptimization,
    ActivationCheckpointing,
    MemoryPlanning,

    // Computation Optimization
    AlgebraicSimplification,
    StrengthReduction,
    LoopFusion,
    VectorizationHints,

    // Quantization
    Int8Quantization,
    Float16Quantization,
    DynamicQuantization,

    // Other
    Custom
}
