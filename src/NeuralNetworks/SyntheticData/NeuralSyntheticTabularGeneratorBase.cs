using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Shared caller-facing shape contract for neural synthetic-tabular generators.
/// </summary>
/// <remarks>
/// These models consume one feature vector or a batch of feature vectors and emit the configured
/// table width. Their internal generators differ, but that public geometry is common and is derived
/// from the architecture rather than copied into every implementation.
/// </remarks>
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public abstract class NeuralSyntheticTabularGeneratorBase<T> : NeuralNetworkBase<T>, IShapeContract
{
    /// <summary>Initializes the shared neural generator base.</summary>
    protected NeuralSyntheticTabularGeneratorBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T> lossFunction,
        double maxGradNorm)
        : base(architecture, lossFunction, maxGradNorm)
    {
    }

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => inputRank switch
        {
            1 =>
            [
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(Architecture.OutputSize)),
            ],
            2 =>
            [
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(Architecture.OutputSize)),
            ],
            _ => null,
        };
}
