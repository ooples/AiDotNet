using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>Shared public shape law for architecture-driven tabular neural networks.</summary>
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public abstract class TabularNeuralNetworkBase<T> : NeuralNetworkBase<T>, IShapeContract
{
    /// <summary>Initializes the common tabular network base.</summary>
    protected TabularNeuralNetworkBase(
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
