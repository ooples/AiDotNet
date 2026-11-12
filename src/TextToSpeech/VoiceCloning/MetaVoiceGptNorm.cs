using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>MetaVoice's bias-free RMSNorm or LayerNorm, with one learned feature scale.</summary>
[LayerCategory(LayerCategory.Normalization)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, TestInputShape = "2, 8", TestConstructorArgs = "8, true, 1e-5")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal sealed partial class MetaVoiceGptNorm<T> : LayerBase<T>, IShapeContract
{
    private readonly int _dimension;
    private readonly bool _rms;
    private readonly double _epsilon;
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _gamma;
    public override bool SupportsTraining => true;

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 2
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Same(TensorAxis.Features))
        }
        : null;

    public MetaVoiceGptNorm([LayerState] int dimension, [LayerState] bool rms, [LayerState] double epsilon = 1e-5)
        : base(new[] { dimension }, new[] { dimension })
    {
        if (dimension <= 0 || epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(dimension));
        _dimension = dimension; _rms = rms; _epsilon = epsilon;
        _gamma = new Tensor<T>(new[] { dimension });
        for (int i = 0; i < dimension; i++) _gamma[i] = NumOps.One;
        RegisterTrainableParameter(_gamma, PersistentTensorRole.NormalizationParams);
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 2 || input.Shape[1] != _dimension)
            throw new ArgumentException("Expected [time, feature] rows.", nameof(input));
        int rows = input.Shape[0];
        var centered = _rms ? input : Engine.TensorSubtract(input,
            Engine.TensorTile(Engine.ReduceMean(input, new[] { 1 }, true), new[] { 1, _dimension }));
        var variance = Engine.ReduceMean(Engine.TensorMultiply(centered, centered), new[] { 1 }, true);
        var inverse = Engine.TensorPow(Engine.TensorAddScalar(variance, NumOps.FromDouble(_epsilon)), NumOps.FromDouble(-.5));
        return Engine.TensorMultiply(Engine.TensorMultiply(centered, Engine.TensorTile(inverse, new[] { 1, _dimension })),
            Engine.TensorTile(Engine.Reshape(_gamma, new[] { 1, _dimension }), new[] { rows, 1 }));
    }

    internal void Load(double[] values)
    {
        if (values.Length != _dimension) throw new ArgumentException("Normalization scale dimension differs.", nameof(values));
        for (int i = 0; i < values.Length; i++) _gamma[i] = NumOps.FromDouble(values[i]);
        Engine.InvalidatePersistentTensor(_gamma);
    }

    public override void ResetState() { }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Dimension"] = _dimension.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Rms"] = _rms.ToString();
        metadata["Epsilon"] = _epsilon.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
