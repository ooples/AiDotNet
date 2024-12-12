using AiDotNet.Normalizers;

namespace AiDotNet.Factories;

public class NormalizerFactory<T>
{
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    public static INormalizer<T> CreateNormalizer(NormalizationMethod method, T? lpNormP = default)
    {
        return method switch
        {
            NormalizationMethod.None => new NoNormalizer<T>(),
            NormalizationMethod.MinMax => new MinMaxNormalizer<T>(),
            NormalizationMethod.ZScore => new ZScoreNormalizer<T>(),
            NormalizationMethod.RobustScaling => new RobustScalingNormalizer<T>(),
            NormalizationMethod.Decimal => new DecimalNormalizer<T>(),
            NormalizationMethod.Binning => new BinningNormalizer<T>(),
            NormalizationMethod.MeanVariance => new MeanVarianceNormalizer<T>(),
            NormalizationMethod.LogMeanVariance => new LogMeanVarianceNormalizer<T>(),
            NormalizationMethod.GlobalContrast => new GlobalContrastNormalizer<T>(),
            NormalizationMethod.LpNorm => new LpNormNormalizer<T>(lpNormP ?? _numOps.FromDouble(2)),
            NormalizationMethod.Log => new LogNormalizer<T>(),
            _ => throw new ArgumentException($"Unsupported normalization method: {method}")
        };
    }
}