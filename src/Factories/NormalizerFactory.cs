global using AiDotNet.Normalizers;

namespace AiDotNet.Factories;

public class NormalizerFactory
{
    public static INormalizer CreateNormalizer(NormalizationMethod method, double lpNormP = 2)
    {
        return method switch
        {
            NormalizationMethod.None => new NoNormalizer(),
            NormalizationMethod.MinMax => new MinMaxNormalizer(),
            NormalizationMethod.ZScore => new ZScoreNormalizer(),
            NormalizationMethod.RobustScaling => new RobustScalingNormalizer(),
            NormalizationMethod.Decimal => new DecimalNormalizer(),
            NormalizationMethod.Binning => new BinningNormalizer(),
            NormalizationMethod.MeanVariance => new MeanVarianceNormalizer(),
            NormalizationMethod.LogMeanVariance => new LogMeanVarianceNormalizer(),
            NormalizationMethod.GlobalContrast => new GlobalContrastNormalizer(),
            NormalizationMethod.LpNorm => new LpNormNormalizer(lpNormP),
            NormalizationMethod.Log => new LogNormalizer(),
            _ => throw new ArgumentException($"Unsupported normalization method: {method}")
        };
    }
}