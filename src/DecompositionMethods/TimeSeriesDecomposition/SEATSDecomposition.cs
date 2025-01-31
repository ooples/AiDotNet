using AiDotNet.Enums.AlgorithmTypes;

namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

public class SEATSDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly SARIMAOptions<T> _sarimaOptions;
    private readonly int _forecastHorizon;
    private readonly SEATSAlgorithmType _algorithm;

    public SEATSDecomposition(Vector<T> timeSeries, SARIMAOptions<T>? sarimaOptions = null, int forecastHorizon = 12, SEATSAlgorithmType algorithm = SEATSAlgorithmType.Standard) 
        : base(timeSeries)
    {
        _sarimaOptions = sarimaOptions ?? new();
        _forecastHorizon = forecastHorizon;
        _algorithm = algorithm;
        Decompose();
    }

    protected override void Decompose()
    {
        switch (_algorithm)
        {
            case SEATSAlgorithmType.Standard:
                DecomposeStandard();
                break;
            case SEATSAlgorithmType.Canonical:
                DecomposeCanonical();
                break;
            case SEATSAlgorithmType.Burman:
                DecomposeBurman();
                break;
            default:
                throw new ArgumentException("Unsupported SEATS algorithm");
        }
    }

    private void DecomposeStandard()
    {
        // Step 1: Fit SARIMA model
        var sarimaModel = new SARIMAModel<T>(_sarimaOptions);
        sarimaModel.Train(new Matrix<T>(TimeSeries.Length, 1), TimeSeries);

        // Step 2: Extract components
        Vector<T> trend = ExtractTrendComponent(sarimaModel);
        Vector<T> seasonal = ExtractSeasonalComponent(sarimaModel);
        Vector<T> irregular = ExtractIrregularComponent(trend, seasonal);

        // Add components
        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Irregular, irregular);
    }

    private void DecomposeCanonical()
    {
        // Step 1: Fit SARIMA model
        var sarimaModel = new SARIMAModel<T>(_sarimaOptions);
        sarimaModel.Train(new Matrix<T>(TimeSeries.Length, 1), TimeSeries);

        // Step 2: Extract components using canonical decomposition
        Vector<T> trend = ExtractCanonicalTrendComponent(sarimaModel);
        Vector<T> seasonal = ExtractCanonicalSeasonalComponent(sarimaModel);
        Vector<T> irregular = ExtractCanonicalIrregularComponent(sarimaModel, trend, seasonal);

        // Add components
        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Irregular, irregular);
    }

    private void DecomposeBurman()
    {
        // Step 1: Fit SARIMA model
        var sarimaModel = new SARIMAModel<T>(_sarimaOptions);
        sarimaModel.Train(new Matrix<T>(TimeSeries.Length, 1), TimeSeries);

        // Step 2: Extract components using Burman's method
        Vector<T> trend = ExtractBurmanTrendComponent(sarimaModel);
        Vector<T> seasonal = ExtractBurmanSeasonalComponent(sarimaModel);
        Vector<T> irregular = ExtractBurmanIrregularComponent(sarimaModel, trend, seasonal);

        // Add components
        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Irregular, irregular);
    }

    private Vector<T> ExtractCanonicalTrendComponent(SARIMAModel<T> model)
    {
        // Extract trend component using canonical decomposition
        var trendFilter = DesignCanonicalTrendFilter(model);
        return ApplyFilter(TimeSeries, trendFilter);
    }

    private Vector<T> ExtractCanonicalSeasonalComponent(SARIMAModel<T> model)
    {
        // Extract seasonal component using canonical decomposition
        var seasonalFilter = DesignCanonicalSeasonalFilter(model);
        return ApplyFilter(TimeSeries, seasonalFilter);
    }

    private Vector<T> ExtractCanonicalIrregularComponent(SARIMAModel<T> model, Vector<T> trend, Vector<T> seasonal)
    {
        // Calculate irregular component as the residual
        return TimeSeries.Subtract(trend).Subtract(seasonal);
    }

    private Vector<T> ExtractBurmanTrendComponent(SARIMAModel<T> model)
    {
        // Extract trend component using Burman's method
        var trendFilter = DesignBurmanTrendFilter(model);
        return ApplyFilter(TimeSeries, trendFilter);
    }

    private Vector<T> ExtractBurmanSeasonalComponent(SARIMAModel<T> model)
    {
        // Extract seasonal component using Burman's method
        var seasonalFilter = DesignBurmanSeasonalFilter(model);
        return ApplyFilter(TimeSeries, seasonalFilter);
    }

    private Vector<T> ExtractBurmanIrregularComponent(SARIMAModel<T> model, Vector<T> trend, Vector<T> seasonal)
    {
        // Calculate irregular component as the residual
        return TimeSeries.Subtract(trend).Subtract(seasonal);
    }

    private Vector<T> DesignCanonicalTrendFilter(SARIMAModel<T> model)
    {
        var arParams = model.GetARParameters();
        var maParams = model.GetMAParameters();
        int p = arParams.Length;
        int q = maParams.Length;

        // Calculate the spectral density at frequency zero
        T spectralDensityAtZero = CalculateSpectralDensityAtZero(arParams, maParams);

        // Design the canonical trend filter
        int filterLength = Math.Max(p, q) + 1;
        var filterCoeffs = new T[filterLength];

        for (int k = 0; k < filterLength; k++)
        {
            T arTerm = k < p ? arParams[k] : NumOps.Zero;
            T maTerm = k < q ? maParams[k] : NumOps.Zero;
            filterCoeffs[k] = NumOps.Divide(NumOps.Subtract(arTerm, maTerm), spectralDensityAtZero);
        }

        return new Vector<T>(filterCoeffs);
    }

    private Vector<T> DesignCanonicalSeasonalFilter(SARIMAModel<T> model)
    {
        var sarParams = model.GetSeasonalARParameters();
        var smaParams = model.GetSeasonalMAParameters();
        int P = sarParams.Length;
        int Q = smaParams.Length;
        int s = model.GetSeasonalPeriod();

        // Calculate the spectral density at seasonal frequencies
        T[] spectralDensities = CalculateSpectralDensitiesAtSeasonalFrequencies(sarParams, smaParams, s);

        // Design the canonical seasonal filter
        int filterLength = s * Math.Max(P, Q) + 1;
        var filterCoeffs = new T[filterLength];

        for (int k = 0; k < filterLength; k++)
        {
            T sum = NumOps.Zero;
            for (int j = 1; j < s; j++)
            {
                T freq = NumOps.Divide(NumOps.FromDouble(2 * Math.PI * j), NumOps.FromDouble(s));
                T cosine = NumOps.FromDouble(Math.Cos(k * Convert.ToDouble(freq)));
                sum = NumOps.Add(sum, NumOps.Divide(cosine, spectralDensities[j]));
            }

            filterCoeffs[k] = NumOps.Divide(sum, NumOps.FromDouble(s - 1));
        }

        return new Vector<T>(filterCoeffs);
    }

    private Vector<T> DesignBurmanTrendFilter(SARIMAModel<T> model)
    {
        var arParams = model.GetARParameters();
        var maParams = model.GetMAParameters();
        int p = arParams.Length;
        int q = maParams.Length;

        // Calculate the spectral density
        Func<T, T> spectralDensity = freq => CalculateSpectralDensity(freq, arParams, maParams);

        // Design the Burman trend filter
        int filterLength = Math.Max(p, q) * 2 + 1;
        var filterCoeffs = new T[filterLength];

        for (int k = -filterLength / 2; k <= filterLength / 2; k++)
        {
            T integral = NumericIntegration(freq => 
            {
                T cosine = NumOps.FromDouble(Math.Cos(k * Convert.ToDouble(freq)));
                return NumOps.Multiply(cosine, 
                    NumOps.Divide(NumOps.One, spectralDensity(NumOps.FromDouble(Convert.ToDouble(freq)))));
            }, NumOps.Zero, NumOps.FromDouble(Math.PI), 1000);

            filterCoeffs[k + filterLength / 2] = NumOps.Divide(integral, NumOps.FromDouble(Math.PI));
        }

        return new Vector<T>(filterCoeffs);
    }

    private T CalculateSpectralDensityAtZero(Vector<T> arParams, Vector<T> maParams)
    {
        T arSum = NumOps.One;
        T maSum = NumOps.One;

        for (int i = 0; i < arParams.Length; i++)
        {
            arSum = NumOps.Subtract(arSum, arParams[i]);
        }

        for (int i = 0; i < maParams.Length; i++)
        {
            maSum = NumOps.Add(maSum, maParams[i]);
        }

        return NumOps.Divide(NumOps.Multiply(maSum, maSum), NumOps.Multiply(arSum, arSum));
    }

    private T[] CalculateSpectralDensitiesAtSeasonalFrequencies(Vector<T> sarParams, Vector<T> smaParams, int s)
    {
        T[] spectralDensities = new T[s];

        for (int j = 0; j < s; j++)
        {
            T freq = NumOps.Divide(NumOps.FromDouble(2 * Math.PI * j), NumOps.FromDouble(s));
            spectralDensities[j] = CalculateSpectralDensity(freq, sarParams, smaParams);
        }

        return spectralDensities;
    }

    private T CalculateSpectralDensity(T freq, Vector<T> arParams, Vector<T> maParams)
    {
        T arTerm = NumOps.One;
        T maTerm = NumOps.One;

        for (int i = 0; i < arParams.Length; i++)
        {
            T cosine = NumOps.FromDouble(Math.Cos(i * Convert.ToDouble(freq)));
            T sine = NumOps.FromDouble(Math.Sin(i * Convert.ToDouble(freq)));
            T realPart = NumOps.Multiply(arParams[i], cosine);
            T imagPart = NumOps.Multiply(arParams[i], sine);
            arTerm = NumOps.Subtract(arTerm, realPart);
            arTerm = NumOps.Add(arTerm, imagPart);
        }

        for (int i = 0; i < maParams.Length; i++)
        {
            T cosine = NumOps.FromDouble(Math.Cos(i * Convert.ToDouble(freq)));
            T sine = NumOps.FromDouble(Math.Sin(i * Convert.ToDouble(freq)));
            T realPart = NumOps.Multiply(maParams[i], cosine);
            T imagPart = NumOps.Multiply(maParams[i], sine);
            maTerm = NumOps.Add(maTerm, realPart);
            maTerm = NumOps.Subtract(maTerm, imagPart);
        }

        T arTermSquared = NumOps.Add(NumOps.Multiply(arTerm, arTerm), NumOps.Multiply(arTerm, arTerm));
        T maTermSquared = NumOps.Add(NumOps.Multiply(maTerm, maTerm), NumOps.Multiply(maTerm, maTerm));

        return NumOps.Divide(maTermSquared, arTermSquared);
    }

    private T NumericIntegration(Func<T, T> function, T start, T end, int steps)
    {
        T stepSize = NumOps.Divide(NumOps.Subtract(end, start), NumOps.FromDouble(steps));
        T sum = NumOps.Zero;

        for (int i = 0; i < steps; i++)
        {
            T x = NumOps.Add(start, NumOps.Multiply(NumOps.FromDouble(i), stepSize));
            sum = NumOps.Add(sum, function(x));
        }

        return NumOps.Multiply(sum, stepSize);
    }

    private Vector<T> DesignBurmanSeasonalFilter(SARIMAModel<T> model)
    {
        var sarParams = model.GetSeasonalARParameters();
        var smaParams = model.GetSeasonalMAParameters();
        int s = model.GetSeasonalPeriod();
        int P = sarParams.Length;
        int Q = smaParams.Length;

        // Calculate the spectral density at seasonal frequencies
        Func<T, T> spectralDensity = freq => CalculateSeasonalSpectralDensity(freq, sarParams, smaParams, s);

        // Design the Burman seasonal filter
        int filterLength = s * (Math.Max(P, Q) + 1);
        var filterCoeffs = new T[filterLength];

        for (int k = 0; k < filterLength; k++)
        {
            T integral = NumericIntegration(freq => 
            {
                T cosine = NumOps.FromDouble(Math.Cos(k * Convert.ToDouble(freq)));
                return NumOps.Multiply(cosine, 
                    NumOps.Divide(NumOps.One, spectralDensity(freq)));
            }, NumOps.Zero, NumOps.FromDouble(2 * Math.PI), 1000);

            filterCoeffs[k] = NumOps.Divide(integral, NumOps.FromDouble(2 * Math.PI));
        }

        // Normalize the filter
        T sum = filterCoeffs.Aggregate(NumOps.Zero, (acc, coeff) => NumOps.Add(acc, NumOps.Abs(coeff)));
        for (int i = 0; i < filterCoeffs.Length; i++)
        {
            filterCoeffs[i] = NumOps.Divide(filterCoeffs[i], sum);
        }

        return new Vector<T>(filterCoeffs);
    }

    private T CalculateSeasonalSpectralDensity(T freq, Vector<T> sarParams, Vector<T> smaParams, int s)
    {
        T arTerm = NumOps.One;
        T maTerm = NumOps.One;

        for (int i = 0; i < sarParams.Length; i++)
        {
            T angle = NumOps.Multiply(NumOps.FromDouble(s * (i + 1)), freq);
            T cosine = NumOps.FromDouble(Math.Cos(Convert.ToDouble(angle)));
            T sine = NumOps.FromDouble(Math.Sin(Convert.ToDouble(angle)));
            T realPart = NumOps.Multiply(sarParams[i], cosine);
            T imagPart = NumOps.Multiply(sarParams[i], sine);
            arTerm = NumOps.Subtract(arTerm, realPart);
            arTerm = NumOps.Add(arTerm, imagPart);
        }

        for (int i = 0; i < smaParams.Length; i++)
        {
            T angle = NumOps.Multiply(NumOps.FromDouble(s * (i + 1)), freq);
            T cosine = NumOps.FromDouble(Math.Cos(Convert.ToDouble(angle)));
            T sine = NumOps.FromDouble(Math.Sin(Convert.ToDouble(angle)));
            T realPart = NumOps.Multiply(smaParams[i], cosine);
            T imagPart = NumOps.Multiply(smaParams[i], sine);
            maTerm = NumOps.Add(maTerm, realPart);
            maTerm = NumOps.Subtract(maTerm, imagPart);
        }

        T arTermSquared = NumOps.Add(NumOps.Multiply(arTerm, arTerm), NumOps.Multiply(arTerm, arTerm));
        T maTermSquared = NumOps.Add(NumOps.Multiply(maTerm, maTerm), NumOps.Multiply(maTerm, maTerm));

        return NumOps.Divide(maTermSquared, arTermSquared);
    }

    private Vector<T> ApplyFilter(Vector<T> signal, Vector<T> filter)
    {
        int signalLength = signal.Length;
        int filterLength = filter.Length;
        int resultLength = signalLength + filterLength - 1;

        var result = new Vector<T>(resultLength);

        // Reverse the filter for convolution
        var reversedFilter = new Vector<T>(filterLength);
        for (int i = 0; i < filterLength; i++)
        {
            reversedFilter[i] = filter[filterLength - 1 - i];
        }

        // Perform convolution with zero-padding
        for (int i = 0; i < resultLength; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < filterLength; j++)
            {
                int signalIndex = i - j;
                if (signalIndex >= 0 && signalIndex < signalLength)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(signal[signalIndex], reversedFilter[j]));
                }
            }
            result[i] = sum;
        }

        // Trim the result to match the original signal length
        var trimmedResult = new Vector<T>(signalLength);
        int startIndex = (filterLength - 1) / 2;
        for (int i = 0; i < signalLength; i++)
        {
            trimmedResult[i] = result[startIndex + i];
        }

        return trimmedResult;
    }

    private Vector<T> ExtractTrendComponent(SARIMAModel<T> model)
    {
        // Extract trend component using the non-seasonal part of the SARIMA model
        var trendOptions = new SARIMAOptions<T>
        {
            P = _sarimaOptions.P,
            D = _sarimaOptions.D,
            Q = _sarimaOptions.Q,
            SeasonalPeriod = 1 // No seasonality for trend
        };

        var trendModel = new SARIMAModel<T>(trendOptions);
        trendModel.Train(new Matrix<T>(TimeSeries.Length, 1), TimeSeries);

        return trendModel.Predict(new Matrix<T>(_forecastHorizon, 1));
    }

    private Vector<T> ExtractSeasonalComponent(SARIMAModel<T> model)
    {
        // Extract seasonal component using the seasonal part of the SARIMA model
        var seasonalOptions = new SARIMAOptions<T>
        {
            P = 0,
            D = 0,
            Q = 0,
            SeasonalP = _sarimaOptions.SeasonalP,
            SeasonalD = _sarimaOptions.SeasonalD,
            SeasonalQ = _sarimaOptions.SeasonalQ,
            SeasonalPeriod = _sarimaOptions.SeasonalPeriod
        };

        var seasonalModel = new SARIMAModel<T>(seasonalOptions);
        seasonalModel.Train(new Matrix<T>(TimeSeries.Length, 1), TimeSeries);

        return seasonalModel.Predict(new Matrix<T>(_forecastHorizon, 1));
    }

    private Vector<T> ExtractIrregularComponent(Vector<T> trend, Vector<T> seasonal)
    {
        // Calculate irregular component as the residual
        return TimeSeries.Subtract(trend).Subtract(seasonal);
    }
}