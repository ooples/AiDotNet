namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

/// <summary>
/// Implements the SEATS (Seasonal Extraction in ARIMA Time Series) decomposition method for time series data.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SEATS is an advanced method for breaking down time series data into different components:
/// trend (long-term direction), seasonal patterns (regular fluctuations), and irregular components (random noise).
/// It uses statistical models to identify these patterns in your data.
/// </para>
/// <para>
/// This implementation supports multiple algorithm variants: Standard, Canonical, and Burman.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SEATSDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly SARIMAOptions<T> _sarimaOptions;
    private readonly int _forecastHorizon;
    private readonly SEATSAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the <see cref="SEATSDecomposition{T}"/> class.
    /// </summary>
    /// <param name="timeSeries">The time series data to decompose.</param>
    /// <param name="sarimaOptions">Options for the SARIMA model. If null, default options will be used.</param>
    /// <param name="forecastHorizon">The number of periods to forecast.</param>
    /// <param name="algorithm">The algorithm variant to use for decomposition.</param>
    /// <remarks>
    /// <b>For Beginners:</b> When creating a SEATS decomposition:
    /// - timeSeries: Your data points arranged in time order
    /// - sarimaOptions: Settings for the statistical model (you can use default settings)
    /// - forecastHorizon: How many future periods you want to predict (e.g., 12 for a year of monthly data)
    /// - algorithm: Which calculation method to use (Standard is a good default)
    /// </remarks>
    public SEATSDecomposition(Vector<T> timeSeries, SARIMAOptions<T>? sarimaOptions = null, int forecastHorizon = 12, SEATSAlgorithmType algorithm = SEATSAlgorithmType.Standard)
        : base(timeSeries)
    {
        _sarimaOptions = sarimaOptions ?? new();
        _forecastHorizon = forecastHorizon;
        _algorithm = algorithm;
        Decompose();
    }

    /// <summary>
    /// Performs the time series decomposition based on the selected algorithm.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified.</exception>
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

    /// <summary>
    /// Performs standard SEATS decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method fits a statistical model to your data and then
    /// separates it into trend (long-term direction), seasonal patterns, and random noise.
    /// </remarks>
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

    /// <summary>
    /// Performs canonical SEATS decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The canonical method is a specific approach that ensures the components
    /// are as independent from each other as possible. This means the trend, seasonal, and
    /// irregular components have minimal correlation with each other.
    /// </remarks>
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

    /// <summary>
    /// Performs Burman's variant of SEATS decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Burman's method is an alternative approach to decomposition that uses
    /// special filters designed to better separate the components in certain types of data.
    /// It can be more effective for some complex time series.
    /// </remarks>
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

    /// <summary>
    /// Extracts the trend component using canonical decomposition.
    /// </summary>
    /// <param name="model">The fitted SARIMA model.</param>
    /// <returns>The trend component of the time series.</returns>
    private Vector<T> ExtractCanonicalTrendComponent(SARIMAModel<T> model)
    {
        // Extract trend component using canonical decomposition
        var trendFilter = DesignCanonicalTrendFilter(model);
        return ApplyFilter(TimeSeries, trendFilter);
    }

    /// <summary>
    /// Extracts the seasonal component using canonical decomposition.
    /// </summary>
    /// <param name="model">The fitted SARIMA model.</param>
    /// <returns>The seasonal component of the time series.</returns>
    private Vector<T> ExtractCanonicalSeasonalComponent(SARIMAModel<T> model)
    {
        // Extract seasonal component using canonical decomposition
        var seasonalFilter = DesignCanonicalSeasonalFilter(model);
        return ApplyFilter(TimeSeries, seasonalFilter);
    }

    /// <summary>
    /// Extracts the irregular component as the residual after removing trend and seasonal components.
    /// </summary>
    /// <param name="model">The fitted SARIMA model.</param>
    /// <param name="trend">The extracted trend component.</param>
    /// <param name="seasonal">The extracted seasonal component.</param>
    /// <returns>The irregular component of the time series.</returns>
    private Vector<T> ExtractCanonicalIrregularComponent(SARIMAModel<T> model, Vector<T> trend, Vector<T> seasonal)
    {
        // Calculate irregular component as the residual
        return TimeSeries.Subtract(trend).Subtract(seasonal);
    }

    /// <summary>
    /// Extracts the trend component using Burman's method.
    /// </summary>
    /// <param name="model">The fitted SARIMA model.</param>
    /// <returns>The trend component of the time series.</returns>
    private Vector<T> ExtractBurmanTrendComponent(SARIMAModel<T> model)
    {
        // Extract trend component using Burman's method
        var trendFilter = DesignBurmanTrendFilter(model);
        return ApplyFilter(TimeSeries, trendFilter);
    }

    /// <summary>
    /// Extracts the seasonal component using Burman's method.
    /// </summary>
    /// <param name="model">The fitted SARIMA model.</param>
    /// <returns>The seasonal component of the time series.</returns>
    private Vector<T> ExtractBurmanSeasonalComponent(SARIMAModel<T> model)
    {
        // Extract seasonal component using Burman's method
        var seasonalFilter = DesignBurmanSeasonalFilter(model);
        return ApplyFilter(TimeSeries, seasonalFilter);
    }

    /// <summary>
    /// Extracts the irregular component as the residual after removing trend and seasonal components.
    /// </summary>
    /// <param name="model">The fitted SARIMA model.</param>
    /// <param name="trend">The extracted trend component.</param>
    /// <param name="seasonal">The extracted seasonal component.</param>
    /// <returns>The irregular component of the time series.</returns>
    private Vector<T> ExtractBurmanIrregularComponent(SARIMAModel<T> model, Vector<T> trend, Vector<T> seasonal)
    {
        // Calculate irregular component as the residual
        return TimeSeries.Subtract(trend).Subtract(seasonal);
    }

    /// <summary>
    /// Designs a canonical trend filter based on the provided SARIMA model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A canonical trend filter helps identify the long-term direction or pattern in your data
    /// by removing short-term fluctuations. It's like looking at the general path of a hiking trail
    /// while ignoring small ups and downs along the way.
    /// </para>
    /// </remarks>
    /// <param name="model">The SARIMA model containing parameters for filter design.</param>
    /// <returns>A vector representing the designed filter coefficients.</returns>
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

    /// <summary>
    /// Designs a canonical seasonal filter based on the provided SARIMA model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A canonical seasonal filter helps identify repeating patterns in your data
    /// that occur at regular intervals (like daily, weekly, or yearly patterns). It's like identifying
    /// that ice cream sales always go up in summer and down in winter, year after year.
    /// </para>
    /// </remarks>
    /// <param name="model">The SARIMA model containing parameters for filter design.</param>
    /// <returns>A vector representing the designed filter coefficients.</returns>
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

    /// <summary>
    /// Designs a Burman trend filter based on the provided SARIMA model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Burman trend filter is an advanced technique to extract the underlying trend
    /// from your time series data. It's particularly good at handling complex data where simpler methods
    /// might fail. Think of it as a sophisticated camera lens that can focus on just the main subject
    /// while blurring out all the background noise.
    /// </para>
    /// </remarks>
    /// <param name="model">The SARIMA model containing parameters for filter design.</param>
    /// <returns>A vector representing the designed filter coefficients.</returns>
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

    /// <summary>
    /// Calculates the spectral density at frequency zero for the given AR and MA parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spectral density is a way to measure how much of each frequency component
    /// contributes to your data. At frequency zero, we're looking at the very long-term component
    /// (like the overall average level). This helps us understand the strength of the trend in your data.
    /// </para>
    /// </remarks>
    /// <param name="arParams">The autoregressive parameters.</param>
    /// <param name="maParams">The moving average parameters.</param>
    /// <returns>The spectral density value at frequency zero.</returns>
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

    /// <summary>
    /// Calculates the spectral densities at seasonal frequencies for the given seasonal AR and MA parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how strong each seasonal pattern is in your data.
    /// For example, if you have monthly data, it measures the strength of patterns that repeat
    /// every month, every two months, etc. This helps identify which seasonal patterns are most important.
    /// </para>
    /// </remarks>
    /// <param name="sarParams">The seasonal autoregressive parameters.</param>
    /// <param name="smaParams">The seasonal moving average parameters.</param>
    /// <param name="s">The seasonal period.</param>
    /// <returns>An array of spectral density values at seasonal frequencies.</returns>
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

    /// <summary>
    /// Calculates the spectral density at a given frequency for the provided AR and MA parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method measures how much a specific pattern (at a particular frequency)
    /// contributes to your data. Think of it like analyzing the ingredients in a recipe - 
    /// this tells you how much of each ingredient (frequency) is present in your data "dish".
    /// </para>
    /// </remarks>
    /// <param name="freq">The frequency at which to calculate the spectral density.</param>
    /// <param name="arParams">The autoregressive parameters.</param>
    /// <param name="maParams">The moving average parameters.</param>
    /// <returns>The spectral density value at the specified frequency.</returns>
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

    /// <summary>
    /// Performs numerical integration of a function over a specified interval.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Numerical integration is a way to calculate the area under a curve
    /// when we can't solve it with a formula. We divide the area into many small rectangles,
    /// calculate the area of each, and add them up to get an approximate total area.
    /// </para>
    /// </remarks>
    /// <param name="function">The function to integrate.</param>
    /// <param name="start">The starting point of the integration interval.</param>
    /// <param name="end">The ending point of the integration interval.</param>
    /// <param name="steps">The number of steps to use in the approximation.</param>
    /// <returns>The approximate value of the integral.</returns>
    private T NumericIntegration(Func<T, T> function, T start, T end, int steps)
    {
        T _stepSize = NumOps.Divide(NumOps.Subtract(end, start), NumOps.FromDouble(steps));
        T _sum = NumOps.Zero;

        for (int i = 0; i < steps; i++)
        {
            T x = NumOps.Add(start, NumOps.Multiply(NumOps.FromDouble(i), _stepSize));
            _sum = NumOps.Add(_sum, function(x));
        }

        return NumOps.Multiply(_sum, _stepSize);
    }

    /// <summary>
    /// Designs a Burman seasonal filter based on the provided SARIMA model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Burman seasonal filter helps identify and extract repeating patterns
    /// in your data (like weekly, monthly, or yearly cycles). It's like having a special lens
    /// that only shows you the seasonal patterns while filtering out other components.
    /// </para>
    /// </remarks>
    /// <param name="model">The SARIMA model containing parameters for filter design.</param>
    /// <returns>A vector representing the designed seasonal filter coefficients.</returns>
    private Vector<T> DesignBurmanSeasonalFilter(SARIMAModel<T> model)
    {
        var _sarParams = model.GetSeasonalARParameters();
        var _smaParams = model.GetSeasonalMAParameters();
        int _seasonalPeriod = model.GetSeasonalPeriod();
        int _seasonalArOrder = _sarParams.Length;
        int _seasonalMaOrder = _smaParams.Length;

        // Calculate the spectral density at seasonal frequencies
        Func<T, T> _spectralDensity = freq => CalculateSeasonalSpectralDensity(freq, _sarParams, _smaParams, _seasonalPeriod);

        // Design the Burman seasonal filter
        int _filterLength = _seasonalPeriod * (Math.Max(_seasonalArOrder, _seasonalMaOrder) + 1);
        var _filterCoeffs = new T[_filterLength];

        for (int k = 0; k < _filterLength; k++)
        {
            T _integral = NumericIntegration(freq =>
            {
                T _cosine = NumOps.FromDouble(Math.Cos(k * Convert.ToDouble(freq)));
                return NumOps.Multiply(_cosine,
                    NumOps.Divide(NumOps.One, _spectralDensity(freq)));
            }, NumOps.Zero, NumOps.FromDouble(2 * Math.PI), 1000);

            _filterCoeffs[k] = NumOps.Divide(_integral, NumOps.FromDouble(2 * Math.PI));
        }

        // Normalize the filter
        T _sum = _filterCoeffs.Aggregate(NumOps.Zero, (acc, coeff) => NumOps.Add(acc, NumOps.Abs(coeff)));
        for (int i = 0; i < _filterCoeffs.Length; i++)
        {
            _filterCoeffs[i] = NumOps.Divide(_filterCoeffs[i], _sum);
        }

        return new Vector<T>(_filterCoeffs);
    }

    /// <summary>
    /// Calculates the spectral density at a given frequency for seasonal components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spectral density for seasonal components measures how strongly
    /// each seasonal pattern (like weekly or monthly cycles) appears in your data.
    /// It's like analyzing which musical notes are loudest in a song, but for time patterns
    /// instead of sound.
    /// </para>
    /// </remarks>
    /// <param name="freq">The frequency at which to calculate the spectral density.</param>
    /// <param name="sarParams">The seasonal autoregressive parameters.</param>
    /// <param name="smaParams">The seasonal moving average parameters.</param>
    /// <param name="s">The seasonal period.</param>
    /// <returns>The spectral density value at the specified frequency.</returns>
    private T CalculateSeasonalSpectralDensity(T freq, Vector<T> sarParams, Vector<T> smaParams, int s)
    {
        T _arTerm = NumOps.One;
        T _maTerm = NumOps.One;

        for (int i = 0; i < sarParams.Length; i++)
        {
            T _angle = NumOps.Multiply(NumOps.FromDouble(s * (i + 1)), freq);
            T _cosine = NumOps.FromDouble(Math.Cos(Convert.ToDouble(_angle)));
            T _sine = NumOps.FromDouble(Math.Sin(Convert.ToDouble(_angle)));
            T _realPart = NumOps.Multiply(sarParams[i], _cosine);
            T _imagPart = NumOps.Multiply(sarParams[i], _sine);
            _arTerm = NumOps.Subtract(_arTerm, _realPart);
            _arTerm = NumOps.Add(_arTerm, _imagPart);
        }

        for (int i = 0; i < smaParams.Length; i++)
        {
            T _angle = NumOps.Multiply(NumOps.FromDouble(s * (i + 1)), freq);
            T _cosine = NumOps.FromDouble(Math.Cos(Convert.ToDouble(_angle)));
            T _sine = NumOps.FromDouble(Math.Sin(Convert.ToDouble(_angle)));
            T _realPart = NumOps.Multiply(smaParams[i], _cosine);
            T _imagPart = NumOps.Multiply(smaParams[i], _sine);
            _maTerm = NumOps.Add(_maTerm, _realPart);
            _maTerm = NumOps.Subtract(_maTerm, _imagPart);
        }

        T _arTermSquared = NumOps.Add(NumOps.Multiply(_arTerm, _arTerm), NumOps.Multiply(_arTerm, _arTerm));
        T _maTermSquared = NumOps.Add(NumOps.Multiply(_maTerm, _maTerm), NumOps.Multiply(_maTerm, _maTerm));

        return NumOps.Divide(_maTermSquared, _arTermSquared);
    }

    /// <summary>
    /// Applies a filter to a signal using convolution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Applying a filter to a signal is like using a special lens on a camera.
    /// The filter (lens) highlights certain features while reducing others. In time series,
    /// we use filters to extract specific patterns like trends or seasonal components.
    /// Convolution is the mathematical operation that combines the signal with the filter.
    /// </para>
    /// </remarks>
    /// <param name="signal">The input signal (time series data).</param>
    /// <param name="filter">The filter to apply.</param>
    /// <returns>The filtered signal.</returns>
    private Vector<T> ApplyFilter(Vector<T> signal, Vector<T> filter)
    {
        int _signalLength = signal.Length;
        int _filterLength = filter.Length;
        int _resultLength = _signalLength + _filterLength - 1;

        var _result = new Vector<T>(_resultLength);

        // Reverse the filter for convolution
        var _reversedFilter = new Vector<T>(_filterLength);
        for (int i = 0; i < _filterLength; i++)
        {
            _reversedFilter[i] = filter[_filterLength - 1 - i];
        }

        // Perform convolution with zero-padding
        for (int i = 0; i < _resultLength; i++)
        {
            T _sum = NumOps.Zero;
            for (int j = 0; j < _filterLength; j++)
            {
                int _signalIndex = i - j;
                if (_signalIndex >= 0 && _signalIndex < _signalLength)
                {
                    _sum = NumOps.Add(_sum, NumOps.Multiply(signal[_signalIndex], _reversedFilter[j]));
                }
            }

            _result[i] = _sum;
        }

        // Trim the result to match the original signal length
        var _trimmedResult = new Vector<T>(_signalLength);
        int _startIndex = (_filterLength - 1) / 2;
        for (int i = 0; i < _signalLength; i++)
        {
            _trimmedResult[i] = _result[_startIndex + i];
        }

        return _trimmedResult;
    }

    /// <summary>
    /// Extracts the trend component from a time series using the SARIMA model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The trend component represents the long-term direction of your data.
    /// Think of it as the general path your data follows when you ignore short-term fluctuations
    /// and seasonal patterns. For example, if your sales are generally increasing over years
    /// despite seasonal ups and downs, that upward direction is the trend.
    /// </para>
    /// </remarks>
    /// <param name="model">The SARIMA model used for extraction.</param>
    /// <returns>A vector representing the trend component.</returns>
    private Vector<T> ExtractTrendComponent(SARIMAModel<T> model)
    {
        // Extract trend component using the non-seasonal part of the SARIMA model
        var _trendOptions = new SARIMAOptions<T>
        {
            P = _sarimaOptions.P,
            D = _sarimaOptions.D,
            Q = _sarimaOptions.Q,
            SeasonalPeriod = 1 // No seasonality for trend
        };

        var _trendModel = new SARIMAModel<T>(_trendOptions);
        _trendModel.Train(new Matrix<T>(TimeSeries.Length, 1), TimeSeries);

        return _trendModel.Predict(new Matrix<T>(TimeSeries.Length, 1));
    }

    /// <summary>
    /// Extracts the seasonal component from a time series using the SARIMA model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The seasonal component represents repeating patterns in your data that occur at
    /// regular intervals (like sales increasing during holidays every year). This method isolates these
    /// recurring patterns from your data, similar to identifying which notes repeat in a musical composition.
    /// </para>
    /// <para>
    /// This method creates a specialized SARIMA model that focuses only on the seasonal aspects of your data
    /// by setting the non-seasonal parameters (P, D, Q) to zero and using only the seasonal parameters.
    /// </para>
    /// </remarks>
    /// <param name="model">The SARIMA model used for extraction.</param>
    /// <returns>A vector representing the seasonal component.</returns>
    private Vector<T> ExtractSeasonalComponent(SARIMAModel<T> model)
    {
        // Extract seasonal component using the seasonal part of the SARIMA model
        var _seasonalOptions = new SARIMAOptions<T>
        {
            P = 0,
            D = 0,
            Q = 0,
            SeasonalP = _sarimaOptions.SeasonalP,
            SeasonalD = _sarimaOptions.SeasonalD,
            SeasonalQ = _sarimaOptions.SeasonalQ,
            SeasonalPeriod = _sarimaOptions.SeasonalPeriod
        };

        var _seasonalModel = new SARIMAModel<T>(_seasonalOptions);
        _seasonalModel.Train(new Matrix<T>(TimeSeries.Length, 1), TimeSeries);

        return _seasonalModel.Predict(new Matrix<T>(TimeSeries.Length, 1));
    }

    /// <summary>
    /// Extracts the irregular component from a time series by removing trend and seasonal components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The irregular component (sometimes called "residual" or "noise") represents the 
    /// random fluctuations in your data that can't be explained by trends or seasonal patterns. 
    /// Think of it like the static on a radio - it's the unpredictable part that remains after 
    /// accounting for all predictable patterns.
    /// </para>
    /// <para>
    /// This method simply subtracts both the trend and seasonal components from the original time series,
    /// leaving only the irregular variations.
    /// </para>
    /// </remarks>
    /// <param name="trend">The trend component to subtract.</param>
    /// <param name="seasonal">The seasonal component to subtract.</param>
    /// <returns>A vector representing the irregular component.</returns>
    private Vector<T> ExtractIrregularComponent(Vector<T> trend, Vector<T> seasonal)
    {
        // Calculate irregular component as the residual
        return TimeSeries.Subtract(trend).Subtract(seasonal);
    }
}
