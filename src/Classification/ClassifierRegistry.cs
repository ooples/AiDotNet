using AiDotNet.Interfaces;

namespace AiDotNet.Classification;

/// <summary>
/// Registry for creating classifier instances by type name, enabling serialization
/// and deserialization of wrapped classifiers in Meta and MultiLabel classifiers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> When a Meta or MultiLabel classifier wraps other classifiers
/// (e.g., BaggingClassifier wraps an array of base classifiers), we need a way to
/// save and restore those wrapped classifiers. The registry maps classifier type names
/// to concrete types that can be instantiated for deserialization.
/// </para>
/// <para>
/// All built-in AiDotNet classifiers are automatically registered. If you create custom
/// classifiers, register them with <see cref="Register{TClassifier}"/> before deserializing
/// models that wrap them.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// // Register a custom classifier type
/// ClassifierRegistry&lt;double&gt;.Register&lt;MyCustomClassifier&lt;double&gt;&gt;();
///
/// // Serialize a wrapped classifier
/// var (typeName, data) = ClassifierRegistry&lt;double&gt;.SerializeClassifier(myClassifier);
///
/// // Deserialize it back
/// var restored = ClassifierRegistry&lt;double&gt;.DeserializeClassifier(typeName, data);
/// </code>
/// </para>
/// </remarks>
public static class ClassifierRegistry<T>
{
    private static readonly Dictionary<string, Type> _registry = new(StringComparer.OrdinalIgnoreCase);
    private static readonly object _lock = new();
    private static bool _initialized;

    /// <summary>
    /// Ensures all built-in classifiers are registered. Called automatically on first use.
    /// </summary>
    public static void EnsureInitialized()
    {
        if (_initialized) return;
        lock (_lock)
        {
            if (_initialized) return;
            RegisterDefaults();
            _initialized = true;
        }
    }

    /// <summary>
    /// Registers a classifier type using generics. The type must have a parameterless constructor.
    /// </summary>
    /// <typeparam name="TClassifier">The classifier type to register.</typeparam>
    public static void Register<TClassifier>() where TClassifier : IClassifier<T>, new()
    {
        EnsureInitialized();
        lock (_lock)
        {
            _registry[typeof(TClassifier).Name] = typeof(TClassifier);
        }
    }

    /// <summary>
    /// Registers a classifier type by name and Type object.
    /// </summary>
    /// <param name="typeName">Short type name used as the registry key.</param>
    /// <param name="classifierType">The concrete classifier type. Must implement IClassifier&lt;T&gt; and have a parameterless constructor.</param>
    public static void Register(string typeName, Type classifierType)
    {
        EnsureInitialized();
        lock (_lock)
        {
            _registry[typeName] = classifierType;
        }
    }

    /// <summary>
    /// Creates a new, untrained classifier instance by type name.
    /// </summary>
    /// <param name="typeName">The type name as returned by <see cref="GetTypeName"/>.</param>
    /// <returns>A new classifier instance ready for deserialization.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the type name is not registered.</exception>
    public static IClassifier<T> Create(string typeName)
    {
        EnsureInitialized();
        Type? type;
        lock (_lock)
        {
            _registry.TryGetValue(typeName, out type);
        }

        if (type is not null)
        {
            return (IClassifier<T>)Activator.CreateInstance(type)!;
        }

        throw new InvalidOperationException(
            $"Classifier type '{typeName}' is not registered in ClassifierRegistry<{typeof(T).Name}>. " +
            $"Call ClassifierRegistry<{typeof(T).Name}>.Register<YourClassifier>() before deserializing.");
    }

    /// <summary>
    /// Gets the short type name for a classifier instance. Used as the key for serialization.
    /// </summary>
    /// <param name="classifier">The classifier instance.</param>
    /// <returns>The short type name (e.g., "GaussianNaiveBayes`1").</returns>
    public static string GetTypeName(IClassifier<T> classifier)
    {
        return classifier.GetType().Name;
    }

    /// <summary>
    /// Checks whether a type name is registered.
    /// </summary>
    /// <param name="typeName">The type name to check.</param>
    /// <returns>True if the type is registered.</returns>
    public static bool IsRegistered(string typeName)
    {
        EnsureInitialized();
        lock (_lock)
        {
            return _registry.ContainsKey(typeName);
        }
    }

    /// <summary>
    /// Serializes a wrapped classifier into a type name and base64-encoded data pair.
    /// </summary>
    /// <param name="classifier">The trained classifier to serialize.</param>
    /// <returns>A tuple of (TypeName, Base64-encoded serialized bytes).</returns>
    public static (string TypeName, string SerializedData) SerializeClassifier(IClassifier<T> classifier)
    {
        var typeName = GetTypeName(classifier);
        var bytes = classifier.Serialize();
        return (typeName, Convert.ToBase64String(bytes));
    }

    /// <summary>
    /// Deserializes a wrapped classifier from a type name and base64-encoded data.
    /// </summary>
    /// <param name="typeName">The type name from <see cref="SerializeClassifier"/>.</param>
    /// <param name="serializedData">The base64-encoded serialized bytes.</param>
    /// <returns>A restored classifier with all trained state.</returns>
    public static IClassifier<T> DeserializeClassifier(string typeName, string serializedData)
    {
        var classifier = Create(typeName);
        var bytes = Convert.FromBase64String(serializedData);
        classifier.Deserialize(bytes);
        return classifier;
    }

    /// <summary>
    /// Returns all registered type names.
    /// </summary>
    public static IReadOnlyCollection<string> RegisteredTypes
    {
        get
        {
            EnsureInitialized();
            lock (_lock)
            {
                return _registry.Keys.ToList().AsReadOnly();
            }
        }
    }

    private static void RegisterDefaults()
    {
        // Linear classifiers
        _registry["RidgeClassifier`1"] = typeof(Linear.RidgeClassifier<T>);
        _registry["SGDClassifier`1"] = typeof(Linear.SGDClassifier<T>);
        _registry["PerceptronClassifier`1"] = typeof(Linear.PerceptronClassifier<T>);
        _registry["PassiveAggressiveClassifier`1"] = typeof(Linear.PassiveAggressiveClassifier<T>);

        // Naive Bayes
        _registry["GaussianNaiveBayes`1"] = typeof(NaiveBayes.GaussianNaiveBayes<T>);
        _registry["MultinomialNaiveBayes`1"] = typeof(NaiveBayes.MultinomialNaiveBayes<T>);
        _registry["BernoulliNaiveBayes`1"] = typeof(NaiveBayes.BernoulliNaiveBayes<T>);
        _registry["ComplementNaiveBayes`1"] = typeof(NaiveBayes.ComplementNaiveBayes<T>);

        // Trees
        _registry["DecisionTreeClassifier`1"] = typeof(Trees.DecisionTreeClassifier<T>);

        // Ensemble
        _registry["RandomForestClassifier`1"] = typeof(Ensemble.RandomForestClassifier<T>);
        _registry["ExtraTreesClassifier`1"] = typeof(Ensemble.ExtraTreesClassifier<T>);
        _registry["AdaBoostClassifier`1"] = typeof(Ensemble.AdaBoostClassifier<T>);
        _registry["GradientBoostingClassifier`1"] = typeof(Ensemble.GradientBoostingClassifier<T>);

        // Boosting
        _registry["DARTClassifier`1"] = typeof(Boosting.DARTClassifier<T>);
        _registry["NGBoostClassifier`1"] = typeof(Boosting.NGBoostClassifier<T>);
        _registry["HistGradientBoostingClassifier`1"] = typeof(Boosting.HistGradientBoostingClassifier<T>);
        _registry["ExplainableBoostingClassifier`1"] = typeof(Boosting.ExplainableBoostingClassifier<T>);

        // SVM
        _registry["SupportVectorClassifier`1"] = typeof(SVM.SupportVectorClassifier<T>);
        _registry["LinearSupportVectorClassifier`1"] = typeof(SVM.LinearSupportVectorClassifier<T>);
        _registry["NuSupportVectorClassifier`1"] = typeof(SVM.NuSupportVectorClassifier<T>);

        // Neighbors
        _registry["KNeighborsClassifier`1"] = typeof(Neighbors.KNeighborsClassifier<T>);

        // Discriminant Analysis
        _registry["LinearDiscriminantAnalysis`1"] = typeof(DiscriminantAnalysis.LinearDiscriminantAnalysis<T>);
        _registry["QuadraticDiscriminantAnalysis`1"] = typeof(DiscriminantAnalysis.QuadraticDiscriminantAnalysis<T>);

        // Online
        _registry["HoeffdingTreeClassifier`1"] = typeof(Online.HoeffdingTreeClassifier<T>);
        _registry["AdaptiveRandomForestClassifier`1"] = typeof(Online.AdaptiveRandomForestClassifier<T>);
        _registry["OnlineNaiveBayesClassifier`1"] = typeof(Online.OnlineNaiveBayesClassifier<T>);

        // TimeSeries
        _registry["TimeSeriesForestClassifier`1"] = typeof(TimeSeries.TimeSeriesForestClassifier<T>);
        _registry["RocketClassifier`1"] = typeof(TimeSeries.RocketClassifier<T>);
        _registry["MiniRocketClassifier`1"] = typeof(TimeSeries.MiniRocketClassifier<T>);

        // SemiSupervised
        _registry["LabelPropagation`1"] = typeof(SemiSupervised.LabelPropagation<T>);
        _registry["LabelSpreading`1"] = typeof(SemiSupervised.LabelSpreading<T>);
        _registry["SelfTrainingClassifier`1"] = typeof(SemiSupervised.SelfTrainingClassifier<T>);

        // Imbalanced Ensemble
        _registry["BalancedBaggingClassifier`1"] = typeof(ImbalancedEnsemble.BalancedBaggingClassifier<T>);
        _registry["EasyEnsembleClassifier`1"] = typeof(ImbalancedEnsemble.EasyEnsembleClassifier<T>);
        _registry["BalancedRandomForestClassifier`1"] = typeof(ImbalancedEnsemble.BalancedRandomForestClassifier<T>);

        // Ordinal
        _registry["OrdinalRidgeRegression`1"] = typeof(Ordinal.OrdinalRidgeRegression<T>);
        _registry["OrdinalLogisticRegression`1"] = typeof(Ordinal.OrdinalLogisticRegression<T>);

        // Calibration
        _registry["CalibratedClassifier`1"] = typeof(Calibration.CalibratedClassifier<T>);

        // Meta (these wrap other classifiers but can also be wrapped themselves)
        _registry["BaggingClassifier`1"] = typeof(Meta.BaggingClassifier<T>);
        _registry["ClassifierChain`1"] = typeof(Meta.ClassifierChain<T>);
        _registry["MultiOutputClassifier`1"] = typeof(Meta.MultiOutputClassifier<T>);
        _registry["OneVsOneClassifier`1"] = typeof(Meta.OneVsOneClassifier<T>);
        _registry["OneVsRestClassifier`1"] = typeof(Meta.OneVsRestClassifier<T>);
        _registry["StackingClassifier`1"] = typeof(Meta.StackingClassifier<T>);
        _registry["VotingClassifier`1"] = typeof(Meta.VotingClassifier<T>);
    }
}
