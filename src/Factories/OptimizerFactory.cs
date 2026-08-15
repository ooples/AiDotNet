namespace AiDotNet.Factories;

/// <summary>
/// A factory class that creates optimizer instances for training machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> An optimizer is an algorithm that adjusts the parameters of a machine learning model
/// to minimize errors and improve performance. Think of it like a navigator that helps your model find the
/// best path to the correct answers.
/// </para>
/// <para>
/// This factory helps you create different types of optimizers without needing to know their internal
/// implementation details. Think of it like ordering a specific tool from a catalog - you just specify
/// what you need, and the factory provides it.
/// </para>
/// </remarks>
public static class OptimizerFactory<T, TInput, TOutput>
{
    /// <summary>
    /// Maps every <see cref="OptimizerType"/> that names a shipped optimizer to its implementation, including
    /// the enum's spelling aliases.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This dictionary stores which class to use for each optimizer type, making it easy
    /// to look up the right implementation when needed.
    /// </remarks>
    private static readonly Dictionary<OptimizerType, Type> _optimizerTypes = new Dictionary<OptimizerType, Type>();

    /// <summary>
    /// The reverse map used by <see cref="GetOptimizerType"/>: one CANONICAL enum value per implementation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Kept separate from <see cref="_optimizerTypes"/> because that map is many-to-one — the enum carries
    /// spelling aliases (<c>Adagrad</c>/<c>AdaGrad</c>, <c>AdaMax</c>/<c>Adamax</c>, <c>AdaDelta</c>/
    /// <c>Adadelta</c>, <c>Adam</c>/<c>AdamOptimizer</c>, <c>Normal</c>/<c>NormalOptimizer</c>). Scanning the
    /// forward map to answer a reverse query would return whichever alias the dictionary happened to enumerate
    /// first, making the result depend on hash ordering — unacceptable for a method whose documented purpose
    /// includes identifying an optimizer when saving a model.
    /// </para>
    /// </remarks>
    private static readonly Dictionary<Type, OptimizerType> _canonicalNames = new Dictionary<Type, OptimizerType>();

    /// <summary>
    /// Static constructor that registers every optimizer this library ships.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This code runs once when the OptimizerFactory is first used, registering
    /// all the available optimizer types so they can be created later.
    /// </remarks>
    static OptimizerFactory()
    {
        // Adam family.
        Register(OptimizerType.Adam, typeof(AdamOptimizer<T, TInput, TOutput>), OptimizerType.AdamOptimizer);
        Register(OptimizerType.AdamW, typeof(AdamWOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.AMSGrad, typeof(AMSGradOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.Nadam, typeof(NadamOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.RAdam, typeof(RAdamOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.AdaMax, typeof(AdaMaxOptimizer<T, TInput, TOutput>), OptimizerType.Adamax);

        // Other adaptive methods.
        Register(OptimizerType.AdaDelta, typeof(AdaDeltaOptimizer<T, TInput, TOutput>), OptimizerType.Adadelta);
        Register(OptimizerType.Adagrad, typeof(AdagradOptimizer<T, TInput, TOutput>), OptimizerType.AdaGrad);
        Register(OptimizerType.RMSProp, typeof(RootMeanSquarePropagationOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.Lion, typeof(LionOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.FTRL, typeof(FTRLOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.ASGD, typeof(ASGDOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.Rprop, typeof(RpropOptimizer<T, TInput, TOutput>));

        // Layer-wise adaptive (large-batch) methods.
        Register(OptimizerType.LAMB, typeof(LAMBOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.LARS, typeof(LARSOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.Adam8Bit, typeof(Adam8BitOptimizer<T, TInput, TOutput>));

        // First-order descent.
        Register(OptimizerType.GradientDescent, typeof(GradientDescentOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.StochasticGradientDescent, typeof(StochasticGradientDescentOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.MiniBatchGradientDescent, typeof(MiniBatchGradientDescentOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.Momentum, typeof(MomentumOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.NesterovAcceleratedGradient, typeof(NesterovAcceleratedGradientOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.ProximalGradientDescent, typeof(ProximalGradientDescentOptimizer<T, TInput, TOutput>));

        // Second-order, quasi-Newton and line-search methods.
        Register(OptimizerType.LBFGS, typeof(LBFGSOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.BFGS, typeof(BFGSOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.DFP, typeof(DFPOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.NewtonMethod, typeof(NewtonMethodOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.LevenbergMarquardt, typeof(LevenbergMarquardtOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.ConjugateGradient, typeof(ConjugateGradientOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.CoordinateDescent, typeof(CoordinateDescentOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.TrustRegion, typeof(TrustRegionOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.ADMM, typeof(ADMMOptimizer<T, TInput, TOutput>));

        // Derivative-free and population methods.
        Register(OptimizerType.NelderMead, typeof(NelderMeadOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.PowellMethod, typeof(PowellOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.BayesianOptimization, typeof(BayesianOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.DifferentialEvolution, typeof(DifferentialEvolutionOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.GeneticAlgorithm, typeof(GeneticAlgorithmOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.AntColony, typeof(AntColonyOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.ParticleSwarm, typeof(ParticleSwarmOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.SimulatedAnnealing, typeof(SimulatedAnnealingOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.CMAES, typeof(CMAESOptimizer<T, TInput, TOutput>));
        Register(OptimizerType.TabuSearch, typeof(TabuSearchOptimizer<T, TInput, TOutput>));

        Register(OptimizerType.Normal, typeof(NormalOptimizer<T, TInput, TOutput>), OptimizerType.NormalOptimizer);
    }

    /// <summary>
    /// Registers an optimizer implementation under a canonical <see cref="OptimizerType"/> plus any aliases.
    /// </summary>
    /// <param name="canonical">The name <see cref="GetOptimizerType"/> reports for this implementation.</param>
    /// <param name="type">The implementation class.</param>
    /// <param name="aliases">Additional enum spellings that resolve to the same implementation.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This adds an entry to the catalog of optimizers, connecting the optimizer type
    /// (like "Adam") with the class that implements it. Some optimizers have more than one accepted spelling;
    /// those extra spellings are the aliases, and they all build the same thing.
    /// </remarks>
    private static void Register(OptimizerType canonical, Type type, params OptimizerType[] aliases)
    {
        _optimizerTypes[canonical] = type;
        _canonicalNames[type] = canonical;
        foreach (var alias in aliases)
        {
            _optimizerTypes[alias] = type;
        }
    }

    /// <summary>
    /// Determines the optimizer type from an existing optimizer instance.
    /// </summary>
    /// <param name="optimizer">The optimizer instance to identify.</param>
    /// <returns>The canonical type of the provided optimizer.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="optimizer"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the optimizer type cannot be determined.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method examines an optimizer object and tells you what type it is.
    /// It's like looking at a tool and identifying whether it's a hammer, screwdriver, or wrench.
    /// </para>
    /// <para>
    /// This is useful when you have an optimizer object but don't know its specific type, such as
    /// when saving or loading models. The answer is always the CANONICAL name, so
    /// <c>CreateOptimizer(GetOptimizerType(x))</c> reliably rebuilds the same implementation even for
    /// optimizers the enum spells more than one way.
    /// </para>
    /// </remarks>
    public static OptimizerType GetOptimizerType(IOptimizer<T, TInput, TOutput> optimizer)
    {
        if (optimizer is null) throw new ArgumentNullException(nameof(optimizer));

        // Exact type first — an O(1) hit for every optimizer this library ships, and immune to the
        // enumeration-order ambiguity a scan would carry.
        if (_canonicalNames.TryGetValue(optimizer.GetType(), out var exact))
        {
            return exact;
        }

        // Fall back to an assignability scan for user-defined subclasses of a registered optimizer. Ordered by
        // the canonical enum value so the result is stable rather than dictionary-order dependent, and narrowed
        // to the most-derived match so a subclass of a subclass reports the closer ancestor.
        Type? best = null;
        foreach (var kvp in _canonicalNames.OrderBy(k => k.Value))
        {
            if (!kvp.Key.IsInstanceOfType(optimizer)) continue;
            if (best is null || best.IsAssignableFrom(kvp.Key)) best = kvp.Key;
        }

        if (best is not null) return _canonicalNames[best];

        throw new ArgumentException(
            $"Unknown optimizer type: {optimizer.GetType().Name}. It is not one of the optimizers registered " +
            $"with {nameof(OptimizerFactory<T, TInput, TOutput>)}, and does not derive from one.");
    }

    /// <summary>
    /// Creates an optimizer of the specified type with default options and no model.
    /// The model should be set later via <see cref="IOptimizer{T, TInput, TOutput}.SetModel"/>
    /// or will be set automatically when used with <see cref="AiModelBuilder{T, TInput, TOutput}"/>.
    /// </summary>
    /// <param name="optimizerTypeEnum">The type of optimizer to create.</param>
    /// <returns>An implementation of IOptimizer for the specified optimizer type with default options.</returns>
    /// <exception cref="ArgumentException">Thrown when an unknown optimizer type is specified.</exception>
    /// <exception cref="InvalidOperationException">Thrown when instance creation fails.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a convenience method that creates an optimizer with sensible default settings.
    /// Use this when you want to select an optimizer type (like "Adam") without specifying detailed options.
    /// The optimizer is created without a model reference; call <c>SetModel()</c> before optimizing.
    /// </para>
    /// </remarks>
    internal static IOptimizer<T, TInput, TOutput> CreateOptimizer(OptimizerType optimizerTypeEnum)
        => Instantiate(optimizerTypeEnum, options: null);

    /// <summary>
    /// Creates an optimizer of the specified type with the given options.
    /// </summary>
    /// <param name="optimizerTypeEnum">The type of optimizer to create.</param>
    /// <param name="options">Configuration options for the optimizer.</param>
    /// <returns>An implementation of IOptimizer for the specified optimizer type.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when an unknown optimizer type is specified, or when the
    /// options object does not match the requested optimizer.</exception>
    /// <exception cref="InvalidOperationException">Thrown when instance creation fails.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a specific type of optimizer based on what you request, using
    /// the settings you supply. Different optimizers use different strategies to improve your model.
    /// </para>
    /// <para>
    /// The options object must be the one belonging to the optimizer you asked for — for example
    /// <c>AdamOptimizerOptions</c> with <see cref="OptimizerType.Adam"/>. Passing a mismatched options type is
    /// rejected with an explanatory error rather than silently ignored.
    /// </para>
    /// </remarks>
    public static IOptimizer<T, TInput, TOutput> CreateOptimizer(
        OptimizerType optimizerTypeEnum, OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        return Instantiate(optimizerTypeEnum, options);
    }

    /// <summary>
    /// Resolves the implementation for <paramref name="optimizerTypeEnum"/> and constructs it, supplying
    /// <paramref name="options"/> to the constructor's options parameter when one is provided.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Both public entry points share this so they cannot drift apart in how they pick a constructor. The
    /// previous options-taking overload called <c>Activator.CreateInstance(type, options)</c>, which looks for
    /// a single-argument constructor — optimizer constructors are <c>(model, options, engine)</c> with the
    /// model FIRST, and Activator does not fill in optional parameters, so that call could never bind and the
    /// overload threw <c>MissingMethodException</c> for every optimizer in the library.
    /// </para>
    /// </remarks>
    private static IOptimizer<T, TInput, TOutput> Instantiate(
        OptimizerType optimizerTypeEnum, OptimizationAlgorithmOptions<T, TInput, TOutput>? options)
    {
        if (!_optimizerTypes.TryGetValue(optimizerTypeEnum, out Type? optimizerGenericType))
        {
            throw new ArgumentException(
                $"Optimizer type '{optimizerTypeEnum}' has no implementation in this library. " +
                $"Available: {string.Join(", ", _optimizerTypes.Keys.OrderBy(k => k.ToString()))}.",
                nameof(optimizerTypeEnum));
        }

        if (optimizerGenericType == null)
        {
            throw new InvalidOperationException($"Optimizer type {optimizerTypeEnum} is registered but null.");
        }

        // When OptimizerFactory<T, TInput, TOutput> is constructed with concrete type args,
        // the registered types are already closed generics. Only call MakeGenericType if needed.
        Type concreteType = optimizerGenericType.IsGenericTypeDefinition
            ? optimizerGenericType.MakeGenericType(typeof(T))
            : optimizerGenericType;

        var constructors = concreteType.GetConstructors();
        if (constructors.Length == 0)
        {
            throw new InvalidOperationException($"No public constructors found on {concreteType.Name}.");
        }

        System.Reflection.ConstructorInfo ctor;
        if (options is null)
        {
            // Optimizer constructors have varying signatures: (model, options?, [engine?], [extras?]).
            // All pass model to OptimizerBase which stores it as IFullModel? (nullable). We pass null for
            // all parameters: model=null (set later via SetModel), options=null (use defaults).
            //
            // Prefer parameterless, then fewest required (non-default) parameters, then fewest total.
            // More stable than purely fewest-parameter selection.
            ctor = constructors
                .OrderBy(c => c.GetParameters().Length == 0 ? 0 : 1)
                .ThenBy(c => c.GetParameters().Count(p => !p.HasDefaultValue))
                .ThenBy(c => c.GetParameters().Length)
                .First();
        }
        else
        {
            // Need a constructor that can actually accept the caller's options object.
            var optionsType = options.GetType();
            var candidate = constructors
                .Where(c => c.GetParameters().Any(p => p.ParameterType.IsAssignableFrom(optionsType)))
                .OrderBy(c => c.GetParameters().Count(p => !p.HasDefaultValue))
                .ThenBy(c => c.GetParameters().Length)
                .FirstOrDefault();

            if (candidate is null)
            {
                throw new ArgumentException(
                    $"Optimizer '{optimizerTypeEnum}' ({concreteType.Name}) has no constructor accepting options " +
                    $"of type {optionsType.Name}. Pass the options type that belongs to this optimizer.",
                    nameof(options));
            }
            ctor = candidate;
        }

        var parameters = ctor.GetParameters();
        // Provide default values based on parameter types to avoid null-related constructor failures.
        // Nullable reference types and optional parameters get null; value types get their defaults.
        var args = new object?[parameters.Length];
        bool optionsPlaced = false;
        for (int i = 0; i < parameters.Length; i++)
        {
            var paramType = parameters[i].ParameterType;

            // Fill the options slot exactly once — an optimizer whose constructor took two options-shaped
            // parameters would otherwise receive the same object twice.
            if (options is not null && !optionsPlaced && paramType.IsAssignableFrom(options.GetType()))
            {
                args[i] = options;
                optionsPlaced = true;
                continue;
            }

            if (parameters[i].HasDefaultValue)
            {
                args[i] = parameters[i].DefaultValue;
            }
            else if (paramType.IsValueType)
            {
                args[i] = Activator.CreateInstance(paramType);
            }
            // else: null for reference types - optimizer constructors should handle null options with defaults
        }

        object? instance;
        try
        {
            instance = ctor.Invoke(args);
        }
        catch (System.Reflection.TargetInvocationException ex) when (ex.InnerException is not null)
        {
            throw new InvalidOperationException(
                $"Failed to create optimizer '{optimizerTypeEnum}': {ex.InnerException.Message}",
                ex.InnerException);
        }

        return instance == null
            ? throw new InvalidOperationException($"Failed to create instance of {concreteType.Name}")
            : (IOptimizer<T, TInput, TOutput>)instance;
    }
}
