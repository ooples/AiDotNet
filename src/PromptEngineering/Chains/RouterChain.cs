using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.PromptEngineering.Chains;

/// <summary>
/// Chain that routes input to specialized sub-chains based on classification.
/// </summary>
/// <typeparam name="TInput">The type of input accepted by the chain.</typeparam>
/// <typeparam name="TOutput">The type of output produced by the chain.</typeparam>
/// <remarks>
/// <para>
/// A router chain analyzes input to determine its type, then routes it to the appropriate
/// specialized sub-chain for processing. This enables building modular systems with
/// specialized handling for different input categories.
/// </para>
/// <para><b>For Beginners:</b> Directs input to the right specialized handler.
///
/// Example - Customer Support Router:
/// <code>
/// var chain = new RouterChain&lt;string, string&gt;("SupportRouter", "Routes support tickets to specialists");
///
/// // Define how to classify input
/// chain.SetClassifier(text => ClassifyTicket(text));
///
/// // Register specialized handlers
/// chain.RegisterRoute("billing", new BillingChain());
/// chain.RegisterRoute("technical", new TechnicalSupportChain());
/// chain.RegisterRoute("general", new GeneralSupportChain());
///
/// // Set default route for unknown classifications
/// chain.SetDefaultRoute(new GeneralSupportChain());
///
/// // Run chain - routes to appropriate handler
/// var result = chain.Run("I have a question about my invoice");
/// // Classifier returns "billing", routes to BillingChain
/// </code>
/// </para>
/// </remarks>
public class RouterChain<TInput, TOutput> : ChainBase<TInput, TOutput>
{
    private readonly Dictionary<string, IChain<TInput, TOutput>> _routes;
    private Func<TInput, string>? _syncClassifier;
    private Func<TInput, CancellationToken, Task<string>>? _asyncClassifier;
    private IChain<TInput, TOutput>? _defaultRoute;

    /// <summary>
    /// Initializes a new instance of the RouterChain class.
    /// </summary>
    /// <param name="name">The name of the chain.</param>
    /// <param name="description">The description of the chain.</param>
    public RouterChain(string name, string description = "")
        : base(name, description)
    {
        _routes = new Dictionary<string, IChain<TInput, TOutput>>(StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Sets the classifier function that determines which route to take.
    /// </summary>
    /// <param name="classifier">Function that returns the route name for the input.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public RouterChain<TInput, TOutput> SetClassifier(Func<TInput, string> classifier)
    {
        Guard.NotNull(classifier);
        _syncClassifier = classifier;
        return this;
    }

    /// <summary>
    /// Sets the async classifier function that determines which route to take.
    /// </summary>
    /// <param name="classifier">Async function that returns the route name for the input.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public RouterChain<TInput, TOutput> SetClassifierAsync(Func<TInput, CancellationToken, Task<string>> classifier)
    {
        Guard.NotNull(classifier);
        _asyncClassifier = classifier;
        return this;
    }

    /// <summary>
    /// Registers a route with its associated chain.
    /// </summary>
    /// <param name="routeName">The name of the route (returned by classifier).</param>
    /// <param name="chain">The chain to execute for this route.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public RouterChain<TInput, TOutput> RegisterRoute(string routeName, IChain<TInput, TOutput> chain)
    {
        if (string.IsNullOrWhiteSpace(routeName))
        {
            throw new ArgumentException("Route name cannot be empty.", nameof(routeName));
        }

        Guard.NotNull(chain);
        _routes[routeName] = chain;
        return this;
    }

    /// <summary>
    /// Registers a route using a function handler.
    /// </summary>
    /// <param name="routeName">The name of the route.</param>
    /// <param name="handler">The function to execute for this route.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public RouterChain<TInput, TOutput> RegisterRoute(string routeName, Func<TInput, TOutput> handler)
    {
        if (string.IsNullOrWhiteSpace(routeName))
        {
            throw new ArgumentException("Route name cannot be empty.", nameof(routeName));
        }

        _routes[routeName] = new FunctionChain<TInput, TOutput>(routeName, handler);
        return this;
    }

    /// <summary>
    /// Sets the default route for unrecognized classifications.
    /// </summary>
    /// <param name="chain">The default chain to execute.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public RouterChain<TInput, TOutput> SetDefaultRoute(IChain<TInput, TOutput> chain)
    {
        Guard.NotNull(chain);
        _defaultRoute = chain;
        return this;
    }

    /// <summary>
    /// Sets the default route using a function handler.
    /// </summary>
    /// <param name="handler">The default function to execute.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public RouterChain<TInput, TOutput> SetDefaultRoute(Func<TInput, TOutput> handler)
    {
        _defaultRoute = new FunctionChain<TInput, TOutput>("default", handler);
        return this;
    }

    /// <summary>
    /// Gets the available route names.
    /// </summary>
    public IReadOnlyList<string> RouteNames => _routes.Keys.ToList().AsReadOnly();

    /// <summary>
    /// Gets the route that was used in the last execution.
    /// </summary>
    public string? LastRouteTaken { get; private set; }

    /// <summary>
    /// Executes the router by classifying input and routing to the appropriate chain.
    /// </summary>
    protected override TOutput RunCore(TInput input)
    {
        if (_syncClassifier == null && _asyncClassifier == null)
        {
            throw new InvalidOperationException("No classifier set. Call SetClassifier before running the chain.");
        }

        if (_asyncClassifier != null && _syncClassifier == null)
        {
            throw new InvalidOperationException("Classifier is async. Use RunAsync for chains with async classifiers.");
        }

        var routeName = _syncClassifier!(input);
        LastRouteTaken = routeName;

        if (_routes.TryGetValue(routeName, out var chain))
        {
            return chain.Run(input);
        }

        if (_defaultRoute != null)
        {
            LastRouteTaken = "default";
            return _defaultRoute.Run(input);
        }

        throw new InvalidOperationException(
            $"No route found for classification '{routeName}' and no default route set.");
    }

    /// <summary>
    /// Executes the router asynchronously by classifying input and routing to the appropriate chain.
    /// </summary>
    protected override async Task<TOutput> RunCoreAsync(TInput input, CancellationToken cancellationToken)
    {
        if (_syncClassifier == null && _asyncClassifier == null)
        {
            throw new InvalidOperationException("No classifier set. Call SetClassifier before running the chain.");
        }

        string routeName;
        if (_asyncClassifier != null)
        {
            routeName = await _asyncClassifier(input, cancellationToken).ConfigureAwait(false);
        }
        else
        {
            routeName = _syncClassifier!(input);
        }

        LastRouteTaken = routeName;

        if (_routes.TryGetValue(routeName, out var chain))
        {
            return await chain.RunAsync(input, cancellationToken).ConfigureAwait(false);
        }

        if (_defaultRoute != null)
        {
            LastRouteTaken = "default";
            return await _defaultRoute.RunAsync(input, cancellationToken).ConfigureAwait(false);
        }

        throw new InvalidOperationException(
            $"No route found for classification '{routeName}' and no default route set.");
    }

    /// <summary>
    /// A simple chain wrapper for function handlers.
    /// </summary>
    private class FunctionChain<TIn, TOut> : IChain<TIn, TOut>
    {
        private readonly Func<TIn, TOut> _handler;

        public FunctionChain(string name, Func<TIn, TOut> handler)
        {
            Name = name;
            Description = string.Empty;
            _handler = handler;
        }

        public string Name { get; }
        public string Description { get; }

        public TOut Run(TIn input) => _handler(input);

        public Task<TOut> RunAsync(TIn input, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(_handler(input));
        }

        public bool ValidateInput(TIn input) => input != null;
    }
}
