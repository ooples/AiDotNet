using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Persistence;
using AiDotNet.Serving.ProgramSynthesis;
using AiDotNet.Serving.Sandboxing.Docker;
using AiDotNet.Serving.Sandboxing.Execution;
using AiDotNet.Serving.Sandboxing.Sql;
using AiDotNet.Serving.Security;
using AiDotNet.Serving.Security.ApiKeys;
using AiDotNet.Serving.Security.Attestation;
using AiDotNet.Serving.Security.Licensing;
using AiDotNet.Serving.Services;
using AiDotNet.Serving.Services.Federated;
using Microsoft.AspNetCore.DataProtection;
using Microsoft.EntityFrameworkCore;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Serialization;
using Pomelo.EntityFrameworkCore.MySql.Infrastructure;

namespace AiDotNet.Serving;

/// <summary>
/// Main entry point for the AiDotNet Model Serving API.
/// This application provides a production-ready REST API for serving trained AiDotNet models
/// with dynamic request batching to maximize throughput.
/// </summary>
public class Program
{
    public static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);

        // Configure options
        builder.Services.Configure<ServingOptions>(
            builder.Configuration.GetSection("ServingOptions"));
        builder.Services.Configure<TierEnforcementOptions>(
            builder.Configuration.GetSection("TierEnforcementOptions"));
        builder.Services.Configure<ServingPersistenceOptions>(
            builder.Configuration.GetSection("ServingPersistenceOptions"));
        builder.Services.Configure<AttestationOptions>(
            builder.Configuration.GetSection("AttestationOptions"));
        builder.Services.Configure<ServingSandboxOptions>(
            builder.Configuration.GetSection("ServingSandbox"));
        builder.Services.Configure<ServingSqlSandboxOptions>(
            builder.Configuration.GetSection("ServingSqlSandbox"));
        builder.Services.Configure<ServingProgramSynthesisOptions>(
            builder.Configuration.GetSection("ServingProgramSynthesis"));
        builder.Services.Configure<StripeOptions>(
            builder.Configuration.GetSection("Stripe"));

        // Get serving options to configure Kestrel
        var servingOptions = new ServingOptions();
        builder.Configuration.GetSection("ServingOptions").Bind(servingOptions);

        // Get persistence options to configure EF Core / Data Protection
        var persistenceOptions = new ServingPersistenceOptions();
        builder.Configuration.GetSection("ServingPersistenceOptions").Bind(persistenceOptions);

        // Configure Kestrel to use the specified port (or PORT env var for Azure/container hosting)
        var port = Environment.GetEnvironmentVariable("PORT") is string envPort && int.TryParse(envPort, out var parsedPort)
            ? parsedPort
            : servingOptions.Port;

        if (port < 1 || port > 65535)
        {
            throw new InvalidOperationException($"Port {port} is out of valid range (1-65535).");
        }

        builder.WebHost.ConfigureKestrel(serverOptions =>
        {
            serverOptions.ListenAnyIP(port);
        });

        // Persistence (DB-backed): API keys, protected artifact keys, and ASP.NET Core Data Protection keys.
        static void ConfigureServingDb(DbContextOptionsBuilder options, ServingPersistenceOptions persistence)
        {
            static ServerVersion CreateMySqlServerVersion(ServingPersistenceOptions persistenceOptions)
            {
                var rawVersion = persistenceOptions.MySqlServerVersion ?? string.Empty;
                if (!Version.TryParse(rawVersion, out var parsedVersion))
                {
                    throw new InvalidOperationException(
                        $"Invalid ServingPersistenceOptions.MySqlServerVersion '{rawVersion}'. Expected a version like '8.0.21'.");
                }

                return new MySqlServerVersion(parsedVersion);
            }

            string connectionString = persistence.ConnectionString ?? string.Empty;
            if (string.IsNullOrWhiteSpace(connectionString) && persistence.Provider == ServingDatabaseProvider.Sqlite)
            {
                var dataDir = Path.Combine(AppContext.BaseDirectory, "data");
                Directory.CreateDirectory(dataDir);
                connectionString = $"Data Source={Path.Combine(dataDir, "aidn-serving.db")};";
            }

            switch (persistence.Provider)
            {
                case ServingDatabaseProvider.PostgreSql:
                    options.UseNpgsql(connectionString);
                    break;
                case ServingDatabaseProvider.SqlServer:
                    options.UseSqlServer(connectionString);
                    break;
                case ServingDatabaseProvider.MySql:
                    options.UseMySql(connectionString, CreateMySqlServerVersion(persistence));
                    break;
                default:
                    options.UseSqlite(connectionString);
                    break;
            }

            options.ConfigureWarnings(w =>
                w.Log(Microsoft.EntityFrameworkCore.Diagnostics.RelationalEventId.PendingModelChangesWarning));
        }

        builder.Services.AddDbContext<ServingDbContext>(
            options => ConfigureServingDb(options, persistenceOptions),
            contextLifetime: ServiceLifetime.Scoped,
            optionsLifetime: ServiceLifetime.Singleton);

        builder.Services.AddDataProtection()
            .SetApplicationName("AiDotNet.Serving")
            .PersistKeysToDbContext<ServingDbContext>();

        // API key authentication (tier enforcement)
        builder.Services.AddScoped<IApiKeyService, ApiKeyService>();

        // License key management
        builder.Services.AddScoped<ILicenseService, LicenseService>();

        // Stripe payment integration
        builder.Services.AddScoped<IStripeService, StripeService>();
        builder.Services.AddAuthentication(ApiKeyAuthenticationDefaults.Scheme)
            .AddScheme<ApiKeyAuthenticationOptions, ApiKeyAuthenticationHandler>(
                ApiKeyAuthenticationDefaults.Scheme,
                options => builder.Configuration.GetSection("ApiKeyAuthenticationOptions").Bind(options));

        builder.Services.AddAuthorization(options =>
        {
            options.AddPolicy("AiDotNetAdmin", policy =>
                policy.RequireClaim(ApiKeyClaimTypes.Scope, ApiKeyScopes.Admin.ToString()));

            // Default authorization is REQUIRED for every endpoint that
            // doesn't explicitly opt out with [AllowAnonymous]. Without
            // this fallback, the AddAuthentication registration above
            // only takes effect on controllers/actions that carry an
            // [Authorize] attribute — leaving inference, embeddings,
            // federated, models, and program-synthesis controllers
            // reachable without credentials.
            //
            // Endpoints that legitimately serve anonymous traffic:
            //   - /health (minimal API below, tagged .AllowAnonymous())
            //   - Stripe checkout-session creation + webhook
            //     (StripeController, [AllowAnonymous] on those actions —
            //     webhook authenticates via Stripe-Signature header,
            //     checkout-session is intentionally public)
            //   - License-key validation (LicenseValidationController,
            //     [AllowAnonymous] on the class — the license key itself
            //     IS the credential, so requiring an API key first would
            //     be circular)
            //
            // Swagger UI is registered via UseSwagger/UseSwaggerUI
            // middleware (NOT an MVC endpoint), so it bypasses the
            // FallbackPolicy entirely — and Swagger is only mounted in
            // Development environments anyway (see UseSwagger gate below).
            options.FallbackPolicy = new Microsoft.AspNetCore.Authorization.AuthorizationPolicyBuilder()
                .RequireAuthenticatedUser()
                .Build();
        });

        // Register services as singletons for thread-safe shared access
        builder.Services.AddSingleton<IModelRepository, ModelRepository>();
        AddConfiguredRequestBatcher(builder.Services);
        builder.Services.AddHttpContextAccessor();
        builder.Services.AddSingleton<ITierResolver, ClaimsTierResolver>();
        builder.Services.AddSingleton<ITierPolicyProvider, DefaultTierPolicyProvider>();
        builder.Services.AddSingleton<IModelArtifactStore, DbModelArtifactStore>();
        builder.Services.AddSingleton<IModelArtifactProtector, AesGcmModelArtifactProtector>();
        builder.Services.AddSingleton<IModelArtifactService, ModelArtifactService>();
        builder.Services.AddSingleton<IAttestationVerifier, DevelopmentAttestationVerifier>();
        builder.Services.AddSingleton<IFederatedRunStore, InMemoryFederatedRunStore>();
        builder.Services.AddSingleton<IFederatedCoordinatorService, FederatedCoordinatorService>();

        // Program synthesis + sandboxing (opt-in endpoints for sandboxed execution via AiDotNet.Serving).
        builder.Services.AddSingleton<IServingRequestContextAccessor, ServingRequestContextAccessor>();
        builder.Services.AddSingleton<IServingRequestContextResolver, ServingRequestContextResolver>();
        builder.Services.AddSingleton<IDockerRunner, DockerRunner>();
        builder.Services.AddSingleton<IProgramSandboxExecutor, DockerProgramSandboxExecutor>();
        builder.Services.AddSingleton<ISqlSandboxExecutor, SqlSandboxExecutor>();
        builder.Services.AddSingleton<IServingSqlExecuteResponseRedactor, ServingSqlExecuteResponseRedactor>();
        builder.Services.AddSingleton<IServingProgramSynthesisConcurrencyLimiter, ServingProgramSynthesisConcurrencyLimiter>();
        builder.Services.AddSingleton<IServingProgramExecuteResponseRedactor, ServingProgramExecuteResponseRedactor>();
        builder.Services.AddSingleton<IServingProgramEvaluateIoResponseRedactor, ServingProgramEvaluateIoResponseRedactor>();
        builder.Services.AddSingleton<IServingProgramEvaluator, ServingProgramEvaluator>();
        builder.Services.AddSingleton<IServingCodeTaskRequestValidator, ServingCodeTaskRequestValidator>();
        builder.Services.AddSingleton<IServingCodeTaskResultRedactor, ServingCodeTaskResultRedactor>();
        builder.Services.AddSingleton<IServingCodeTaskExecutor, ServingCodeTaskExecutor>();

        // Register hosted service to load startup models at application start
        builder.Services.AddHostedService<ModelStartupService>();

        // Add controllers and API documentation
        builder.Services.AddControllers().AddNewtonsoftJson(options =>
        {
            options.SerializerSettings.Converters.Add(new StringEnumConverter(new CamelCaseNamingStrategy(), allowIntegerValues: false));
        });
        builder.Services.AddEndpointsApiExplorer();

        builder.Services.AddSwaggerGen(options =>
        {
            options.SwaggerDoc("v1", new Microsoft.OpenApi.OpenApiInfo
            {
                Title = "AiDotNet Model Serving API",
                Version = "v1",
                Description = "Production-ready REST API for serving trained AiDotNet models with dynamic request batching",
                Contact = new Microsoft.OpenApi.OpenApiContact
                {
                    Name = "AiDotNet",
                    Url = new Uri("https://github.com/ooples/AiDotNet")
                }
            });

            // Include XML comments for API documentation
            var xmlFile = $"{System.Reflection.Assembly.GetExecutingAssembly().GetName().Name}.xml";
            var xmlPath = Path.Combine(AppContext.BaseDirectory, xmlFile);
            if (File.Exists(xmlPath))
            {
                options.IncludeXmlComments(xmlPath);
            }
        });

        // Configure CORS. Wide-open (AllowAnyOrigin/Method/Header) is
        // appropriate for local development but a real attack surface
        // in production — a misconfigured proxy + this policy is enough
        // for cross-origin POST against unprotected serving endpoints.
        // Production reads an allow-list from configuration
        // ("Cors:AllowedOrigins": ["https://app.example.com", ...]) and
        // applies it ONLY to that origin list. The dev fallback stays
        // unchanged so local hacking experience is unaffected.
        // Resolve the production CORS allow-list once during service
        // registration so we can both (a) bind it into the AddCors
        // policy and (b) log a startup warning via the host logger AFTER
        // builder.Build() when the list is empty. The warning lives on
        // the host logger (not System.Diagnostics.Trace) so it shows up
        // in standard ASP.NET Core logging pipelines that operators
        // configure (e.g. Serilog / OpenTelemetry log exporters).
        string[] productionCorsOrigins = builder.Environment.IsDevelopment()
            ? System.Array.Empty<string>()
            : builder.Configuration.GetSection("Cors:AllowedOrigins")
                .Get<string[]>() ?? System.Array.Empty<string>();

        builder.Services.AddCors(options =>
        {
            if (builder.Environment.IsDevelopment())
            {
                options.AddDefaultPolicy(policy =>
                {
                    policy.AllowAnyOrigin()
                          .AllowAnyMethod()
                          .AllowAnyHeader();
                });
            }
            else
            {
                options.AddDefaultPolicy(policy =>
                {
                    if (productionCorsOrigins.Length > 0)
                    {
                        policy.WithOrigins(productionCorsOrigins)
                              .AllowAnyMethod()
                              .AllowAnyHeader();
                    }
                    // else: no CORS at all in production unless explicitly
                    // configured — fail closed (warning logged below).
                });
            }
        });

        var app = builder.Build();

        // Surface the fail-closed default through the host logger so
        // operators who deployed without configuring Cors:AllowedOrigins
        // see a clear startup warning in their standard logs instead of
        // debugging "why does my cross-origin request silently fail in
        // production".
        if (!app.Environment.IsDevelopment() && productionCorsOrigins.Length == 0)
        {
            app.Logger.LogWarning(
                "CORS Cors:AllowedOrigins is empty in a non-Development environment. " +
                "Cross-origin requests will be REJECTED. Configure Cors:AllowedOrigins " +
                "(e.g. [\"https://app.example.com\"]) to enable CORS from specific origins, " +
                "or accept the fail-closed default for same-origin-only deployments.");
        }

        // Apply migrations on startup (configurable)
        if (persistenceOptions.MigrateOnStartup)
        {
            using var scope = app.Services.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<ServingDbContext>();
            db.Database.Migrate();
        }

        // Configure the HTTP request pipeline
        if (app.Environment.IsDevelopment())
        {
            app.UseSwagger();
            app.UseSwaggerUI(options =>
            {
                options.SwaggerEndpoint("/swagger/v1/swagger.json", "AiDotNet Serving API v1");
                options.RoutePrefix = string.Empty; // Serve Swagger UI at root
            });
        }

        app.UseCors();
        app.UseAuthentication();
        app.UseMiddleware<ServingRequestContextMiddleware>();
        app.UseAuthorization();
        app.MapControllers();

        // Health check endpoint for Azure warmup probes. AllowAnonymous
        // so the FallbackPolicy (RequireAuthenticatedUser) doesn't block
        // load-balancer / Azure App Service health probes that arrive
        // without credentials.
        app.MapGet("/health", () => Results.Ok(new { status = "healthy" }))
            .AllowAnonymous();

        // Log startup information
        var logger = app.Services.GetRequiredService<ILogger<Program>>();
        logger.LogInformation("AiDotNet Model Serving API is starting");
        logger.LogInformation("Swagger UI available at: http://localhost:{Port}", servingOptions.Port);

        app.Run();
    }

    /// <summary>
    /// Registers the <see cref="IRequestBatcher"/> chosen by the configured
    /// <see cref="ServingOptions.BatchingStrategy"/>. The <see cref="BatchingStrategyType.Continuous"/>
    /// strategy is true vLLM-style continuous batching — per-iteration admission/eviction of requests
    /// through a running scheduler loop — which only <see cref="ContinuousBatchingRequestBatcher"/>
    /// implements. The other strategies (Adaptive/Timeout/Size/Bucket) are batch-sizing policies that
    /// the static <see cref="RequestBatcher"/> applies internally, so they resolve to it. Both batchers
    /// share the same constructor dependencies, so each is built via <see cref="ActivatorUtilities"/>.
    /// </summary>
    internal static void AddConfiguredRequestBatcher(IServiceCollection services)
    {
        services.AddSingleton<IRequestBatcher>(sp =>
        {
            var servingOptions = sp.GetRequiredService<Microsoft.Extensions.Options.IOptions<ServingOptions>>();
            return servingOptions.Value.BatchingStrategy == BatchingStrategyType.Continuous
                ? ActivatorUtilities.CreateInstance<ContinuousBatchingRequestBatcher>(sp)
                : ActivatorUtilities.CreateInstance<RequestBatcher>(sp);
        });
    }
}
