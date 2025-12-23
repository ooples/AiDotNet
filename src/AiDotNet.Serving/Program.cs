using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Persistence;
using AiDotNet.Serving.ProgramSynthesis;
using AiDotNet.Serving.Sandboxing.Docker;
using AiDotNet.Serving.Sandboxing.Execution;
using AiDotNet.Serving.Sandboxing.Sql;
using AiDotNet.Serving.Security;
using AiDotNet.Serving.Security.ApiKeys;
using AiDotNet.Serving.Security.Attestation;
using AiDotNet.Serving.Services;
using AiDotNet.Serving.Services.Federated;
using Microsoft.AspNetCore.DataProtection;
using Microsoft.EntityFrameworkCore;
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

        // Get serving options to configure Kestrel
        var servingOptions = new ServingOptions();
        builder.Configuration.GetSection("ServingOptions").Bind(servingOptions);

        // Get persistence options to configure EF Core / Data Protection
        var persistenceOptions = new ServingPersistenceOptions();
        builder.Configuration.GetSection("ServingPersistenceOptions").Bind(persistenceOptions);

        // Configure Kestrel to use the specified port
        builder.WebHost.ConfigureKestrel(serverOptions =>
        {
            serverOptions.ListenAnyIP(servingOptions.Port);
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
        builder.Services.AddAuthentication(ApiKeyAuthenticationDefaults.Scheme)
            .AddScheme<ApiKeyAuthenticationOptions, ApiKeyAuthenticationHandler>(
                ApiKeyAuthenticationDefaults.Scheme,
                options => builder.Configuration.GetSection("ApiKeyAuthenticationOptions").Bind(options));

        builder.Services.AddAuthorization(options =>
        {
            options.AddPolicy("AiDotNetAdmin", policy =>
                policy.RequireClaim(ApiKeyClaimTypes.Scope, ApiKeyScopes.Admin.ToString()));
        });

        // Register services as singletons for thread-safe shared access
        builder.Services.AddSingleton<IModelRepository, ModelRepository>();
        builder.Services.AddSingleton<IRequestBatcher, RequestBatcher>();
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
        builder.Services.AddControllers().AddJsonOptions(options =>
        {
            _ = options;
        });
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen(options =>
        {
            options.SwaggerDoc("v1", new Microsoft.OpenApi.Models.OpenApiInfo
            {
                Title = "AiDotNet Model Serving API",
                Version = "v1",
                Description = "Production-ready REST API for serving trained AiDotNet models with dynamic request batching",
                Contact = new Microsoft.OpenApi.Models.OpenApiContact
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

        // Configure CORS for development
        builder.Services.AddCors(options =>
        {
            options.AddDefaultPolicy(policy =>
            {
                policy.AllowAnyOrigin()
                      .AllowAnyMethod()
                      .AllowAnyHeader();
            });
        });

        var app = builder.Build();

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

        // Log startup information
        var logger = app.Services.GetRequiredService<ILogger<Program>>();
        logger.LogInformation("AiDotNet Model Serving API is starting");
        logger.LogInformation("Swagger UI available at: http://localhost:{Port}", servingOptions.Port);

        app.Run();
    }
}
