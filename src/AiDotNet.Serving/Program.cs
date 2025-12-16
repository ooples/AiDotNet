using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Services;
using AiDotNet.Serving.Security;
using AiDotNet.Serving.Security.Attestation;

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
        builder.Services.Configure<AttestationOptions>(
            builder.Configuration.GetSection("AttestationOptions"));

        // Get serving options to configure Kestrel
        var servingOptions = new ServingOptions();
        builder.Configuration.GetSection("ServingOptions").Bind(servingOptions);

        // Configure Kestrel to use the specified port
        builder.WebHost.ConfigureKestrel(serverOptions =>
        {
            serverOptions.ListenAnyIP(servingOptions.Port);
        });

        // Register services as singletons for thread-safe shared access
        builder.Services.AddSingleton<IModelRepository, ModelRepository>();
        builder.Services.AddSingleton<IRequestBatcher, RequestBatcher>();
        builder.Services.AddSingleton<ITierResolver, HeaderTierResolver>();
        builder.Services.AddSingleton<ITierPolicyProvider, DefaultTierPolicyProvider>();
        builder.Services.AddSingleton<IModelArtifactStore, InMemoryModelArtifactStore>();
        builder.Services.AddSingleton<IModelArtifactProtector, AesGcmModelArtifactProtector>();
        builder.Services.AddSingleton<IModelArtifactService, ModelArtifactService>();
        builder.Services.AddSingleton<IAttestationVerifier, DevelopmentAttestationVerifier>();

        // Register hosted service to load startup models at application start
        builder.Services.AddHostedService<ModelStartupService>();

        // Add controllers and API documentation
        builder.Services.AddControllers();
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
        app.UseAuthorization();
        app.MapControllers();

        // Log startup information
        var logger = app.Services.GetRequiredService<ILogger<Program>>();
        logger.LogInformation("AiDotNet Model Serving API is starting");
        logger.LogInformation("Swagger UI available at: http://localhost:{Port}", servingOptions.Port);

        app.Run();
    }
}
