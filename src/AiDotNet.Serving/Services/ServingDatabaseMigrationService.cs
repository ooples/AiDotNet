using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Persistence;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using AiDotNet.Validation;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Applies EF Core migrations for the Serving persistence database at startup.
/// </summary>
public sealed class ServingDatabaseMigrationService : IHostedService
{
    private readonly IServiceScopeFactory _scopeFactory;
    private readonly ServingDatabaseOptions _dbOptions;
    private readonly ILogger<ServingDatabaseMigrationService> _logger;

    public ServingDatabaseMigrationService(
        IServiceScopeFactory scopeFactory,
        IOptions<ServingDatabaseOptions> dbOptions,
        ILogger<ServingDatabaseMigrationService> logger)
    {
        Guard.NotNull(scopeFactory);
        _scopeFactory = scopeFactory;
        Guard.NotNull(dbOptions);
        _dbOptions = dbOptions.Value;
        Guard.NotNull(logger);
        _logger = logger;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        if (!_dbOptions.ApplyMigrationsAtStartup)
        {
            return;
        }

        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<ServingDbContext>();

        _logger.LogInformation("Applying Serving DB migrations (Provider={Provider}).", _dbOptions.Provider);
        await db.Database.MigrateAsync(cancellationToken).ConfigureAwait(false);
        _logger.LogInformation("Serving DB migrations applied.");
    }

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}

