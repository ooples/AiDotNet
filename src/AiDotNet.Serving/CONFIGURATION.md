# AiDotNet.Serving Configuration Guide

## Azure App Service Environment Variables

### Database Connection

| Variable | Required | Description |
|----------|----------|-------------|
| `ConnectionStrings__ServingDb` | Yes | EF Core connection string for the persistence database (SQLite, PostgreSQL, SQL Server, or MySQL) |
| `Persistence__Provider` | No | Database provider: `Sqlite` (default), `PostgreSQL`, `SqlServer`, `MySql` |

Example (PostgreSQL):
```
ConnectionStrings__ServingDb=Host=myserver;Database=aidotnet;Username=app;Password=secret
Persistence__Provider=PostgreSQL
```

### Stripe Payment Integration

| Variable | Required | Description |
|----------|----------|-------------|
| `Stripe__SecretKey` | Yes | Stripe secret API key (`sk_live_...` or `sk_test_...`) |
| `Stripe__WebhookSigningSecret` | Yes | Stripe webhook endpoint signing secret (`whsec_...`) |
| `Stripe__ProPriceId` | Yes | Stripe Price ID for Pro tier monthly (`price_...`) |
| `Stripe__ProAnnualPriceId` | Yes | Stripe Price ID for Pro tier annual (`price_...`) |
| `Stripe__EnterprisePriceId` | Yes | Stripe Price ID for Enterprise tier (`price_...`) |
| `Stripe__SuccessUrl` | Yes | Redirect URL after successful checkout |
| `Stripe__CancelUrl` | Yes | Redirect URL after cancelled checkout |

### Authentication (JWT)

| Variable | Required | Description |
|----------|----------|-------------|
| `Authentication__Authority` | Yes | OAuth2/OIDC authority URL |
| `Authentication__Audience` | Yes | Expected JWT audience |

### Model Serving

| Variable | Required | Description |
|----------|----------|-------------|
| `Serving__Port` | No | Server port (default: 52432) |
| `Serving__ModelDirectory` | No | Root directory for model files (default: `models`) |
| `Serving__MaxBatchSize` | No | Maximum inference batch size (default: 100) |

## GitHub Repository Secrets

All secrets required for CI/CD pipelines:

| Secret | Used By | Description |
|--------|---------|-------------|
| `AIDOTNET_BUILD_KEY` | `automated-release.yml` | 32-byte hex-encoded signing key for Layer 1 build signing and Layer 3 integrity hash |
| `NUGET_API_KEY` | `automated-release.yml` | NuGet.org API key for package publishing |
| `AUTOFIX_PAT` | `autofix.yml` | GitHub PAT for automated fix commits |
| `AUTOFIX_GPG_PRIVATE_KEY` | `autofix.yml` | GPG private key for signed autofix commits |
| `AUTOFIX_GPG_PASSPHRASE` | `autofix.yml` | Passphrase for the GPG private key |
| `AZURE_WEBAPP_PUBLISH_PROFILE` | `deploy.yml` | Azure App Service publish profile for the Serving API |
| `AZURE_FUNCTIONAPP_PUBLISH_PROFILE` | `deploy.yml` | Azure Functions publish profile |
| `CODACY_PROJECT_TOKEN` | `quality.yml` | Codacy project analysis token |
| `CODECOV_TOKEN` | `quality.yml` | Codecov upload token |
| `SONAR_TOKEN` | `quality.yml` | SonarCloud analysis token |
| `OPENAI_API_KEY` | Tests | OpenAI API key for integration tests |
| `VERCEL_TOKEN` | Website deploy | Vercel deployment token |
| `VERCEL_ORG_ID` | Website deploy | Vercel organization ID |
| `VERCEL_PROJECT_ID` | Website deploy | Vercel project ID |

## Local Development Setup

### 1. Database (SQLite - no setup needed)

SQLite is the default provider. The database file is created automatically.

### 2. Stripe (test mode)

1. Create a [Stripe test account](https://dashboard.stripe.com/test/dashboard)
2. Copy the test secret key from Developers > API keys
3. Create test products and prices in the Stripe dashboard
4. Set up a webhook endpoint pointing to `https://localhost:{port}/api/webhooks/stripe`
5. Copy the webhook signing secret

### 3. User Secrets

```bash
cd src/AiDotNet.Serving

dotnet user-secrets set "Stripe:SecretKey" "sk_test_..."
dotnet user-secrets set "Stripe:WebhookSigningSecret" "whsec_..."
dotnet user-secrets set "Stripe:ProPriceId" "price_..."
dotnet user-secrets set "Stripe:ProAnnualPriceId" "price_..."
dotnet user-secrets set "Stripe:EnterprisePriceId" "price_..."
dotnet user-secrets set "Stripe:SuccessUrl" "https://localhost:5001/success"
dotnet user-secrets set "Stripe:CancelUrl" "https://localhost:5001/cancel"
```

### 4. Build Key (optional for development)

Development builds work without a build key (Layer 1 signing and Layer 3 integrity checks are skipped). To test with a build key:

```bash
mkdir -p src/BuildKey
openssl rand 32 > src/BuildKey/BuildKey.bin

# Generate integrity hash
openssl dgst -sha256 -mac HMAC -macopt hexkey:$(xxd -p -c 256 src/BuildKey/BuildKey.bin | tr -d '\n') -binary < src/BuildKey/BuildKey.bin > src/IntegrityHash/IntegrityHash.bin
```

### 5. Run

```bash
dotnet run --project src/AiDotNet.Serving
```

## Three-Layer Encryption Architecture

| Layer | Component | Key Source | CI Requirement |
|-------|-----------|-----------|----------------|
| Layer 1 | `BuildKeyProvider` | `AIDOTNET_BUILD_KEY` secret embedded as `BuildKey.bin` | Secret must be configured |
| Layer 2 | `LicenseService` escrow | Server-side `EscrowSecret` + client `LicenseKey` | Database tables must exist |
| Layer 3 | `AssemblyIntegrityChecker` | HMAC of build key embedded as `IntegrityHash.bin` | CI step must generate hash |
