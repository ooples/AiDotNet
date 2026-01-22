# AiDotNet Documentation Local Testing Script
# This script builds and serves the documentation locally for testing before CI/CD

param(
    [switch]$SkipBuild,
    [switch]$SkipPlayground,
    [switch]$ServeOnly,
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "AiDotNet Documentation Local Testing" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Change to root directory
Push-Location $RootDir

try {
    # Step 1: Check prerequisites
    Write-Host "[1/6] Checking prerequisites..." -ForegroundColor Yellow

    # Check for DocFX
    $docfxInstalled = Get-Command docfx -ErrorAction SilentlyContinue
    if (-not $docfxInstalled) {
        Write-Host "  Installing DocFX globally..." -ForegroundColor Gray
        dotnet tool install --global docfx
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  DocFX already installed, updating..." -ForegroundColor Gray
            dotnet tool update --global docfx
        }
    }
    else {
        Write-Host "  DocFX found: $((docfx --version) 2>&1)" -ForegroundColor Green
    }

    # Check for .NET SDK
    $dotnetVersion = dotnet --version
    Write-Host "  .NET SDK: $dotnetVersion" -ForegroundColor Green

    # Step 2: Build the main project (required for API docs)
    if (-not $SkipBuild -and -not $ServeOnly) {
        Write-Host ""
        Write-Host "[2/6] Building AiDotNet..." -ForegroundColor Yellow
        dotnet build src/AiDotNet.csproj -c Release --framework net10.0
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed!"
        }
        Write-Host "  Build successful" -ForegroundColor Green
    }
    else {
        Write-Host "[2/6] Skipping build (--SkipBuild or --ServeOnly)" -ForegroundColor Gray
    }

    # Step 3: Build DocFX documentation
    if (-not $ServeOnly) {
        Write-Host ""
        Write-Host "[3/6] Building documentation with DocFX..." -ForegroundColor Yellow

        # Clean previous build
        if (Test-Path "_site") {
            Remove-Item -Recurse -Force "_site"
        }

        docfx docfx.json
        if ($LASTEXITCODE -ne 0) {
            throw "DocFX build failed!"
        }
        Write-Host "  Documentation built successfully" -ForegroundColor Green
    }
    else {
        Write-Host "[3/6] Skipping DocFX build (--ServeOnly)" -ForegroundColor Gray
    }

    # Step 4: Build and copy Playground
    if (-not $SkipPlayground -and -not $ServeOnly) {
        Write-Host ""
        Write-Host "[4/6] Building Playground..." -ForegroundColor Yellow

        $playgroundDir = "_playground"
        if (Test-Path $playgroundDir) {
            Remove-Item -Recurse -Force $playgroundDir
        }

        dotnet publish src/AiDotNet.Playground/AiDotNet.Playground.csproj -c Release -o $playgroundDir
        if ($LASTEXITCODE -ne 0) {
            throw "Playground build failed!"
        }

        # Copy playground to _site
        Write-Host "  Copying Playground to _site/playground..." -ForegroundColor Gray
        $playgroundDest = "_site/playground"
        if (-not (Test-Path $playgroundDest)) {
            New-Item -ItemType Directory -Force -Path $playgroundDest | Out-Null
        }

        Copy-Item -Path "$playgroundDir/wwwroot/*" -Destination $playgroundDest -Recurse -Force

        # Update base href for local playground subdirectory (matches production structure)
        $playgroundIndex = Join-Path $playgroundDest "index.html"
        if (Test-Path $playgroundIndex) {
            $content = Get-Content -Path $playgroundIndex -Raw
            $content = $content -replace '<base href="/" />', '<base href="/playground/" />'
            Set-Content -Path $playgroundIndex -Value $content -NoNewline
            Write-Host "  Updated base href for local playground" -ForegroundColor Gray
        }

        Write-Host "  Playground integrated successfully" -ForegroundColor Green
    }
    else {
        Write-Host "[4/6] Skipping Playground build (--SkipPlayground or --ServeOnly)" -ForegroundColor Gray
    }

    # Step 5: Verify the build
    Write-Host ""
    Write-Host "[5/6] Verifying build..." -ForegroundColor Yellow

    $requiredPaths = @(
        "_site/index.html",
        "_site/docs/index.html"
    )

    $optionalPaths = @(
        "_site/api/index.html",
        "_site/docs/tutorials/index.html",
        "_site/docs/examples/MixtureOfExpertsExample.html",
        "_site/playground/index.html"
    )

    $allValid = $true
    foreach ($path in $requiredPaths) {
        if (Test-Path $path) {
            Write-Host "  [OK] $path" -ForegroundColor Green
        }
        else {
            Write-Host "  [MISSING] $path" -ForegroundColor Red
            $allValid = $false
        }
    }

    foreach ($path in $optionalPaths) {
        if (Test-Path $path) {
            Write-Host "  [OK] $path" -ForegroundColor Green
        }
        else {
            Write-Host "  [WARNING] $path (optional)" -ForegroundColor Yellow
        }
    }

    if (-not $allValid) {
        Write-Host ""
        Write-Host "  Some required files are missing. Check the DocFX output above." -ForegroundColor Red
    }

    # Step 6: Serve the documentation
    Write-Host ""
    Write-Host "[6/6] Starting local server..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host "Documentation is being served at:" -ForegroundColor White
    Write-Host "  http://localhost:$Port" -ForegroundColor Green
    Write-Host ""
    Write-Host "Test these URLs:" -ForegroundColor White
    Write-Host "  Main:       http://localhost:$Port" -ForegroundColor Gray
    Write-Host "  Docs:       http://localhost:$Port/docs/" -ForegroundColor Gray
    Write-Host "  API:        http://localhost:$Port/api/" -ForegroundColor Gray
    Write-Host "  Examples:   http://localhost:$Port/docs/examples/" -ForegroundColor Gray
    Write-Host "  Tutorials:  http://localhost:$Port/docs/tutorials/" -ForegroundColor Gray
    Write-Host "  Playground: http://localhost:$Port/playground/" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host ""

    docfx serve _site -p $Port
}
catch {
    Write-Host ""
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
