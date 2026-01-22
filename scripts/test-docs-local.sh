#!/bin/bash
# AiDotNet Documentation Local Testing Script
# This script builds and serves the documentation locally for testing before CI/CD

set -e

SKIP_BUILD=false
SKIP_PLAYGROUND=false
SERVE_ONLY=false
PORT=8080

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build) SKIP_BUILD=true; shift ;;
        --skip-playground) SKIP_PLAYGROUND=true; shift ;;
        --serve-only) SERVE_ONLY=true; shift ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Get script directory and root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "======================================"
echo "AiDotNet Documentation Local Testing"
echo "======================================"
echo ""

cd "$ROOT_DIR"

# Step 1: Check prerequisites
echo "[1/6] Checking prerequisites..."

# Check for DocFX
if ! command -v docfx &> /dev/null; then
    echo "  Installing DocFX globally..."
    dotnet tool install --global docfx || dotnet tool update --global docfx
fi
echo "  DocFX found: $(docfx --version 2>&1 || echo 'installed')"

# Check for .NET SDK
DOTNET_VERSION=$(dotnet --version)
echo "  .NET SDK: $DOTNET_VERSION"

# Step 2: Build the main project (required for API docs)
if [ "$SKIP_BUILD" = false ] && [ "$SERVE_ONLY" = false ]; then
    echo ""
    echo "[2/6] Building AiDotNet..."
    dotnet build src/AiDotNet.csproj -c Release --framework net8.0
    echo "  Build successful"
else
    echo "[2/6] Skipping build (--skip-build or --serve-only)"
fi

# Step 3: Build DocFX documentation
if [ "$SERVE_ONLY" = false ]; then
    echo ""
    echo "[3/6] Building documentation with DocFX..."

    # Clean previous build
    rm -rf _site

    docfx docfx.json
    echo "  Documentation built successfully"
else
    echo "[3/6] Skipping DocFX build (--serve-only)"
fi

# Step 4: Build and copy Playground
if [ "$SKIP_PLAYGROUND" = false ] && [ "$SERVE_ONLY" = false ]; then
    echo ""
    echo "[4/6] Building Playground..."

    rm -rf _playground

    dotnet publish src/AiDotNet.Playground/AiDotNet.Playground.csproj -c Release -o _playground

    # Copy playground to _site
    echo "  Copying Playground to _site/playground..."
    mkdir -p _site/playground
    cp -r _playground/wwwroot/* _site/playground/

    # Update base href for local playground subdirectory (matches production structure)
    if [ -f "_site/playground/index.html" ]; then
        # Handle sed -i portability between macOS (BSD) and Linux (GNU)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' 's|<base href="/" />|<base href="/playground/" />|g' _site/playground/index.html
        else
            sed -i 's|<base href="/" />|<base href="/playground/" />|g' _site/playground/index.html
        fi
        echo "  Updated base href for local playground"
    fi

    echo "  Playground integrated successfully"
else
    echo "[4/6] Skipping Playground build (--skip-playground or --serve-only)"
fi

# Step 5: Verify the build
echo ""
echo "[5/6] Verifying build..."

REQUIRED_PATHS=(
    "_site/index.html"
    "_site/docs/index.html"
)

OPTIONAL_PATHS=(
    "_site/api/index.html"
    "_site/docs/tutorials/index.html"
    "_site/docs/examples/MixtureOfExpertsExample.html"
    "_site/playground/index.html"
)

ALL_VALID=true
for path in "${REQUIRED_PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "  [OK] $path"
    else
        echo "  [MISSING] $path"
        ALL_VALID=false
    fi
done

for path in "${OPTIONAL_PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "  [OK] $path"
    else
        echo "  [WARNING] $path (optional)"
    fi
done

if [ "$ALL_VALID" = false ]; then
    echo ""
    echo "  Some required files are missing. Check the DocFX output above."
fi

# Step 6: Serve the documentation
echo ""
echo "[6/6] Starting local server..."
echo ""
echo "======================================"
echo "Documentation is being served at:"
echo "  http://localhost:$PORT"
echo ""
echo "Test these URLs:"
echo "  Main:       http://localhost:$PORT"
echo "  Docs:       http://localhost:$PORT/docs/"
echo "  API:        http://localhost:$PORT/api/"
echo "  Examples:   http://localhost:$PORT/docs/examples/"
echo "  Tutorials:  http://localhost:$PORT/docs/tutorials/"
echo "  Playground: http://localhost:$PORT/playground/"
echo ""
echo "Press Ctrl+C to stop the server"
echo "======================================"
echo ""

docfx serve _site -p $PORT
