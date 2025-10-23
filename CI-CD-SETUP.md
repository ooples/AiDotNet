# CI/CD Pipeline Setup Instructions

This document provides step-by-step instructions to configure the CI/CD pipeline for AiDotNet.

## Required GitHub Secrets

You need to add these secrets to your GitHub repository:

### 1. NUGET_API_KEY (Required)
**Purpose**: Publish packages to NuGet.org

**Steps**:
1. Go to https://www.nuget.org/account/apikeys
2. Click "Create"
3. Name: "AiDotNet CI/CD"
4. Select "Push" permission
5. Glob pattern: `AiDotNet*`
6. Click "Create" and copy the key
7. Go to GitHub repo ’ Settings ’ Secrets and variables ’ Actions
8. Click "New repository secret"
9. Name: `NUGET_API_KEY`
10. Paste the key and click "Add secret"

### 2. SONAR_TOKEN (Required)
**Purpose**: Code quality analysis with SonarCloud

**Steps**:
1. Go to https://sonarcloud.io
2. Click "Log in" ’ "With GitHub"
3. Click "+" icon ’ "Analyze new project"
4. Select "ooples/AiDotNet"
5. Choose "With GitHub Actions"
6. Copy the token shown
7. In GitHub repo ’ Settings ’ Secrets and variables ’ Actions
8. Create secret named `SONAR_TOKEN` with the copied token

**Configuration** (already set in workflows):
- Organization: `ooples`
- Project Key: `ooples_AiDotNet`

### 3. EMAIL_USERNAME (Optional)
**Purpose**: Send email notifications

**Steps**:
1. Use a Gmail account
2. In GitHub repo ’ Settings ’ Secrets and variables ’ Actions
3. Create secret named `EMAIL_USERNAME`
4. Value: your Gmail address (e.g., noreply.aidotnet@gmail.com)

### 4. EMAIL_PASSWORD (Optional)
**Purpose**: Gmail App Password for email notifications

**Steps**:
1. Go to https://myaccount.google.com/apppasswords
2. Sign in with your Gmail
3. App name: "AiDotNet GitHub Actions"
4. Click "Create" and copy the 16-character password
5. In GitHub ’ Create secret named `EMAIL_PASSWORD`
6. Paste the app password

**Note**: Emails will be sent to cheatcountry@gmail.com

### 5. SLACK_WEBHOOK_URL (Optional)
**Purpose**: Send notifications to Slack

**Steps**:
1. Go to https://api.slack.com/messaging/webhooks
2. Click "Create your Slack app" ’ "From scratch"
3. App name: "AiDotNet CI/CD"
4. Choose your workspace
5. Enable "Incoming Webhooks" ’ Toggle ON
6. Click "Add New Webhook to Workspace"
7. Select channel (e.g., #builds)
8. Copy the webhook URL (starts with https://hooks.slack.com/services/)
9. In GitHub ’ Create secret named `SLACK_WEBHOOK_URL`
10. Paste the webhook URL

## GitHub Pages Setup

**Purpose**: Host auto-generated API documentation

**Steps**:
1. Go to repository Settings ’ Pages
2. Source: "Deploy from a branch"
3. Branch: `gh-pages`
4. Folder: `/ (root)`
5. Click "Save"

The `gh-pages` branch will be created automatically on first documentation build.

Documentation will be available at: https://ooples.github.io/AiDotNet/

## Pipeline Overview

### Build Workflow (Every Commit)
- Builds all frameworks (net462, net6.0, net7.0, net8.0)
- Runs Roslyn analyzers
- Checks code formatting
- **Runtime**: ~2-5 minutes

### PR Validation Workflow (Every Pull Request)
- Everything from Build workflow
- SonarCloud code quality analysis
- CodeQL security scanning
- Runs tests with coverage
- Creates NuGet package artifact
- Sends failure notifications
- **Runtime**: ~10-20 minutes

### Publish Workflow (Master Branch & Tags)
**On master branch commits:**
- Creates preview package (e.g., 0.0.5-preview.123)
- Publishes to NuGet.org and GitHub Packages

**On tagged releases (e.g., v1.0.0):**
- Creates release package (e.g., 1.0.0)
- Publishes to NuGet.org
- Creates GitHub Release with changelog
- Attaches package files

### Documentation Workflow (Master Branch)
- Generates API docs with DocFX
- Publishes to GitHub Pages
- Runs when .cs files change

## Verify Setup

### Test the Pipeline
1. Create a test branch: `git checkout -b test-ci`
2. Make a small change to any .cs file
3. Commit and push: `git commit -am "Test CI/CD" && git push -u origin test-ci`
4. Create a Pull Request
5. Check Actions tab - you should see workflows running

### Check Required Secrets
Go to Settings ’ Secrets and variables ’ Actions

You should have:
-  NUGET_API_KEY (required)
-  SONAR_TOKEN (required)
-    EMAIL_USERNAME (optional)
-    EMAIL_PASSWORD (optional)
-    SLACK_WEBHOOK_URL (optional)

## Troubleshooting

### "Secret not found" Error
- Verify secret names are exactly as shown (case-sensitive)
- Secrets must be added to repository settings, not organization

### SonarCloud Fails
- Check SONAR_TOKEN is valid
- Verify organization is `ooples`
- Verify project key is `ooples_AiDotNet`

### NuGet Publish Fails
- Check NUGET_API_KEY is valid and has Push permission
- Version must not already exist on NuGet.org
- API key glob pattern must match `AiDotNet*`

### Email Notifications Don't Work
- Must use Gmail App Password, not regular password
- Gmail account must have 2FA enabled
- Check EMAIL_USERNAME and EMAIL_PASSWORD are correct

### Self-Hosted Runner Issues
- Check Settings ’ Actions ’ Runners (runner must be online)
- Runner needs .NET 6, 7, and 8 SDKs installed
- Runner needs internet access for NuGet restore

## Next Steps

Once pipeline is configured:

1. **Merge PR #137** (Fix interpretability initialization errors)
2. **Run create-user-stories** for 73 placeholder implementations
3. **Create test coverage** to achieve 80% minimum
4. **First release**: Tag v1.0.0 when ready

## Support

For issues:
1. Check Actions tab for detailed logs
2. Verify all secret names match exactly
3. Check runner is online (for self-hosted)
4. Review SonarCloud dashboard for quality issues
