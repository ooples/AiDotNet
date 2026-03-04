# AiDotNet Launch Plan - Complete Setup Instructions

## Current State (Updated)

| Component | Status | Notes |
|-----------|--------|-------|
| AiDotNet (NuGet) | v0.0.5-preview | 7,300+ classes, 139 modules. Needs 1.0.0 release |
| AiDotNet.Tensors (NuGet) | v0.8.0 | Separate repo. Custom GPU kernels via P/Invoke (CUDA, OpenCL, HIP, Metal, Vulkan, WebGPU) |
| AiDotNet.Serving | Code only, NOT deployed | 193 files, 11 controllers, Stripe/auth/licensing built in |
| Website | **DEPLOYED to Vercel** | **67 pages** built & deployed. Project: `franklins-projects-02a0b5a0/website` |
| Auth Pages | **BUILT & DEPLOYED** | Login, signup, callback, dashboard, API keys, usage, billing, settings |
| Supabase Client | **INSTALLED** | `@supabase/supabase-js` added, `src/lib/supabase.ts` created |
| Supabase DB | **NOT SET UP** | Need correct SUPABASE_ACCESS_TOKEN (current one is wrong/too short) |
| Navbar Auth | **BUILT & DEPLOYED** | Shows Sign In when logged out, avatar dropdown when logged in |
| Azure API | **App created** | `aidotnet-serving.azurewebsites.net` on F1 (Free), CORS configured, needs code deploy |
| Vercel Env Vars | **Partial** | `PUBLIC_SUPABASE_URL` set. Still need `PUBLIC_SUPABASE_ANON_KEY`, Stripe links |
| DNS / Domains | **NOT CONFIGURED** | All 3 domains still at registrar defaults. Need A/CNAME records for Vercel |
| Stripe | Code exists, not connected | Payment Links need to be created |
| Email | NONE | Need email forwarding for all 3 domains |

### What YOU Need To Do (Manual Steps)

**CRITICAL - DNS must be done first** so the website is accessible:

- [ ] **Configure DNS for all 3 domains** (Section 2) - A record: `76.76.21.21`, CNAME www: `cname.vercel-dns.com`
- [ ] **Add domains in Vercel dashboard** (Settings > Domains) - the CLI lacks permission
- [ ] **Fix SUPABASE_ACCESS_TOKEN** - get from https://supabase.com/dashboard/account/tokens (current one is wrong)
- [ ] Set up ImprovMX email forwarding (Section 1)
- [ ] Create Stripe products and payment links (Section 4)
- [ ] Run Supabase SQL migration (Section 5) - I can do this once token is fixed
- [ ] Set up GitHub OAuth app (Section 6)
- [ ] Add remaining Vercel env vars: `PUBLIC_SUPABASE_ANON_KEY`, Stripe links (Section 7)
- [ ] Deploy AiDotNet.Serving code to Azure (Section 8) - App Service already created
- [ ] Set up monitoring (Section 9)

### GitHub Secrets Needed (repo Settings > Secrets and variables > Actions)

| Secret | Value | For |
|--------|-------|-----|
| `VERCEL_TOKEN` | Get from https://vercel.com/account/tokens | Website deploy workflow |
| `VERCEL_ORG_ID` | `team_5HDGYIA59nu3JwRl3ZYCMGYV` | Website deploy workflow |
| `VERCEL_PROJECT_ID` | `prj_JyDpz7IdLTf5WcEWHyxUKq7TcCSH` | Website deploy workflow |
| `AZURE_WEBAPP_PUBLISH_PROFILE` | Run: `az webapp deployment list-publishing-profiles --name aidotnet-serving --resource-group aidotnet-rg --xml` | Serving API deploy |

### What's Already Done (Automated)

- [x] Website code deployed to Vercel production (67 pages)
- [x] Azure resource group exists (`aidotnet-rg`)
- [x] Azure App Service created (`aidotnet-serving.azurewebsites.net`, F1 tier)
- [x] Azure CORS configured for all 3 domains + vercel
- [x] Vercel `PUBLIC_SUPABASE_URL` env var set
- [x] Logo JPEG created for Stripe (`public/logo.jpg`)
- [x] GitHub Actions workflow: `deploy-website.yml` (Vercel)
- [x] GitHub Actions workflow: `deploy-serving.yml` (Azure App Service)

---

## Section 1: Email Setup (ImprovMX)

ImprovMX provides free email forwarding so you can receive email at `support@aidotnet.dev` etc.

### Step 1: Create ImprovMX Account

1. Go to https://improvmx.com
2. Click **Sign Up** (free tier supports up to 25 aliases per domain)
3. Sign up with your personal email address

### Step 2: Add Your Primary Domain

1. After logging in, click **Add Domain**
2. Enter `aidotnet.dev`
3. Set the default forwarding destination to your personal email
4. ImprovMX will show you the DNS records you need to add

### Step 3: Add DNS Records for aidotnet.dev

In **Namecheap** dashboard:

1. Go to **Domain List** > `aidotnet.dev` > **Manage** > **Advanced DNS**
2. Add these **MX Records**:

| Type | Host | Value | Priority |
|------|------|-------|----------|
| MX | @ | `mx1.improvmx.com` | 10 |
| MX | @ | `mx2.improvmx.com` | 20 |

3. Add this **TXT Record** for SPF (allows ImprovMX to send on your behalf):

| Type | Host | Value |
|------|------|-------|
| TXT | @ | `v=spf1 include:spf.improvmx.com ~all` |

4. **Important**: If there's an existing MX record (like Namecheap's email forwarding), **delete it first**

### Step 4: Create Email Aliases

In the ImprovMX dashboard for `aidotnet.dev`, create these aliases:

| Alias | Forwards To |
|-------|------------|
| `support@aidotnet.dev` | your personal email |
| `hello@aidotnet.dev` | your personal email |
| `billing@aidotnet.dev` | your personal email |
| `security@aidotnet.dev` | your personal email |
| `*` (catch-all) | your personal email |

### Step 5: Repeat for aidotnet.ai and aidotnet.io

1. In ImprovMX, click **Add Domain** again for each
2. Add the same MX and TXT records in Namecheap for each domain
3. Add a catch-all alias `*` -> your personal email for each

### Step 6: Set Up "Send As" in Gmail (Optional but Recommended)

This lets you reply to emails FROM `support@aidotnet.dev` instead of your personal email:

1. In Gmail, go to **Settings** (gear icon) > **See all settings** > **Accounts and Import**
2. Under "Send mail as", click **Add another email address**
3. Enter:
   - Name: `AiDotNet Support`
   - Email: `support@aidotnet.dev`
   - Uncheck "Treat as an alias"
4. Click **Next Step**
5. SMTP Server: `smtp.improvmx.com`
6. Port: `587`
7. Username: your ImprovMX email (the one you signed up with)
8. Password: your ImprovMX password (or create an SMTP credential in ImprovMX dashboard > SMTP)
9. Select **TLS**
10. Click **Add Account**
11. ImprovMX will send a confirmation email to `support@aidotnet.dev` which forwards to your personal email
12. Click the confirmation link or enter the code

**Note**: ImprovMX free tier does NOT include SMTP sending. You need **ImprovMX Premium ($9/month)** for the "Send As" feature. Skip this step if you don't want the premium plan - you can still receive all emails on the free tier.

### Step 7: Verify Email Works

1. Wait 15-60 minutes for DNS propagation
2. Send a test email to `test@aidotnet.dev` from a different email account
3. Check that it arrives in your personal inbox
4. Check ImprovMX dashboard for delivery logs if it doesn't arrive

### Troubleshooting

- **Emails not arriving**: Check that MX records are correctly set and no conflicting MX records exist
- **SPF failures**: Make sure the TXT record with `v=spf1 include:spf.improvmx.com ~all` is set
- **DNS not propagating**: Use https://mxtoolbox.com/SuperTool.aspx to check MX records for your domain
- **ImprovMX shows "DNS not configured"**: DNS can take up to 48 hours to propagate globally

---

## Section 2: Domain DNS Configuration

You've purchased 3 domains. Here's the complete DNS setup for each.

### aidotnet.dev (Primary Domain - Website + Email)

In Namecheap > `aidotnet.dev` > **Advanced DNS**, set these records:

| Type | Host | Value | TTL | Notes |
|------|------|-------|-----|-------|
| A | @ | `76.76.21.21` | Automatic | Vercel IP |
| CNAME | www | `cname.vercel-dns.com` | Automatic | Vercel www redirect |
| CNAME | api | `aidotnet-serving.azurewebsites.net` | Automatic | Azure backend (add later) |
| MX | @ | `mx1.improvmx.com` | 10 | Email forwarding |
| MX | @ | `mx2.improvmx.com` | 20 | Email forwarding |
| TXT | @ | `v=spf1 include:spf.improvmx.com ~all` | Automatic | Email SPF |

**Alternative approach**: Instead of individual DNS records, you can point nameservers to Vercel:

1. In Namecheap > `aidotnet.dev` > **Nameservers**, select **Custom DNS**
2. Enter:
   ```text
   ns1.vercel-dns.com
   ns2.vercel-dns.com
   ```
3. Then manage ALL DNS records in Vercel dashboard instead of Namecheap
4. **Warning**: If you use Vercel nameservers, you must add MX/TXT records for email in Vercel's DNS settings instead

**Recommendation**: Use the individual records approach (first table above) so you can manage email DNS separately in Namecheap.

### aidotnet.ai (Redirect to primary)

Option A - Namecheap redirect:
1. Go to Namecheap > `aidotnet.ai` > **Manage** > **Redirect Domain**
2. Set redirect to: `https://aidotnet.dev`
3. Type: **Permanent (301)**

Option B - Vercel redirect (more reliable):
1. Add the same A and CNAME records as the primary domain
2. Add domain in Vercel dashboard (it will auto-redirect to primary)

Also add email records:

| Type | Host | Value | Priority |
|------|------|-------|----------|
| MX | @ | `mx1.improvmx.com` | 10 |
| MX | @ | `mx2.improvmx.com` | 20 |
| TXT | @ | `v=spf1 include:spf.improvmx.com ~all` | - |

### aidotnet.io (Redirect to primary)

Same as aidotnet.ai above - either Namecheap redirect or Vercel redirect, plus email records.

---

## Section 3: Connect Domains to Vercel

### Step 1: Add Primary Domain

1. Go to https://vercel.com
2. Select your website project
3. Go to **Settings** > **Domains**
4. Click **Add**
5. Enter `aidotnet.dev`
6. Vercel will check DNS and show status (green = configured, yellow = pending)

### Step 2: Add www Redirect

1. Still in Domains settings, click **Add**
2. Enter `www.aidotnet.dev`
3. Vercel will auto-configure it to redirect to `aidotnet.dev`

### Step 3: Add Redirect Domains

1. Click **Add** > enter `aidotnet.ai`
2. Click **Add** > enter `aidotnet.io`
3. Vercel will redirect these to your primary domain

### Step 4: SSL Certificates

- Vercel automatically provisions free SSL certificates for all domains
- No action needed - certificates are auto-renewed

### Step 5: Verify

After DNS propagates (15 min to 48 hours):
1. Visit `https://aidotnet.dev` - should show your website
2. Visit `https://www.aidotnet.dev` - should redirect to `https://aidotnet.dev`
3. Visit `https://aidotnet.ai` - should redirect to `https://aidotnet.dev`
4. Visit `https://aidotnet.io` - should redirect to `https://aidotnet.dev`

### Step 6: Update Stripe Payment Link Redirect URLs

Once the domain is live, update the redirect URLs in your Stripe Payment Links:
- Change from `https://website-plum-chi-82.vercel.app/pricing/?success=true`
- To `https://aidotnet.dev/pricing/?success=true`

You can edit existing payment links in the Stripe dashboard.

---

## Section 4: Stripe Setup

### Step 1: Create Stripe Account (if needed)

1. Go to https://dashboard.stripe.com
2. Sign up or log in
3. Complete account setup (business name, bank account for payouts)

### Step 2: Create the AiDotNet Pro Product

1. Go to https://dashboard.stripe.com/products
2. Click **+ Add product**
3. Fill in:
   - **Name**: `AiDotNet Pro`
   - **Description**: `Cloud-hosted inference API, priority support, and direct team access for AiDotNet`
   - **Image**: Upload the AiDotNet logo (optional)
4. Add **two prices**:
   - Click **Add another price**
   - Price 1: `$29.00 USD`, Recurring, Billing period: **Monthly**
   - Price 2: `$290.00 USD`, Recurring, Billing period: **Yearly** (saves ~17%)
5. Click **Save product**

### Step 3: Create Payment Link - Pro Monthly

1. Go to https://dashboard.stripe.com/payment-links
2. Click **+ New payment link**
3. Search for and select `AiDotNet Pro`
4. Choose the **$29.00/month** price
5. Configure settings:
   - **After payment**: "Don't show confirmation page" > Redirect to:
     - `https://aidotnet.dev/pricing/?success=true` (if domain is live)
     - OR `https://website-plum-chi-82.vercel.app/pricing/?success=true` (temp URL)
   - **Options** (gear icon):
     - Check **Allow promotion codes** (lets you offer launch discounts later)
     - Check **Collect billing address** (needed for tax compliance)
     - Check **Collect email address** (Stripe needs this for subscriptions)
6. Click **Create link**
7. **Copy the URL** - format: `https://buy.stripe.com/xxxxxxxx`
8. Save this as `PRO_MONTHLY_LINK`

### Step 4: Create Payment Link - Pro Annual

1. Same process as Step 3
2. Select the **$290.00/year** price instead
3. Same redirect URL and settings
4. **Copy the URL** and save as `PRO_YEARLY_LINK`

### Step 5: Create Stripe Customer Portal

This lets subscribers manage their subscription (update card, cancel, view invoices):

1. Go to https://dashboard.stripe.com/settings/billing/portal
2. Configure:
   - **Customer information**: Allow customers to update email and payment method
   - **Subscriptions**: Allow customers to cancel, switch plans
   - **Invoices**: Show invoice history
3. Enable **Payment method update**
4. Enable **Subscription cancellation**
5. Click **Save**
6. Note the portal link URL - you'll need this for the billing page

### Step 6: Set Up Webhooks (Do This After Serving Is Deployed)

Skip this step until AiDotNet.Serving is deployed to Azure (Section 8).

1. Go to https://dashboard.stripe.com/webhooks
2. Click **Add endpoint**
3. Endpoint URL: `https://api.aidotnet.dev/api/webhooks/stripe`
4. Select these events:
   - `checkout.session.completed`
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
   - `invoice.paid`
   - `invoice.payment_failed`
5. Click **Add endpoint**
6. Copy the **Signing secret** (starts with `whsec_`) - needed for Serving config

### Step 7: Note Your Stripe Keys

From https://dashboard.stripe.com/apikeys, note:
- **Publishable key**: `pk_live_...` (safe for client-side, but not needed for Payment Links)
- **Secret key**: `sk_live_...` (server-side only, needed for Serving)
- **Webhook signing secret**: `whsec_...` (from Step 6)

---

## Section 5: Supabase Setup

### Step 1: Create Supabase Project

1. Go to https://supabase.com/dashboard
2. Click **New project**
3. Fill in:
   - **Organization**: Create one or use existing
   - **Project name**: `aidotnet`
   - **Database password**: Generate a strong password and **save it somewhere safe**
   - **Region**: Choose closest to your users (e.g., East US)
4. Click **Create new project**
5. Wait for project to be created (~2 minutes)

### Step 2: Note Your Credentials

After project is created, go to **Settings** > **API**:

- **Project URL**: `https://xxxxx.supabase.co` (save this)
- **anon/public key**: `eyJhbG...` (save this - safe for client-side)
- **service_role key**: `eyJhbG...` (save this - server-side only, NEVER expose to client)

### Step 3: Run Database Setup SQL

1. In Supabase dashboard, go to **SQL Editor**
2. Click **New query**
3. Paste this entire SQL block and click **Run**:

```sql
-- ============================================
-- AiDotNet User Profile & API Management Schema
-- ============================================

-- User profiles (extends Supabase auth.users)
CREATE TABLE public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT NOT NULL,
  full_name TEXT,
  avatar_url TEXT,
  stripe_customer_id TEXT,
  subscription_tier TEXT DEFAULT 'free' CHECK (subscription_tier IN ('free', 'pro', 'enterprise')),
  subscription_status TEXT DEFAULT 'active',
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- API keys managed by users
CREATE TABLE public.user_api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  key_name TEXT NOT NULL,
  key_prefix TEXT NOT NULL,
  key_hash TEXT NOT NULL,
  scopes TEXT[] DEFAULT ARRAY['inference'],
  is_active BOOLEAN DEFAULT true,
  last_used_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT now(),
  expires_at TIMESTAMPTZ
);

-- API usage tracking
CREATE TABLE public.api_usage (
  id BIGSERIAL PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  api_key_id UUID REFERENCES public.user_api_keys(id) ON DELETE SET NULL,
  endpoint TEXT NOT NULL,
  model_id TEXT,
  request_tokens INTEGER DEFAULT 0,
  response_tokens INTEGER DEFAULT 0,
  latency_ms INTEGER,
  status_code INTEGER,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes for performance
CREATE INDEX idx_api_keys_user ON public.user_api_keys(user_id);
CREATE INDEX idx_api_usage_user ON public.api_usage(user_id);
CREATE INDEX idx_api_usage_created ON public.api_usage(created_at);
CREATE INDEX idx_profiles_stripe ON public.profiles(stripe_customer_id);

-- Enable Row Level Security
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.api_usage ENABLE ROW LEVEL SECURITY;

-- RLS Policies: users can only access their own data
CREATE POLICY "Users can view own profile" ON public.profiles
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.profiles
  FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can manage own API keys" ON public.user_api_keys
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can view own usage" ON public.api_usage
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own usage" ON public.api_usage
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Auto-create profile when a new user signs up
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, email, full_name, avatar_url)
  VALUES (
    NEW.id,
    NEW.email,
    NEW.raw_user_meta_data->>'full_name',
    NEW.raw_user_meta_data->>'avatar_url'
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
```

4. You should see "Success. No rows returned" - this means it worked
5. Go to **Table Editor** to verify the 3 tables were created: `profiles`, `user_api_keys`, `api_usage`

### Step 4: Enable Email Auth

1. Go to **Authentication** > **Providers**
2. **Email** should already be enabled by default
3. Go to **Authentication** > **Settings**
4. Under **Site URL**, set: `https://aidotnet.dev` (or temp Vercel URL if domain isn't ready)
5. Under **Redirect URLs**, add:
   - `https://aidotnet.dev/auth/callback/`
   - `https://website-plum-chi-82.vercel.app/auth/callback/` (temp URL as fallback)

---

## Section 6: GitHub OAuth Setup

### Step 1: Create GitHub OAuth App

1. Go to https://github.com/settings/developers
2. Click **OAuth Apps** > **New OAuth App**
3. Fill in:
   - **Application name**: `AiDotNet`
   - **Homepage URL**: `https://aidotnet.dev`
   - **Application description**: `Sign in to AiDotNet with your GitHub account`
   - **Authorization callback URL**: `https://xxxxx.supabase.co/auth/v1/callback`
     - Replace `xxxxx` with your Supabase project ID from Section 5 Step 2
4. Click **Register application**
5. On the app page:
   - Copy the **Client ID**
   - Click **Generate a new client secret**
   - Copy the **Client Secret** (you won't see it again)

### Step 2: Add GitHub to Supabase

1. In Supabase dashboard > **Authentication** > **Providers**
2. Find **GitHub** and click to expand
3. Toggle **Enable GitHub provider** ON
4. Paste:
   - **Client ID**: from Step 1
   - **Client Secret**: from Step 1
5. Click **Save**

### Step 3: (Optional) Google OAuth Setup

1. Go to https://console.cloud.google.com/apis/credentials
2. Create a new project (or select existing)
3. Click **Create Credentials** > **OAuth 2.0 Client ID**
4. Application type: **Web application**
5. Name: `AiDotNet`
6. Authorized redirect URIs: `https://xxxxx.supabase.co/auth/v1/callback`
7. Click **Create**
8. Copy **Client ID** and **Client Secret**
9. In Supabase > **Authentication** > **Providers** > **Google**:
   - Toggle ON
   - Paste Client ID and Client Secret
   - Click **Save**

---

## Section 7: Set Vercel Environment Variables

This connects the website to Supabase and Stripe.

### Step 1: Go to Vercel Settings

1. Go to https://vercel.com
2. Select your website project
3. Go to **Settings** > **Environment Variables**

### Step 2: Add These Variables

Add each variable with "Production" and "Preview" checked:

| Name | Value | Where to find it |
|------|-------|-------------------|
| `PUBLIC_SUPABASE_URL` | `https://xxxxx.supabase.co` | Supabase > Settings > API > Project URL |
| `PUBLIC_SUPABASE_ANON_KEY` | `eyJhbG...` | Supabase > Settings > API > anon/public key |
| `PUBLIC_STRIPE_PRO_MONTHLY_LINK` | `https://buy.stripe.com/xxxxx` | From Section 4 Step 3 |
| `PUBLIC_STRIPE_PRO_YEARLY_LINK` | `https://buy.stripe.com/xxxxx` | From Section 4 Step 4 |

### Step 3: Redeploy

1. Go to **Deployments** tab
2. Find the latest deployment
3. Click **...** > **Redeploy**
4. Wait for build to complete

### Step 4: Verify

1. Visit your site
2. Click "Sign In" in the navbar - should go to `/login/`
3. Try signing up with email - should work if Supabase is configured
4. Click "Subscribe" on pricing - should redirect to Stripe checkout

---

## Section 8: Deploy AiDotNet.Serving to Azure

This deploys the API backend that handles inference, model serving, and Stripe webhooks.

**Prerequisites**: Azure CLI installed (`az --version` should work)

### Step 1: Login and Create Resources

Open a terminal and run each command:

```bash
# Login to Azure (opens browser)
az login

# Create resource group
az group create --name aidotnet-rg --location eastus

# Create App Service plan (B1 = ~$13/month, Linux)
az appservice plan create \
  --name aidotnet-plan \
  --resource-group aidotnet-rg \
  --sku B1 \
  --is-linux

# Create web app for .NET 10
az webapp create \
  --name aidotnet-serving \
  --resource-group aidotnet-rg \
  --plan aidotnet-plan \
  --runtime "DOTNETCORE:10.0"
```

### Step 2: Create PostgreSQL Database

```bash
# Create PostgreSQL Flexible Server (~$13/month)
az postgres flexible-server create \
  --name aidotnet-db \
  --resource-group aidotnet-rg \
  --sku-name Standard_B1ms \
  --storage-size 32 \
  --admin-user aidotnetadmin \
  --admin-password "$(openssl rand -base64 32)" \
  --version 16

# Create the database
az postgres flexible-server db create \
  --resource-group aidotnet-rg \
  --server-name aidotnet-db \
  --database-name aidotnet_serving

# Allow Azure services to connect
az postgres flexible-server firewall-rule create \
  --resource-group aidotnet-rg \
  --name aidotnet-db \
  --rule-name AllowAzure \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0
```

### Step 3: Configure App Settings

Replace all `<PLACEHOLDER>` values with your actual values:

```bash
az webapp config appsettings set \
  --name aidotnet-serving \
  --resource-group aidotnet-rg \
  --settings \
    "ServingPersistenceOptions__Provider=PostgreSql" \
    "ServingPersistenceOptions__ConnectionString=@Microsoft.KeyVault(SecretUri=https://aidotnet-kv.vault.azure.net/secrets/ServingDbConnectionString)" \
    "ServingPersistenceOptions__MigrateOnStartup=true" \
    "ServingOptions__Port=8080" \
    "StripeOptions__SecretKey=@Microsoft.KeyVault(SecretUri=https://aidotnet-kv.vault.azure.net/secrets/StripeSecretKey)" \
    "StripeOptions__WebhookSigningSecret=@Microsoft.KeyVault(SecretUri=https://aidotnet-kv.vault.azure.net/secrets/StripeWebhookSecret)" \
    "StripeOptions__ProPriceId=@Microsoft.KeyVault(SecretUri=https://aidotnet-kv.vault.azure.net/secrets/StripeProPriceId)" \
    "StripeOptions__ProAnnualPriceId=@Microsoft.KeyVault(SecretUri=https://aidotnet-kv.vault.azure.net/secrets/StripeProAnnualPriceId)" \
    "StripeOptions__SuccessUrl=https://aidotnet.dev/pricing/?success=true" \
    "StripeOptions__CancelUrl=https://aidotnet.dev/pricing/?cancelled=true" \
    "ASPNETCORE_ENVIRONMENT=Production"
```

### Step 4: Configure Custom Domain

```bash
# Add custom domain
az webapp config hostname add \
  --webapp-name aidotnet-serving \
  --resource-group aidotnet-rg \
  --hostname api.aidotnet.dev

# Enable managed SSL certificate
az webapp config ssl create \
  --name aidotnet-serving \
  --resource-group aidotnet-rg \
  --hostname api.aidotnet.dev
```

### Step 5: Set Up GitHub Actions for Auto-Deploy

1. Get the publish profile:
```bash
az webapp deployment list-publishing-profiles \
  --name aidotnet-serving \
  --resource-group aidotnet-rg \
  --xml
```

2. Copy the entire XML output
3. Go to GitHub repo > **Settings** > **Secrets and variables** > **Actions**
4. Click **New repository secret**
5. Name: `AZURE_WEBAPP_PUBLISH_PROFILE`
6. Value: paste the XML

The deploy workflow file (`.github/workflows/deploy-serving.yml`) has been created and is checked into the repository.

### Step 6: Verify Deployment

```bash
# Check if the app is running
curl https://api.aidotnet.dev/swagger/v1/swagger.json

# Test an endpoint (will return error without models, but confirms API works)
curl -X POST https://api.aidotnet.dev/api/inference/predict \
  -H "Content-Type: application/json" \
  -d '{"modelId": "test"}'
```

---

## Section 9: Monitoring Setup

### Uptime Monitoring (Free)

1. Go to https://uptimerobot.com and create a free account
2. Click **Add New Monitor**
3. Configure:
   - Monitor Type: **HTTPS**
   - Friendly Name: `AiDotNet API`
   - URL: `https://api.aidotnet.dev/swagger/v1/swagger.json`
   - Monitoring Interval: **5 minutes**
4. Set up alerts:
   - Alert when **down for 2 consecutive checks**
   - Notify via email

### Azure Application Insights (Free tier)

```bash
# Create Application Insights
az monitor app-insights component create \
  --app aidotnet-insights \
  --location eastus \
  --resource-group aidotnet-rg

# Get the connection string
az monitor app-insights component show \
  --app aidotnet-insights \
  --resource-group aidotnet-rg \
  --query connectionString
```

Add the connection string to App Service settings:
```bash
az webapp config appsettings set \
  --name aidotnet-serving \
  --resource-group aidotnet-rg \
  --settings "ApplicationInsights__ConnectionString=<CONNECTION_STRING>"
```

---

## Section 10: Post-Setup Verification Checklist

After completing all sections above, verify everything works:

### Website
- [ ] `https://aidotnet.dev` loads correctly
- [ ] `https://www.aidotnet.dev` redirects to `https://aidotnet.dev`
- [ ] `https://aidotnet.ai` redirects to `https://aidotnet.dev`
- [ ] `https://aidotnet.io` redirects to `https://aidotnet.dev`

### Email
- [ ] Send test email to `test@aidotnet.dev` - arrives in your inbox
- [ ] Send test email to `support@aidotnet.dev` - arrives in your inbox
- [ ] MX records verified at https://mxtoolbox.com

### Auth
- [ ] Sign up with email at `/signup/` - receive confirmation email
- [ ] Sign in with email at `/login/` - redirects to `/account/`
- [ ] Sign in with GitHub - OAuth flow works end to end
- [ ] Dashboard shows user name and plan badge
- [ ] Sign out works

### Stripe
- [ ] Subscribe button on pricing page redirects to Stripe checkout
- [ ] Monthly and yearly payment links work
- [ ] After payment, redirects to `/pricing/?success=true`

### API (after Azure deployment)
- [ ] `https://api.aidotnet.dev` responds
- [ ] Swagger page loads
- [ ] API key authentication works

### Subscriber Portal
- [ ] Create API key at `/account/api-keys/` - key displayed once
- [ ] Copy button works
- [ ] Revoke key works
- [ ] Usage page loads at `/account/usage/`
- [ ] Billing page loads at `/account/billing/`
- [ ] Settings page loads at `/account/settings/`

---

## Recommended Order of Operations

Do these in order - each step depends on the previous:

1. **Stripe** (Section 4) - Create products and payment links (15 min)
2. **Supabase** (Section 5) - Create project and run SQL (15 min)
3. **GitHub OAuth** (Section 6) - Create OAuth app (10 min)
4. **Vercel Env Vars** (Section 7) - Connect everything + redeploy (5 min)
5. **DNS** (Section 2) - Configure all 3 domains (15 min)
6. **Vercel Domains** (Section 3) - Add domains to Vercel (5 min)
7. **Email** (Section 1) - Set up ImprovMX forwarding (15 min)
8. **Azure** (Section 8) - Deploy Serving backend (30 min)
9. **Monitoring** (Section 9) - Set up uptime checks (10 min)
10. **Verify** (Section 10) - Run through checklist (15 min)

**Total estimated time: ~2.5 hours**

---

## Cost Summary (Monthly)

| Service | Tier | Monthly Cost |
|---------|------|-------------|
| Vercel | Free (Hobby) | $0 |
| Supabase | Free tier | $0 |
| ImprovMX | Free (receiving only) | $0 |
| Azure App Service | B1 (Linux) | ~$13 |
| Azure PostgreSQL | Burstable B1ms | ~$13 |
| Azure App Insights | Free (5GB/month) | $0 |
| UptimeRobot | Free | $0 |
| Namecheap (aidotnet.dev) | Annual | ~$1/month |
| Namecheap (aidotnet.ai) | Annual | ~$5/month |
| Namecheap (aidotnet.io) | Annual | ~$3/month |
| Stripe | 2.9% + 30c per transaction | Variable |
| **Total fixed** | | **~$35/month** |

Break-even: **2 Pro subscribers** covers all infrastructure costs.

---

## What Claude Will Build Next (No Action Needed From You)

These are coding tasks that will be done for you:

- [ ] `.github/workflows/deploy-serving.yml` - Auto-deploy workflow
- [ ] Enable Swagger in production in `Program.cs`
- [ ] Update `astro.config.mjs` site URL after domain is live
- [ ] Auto-generate API reference docs (500+ pages from XML doc comments)
- [ ] Write 20 tutorials (priority-ordered)
- [ ] Build interactive playground
- [ ] Update NuGet package metadata for 1.0.0
- [ ] Rewrite GitHub README
- [ ] Draft 4 blog/launch posts (HN, Reddit, Dev.to, .NET Blog)
- [ ] E2E tests for new auth/portal pages
- [ ] Stripe Customer Portal integration in billing page
