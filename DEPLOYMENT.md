# 🚀 Deployment Guide - PlaceDiscover Search Agent

## Repository Information

- **GitHub**: https://github.com/sriramnalla30/PlaceDiscover
- **Netlify Site**: ramplacesearch.netlify.app
- **Frontend Path**: `/frontend` directory
- **Backend**: New Render deployment (to be created)

---

## 📋 Pre-Deployment Checklist

- [x] `netlify.toml` created (publish: frontend/)
- [x] `render.yaml` created (auto-deploy config)
- [x] `frontend/index.html` created (from dashboard.html)
- [ ] Update API_BASE_URL in `frontend/index.html` after Render deployment
- [ ] Push to GitHub
- [ ] Create Render service
- [ ] Add environment variables to Render

---

## 🔧 Step 1: Update GitHub Repository

### Option A: Replace Existing Project (Recommended - Preserves Domain)

```powershell
# Navigate to a temporary location
cd D:\temp

# Clone your existing repo
git clone https://github.com/sriramnalla30/PlaceDiscover.git
cd PlaceDiscover

# Backup current project (optional)
git checkout -b backup-old-project
git push origin backup-old-project

# Switch back to main
git checkout main

# Remove all files except .git
Get-ChildItem -Exclude .git | Remove-Item -Recurse -Force

# Copy new Search Agent files
Copy-Item "D:\gitcode\Projects\Agentic_Projects\Search_Agent\*" -Destination . -Recurse -Exclude .git,.env

# Add and commit
git add .
git commit -m "Deploy Search Agent - LangGraph agentic workflow"
git push origin main
```

### Option B: Fresh Start (If you prefer clean history)

```powershell
cd D:\gitcode\Projects\Agentic_Projects\Search_Agent

# Initialize git if not already
git init
git add .
git commit -m "Initial commit - Search Agent"

# Connect to your existing repo
git remote add origin https://github.com/sriramnalla30/PlaceDiscover.git

# Force push (WARNING: This erases old history)
git push -f origin main
```

---

## 🌐 Step 2: Create Render Backend

### 2.1 Go to Render Dashboard

1. Visit: https://dashboard.render.com/
2. Sign in (or create account)
3. Click **"New +"** → **"Web Service"**

### 2.2 Connect GitHub Repository

1. Select **"Build and deploy from a Git repository"**
2. Click **"Connect account"** → Authorize GitHub
3. Find and select: **sriramnalla30/PlaceDiscover**
4. Click **"Connect"**

### 2.3 Configure Service Settings

**Basic Settings:**

```
Name: search-agent-api
Region: Choose closest to your users (e.g., Singapore, Oregon)
Branch: main
Runtime: Python 3
```

**Build Settings:**

```
Build Command: pip install -r requirements.txt
Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

**Instance Type:**

```
Free (or Starter if you need better performance)
```

### 2.4 Add Environment Variables

Click **"Advanced"** → **Environment Variables** → Add these:

| Key                      | Value           | Get From                         |
| ------------------------ | --------------- | -------------------------------- |
| `GROQ_API_KEY`           | `your_key_here` | https://console.groq.com/        |
| `GROQ_API_KEY_2`         | `your_key_here` | https://console.groq.com/        |
| `SERPSTACK_API_KEY`      | `your_key_here` | https://serpstack.com/dashboard  |
| `WEBSCRAPING_AI_API_KEY` | `your_key_here` | https://webscraping.ai/dashboard |

### 2.5 Deploy

1. Click **"Create Web Service"**
2. Wait for deployment (3-5 minutes)
3. **Copy your service URL**: `https://search-agent-api-xxxx.onrender.com`

---

## 🔗 Step 3: Update Frontend API URL

### 3.1 Edit `frontend/index.html`

Find line ~965 (after line `let currentThreadId = null;`):

```javascript
// BEFORE:
const API_BASE_URL =
  window.location.hostname === "localhost"
    ? "http://localhost:8000"
    : "https://YOUR_RENDER_APP_NAME.onrender.com";

// AFTER (replace with your actual Render URL):
const API_BASE_URL =
  window.location.hostname === "localhost"
    ? "http://localhost:8000"
    : "https://search-agent-api-xxxx.onrender.com"; // Your Render URL here
```

### 3.2 Commit and Push Update

```powershell
cd D:\temp\PlaceDiscover  # Or your repo location

git add frontend/index.html
git commit -m "Update API endpoint to Render backend"
git push origin main
```

---

## 🎯 Step 4: Deploy to Netlify

### 4.1 Automatic Deployment

- Netlify detects push to `main` branch
- Reads `netlify.toml` (publish: `frontend/`)
- Deploys `frontend/index.html`
- **Your domain stays the same**: `ramplacesearch.netlify.app`

### 4.2 Manual Trigger (if needed)

1. Go to: https://app.netlify.com/sites/ramplacesearch
2. Click **"Deploys"** → **"Trigger deploy"** → **"Deploy site"**

### 4.3 Verify Deployment

1. Wait 1-2 minutes for build
2. Visit: **https://ramplacesearch.netlify.app**
3. Test the agent with a query

---

## ✅ Step 5: Verification Checklist

### Frontend (Netlify)

- [ ] Site loads: https://ramplacesearch.netlify.app
- [ ] No console errors (F12 → Console)
- [ ] UI displays correctly

### Backend (Render)

- [ ] Service running: https://your-app.onrender.com
- [ ] Health check passes: Visit root URL
- [ ] Environment variables set

### Integration Test

- [ ] Enter query: "Find the best gym in Bangalore"
- [ ] Agent starts processing (see logs)
- [ ] SerpStack tool badge appears
- [ ] Reviews fetched (WebScraping.AI badge)
- [ ] Winner recommendation shows

---

## 🔧 Troubleshooting

### Issue: Netlify shows "Page not found"

**Solution**: Check `netlify.toml` publish directory is `frontend`

### Issue: CORS errors in browser console

**Solution**: Ensure Render service is running and URL is correct in `frontend/index.html`

### Issue: Backend returns 500 errors

**Solution**:

1. Check Render logs: Dashboard → Service → "Logs"
2. Verify all environment variables are set
3. Check API quotas (SerpStack, Groq, WebScraping.AI)

### Issue: Free tier Render spins down

**Solution**:

- Free tier sleeps after 15 min inactivity
- First request takes 30-60s to wake up
- Upgrade to Starter ($7/mo) for always-on

---

## 🎉 Success!

Once deployed:

- **Frontend**: https://ramplacesearch.netlify.app ✅
- **Backend**: https://your-app.onrender.com ✅
- **Domain unchanged**: Company can continue using same URL ✅

---

## 📝 Quick Reference

### File Structure After Deployment

```
PlaceDiscover/
├── frontend/
│   └── index.html          # Main UI (deployed to Netlify)
├── app/                    # Backend (deployed to Render)
├── netlify.toml           # Netlify config
├── render.yaml            # Render config
├── requirements.txt       # Python deps
└── .env.example           # Env template
```

### Important URLs

- Netlify Dashboard: https://app.netlify.com/sites/ramplacesearch
- Render Dashboard: https://dashboard.render.com/
- GitHub Repo: https://github.com/sriramnalla30/PlaceDiscover

---

**Need Help?** Check logs:

- **Netlify**: Site → Deploys → Click latest deploy → "Deploy log"
- **Render**: Service → Logs tab
