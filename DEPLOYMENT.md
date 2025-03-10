# Deploying to Hugging Face Spaces

This guide provides step-by-step instructions for deploying the Stock Sentiment Dashboard to Hugging Face Spaces.

## Prerequisites

- A Hugging Face account
- Git installed on your local machine
- The complete codebase downloaded from the repository

## Steps to Deploy

### 1. Create a New Space on Hugging Face

1. Log in to your Hugging Face account
2. Click on your profile picture > "New Space"
3. Configure your Space:
   - **Owner**: Your username or organization
   - **Space name**: Choose a name (e.g., "stock-sentiment-dashboard")
   - **License**: MIT
   - **SDK**: Select "Streamlit"
   - **Hardware**: CPU (recommended: 2CPU, 16GB RAM)
   - **Privacy**: Public or Private based on your preference

### 2. Clone the Space Repository

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/stock-sentiment-dashboard
cd stock-sentiment-dashboard
```

### 3. Prepare Your Files

1. Copy all project files to the cloned repository folder
2. Ensure you have the following key files:
   - `main.py`
   - All folders (`components`, `utils`, etc.)
   - Stock CSV file in `attached_assets` folder

### 4. Create Requirements File

Create a file named `requirements.txt` with the contents of `deployment-requirements.txt`:

```bash
cp deployment-requirements.txt requirements.txt
```

### 5. Configure Streamlit (Optional)

Create a `.streamlit` directory and add a `config.toml` file:

```bash
mkdir -p .streamlit
```

Add the following to `.streamlit/config.toml`:

```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
port = 7860
```

### 6. Commit and Push Your Changes

```bash
git add .
git commit -m "Initial deployment"
git push
```

### 7. Monitor Deployment

1. Go to your Space on Hugging Face
2. Click on "Settings" > "Factory" to monitor the build process
3. Once complete, your app will be live at:
   `https://huggingface.co/spaces/YOUR_USERNAME/stock-sentiment-dashboard`

## Troubleshooting

- **Build Failures**: Check logs in the "Factory" section
- **Memory Issues**: Consider upgrading to a higher RAM tier if needed
- **Missing Dependencies**: Update the requirements.txt file
- **Streamlit Issues**: Ensure streamlit config is correctly set

## Updating Your Deployment

To update your deployed app:

1. Make changes to your local repository
2. Commit and push changes
3. Hugging Face will automatically rebuild and update your Space

## Custom Domain (Optional)

You can set up a custom domain in the Space settings if you prefer a branded URL.