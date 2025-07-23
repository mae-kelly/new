#!/bin/bash
# Corporate Network Configuration Script for AO1 Scanner

echo "🏢 AO1 Scanner - Corporate Network Configuration"
echo "================================================"

# Function to detect corporate proxy
detect_proxy() {
    echo "🔍 Detecting corporate proxy settings..."
    
    # Check environment variables
    if [[ -n "$HTTP_PROXY" ]] || [[ -n "$http_proxy" ]]; then
        echo "✅ HTTP proxy detected: ${HTTP_PROXY:-$http_proxy}"
    else
        echo "⚠️  No HTTP proxy found in environment"
    fi
    
    if [[ -n "$HTTPS_PROXY" ]] || [[ -n "$https_proxy" ]]; then
        echo "✅ HTTPS proxy detected: ${HTTPS_PROXY:-$https_proxy}"
    else
        echo "⚠️  No HTTPS proxy found in environment"
    fi
}

# Function to configure proxy
configure_proxy() {
    echo ""
    echo "🔧 Configuring corporate proxy..."
    
    # Prompt for proxy if not set
    if [[ -z "$HTTP_PROXY" ]] && [[ -z "$http_proxy" ]]; then
        echo "Please enter your corporate proxy URL (e.g., http://proxy.company.com:8080):"
        read -r proxy_url
        
        if [[ -n "$proxy_url" ]]; then
            export HTTP_PROXY="$proxy_url"
            export HTTPS_PROXY="$proxy_url"
            export http_proxy="$proxy_url"
            export https_proxy="$proxy_url"
            echo "✅ Proxy configured: $proxy_url"
        fi
    fi
}

# Function to configure CA certificates
configure_certificates() {
    echo ""
    echo "🔒 Configuring SSL certificates..."
    
    # Check for existing CA bundle
    ca_locations=(
        "$REQUESTS_CA_BUNDLE"
        "$CURL_CA_BUNDLE" 
        "$SSL_CERT_FILE"
        "/etc/ssl/certs/ca-certificates.crt"
        "/etc/ssl/certs/ca-bundle.crt"
        "/etc/pki/tls/certs/ca-bundle.crt"
        "/usr/local/share/certs/ca-root-nss.crt"
        "/etc/ssl/cert.pem"
    )
    
    ca_found=""
    for ca_path in "${ca_locations[@]}"; do
        if [[ -n "$ca_path" ]] && [[ -f "$ca_path" ]]; then
            ca_found="$ca_path"
            echo "✅ Found CA bundle: $ca_path"
            break
        fi
    done
    
    if [[ -n "$ca_found" ]]; then
        export REQUESTS_CA_BUNDLE="$ca_found"
        export CURL_CA_BUNDLE="$ca_found"
        export SSL_CERT_FILE="$ca_found"
        echo "✅ CA bundle configured: $ca_found"
    else
        echo "⚠️  No CA bundle found. You may need to:"
        echo "   export REQUESTS_CA_BUNDLE=/path/to/corporate-ca-bundle.crt"
    fi
}

# Function to test connectivity
test_connectivity() {
    echo ""
    echo "🌐 Testing connectivity to required services..."
    
    # Test basic connectivity
    urls=(
        "https://huggingface.co"
        "https://pypi.org"
        "https://files.pythonhosted.org"
    )
    
    for url in "${urls[@]}"; do
        echo -n "Testing $url... "
        if curl -s --max-time 10 --head "$url" > /dev/null 2>&1; then
            echo "✅ OK"
        else
            echo "❌ FAILED"
        fi
    done
}

# Function to manually download models
download_models_manually() {
    echo ""
    echo "📦 Manual model download option..."
    echo "If automatic download fails, you can manually download models:"
    echo ""
    echo "1. Create directory:"
    echo "   mkdir -p ~/.cache/sentence_transformers/"
    echo ""
    echo "2. Download from corporate network:"
    echo "   wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json"
    echo "   wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin"
    echo "   # ... (download all required files)"
    echo ""
    echo "3. Or ask IT to whitelist these domains:"
    echo "   - huggingface.co"
    echo "   - cdn-lfs.huggingface.co"
    echo "   - pypi.org"
    echo "   - files.pythonhosted.org"
}

# Function to create environment setup script
create_env_script() {
    echo ""
    echo "📝 Creating environment setup script..."
    
    cat > setup_corporate_env.sh << 'EOF'
#!/bin/bash
# AO1 Scanner Corporate Environment Setup

# Set these to your corporate values
export HTTP_PROXY="http://proxy.company.com:8080"
export HTTPS_PROXY="http://proxy.company.com:8080"
export http_proxy="http://proxy.company.com:8080"
export https_proxy="http://proxy.company.com:8080"

# Set CA bundle path (update this path)
export REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
export CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
export SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"

# No proxy for internal domains
export NO_PROXY="localhost,127.0.0.1,.company.com"
export no_proxy="localhost,127.0.0.1,.company.com"

echo "✅ Corporate environment configured"
echo "HTTP_PROXY: $HTTP_PROXY"
echo "HTTPS_PROXY: $HTTPS_PROXY"
echo "CA_BUNDLE: $REQUESTS_CA_BUNDLE"
EOF
    
    chmod +x setup_corporate_env.sh
    echo "✅ Created setup_corporate_env.sh"
    echo "   Edit this file with your corporate proxy settings"
    echo "   Then run: source setup_corporate_env.sh"
}

# Function to install dependencies with corporate settings
install_dependencies() {
    echo ""
    echo "📦 Installing Python dependencies with corporate settings..."
    
    pip_args=""
    
    # Add proxy if configured
    if [[ -n "$HTTP_PROXY" ]]; then
        pip_args="$pip_args --proxy $HTTP_PROXY"
    fi
    
    # Add trusted hosts for corporate networks
    pip_args="$pip_args --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org"
    
    echo "Running: pip install $pip_args -r requirements.txt"
    
    # Install with corporate settings
    pip install $pip_args -r requirements.txt
    
    if [[ $? -eq 0 ]]; then
        echo "✅ Dependencies installed successfully"
    else
        echo "❌ Dependency installation failed"
        echo "Try running manually:"
        echo "pip install $pip_args sentence-transformers torch numpy scikit-learn"
    fi
}

# Main execution
main() {
    detect_proxy
    configure_proxy
    configure_certificates
    test_connectivity
    create_env_script
    download_models_manually
    
    echo ""
    echo "🎯 NEXT STEPS:"
    echo "1. Edit setup_corporate_env.sh with your corporate settings"
    echo "2. Run: source setup_corporate_env.sh"
    echo "3. Run: python run_ao1_scan_corporate_fixed.py"
    echo ""
    echo "If models still fail to download, contact IT to whitelist:"
    echo "- huggingface.co"
    echo "- cdn-lfs.huggingface.co"
    echo "- pypi.org"
    echo "- files.pythonhosted.org"
}

# Run main function
main