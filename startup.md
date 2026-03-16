
## **Ollama Setup & Usage**

### **Step 1: Install Ollama**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
ollama --version
# Should output: ollama version is 0.x.x
```


### **Step 2: Start Ollama Server**

Run Ollama server **BEFORE** running the benchmark.

#### **Option A: Run in Current Terminal (Foreground)**
```bash
# This will block the terminal
ollama serve
```

Keep this terminal open, then open a NEW terminal for the benchmark.

#### **Option B: Run in Background**
```bash
# macOS/Linux - run as background process
ollama serve > /dev/null 2>&1 &

# Verify it's running
ps aux | grep ollama
# Or check with curl:
curl http://localhost:11434/api/tags


---

### **Step 3: Pull the Model**

Download the LLM model you want to use:

```bash
# Recommended: llama3.2 (3B params, fast, good quality)
ollama pull llama3.2
```

#### **Verify Model is Downloaded:**
```bash
ollama list

# Should show:
# NAME              ID              SIZE      MODIFIED
# llama3.2:latest   a80c4f17acd5    2.0 GB    2 minutes ago
```

---

### **Step 4: Test Ollama**

```bash
# Quick test
ollama run llama3.2 "What is 2+2?"
```
---

### **Step 5: Configure RapidIndex**

Make sure your `.env` file has:

```bash
# .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_URL=http://localhost:11434
```

---

### **Step 6: Run Benchmark**

Now you can run the benchmark:

```bash
python -m benchmarks.run_benchmark
```

---

## **All-in-One Setup Script**

Create this helper script for easy setup:

```bash
# scripts/start_ollama_and_benchmark.sh
#!/bin/bash

echo "=========================================="
echo "RapidIndex Benchmark with Ollama"
echo "=========================================="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo " Ollama not installed"
    echo ""
    echo "Install with:"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

echo "Ollama installed"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    sleep 3
    
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama server started (PID: $OLLAMA_PID)"
    else
        echo "Failed to start Ollama"
        exit 1
    fi
else
    echo " Ollama server already running"
fi

# Check if model is downloaded
if ! ollama list | grep -q "llama3.2"; then
    echo ""
    echo "Downloading llama3.2 model (~2GB)..."
    ollama pull llama3.2
    echo "Model downloaded"
else
    echo "Model llama3.2 ready"
fi

# Test Ollama
echo ""
echo "Testing Ollama..."
TEST_RESPONSE=$(ollama run llama3.2 "Say 'ready'" --verbose=false 2>/dev/null)
if [ -n "$TEST_RESPONSE" ]; then
    echo " Ollama responding correctly"
else
    echo " Ollama test failed, but continuing..."
fi

echo ""
echo "=========================================="
echo "Running Benchmark"
echo "=========================================="
echo ""

# Run benchmark
python -m benchmarks.run_benchmark

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
```

Make it executable and run:
```bash
chmod +x scripts/start_ollama_and_benchmark.sh
./scripts/start_ollama_and_benchmark.sh
```

---

##  **Troubleshooting**


**Check if Ollama is running:**
```bash
curl http://localhost:11434/api/tags

# Should return JSON with available models
# If it fails, Ollama is not running
```

**Solution:**
```bash
# Kill any existing Ollama processes
pkill ollama

# Start fresh
ollama serve &
sleep 3

# Test again
curl http://localhost:11434/api/tags
```

---

### **Problem: "Model not found"**

**Check downloaded models:**
```bash
ollama list
```

**If llama3.2 is missing:**
```bash
ollama pull llama3.2
```

---

### **Problem: Slow responses**

**Try a smaller model:**
```bash
# Use 1B parameter model (faster)
ollama pull llama3.2:1b

# Update .env
OLLAMA_MODEL=llama3.2:1b
```

---

### **Problem: Port 11434 already in use**

**Check what's using the port:**
```bash
lsof -i :11434
```

**Use a different port:**
```bash
# Start Ollama on different port
OLLAMA_HOST=0.0.0.0:11435 ollama serve &

# Update .env
OLLAMA_URL=http://localhost:11435
```


---

## **Quick Reference Commands**

```bash
# 1. Start Ollama (one of these)
ollama serve                    # Foreground (blocks terminal)
ollama serve &                  # Background
nohup ollama serve &            # Background + persist after logout

# 2. Check status
curl http://localhost:11434/api/tags

# 3. List models
ollama list

# 4. Pull model
ollama pull llama3.2

# 5. Test model
ollama run llama3.2 "Hello"

# 6. Run benchmark
python -m benchmarks.run_benchmark

# 7. Stop Ollama
pkill ollama
```

---

## **Step-by-Step Checklist**

```
□ 1. Install Ollama
    curl -fsSL https://ollama.com/install.sh | sh

□ 2. Start Ollama server
    ollama serve &

□ 3. Pull model
    ollama pull llama3.2

□ 4. Verify it's running
    curl http://localhost:11434/api/tags

□ 5. Configure .env
    LLM_PROVIDER=ollama
    OLLAMA_MODEL=llama3.2

□ 6. Setup test data
    python scripts/setup_test_data.py

□ 7. Run benchmark
    python -m benchmarks.run_benchmark
```

---

## **Tips**

### **Keep Ollama Running Permanently:**
```bash
# Add to ~/.bashrc or ~/.zshrc
alias start-ollama='ollama serve > /dev/null 2>&1 &'

# Then just run:
start-ollama
```

### **Monitor Ollama:**
```bash
tail -f ~/.ollama/logs/server.log
top -pid $(pgrep ollama)
```

### **Switch Models:**
```bash
# In .env,  change:
OLLAMA_MODEL=llama3.2      # Fast
OLLAMA_MODEL=mistral       # Better quality
OLLAMA_MODEL=llama3.2:1b   # Very fast
```

