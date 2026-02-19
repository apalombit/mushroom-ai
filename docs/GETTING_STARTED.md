# Getting Started

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | >= 3.11 | [python.org](https://www.python.org/) |
| Docker + Compose | >= 24.x | [docker.com](https://docs.docker.com/get-docker/) |
| Git | >= 2.x | [git-scm.com](https://git-scm.com/) |

## Step 1: Clone and Configure

```bash
git clone https://github.com/<your-username>/mushroom-ai.git
cd mushroom-ai
cp .env.example .env
```

Edit `.env` if needed (defaults work for local development).

## Step 2: Install Ollama

```bash
# Option A: On host (recommended if you have a GPU)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b

# Option B: Via Docker (CPU-only, included in docker-compose)
# After docker compose up, run:
# docker compose exec ollama ollama pull llama3.1:8b
```

## Step 3: Start Docker Services

```bash
docker compose up -d
docker compose ps      # verify all healthy
```

Services:
| Service | URL | Purpose |
|---------|-----|---------|
| PostgreSQL + pgvector | localhost:5432 | Feature database + vector similarity |
| MLflow | http://localhost:5000 | Experiment tracking |
| Ollama | http://localhost:11434 | Local LLM inference |

## Step 4: Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 5: Verify Setup

```bash
make test-setup    # or: python -m scripts.test_setup
```

Should show ✓ for PostgreSQL, Ollama, Instructor, and MLflow.

## Step 6: Initialize Database

```bash
make db-init       # or: python -m scripts.init_db
```

## Development Workflow

```bash
make up            # start Docker services
make serve         # FastAPI at http://localhost:8001
make ui            # Streamlit at http://localhost:8501 (separate terminal)
make test          # run tests
make lint          # check code style
```
