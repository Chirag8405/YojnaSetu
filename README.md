# YojanaSetu

YojanaSetu is a voice-first PMJAY navigation platform focused on Maharashtra users in Hindi and Marathi. It combines retrieval-augmented generation, strict citation guardrails, and telephony integration so citizens can ask scheme questions through web and IVR channels with accessible, low-literacy-first experiences.

The project includes a FastAPI backend, ingestion pipeline for PMJAY notifications and hospital empanelment data, a Next.js public experience, and an admin dashboard for confidence and freshness monitoring. Every answer path is citation-aware and fallback-safe to minimize false benefit claims.

## Architecture Diagram

- See architecture: [docs/architecture.md](docs/architecture.md)

## Quick Start

1. Clone repository:

```bash
git clone <your-repo-url>
cd yojanasetu
```

2. Create environment file:

```bash
cp .env.example .env
```

3. Start local stack:

```bash
docker-compose up --build
```

4. Verify health endpoint:

```bash
curl -s http://localhost:8000/health
```

## Local Dev (Single Command)

Start both backend and frontend from the repository root:

```bash
npm run dev
```

This command will:
- create `.env` from `.env.example` if missing,
- set up `backend/.venv311` with Python 3.11 using `uv` (first run only),
- start backend on `http://localhost:8000`,
- start frontend on `http://localhost:3000`.

## Data Loading

```bash
python scripts/run_ingestion.py --help
```

## Validation Commands

```bash
python scripts/validate_golden_set.py --golden-set backend/tests/golden_set.json
python scripts/validate_denial_decoder.py --letters-dir backend/tests/denial_letters
```

## API Reference

### Query Agent

- Endpoint: `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"PMJAY eligibility?","language":"hi","hospital_name":null}'
```

### System Health

- Endpoint: `GET /health`

```bash
curl -s http://localhost:8000/health
```

### Hospital Status

- Endpoint: `GET /hospital/{name}`

```bash
curl -s "http://localhost:8000/hospital/Ruby%20Hall"
```

### Denial Decoder

- Endpoint: `POST /denial`

```bash
curl -X POST http://localhost:8000/denial \
  -H "Content-Type: application/json" \
  -d '{"text":"Claim rejected due to missing document"}'
```

### IVR Incoming Hook

- Endpoint: `POST /ivr/incoming`

```bash
curl -X POST http://localhost:8000/ivr/incoming
```

### IVR Transcription Hook

- Endpoint: `POST /ivr/transcription`

```bash
curl -X POST http://localhost:8000/ivr/transcription \
  -d "CallSid=CA123" \
  -d "RecordingUrl=https://example.com/audio.mp3" \
  -d "TranscriptionText=PMJAY question" \
  -d "TranscriptionConfidence=0.92"
```

### IVR DTMF Fallback

- Endpoint: `POST /ivr/dtmf`

```bash
curl -X POST http://localhost:8000/ivr/dtmf -d "Digits=1"
```

### Legacy PMJAY Navigate Endpoint

- Endpoint: `POST /api/v1/pmjay/navigate`

```bash
curl -X POST http://localhost:8000/api/v1/pmjay/navigate \
  -H "Content-Type: application/json" \
  -d '{"query":"What documents are required for PMJAY?","language":"hi","state":"maharashtra","scheme":"pmjay"}'
```
