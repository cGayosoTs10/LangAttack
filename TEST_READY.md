# TEST_READY: LangAttack E2E Test Suite Specification & Inventory

**Status**: READY & VERIFIED  
**Runner**: `~/.local/bin/uv run pytest`  
**Total Tests**: 145 tests across unit and E2E tiers  
**Current Execution Result**: 34 passed, 111 skipped (progressive testability across M1-M4), 0 failed  
**Lint Status**: `~/.local/bin/uv run ruff check .` clean (0 errors)  
**Security / Telemetry AST**: 0 pdb violations across entire repository  

---

## 1. Test Architecture & Directory Layout

```
tests/
├── conftest.py                    # Offscreen Qt, hardware mocks (sounddevice, GPU, AI engines)
├── test_basic.py                  # M1 repository foundation & path sanity
├── test_schemas.py                # Tier 1 & 2 Pydantic schemas (phonetics, exercise)
├── test_db.py                     # Tier 1 & 2 SQLite DDL, WAL, language flexibility
├── test_telemetry.py              # Tier 1 & 2 log rotation, latency tracker, zero-pdb AST
├── test_ingestion.py              # Tier 1 & 2 web article & YouTube transcript extraction
├── test_workers.py                # Tier 1 & 2 GenericWorker & QThreadPool concurrency
├── test_core.py                   # Tier 1 & 2 TTS, ASR, Phonetics GOP, LLM provider
├── test_ui.py                     # Tier 1 & 2 PySide6 GUI panels, QSplitter, actions
└── e2e/
    ├── test_tier1_features.py     # Tier 1: Feature Coverage (>=5 test cases per area, 40 tests)
    ├── test_tier2_boundary.py     # Tier 2: Boundary & Corner Cases (>=5 test cases per area, 40 tests)
    ├── test_tier3_pairwise.py     # Tier 3: Cross-Feature Interactions (5 pair workflows)
    └── test_tier4_workloads.py    # Tier 4: Real-World Workloads (Scenarios A, B, C)
```

---

## 2. Feature Inventory Coverage Matrix

| Feature Area | Tier 1 (Coverage) | Tier 2 (Boundary) | Tier 3 (Pairwise) | Tier 4 (Workloads) | Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **FEAT-ENV** (uv, pyproject, gitignore) | 5 tests | 5 tests | ✓ | ✓ | Passed (M1) |
| **FEAT-TEL** (logs, latency_tracker, zero-pdb) | 5 tests | 5 tests | ✓ | ✓ | Passed (M1) |
| **FEAT-DAT** (SQLite DDL, WAL, 4 tables, language) | 5 tests | 5 tests | ✓ | ✓ | Ready (M2) |
| **FEAT-SCH** (Pydantic Phonetics & Exercise) | 5 tests | 5 tests | ✓ | ✓ | Ready (M2) |
| **FEAT-CON** (GenericWorker, QThreadPool) | 5 tests | 5 tests | ✓ | ✓ | Ready (M3) |
| **FEAT-ING** (Web & YouTube extraction) | 5 tests | 5 tests | ✓ | ✓ | Ready (M3) |
| **FEAT-AIE** (TTS CPU 16kHz, Whisper, Wav2Vec, LLM) | 5 tests | 5 tests | ✓ | ✓ | Ready (M3) |
| **FEAT-GUI** (PySide6 QSplitter, 3 panes, dark theme) | 5 tests | 5 tests | ✓ | ✓ | Ready (M4) |

---

## 3. Mock & Headless Infrastructure (`tests/conftest.py`)

- **Offscreen Display**: `QT_QPA_PLATFORM=offscreen` ensures PySide6 initializes safely in headless environments without X11/Wayland display server.
- **Audio Mocking (`sounddevice`)**: Automatic autouse fixture intercepts `sounddevice.rec`, `sounddevice.play`, `sounddevice.wait`, `sounddevice.query_devices`, `sounddevice.InputStream`.
- **GPU Mocking (`torch.cuda`)**: Emulates CUDA availability without requiring physical NVIDIA hardware.
- **AI Pipelines**: Mock fixtures provide realistic stubs for Kokoro-82M (valid 16kHz WAV generation), faster-whisper medium ASR, Wav2Vec GOP evaluation, and Gemini/Ollama LLM responses.
- **Network Isolation**: Trafilatura web fetching and YouTube subtitle requests are mocked for complete offline execution.

---

## 4. Verification Command

```bash
~/.local/bin/uv run pytest
```

