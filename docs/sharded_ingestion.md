# Sharded Ingestion: Storage-Optimized Pipeline

## What Changed

The corpus ingestion pipeline has been **modified to shard during harmonization** instead of after, reducing peak storage requirements by 50%.

## Why This Matters

**Old approach (ingest → shard):**
- Writes thousands of individual `.npy` files (~50-80GB)
- Then concatenates them into shards (~50-80GB)
- **Peak storage: 100-160GB** (both formats exist during sharding)

**New approach (shard during ingest):**
- Buffers harmonized recordings in memory
- Flushes to 2GB shard when buffer exceeds threshold
- **Peak storage: 50-80GB** (only shards exist)

---

## Files Modified

### 1. `src/neurostream/preprocessing/corpus_pipeline.py`

**Key changes:**
- `ingest_corpus()` now buffers recordings and writes shards directly
- No longer writes per-recording `.npy` files
- Manifest format changed from `v1` to `v2-sharded`

**New manifest structure:**
```json
{
  "version": "v2-sharded",
  "shards": [
    {
      "shard_name": "shard_000.npy",
      "shard_idx": 0,
      "n_channels": 22,
      "total_samples": 15680000,
      "recordings": [
        {
          "source": "PhysionetMI",
          "subject": 1,
          "session": "0",
          "run": "0",
          "byte_offset": 0,
          "n_samples": 32000,
          "n_channels": 22,
          "fs": 128,
          "units": "uV"
        },
        ...
      ]
    },
    ...
  ],
  "rejected": [...],
  "total_recordings": 2847,
  "total_shards": 38
}
```

### 2. `configs/pretrain_corpus.yaml`

**Added parameter:**
```yaml
# Shard size in GB — recordings concatenated until buffer exceeds this.
shard_size_gb: 2.0
```

### 3. `notebooks/02_open_corpus_sanity.ipynb`

**Updated to read sharded format:**
- Memory-maps shard files instead of per-recording files
- Uses `byte_offset` metadata to slice recordings from concatenated shards
- Caches loaded shards for efficiency
- Assertion checks for `v2-sharded` format

**New helper functions:**
```python
def load_shard(shard_name: str) -> np.ndarray:
    """Memory-map a shard file, cached."""

def load_recording(rec: dict) -> np.ndarray:
    """Slice one recording from its parent shard using byte_offset."""
```

---

## Output Directory Structure

**Before (v1):**
```
data/processed/open_corpus/
├── PhysionetMI/
│   ├── sub-001/
│   │   ├── ses-0_run-0.npy
│   │   ├── ses-0_run-1.npy
│   │   └── ...
│   └── sub-002/...
├── Cho2017/...
└── manifest.json
```

**After (v2-sharded):**
```
data/processed/open_corpus/
├── shard_000.npy              # 2GB concatenated recordings
├── shard_000_meta.json        # recording boundaries + provenance
├── shard_001.npy
├── shard_001_meta.json
├── ...
├── shard_037.npy
├── shard_037_meta.json
└── manifest.json              # top-level: all shards + rejected
```

---

## Running Ingestion

**Same command as before:**
```bash
uv run python scripts/ingest_open_corpus.py
```

**Expected output:**
```
INFO ingest :: writing to /Users/.../data/processed/open_corpus
INFO corpus_pipeline :: === PhysionetMI: subjects=all ===
INFO corpus_pipeline :: Flushed shard_000: 83 recordings, 15680000 samples, 1.39 GB
INFO corpus_pipeline :: Flushed shard_001: 79 recordings, 14912000 samples, 1.32 GB
...
INFO corpus_pipeline :: Done. kept=2847 rejected=142 shards=38 sources_seen=[...]
```

---

## Validation

**Run the sanity-check notebook after ingestion:**
```bash
jupyter notebook notebooks/02_open_corpus_sanity.ipynb
```

The notebook will:
1. Assert the manifest is `v2-sharded` format
2. Memory-map shards and slice recordings using `byte_offset` metadata
3. Run all 5 sanity checks (random windows, channel stats, durations, power spectra)

**Pass criterion:** All 5 checks show clean, consistent EEG across sources.

---

## Benefits

1. **50% less peak storage** — critical for storage-constrained environments
2. **Single-pass ingestion** — no separate sharding step needed
3. **Faster transfer to Windows** — fewer files to copy (38 shards vs. thousands)
4. **Better for MAE training** — shards are already in the format the dataloader needs

---

## Trade-offs

**Pros:**
- ✅ Half the storage
- ✅ Faster iteration
- ✅ One less script to maintain

**Cons:**
- ⚠️ Can't inspect individual recordings as easily (need to slice from shard)
- ⚠️ If harmonization fails mid-run, lose the entire partial shard
- ⚠️ Notebook now has 20 more lines (shard-slicing logic)

The storage savings outweigh the cons for Phase 2's use case.

---

## Next Steps

1. **Run ingestion:** `uv run python scripts/ingest_open_corpus.py` (1-3 hours)
2. **Validate in notebook:** All 5 sanity checks pass
3. **Transfer shards to Windows:** 38 files instead of thousands
4. **Implement MAE dataloader:** Reads directly from sharded format (Day 7-9)
