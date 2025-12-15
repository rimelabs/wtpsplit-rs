# wtpsplit-rs

A Rust port of [wtpsplit](https://github.com/segment-any-text/wtpsplit) - universal sentence and paragraph segmentation using ONNX models.

## Overview

wtpsplit-rs provides high-quality sentence segmentation for 85+ languages using pre-trained transformer models. It's a native Rust implementation that runs ONNX models for inference, making it suitable for production deployments without Python dependencies.

### Supported Models

- **SaT (Segment any Text)** - Modern subword-based models using XLM-RoBERTa backbone (recommended)
- **WtP (Where's the Point)** - Legacy character-based models (deprecated)

## Features

- Native Rust implementation with minimal dependencies
- ONNX Runtime inference for cross-platform deployment
- Automatic model downloading from HuggingFace Hub
- Support for local model files
- Configurable sentence boundary threshold
- Overlapping chunk processing for long documents
- Batch processing support

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
wtpsplit = { path = "path/to/wtpsplit-rs" }
```

### ONNX Runtime Setup

wtpsplit-rs uses dynamic linking to ONNX Runtime. You need to have the ONNX Runtime library available:

**Option 1: Install via pip**
```bash
pip install onnxruntime
# or for GPU support:
pip install onnxruntime-gpu
```

Then set the library path:
```bash
export ORT_DYLIB_PATH=$(python -c "import onnxruntime; print(onnxruntime.__path__[0])")/capi/libonnxruntime.so
```

**Option 2: Download from GitHub releases**

Download from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) and set:
```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
```

## Quick Start

```rust
use wtpsplit::{SaT, SaTOptions};

fn main() -> anyhow::Result<()> {
    // Load model (downloads automatically from HuggingFace Hub)
    let mut sat = SaT::new("sat-3l-sm", None)?;

    // Split text into sentences
    let text = "Hello world. This is a test. How are you?";
    let sentences = sat.split(text, None)?;

    for sentence in sentences {
        println!("{}", sentence);
    }

    Ok(())
}
```

## API Reference

### SaT

The main struct for sentence segmentation.

#### Creating a SaT Instance

```rust
// From HuggingFace Hub (auto-download)
let sat = SaT::new("sat-3l-sm", None)?;

// With custom hub prefix
let sat = SaT::new("sat-3l-sm", Some("segment-any-text"))?;

// From local directory
let sat = SaT::new("/path/to/model", None)?;
// or
let sat = SaT::from_dir(Path::new("/path/to/model"))?;
```

#### Splitting Text

```rust
// Basic splitting
let sentences = sat.split("Your text here.", None)?;

// With options
let options = SaTOptions {
    threshold: Some(0.5),        // Custom threshold (default: 0.01)
    strip_whitespace: true,      // Trim sentences
    ..Default::default()
};
let sentences = sat.split("Your text here.", Some(&options))?;

// Batch processing
let texts = vec!["First text.", "Second text."];
let results = sat.split_batch(&texts, None)?;

// Paragraph segmentation
let paragraphs = sat.split_paragraphs("First para sentence.\n\nSecond para.", None)?;
// Returns Vec<Vec<String>> - paragraphs containing sentences
```

#### Getting Probabilities

```rust
// Get per-character boundary probabilities
let probs = sat.predict_proba("Hello world. Test.", None)?;
// Returns Vec<f32> with probability for each character
```

### SaTOptions

Configuration for sentence splitting:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `threshold` | `Option<f32>` | `None` (0.01) | Probability threshold for sentence boundaries |
| `stride` | `usize` | `64` | Stride for overlapping chunks |
| `block_size` | `usize` | `512` | Maximum chunk size in tokens |
| `batch_size` | `usize` | `32` | Batch size for inference |
| `weighting` | `Weighting` | `Uniform` | Weight scheme for overlapping predictions (`Uniform` or `Hat`) |
| `strip_whitespace` | `bool` | `false` | Trim whitespace from sentences |
| `split_on_input_newlines` | `bool` | `true` | Split on newlines in addition to model predictions |
| `remove_whitespace_before_inference` | `bool` | `false` | Remove spaces before inference (for some languages) |
| `paragraph_threshold` | `f32` | `0.5` | Probability threshold for paragraph boundaries |
| `do_paragraph_segmentation` | `bool` | `false` | Enable paragraph segmentation mode |

## Available Models

Models are automatically downloaded from HuggingFace Hub on first use.

### SaT Models (Recommended)

| Model | Parameters | Description |
|-------|------------|-------------|
| `sat-1l` | ~85M | Single layer, fastest |
| `sat-3l` | ~95M | 3 layers, good balance |
| `sat-6l` | ~110M | 6 layers |
| `sat-12l` | ~135M | 12 layers, most accurate |
| `sat-1l-sm` | ~45M | Single layer, small |
| `sat-3l-sm` | ~50M | 3 layers, small (default) |
| `sat-12l-sm` | ~85M | 12 layers, small |

### WtP Models (Deprecated)

Legacy character-based models. Use SaT models for new projects.

## Command-Line Example

The crate includes a CLI example:

```bash
# Build
cargo build --release --example split

# Run with text
./target/release/examples/split "Hello world. This is a test."

# Run with file
./target/release/examples/split --file input.txt

# With options
./target/release/examples/split --model sat-12l-sm --threshold 0.5 --strip "Your text here."

# Paragraph segmentation
./target/release/examples/split --paragraphs --file document.txt

# With custom paragraph threshold
./target/release/examples/split --paragraphs --para-threshold 0.7 --file document.txt

# Show help
./target/release/examples/split --help
```

## Using Local Models

To use locally stored ONNX models:

1. Export your model to ONNX format (see [wtpsplit documentation](https://github.com/segment-any-text/wtpsplit))

2. Ensure the directory contains:
   - `model.onnx` - The ONNX model file
   - `config.json` - Model configuration

3. Load from path:
```rust
let sat = SaT::new("/path/to/model/directory", None)?;
```

## Performance Tips

1. **Use release builds** - Debug builds are significantly slower
2. **Batch processing** - Use `split_batch` for multiple texts
3. **Appropriate model size** - `sat-3l-sm` offers good speed/accuracy tradeoff
4. **GPU acceleration** - Use `onnxruntime-gpu` for CUDA support

## Error Handling

The library uses a custom `Error` type with variants:

```rust
use wtpsplit::{Error, Result};

match sat.split(text, None) {
    Ok(sentences) => { /* ... */ }
    Err(Error::ModelLoad(msg)) => eprintln!("Failed to load model: {}", msg),
    Err(Error::Inference(msg)) => eprintln!("Inference error: {}", msg),
    Err(Error::Tokenization(msg)) => eprintln!("Tokenization error: {}", msg),
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Comparison with Python wtpsplit

| Feature | Python | Rust |
|---------|--------|------|
| Sentence segmentation | Yes | Yes |
| Paragraph segmentation | Yes | Yes |
| Language adapters | Yes | No |
| Style adapters | Yes | No |
| PyTorch backend | Yes | No |
| ONNX backend | Yes | Yes |
| Punctuation prediction | Yes | No |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Credits

- Original [wtpsplit](https://github.com/segment-any-text/wtpsplit) by Markus Frohmann, Igor Sterner, Benjamin Minixhofer, et al.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) for inference
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) for XLM-RoBERTa tokenization

## References

```bibtex
@inproceedings{frohmann-etal-2024-segment,
    title = "Segment Any Text: A Universal Approach for Robust, Efficient and Adaptable Sentence Segmentation",
    author = "Frohmann, Markus  and
      Sterner, Igor  and
      Vuli{\'c}, Ivan  and
      Minixhofer, Benjamin  and
      Schedl, Markus",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.665",
    pages = "11908--11941"
}
```
