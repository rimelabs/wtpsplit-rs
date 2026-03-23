//! Common model file utilities (no network access required)

use std::path::{Path, PathBuf};

use crate::error::Error;
use crate::Result;

/// Default hub prefix for SaT models
pub const SAT_HUB_PREFIX: &str = "segment-any-text";

/// Default hub prefix for WtP models
pub const WTP_HUB_PREFIX: &str = "benjamin";

/// Model files that need to be downloaded
pub struct ModelFiles {
    pub config_path: PathBuf,
    pub onnx_path: PathBuf,
    pub tokenizer_path: Option<PathBuf>,
}

/// Check if a path is a local directory
pub fn is_local_path(path: &str) -> bool {
    Path::new(path).is_dir()
}

/// Get model files from a local directory
pub fn get_local_model_files(model_dir: &Path, use_optimized: bool) -> Result<ModelFiles> {
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Err(Error::ModelNotFound(format!(
            "config.json not found in {}",
            model_dir.display()
        )));
    }

    let onnx_filename = if use_optimized {
        "model_optimized.onnx"
    } else {
        "model.onnx"
    };

    let onnx_path = model_dir.join(onnx_filename);
    if !onnx_path.exists() {
        // Try the other variant
        let alt_filename = if use_optimized {
            "model.onnx"
        } else {
            "model_optimized.onnx"
        };
        let alt_path = model_dir.join(alt_filename);
        if alt_path.exists() {
            return Ok(ModelFiles {
                config_path,
                onnx_path: alt_path,
                tokenizer_path: model_dir.join("tokenizer.json").exists().then(|| model_dir.join("tokenizer.json")),
            });
        }
        return Err(Error::ModelNotFound(format!(
            "ONNX model not found in {}",
            model_dir.display()
        )));
    }

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer_path = if tokenizer_path.exists() {
        Some(tokenizer_path)
    } else {
        None
    };

    Ok(ModelFiles {
        config_path,
        onnx_path,
        tokenizer_path,
    })
}
