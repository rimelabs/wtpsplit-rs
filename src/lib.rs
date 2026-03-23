//! # wtpsplit
//!
//! Rust port of wtpsplit - Universal sentence and paragraph segmentation.
//!
//! This library provides fast, accurate sentence boundary detection using ONNX models.
//!
//! ## Features
//!
//! - **SaT (Segment any Text)**: Modern subword-based models using XLM-RoBERTa
//! - **WtP (Where's the Point)**: Legacy character-based models (deprecated)
//! - ONNX Runtime for efficient inference
//! - Automatic model downloading from HuggingFace Hub
//!
//! ## Example
//!
//! ```no_run
//! use wtpsplit::SaT;
//!
//! let sat = SaT::new("sat-3l-sm", None)?;
//! let sentences = sat.split("This is a test. Another sentence here.", None)?;
//! for sentence in sentences {
//!     println!("{}", sentence);
//! }
//! # Ok::<(), wtpsplit::Error>(())
//! ```

pub mod config;
pub mod constants;
pub mod error;
pub mod extract;
pub mod hub_common;
#[cfg(feature = "hub")]
pub mod hub;
pub mod model;
pub mod sat;
pub mod utils;
pub mod wtp;

pub use config::ModelConfig;
pub use error::Error;
pub use extract::Weighting;
pub use sat::{SaT, SaTOptions};
#[allow(deprecated)]
pub use wtp::{WtP, WtPOptions};

/// Result type alias for wtpsplit operations
pub type Result<T> = std::result::Result<T, Error>;
