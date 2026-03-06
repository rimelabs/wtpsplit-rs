//! ONNX model wrapper for inference

use half::f16;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;

use crate::config::ModelConfig;
use crate::error::Error;
use crate::Result;

/// ONNX model wrapper that handles both SaT and WtP models
pub struct OnnxModel {
    session: Session,
    pub config: ModelConfig,
}

impl OnnxModel {
    /// Load an ONNX model from a file
    pub fn new(onnx_path: &Path, config: ModelConfig) -> Result<Self> {
        let session = Session::builder()
            .map_err(ort::Error::from)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort::Error::from)?
            .commit_from_file(onnx_path)
            .map_err(ort::Error::from)?;

        Ok(Self { session, config })
    }

    /// Run inference for SaT models (subword-based)
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs (batch_size x seq_len)
    /// * `attention_mask` - Attention mask (batch_size x seq_len)
    ///
    /// # Returns
    /// Logits tensor (batch_size x seq_len x num_labels)
    pub fn forward_sat(
        &mut self,
        input_ids: ArrayView2<'_, i64>,
        attention_mask: ArrayView2<'_, f32>,
    ) -> Result<Array3<f32>> {
        // Create input_ids tensor (shape, data) for ort 2.0.0-rc.11 compatibility
        let shape_ids = [input_ids.dim().0, input_ids.dim().1];
        let data_ids: Box<[i64]> =
            input_ids.iter().copied().collect::<Vec<_>>().into_boxed_slice();
        let input_ids_value = Value::from_array((shape_ids, data_ids))?;

        // Convert attention mask to f16
        let attention_mask_f16: Array2<f16> = attention_mask.mapv(f16::from_f32);
        let shape_mask = [attention_mask_f16.dim().0, attention_mask_f16.dim().1];
        let data_mask: Box<[f16]> = attention_mask_f16
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let attention_mask_value = Value::from_array((shape_mask, data_mask))?;

        // Run inference
        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_value,
            "attention_mask" => attention_mask_value
        ])?;

        // Extract logits from output (model outputs f16, convert to f32)
        let logits_value = &outputs["logits"];
        let (out_shape, data_f16) = logits_value
            .try_extract_tensor::<f16>()
            .map_err(|e| Error::Inference(format!("Failed to extract logits: {}", e)))?;

        // Convert f16 to f32
        let data_f32: Vec<f32> = data_f16.iter().map(|&x| x.to_f32()).collect();

        // Convert to owned Array3
        let logits_array = Array3::from_shape_vec(
            (out_shape[0] as usize, out_shape[1] as usize, out_shape[2] as usize),
            data_f32,
        )
        .map_err(|e| Error::Inference(format!("Failed to reshape logits: {}", e)))?;

        Ok(logits_array)
    }

    /// Run inference for WtP models (character-based)
    ///
    /// # Arguments
    /// * `hashed_ids` - Hash-encoded character IDs (batch_size x seq_len x num_hashes)
    /// * `attention_mask` - Attention mask (batch_size x seq_len)
    ///
    /// # Returns
    /// Logits tensor (batch_size x seq_len x num_labels)
    pub fn forward_wtp(
        &mut self,
        hashed_ids: ArrayView3<'_, i64>,
        attention_mask: ArrayView2<'_, f32>,
    ) -> Result<Array3<f32>> {
        // Create hashed_ids tensor (shape, data) for ort 2.0.0-rc.11 compatibility
        let dim = hashed_ids.dim();
        let shape_ids = [dim.0, dim.1, dim.2];
        let data_ids: Box<[i64]> =
            hashed_ids.iter().copied().collect::<Vec<_>>().into_boxed_slice();
        let hashed_ids_value = Value::from_array((shape_ids, data_ids))?;

        // Convert attention mask to f16
        let attention_mask_f16: Array2<f16> = attention_mask.mapv(f16::from_f32);
        let shape_mask = [attention_mask_f16.dim().0, attention_mask_f16.dim().1];
        let data_mask: Box<[f16]> = attention_mask_f16
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let attention_mask_value = Value::from_array((shape_mask, data_mask))?;

        // Run inference
        let outputs = self.session.run(ort::inputs![
            "hashed_ids" => hashed_ids_value,
            "attention_mask" => attention_mask_value
        ])?;

        // Extract logits from output (model outputs f16, convert to f32)
        let logits_value = &outputs["logits"];
        let (out_shape, data_f16) = logits_value
            .try_extract_tensor::<f16>()
            .map_err(|e| Error::Inference(format!("Failed to extract logits: {}", e)))?;

        // Convert f16 to f32
        let data_f32: Vec<f32> = data_f16.iter().map(|&x| x.to_f32()).collect();

        // Convert to owned Array3
        let logits_array = Array3::from_shape_vec(
            (out_shape[0] as usize, out_shape[1] as usize, out_shape[2] as usize),
            data_f32,
        )
        .map_err(|e| Error::Inference(format!("Failed to reshape logits: {}", e)))?;

        Ok(logits_array)
    }
}

/// Batch processor for efficient inference
pub struct BatchProcessor {
    pub batch_size: usize,
}

impl BatchProcessor {
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }

    /// Split chunks into batches for processing
    pub fn create_batches<T: Clone>(&self, items: &[T]) -> Vec<Vec<T>> {
        items.chunks(self.batch_size).map(|c| c.to_vec()).collect()
    }
}
