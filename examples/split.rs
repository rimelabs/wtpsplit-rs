//! Example: Split text into sentences using SaT
//!
//! Usage:
//!   cargo run --example split -- "Your text here. Another sentence."
//!   cargo run --example split -- --model sat-12l-sm "Text to split."
//!   cargo run --example split -- --file input.txt

use std::env;
use std::fs;

use wtpsplit::{SaT, SaTOptions, Weighting};

fn print_usage() {
    eprintln!("Usage: split [OPTIONS] <TEXT or --file PATH>");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --model <NAME>      Model name (default: sat-3l-sm)");
    eprintln!("  --threshold <FLOAT> Sentence boundary threshold");
    eprintln!("  --file <PATH>       Read text from file");
    eprintln!("  --strip             Strip whitespace from sentences");
    eprintln!("  --weighting <TYPE>  Weighting scheme: uniform or hat (default: uniform)");
    eprintln!("  --help              Show this help message");
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    // Parse arguments
    let mut model_name = "sat-3l-sm".to_string();
    let mut threshold: Option<f32> = None;
    let mut text: Option<String> = None;
    let mut strip_whitespace = false;
    let mut weighting = Weighting::Uniform;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            "--model" | "-m" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --model requires a value");
                    std::process::exit(1);
                }
                model_name = args[i].clone();
            }
            "--threshold" | "-t" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --threshold requires a value");
                    std::process::exit(1);
                }
                threshold = Some(args[i].parse()?);
            }
            "--file" | "-f" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --file requires a path");
                    std::process::exit(1);
                }
                text = Some(fs::read_to_string(&args[i])?);
            }
            "--strip" | "-s" => {
                strip_whitespace = true;
            }
            "--weighting" | "-w" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --weighting requires a value (uniform or hat)");
                    std::process::exit(1);
                }
                weighting = match args[i].to_lowercase().as_str() {
                    "uniform" => Weighting::Uniform,
                    "hat" => Weighting::Hat,
                    _ => {
                        eprintln!("Error: Invalid weighting scheme. Use 'uniform' or 'hat'");
                        std::process::exit(1);
                    }
                };
            }
            arg if arg.starts_with('-') => {
                eprintln!("Error: Unknown option: {}", arg);
                print_usage();
                std::process::exit(1);
            }
            _ => {
                // Treat as text input
                if text.is_none() {
                    text = Some(args[i].clone());
                } else {
                    // Append to existing text
                    text = Some(format!("{} {}", text.unwrap(), args[i]));
                }
            }
        }
        i += 1;
    }

    let text = match text {
        Some(t) => t,
        None => {
            eprintln!("Error: No text provided");
            print_usage();
            std::process::exit(1);
        }
    };

    println!("Loading model: {}...", model_name);
    let mut sat = SaT::new(&model_name, None)?;

    let options = SaTOptions {
        threshold,
        strip_whitespace,
        weighting,
        ..Default::default()
    };

    println!("Splitting text...\n");
    let sentences = sat.split(&text, Some(&options))?;

    println!("Found {} sentences:\n", sentences.len());
    for (i, sentence) in sentences.iter().enumerate() {
        println!("[{}] {}", i + 1, sentence);
    }

    Ok(())
}
