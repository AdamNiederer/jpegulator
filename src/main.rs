// -*- rustic-run-arguments: "test4.jpg"; -*-
// (setq-local rustic-run-arguments "test4.jpg")

use std::ffi::OsStr;
use std::path::PathBuf;
use clap::Parser;

mod constants;
mod decoder;
mod encoder;
mod jpeg;
mod bmp;

use crate::decoder::decode;
use crate::encoder::encode;

#[derive(Parser)]
struct Cli {
    file: PathBuf,
    #[arg(short, long)]
    decode: bool,
    #[arg(long)]
    output_coefficients: Option<PathBuf>,
    #[arg(long)]
    output_y: Option<PathBuf>,
    #[arg(long)]
    output_cb: Option<PathBuf>,
    #[arg(long)]
    output_cr: Option<PathBuf>,
}

fn main() -> Result<(), anyhow::Error> {
    let args = Cli::parse();
    let input = std::fs::read(&args.file).unwrap();
    let extension = args.file.extension().map(OsStr::to_str).unwrap_or(Some("")).unwrap_or("");

    if !args.decode && extension != "jpg" && extension != "jpeg" && extension != "jfif" {
        std::fs::write(args.file.with_extension("jpg"), encode(input)?).unwrap();
    } else {
        std::fs::write(args.file.with_extension("bmp"), decode(input)?).unwrap();
    }

    Ok(())
}
