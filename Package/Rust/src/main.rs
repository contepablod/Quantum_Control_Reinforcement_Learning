use pyo3::prelude::*;
use std::process::Command;

fn main() -> PyResult<()> {
    // Path to your Python script
    let script_path = "/home/pdconte/Desktop/DUTh_Thesis/Package/main.py";

    // Run the Python script using the system's Python interpreter
    let output = Command::new("python3")
        .arg(script_path)
        .output()
        .expect("Failed to execute Python script");

    // Check if the script ran successfully
    if output.status.success() {
        println!("Python script output:\n{}", String::from_utf8_lossy(&output.stdout));
    } else {
        eprintln!("Error:\n{}", String::from_utf8_lossy(&output.stderr));
    }

    Ok(())
}
