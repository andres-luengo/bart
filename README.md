# RFI Pipeline

A Python package for detecting and analyzing Radio Frequency Interference (RFI) in astronomical observation data.

## Description

The RFI Pipeline is designed to process large volumes of astronomical data stored in HDF5 format, identifying potential radio frequency interference through statistical analysis. The pipeline uses multi-processing capabilities to efficiently handle large datasets and provides detailed logging for monitoring the analysis process.

## Features

- **Multi-processing support**: Parallel processing of data batches for improved performance
- **Statistical filtering**: Two-stage filtering process using warm and hot significance thresholds
- **Batch processing**: Divides large datasets into manageable batches for reliability
- **Comprehensive logging**: Detailed logging with configurable verbosity levels
- **Memory management**: Configurable memory limits to prevent system overload
- **Flexible configuration**: Command-line interface with extensive parameter customization

## Installation

### From PyPI (when published)
```bash
pip install rfi-pipeline
```

### Development Installation
```bash
git clone https://github.com/breakthrough-listen/rfi-pipeline.git
cd rfi-pipeline
pip install -e .
```

### Development with optional dependencies
```bash
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

```bash
rfi-finder input_files.txt output_directory [options]
```

#### Basic Example
```bash
rfi-finder file_list.txt /path/to/output --num-processes 4 --max-rss-gb 64
```

#### Parameters

**Input/Output:**
- `infile`: Path to file containing paths to .h5 files (one per line)
- `outdir`: Output directory for results
- `-f, --force`: Overwrite existing output directory

**Processing:**
- `--frequency-block-size`: Size of frequency blocks (default: 1024)
- `--warm-significance`: Sigma threshold for initial filtering (default: 5.0)
- `--hot-significance`: MAD threshold for secondary filtering (default: 10.0)

**Resource Management:**
- `-n, --num-processes`: Number of parallel processes (default: 1)
- `-m, --max-rss-gb`: Maximum memory usage in GB (default: 32.0)
- `--num-batches`: Number of processing batches (default: 100)

**Logging:**
- `-v, --verbose`: Increase verbosity (can be repeated)
- `-q, --quiet`: Decrease verbosity (can be repeated)

### Python API

```python
from rfi_pipeline import Manager
from pathlib import Path

# Set up processing parameters
process_params = {
    'freq_window': 1024,
    'warm_significance': 5.0,
    'hot_significance': 10.0
}

# Initialize manager
files = [Path("data1.h5"), Path("data2.h5")]
manager = Manager(
    process_params=process_params,
    num_batches=10,
    num_processes=4,
    files=tuple(files),
    outdir=Path("output")
)

# Run processing
manager.run()
```

### Merge Tool

After running the main RFI pipeline, you can merge all batch results into a single file using the merge tool:

#### Command Line Interface
```bash
rfi-pipeline-merge rundir [outdir] [options]
```

#### Examples
```bash
# Merge batches in run directory to the same directory
rfi-pipeline-merge /path/to/pipeline/output

# Merge to custom output directory
rfi-pipeline-merge /path/to/pipeline/output /path/to/merged/results

# Compress output and use verbose logging
rfi-pipeline-merge /path/to/pipeline/output --compress --verbose

# Output in different formats
rfi-pipeline-merge /path/to/pipeline/output --format parquet --compress
```

#### Parameters
- `rundir`: Path to RFI pipeline output directory (must contain `batches/` subdirectory)
- `outdir`: Output directory for merged data (optional, defaults to the run directory itself)
- `-f, --force`: Overwrite existing output files without confirmation
- `-v, --verbose`: Increase verbosity level (use `-vv` for debug output)
- `--format`: Output format: `csv` (default), `parquet`, or `hdf5`
- `--sort-by`: Column to sort merged data by (default: `frequency`)
- `--compress`: Compress output file (format-dependent compression)

#### Python API
```python
from pathlib import Path
from rfi_pipeline.merge import merge_rfi_run

# Merge batch files programmatically
output_path = merge_rfi_run(
    rundir=Path("/path/to/pipeline/output"),
    outdir=Path("/path/to/merged/results"),  # Optional, defaults to rundir
    format_type='csv',
    compress=True,
    sort_by='frequency',
    force=True
)
print(f"Merged data saved to: {output_path}")
```

The merge tool will:
- Combine all `batch_*.csv` files from the `batches/` directory
- Add a `batch_number` column to track the source batch
- Sort the data by the specified column (default: frequency)
- Update the existing `meta.json` file with merge information
- Support multiple output formats with optional compression

## Algorithm

The RFI detection algorithm works in two stages:

1. **Warm Filtering**: Identifies frequency blocks where the maximum value exceeds the median by more than `warm_significance` standard deviations
2. **Hot Filtering**: Further filters blocks where the maximum value exceeds the median by more than `hot_significance` median absolute deviations

For each detected RFI candidate, the pipeline computes:
- Central frequency of the detection
- Kurtosis of the normalized signal

## Output

Results are saved as CSV files in the specified output directory:
- `batches/batch_XXX.csv`: Individual batch results
- `logs/all_logs.log`: Comprehensive processing logs
- `logs/error_logs.log`: Error-specific logs

Each CSV contains columns:
- `frequency`: Central frequency of the detection
- `kurtosis`: Statistical kurtosis of the signal
- `source file`: Path to the source data file

## Requirements

- Python ≥ 3.8
- numpy ≥ 1.20.0
- pandas ≥ 1.3.0
- scipy ≥ 1.7.0
- h5py ≥ 3.0.0
- hdf5plugin ≥ 3.0.0

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black .
```

### Type Checking
```bash
mypy .
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Support

For issues and questions, please use the GitHub issue tracker.
