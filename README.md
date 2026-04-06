# feature_extraction_and_matching

SIFT feature extraction and descriptor matching (SSD + NCC) pipeline.

## Requirements

- Python 3.8+
- `opencv-python`
- `numpy`

Install:

```bash
pip install opencv-python numpy
```

## Run

Process all unique pairs from a list of images:

```bash
python feature_pipeline.py image1.jpg image2.jpg image3.png --output-dir output_matches --top-k 50
```

For each image pair, the script prints:

- `SIFT_Descriptor_Time: [X.XX] seconds`
- `Matching_SSD_Time: [X.XX] seconds`
- `Matching_NCC_Time: [X.XX] seconds`

It also writes match visualizations for SSD and NCC to the output directory.
