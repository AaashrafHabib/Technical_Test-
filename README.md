# Drawing Quality Analyzer

This repository contains a Python project that analyzes the quality of technical drawings. The analyzer evaluates various aspects of an image, such as technical and structural features, and calculates a comprehensive quality score. The tool can be used for patent drawings, technical illustrations, or any type of graphical artwork.

## Features

- **Technical Score Calculation**:
  - Line Clarity: Evaluates the straightness of lines in the drawing.
  - Contrast: Measures the contrast between light and dark areas of the image.
  - Resolution: Assesses the resolution of the image and its detail preservation.

- **Structural Score Calculation**:
  - Composition: Analyzes the image composition, such as framing and layout.
  - Legibility: Measures the clarity of text and symbols in the image.
  - Conformity: Checks if the image meets predefined technical standards.

- **Quality Score**: Combines the technical and structural scores to provide an overall image quality score on a scale of 0 to 1.

## Requirements

- Python 3.6+
- OpenCV (`opencv-python`)
- Numpy
- JSON

You can install the required dependencies by running:

```bash
pip install -r requirements.txt

![image](https://github.com/user-attachments/assets/61e48e06-58eb-459c-8004-fdadb157b631)

