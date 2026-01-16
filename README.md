# topsis-python

## Package Description
Python package implementing TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) for ranking machine learning models and multi-criteria decision making.

**Author:** Raunaq Mittal  
**Roll No:** 102303752

## Algorithm Overview

### STEP 1: Build Evaluation Matrix
Construct a decision matrix with m alternatives (rows) and n criteria (columns). Preprocess data as needed.

### STEP 2: Normalize the Matrix
Apply vector normalization to make criteria comparable.

### STEP 3: Apply Weights
Multiply normalized values by their respective criterion weights.

### STEP 4: Identify Ideal Solutions
Determine the ideal best and ideal worst values for each criterion based on impact direction.

### STEP 5: Compute Separation Measures
Calculate Euclidean distances from each alternative to the ideal best and ideal worst solutions.

### STEP 6: Calculate Performance Score
Compute the relative closeness to the ideal solution for each alternative.

### STEP 7: Rank Alternatives
Sort alternatives by their performance scores in descending order.

## Installation

```bash
pip install topsis-python-raunaqmittal