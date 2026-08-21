# Analysing earthquake damage using texture analysis and CNN
This repository is built for earthquake damage detection from high-resolution aerial or satellite imagery. It combines image preprocessing, texture feature extraction, and deep learning classification to distinguish damaged from undamaged buildings.

## Project goal
The workflow prepares building crops from annotated imagery, extracts texture information, trains a ResNet-based model, and then runs inference over larger raster data using building footprints.

## Repository structure

- [data_pre-processing/extract.py](data_pre-processing/extract.py)  
  Extracts and crops building patches from COCO-style annotation data into training folders.

- [texture_analysis](texture_analysis)  
  Contains texture/feature-processing scripts such as GLCM, Sobel filtering and dataset loading utilities.

- [train_and_classify/train_model.py](train_and_classify/train_model.py)  
  Trains a modified ResNet50 model for binary classification using multi-band imagery and texture inputs.

- [train_and_classify/classify.py](train_and_classify/classify.py)  
  Runs inference on raster data, evaluates building footprints, and exports classified results.

- [models](models)  
  Stores trained model weights.

- [test_images](test_images)  
  Sample images for testing or validation.

## Typical workflow

1. Prepare the annotation dataset and crop building images.
2. Generate texture features and organize training data.
3. Train the model with multi-band inputs.
4. Run classification across a raster using a building footprint dataset.
5. Save results as geospatial outputs.



## Notes

- Several file paths are currently hardcoded to local Windows directories.
- Update the data and output paths before running the scripts in a new environment.
- The model expects multi-band image inputs and uses a binary damaged/undamaged classification setup.
- If no building footprint data is available, an example can be found here: https://zenodo.org/records/18437501.

## Example usage

- Run [extract.py](data_pre-processing/extract.py) to create training crops.
- Prepare training data by calculating feature stacks with texture analysis features using [calculate_features.py].(texture_analysis/texture_analysis.py)
- Update paths in [train_model.py](train_and_classify/train_model.py) and train the model.
- Run [classify.py](train_and_classify/classify.py) to classify imagery.
