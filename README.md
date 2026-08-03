# Thermal Defect Detection Using YOLO

## Project Overview

This project focuses on detecting  defects in 3D printing processes using YOLO object detection models. Thermal image sequences from multiple print histories are used to identify manufacturing defects such as Blob and Underextrusion.

## Objectives

- Build an end-to-end object detection pipeline.
- Detect thermal defects from infrared image sequences.
- Track experiments and model performance.
- Create a scalable MLOps workflow using Git, DVC, and MLflow.

## Dataset

The dataset consists of thermal image sequences collected from three different print runs.

### Classes

- Blob
- Underextrusion

## Project Structure

```text
src/
├── data_validation/
├── data_preparation/
├── model_training/
├── model_prediction/
├── model_evaluation/
├── model_validation/
└── model_analysis/

configs/
├── config.yaml
└── data.yaml

pipelines/