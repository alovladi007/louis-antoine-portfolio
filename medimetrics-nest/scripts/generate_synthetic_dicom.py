#!/usr/bin/env python3
"""
Generate synthetic DICOM files for testing
"""

import os
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from datetime import datetime
import argparse
from pathlib import Path

def create_synthetic_dicom(
    output_path: str,
    patient_id: str = "SYNTH001",
    study_id: str = "STUDY001",
    series_id: str = "SERIES001",
    modality: str = "CR",
    body_part: str = "CHEST",
    image_size: tuple = (512, 512),
    pattern: str = "gradient"
):
    """Create a synthetic DICOM file with specified parameters"""
    
    # Create synthetic image data
    if pattern == "gradient":
        # Gradient pattern
        x = np.linspace(0, 1, image_size[0])
        y = np.linspace(0, 1, image_size[1])
        X, Y = np.meshgrid(x, y)
        pixel_array = ((X + Y) * 2048).astype(np.uint16)
    elif pattern == "circles":
        # Circular pattern
        center = (image_size[0] // 2, image_size[1] // 2)
        Y, X = np.ogrid[:image_size[0], :image_size[1]]
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        pixel_array = (np.sin(dist / 20) * 2048 + 2048).astype(np.uint16)
    elif pattern == "noise":
        # Random noise pattern
        pixel_array = (np.random.rand(*image_size) * 4096).astype(np.uint16)
    else:
        # Checkerboard pattern
        pixel_array = np.zeros(image_size, dtype=np.uint16)
        block_size = 32
        for i in range(0, image_size[0], block_size * 2):
            for j in range(0, image_size[1], block_size * 2):
                pixel_array[i:i+block_size, j:j+block_size] = 3000
                pixel_array[i+block_size:i+2*block_size, j+block_size:j+2*block_size] = 3000
    
    # Add some synthetic "anatomy"
    if body_part == "CHEST":
        # Add lung-like regions
        left_lung = (image_size[0] // 3, image_size[1] // 3)
        right_lung = (2 * image_size[0] // 3, image_size[1] // 3)
        Y, X = np.ogrid[:image_size[0], :image_size[1]]
        
        # Left lung
        dist_left = np.sqrt((X - left_lung[0])**2 + (Y - left_lung[1])**2)
        mask_left = dist_left < 80
        pixel_array[mask_left] = pixel_array[mask_left] * 0.3
        
        # Right lung
        dist_right = np.sqrt((X - right_lung[0])**2 + (Y - right_lung[1])**2)
        mask_right = dist_right < 80
        pixel_array[mask_right] = pixel_array[mask_right] * 0.3
        
        # Heart region
        heart = (image_size[0] // 2, image_size[1] // 2)
        dist_heart = np.sqrt((X - heart[0])**2 + (Y - heart[1])**2)
        mask_heart = dist_heart < 40
        pixel_array[mask_heart] = pixel_array[mask_heart] * 1.5
    
    # Create DICOM dataset
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.DigitalXRayImageStorageForPresentation
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
    
    # Create the FileDataset
    ds = FileDataset(
        output_path,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128
    )
    
    # Add required DICOM tags
    ds.PatientName = f"Synthetic^Patient^{patient_id}"
    ds.PatientID = patient_id
    ds.PatientBirthDate = "19800101"
    ds.PatientSex = "O"
    
    # Study information
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.StudyID = study_id
    ds.StudyDate = datetime.now().strftime("%Y%m%d")
    ds.StudyTime = datetime.now().strftime("%H%M%S")
    ds.StudyDescription = f"Synthetic {body_part} Study"
    
    # Series information
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesNumber = 1
    ds.SeriesDescription = f"Synthetic {modality} Series"
    
    # Image information
    ds.SOPClassUID = pydicom.uid.DigitalXRayImageStorageForPresentation
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.Modality = modality
    ds.BodyPartExamined = body_part
    ds.InstanceNumber = 1
    
    # Image pixel data
    ds.Rows = image_size[0]
    ds.Columns = image_size[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = pixel_array.tobytes()
    
    # Window/Level for display
    ds.WindowCenter = 2048
    ds.WindowWidth = 4096
    
    # Additional metadata
    ds.InstitutionName = "MediMetrics Synthetic Data"
    ds.Manufacturer = "MediMetrics"
    ds.ManufacturerModelName = "Synthetic Generator v1.0"
    
    # Save the DICOM file
    ds.save_as(output_path, write_like_original=False)
    print(f"Created synthetic DICOM: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic DICOM files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/fixtures/synthetic",
        help="Output directory for DICOM files"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of DICOM files to generate"
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="CR",
        choices=["CR", "DX", "CT", "MR"],
        help="DICOM modality"
    )
    parser.add_argument(
        "--body-part",
        type=str,
        default="CHEST",
        choices=["CHEST", "ABDOMEN", "HEAD", "EXTREMITY"],
        help="Body part examined"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate DICOM files
    patterns = ["gradient", "circles", "noise", "checkerboard"]
    
    for i in range(args.count):
        pattern = patterns[i % len(patterns)]
        filename = f"synthetic_{args.modality.lower()}_{i+1:03d}.dcm"
        output_path = output_dir / filename
        
        create_synthetic_dicom(
            str(output_path),
            patient_id=f"SYNTH{i+1:03d}",
            study_id=f"STUDY{i+1:03d}",
            series_id=f"SERIES{i+1:03d}",
            modality=args.modality,
            body_part=args.body_part,
            pattern=pattern
        )
    
    print(f"\nGenerated {args.count} synthetic DICOM files in {output_dir}")
    
    # Also create sample PNG images for testing
    from PIL import Image
    
    # Create frontal chest X-ray simulation
    frontal = np.zeros((512, 512), dtype=np.uint8)
    # Add lung fields
    frontal[100:400, 50:200] = 30  # Left lung
    frontal[100:400, 312:462] = 30  # Right lung
    # Add heart shadow
    frontal[200:350, 180:332] = 100
    # Add some ribs
    for i in range(5):
        y = 120 + i * 50
        frontal[y:y+10, 50:462] = 150
    
    img_frontal = Image.fromarray(frontal, mode='L')
    img_frontal.save(output_dir / "sample_frontal.png")
    print(f"Created sample_frontal.png")
    
    # Create lateral chest X-ray simulation
    lateral = np.zeros((512, 512), dtype=np.uint8)
    # Add lung field
    lateral[100:400, 100:400] = 40
    # Add spine
    lateral[50:450, 380:420] = 180
    # Add heart
    lateral[200:320, 150:280] = 120
    
    img_lateral = Image.fromarray(lateral, mode='L')
    img_lateral.save(output_dir / "sample_lateral.png")
    print(f"Created sample_lateral.png")

if __name__ == "__main__":
    main()