import cv2
import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import Dict

@dataclass
class TechnicalScore:
    line_clarity: Dict
    contrast: Dict
    resolution: Dict
    total: float

@dataclass
class StructuralScore:
    composition: Dict
    legibility: Dict
    conformity: Dict
    total: float

@dataclass
class ImageQuality:
    technical: TechnicalScore
    structural: StructuralScore
    quality_score: float

class DrawingQualityAnalyzer:

    def analyze_image(self, image_path: str) -> ImageQuality:
        """Full image analysis."""
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read the image in color
        if img is None:
            raise ValueError("Invalid image path or file format.")
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for analysis
        tech_score = self._calculate_technical_score(gray_img)
        struct_score = self._calculate_structural_score(img)
        quality_score = (tech_score.total + struct_score.total) / 100  # Combine scores to calculate final score [0,1]
        return ImageQuality(tech_score, struct_score, quality_score)

    def _calculate_technical_score(self, img) -> TechnicalScore:
        """Calculate the technical score."""
        line_clarity = self._calculate_line_clarity(img)
        contrast = self._calculate_contrast(img)
        resolution = self._calculate_resolution(img)
        total_technical_score = line_clarity['score'] + contrast['score'] + resolution['score']
        return TechnicalScore(line_clarity, contrast, resolution, total_technical_score)

    def _calculate_structural_score(self, img) -> StructuralScore:
        """Calculate the structural score."""
        composition = self._calculate_composition(img)
        legibility = self._calculate_legibility(img)
        conformity = self._calculate_conformity(img)
        total_structural_score = composition['score'] + legibility['score'] + conformity['score']
        return StructuralScore(composition, legibility, conformity, total_structural_score)

    def _calculate_line_clarity(self, img) -> Dict:
        """Calculate line clarity score."""
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        straight_lines_detected = len(lines) if lines is not None else 0
        score = min((straight_lines_detected / 100) * 20, 20)  # Maximum score of 20
        return {"straight_lines_detected": straight_lines_detected, "score": score}

    def _calculate_contrast(self, img) -> Dict:
        """Calculate contrast score."""
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        black_white_ratio = np.sum(hist[:50]) / max(np.sum(hist[200:]), 1)
        grayscale_distribution = np.std(img) / 128  # Normalize (0 to 1 scale)
        peak1 = np.argmax(hist[:128])  # Dominant peak in the lower range
        peak2 = np.argmax(hist[128:]) + 128  # Dominant peak in the higher range
        background_separation = abs(peak2 - peak1) / 255  # Normalize (0 to 1 scale)
        # I choosed weights of different scores  byMyself 
        score = min(
            (black_white_ratio * 4) +
            (grayscale_distribution * 5) +
            (background_separation * 6), 20  # Maximum score of 20
        )
        return {
            "black_white_ratio": round(black_white_ratio, 2),
            "grayscale_distribution": round(grayscale_distribution, 2),
            "background_separation": round(background_separation, 2),
            "score": round(score, 2)
        }

    def _calculate_resolution(self, img) -> Dict:
        """Calculate resolution score."""
        dimensions = img.shape
        # pixel_density = dimensions[0] * dimensions[1]
        # 1. Pixel Density (PPI) calculation:
        # Using the pixel count as a rough proxy for pixel density, here we assume it's a high-density display
        pixel_density = (dimensions[0] * dimensions[1]) / 10000000  # pixels per 10 million
        #I choosed 15 as a weight for the first score
        pixel_density_score = min((pixel_density) * 15, 15)  # Maximum score of 15
        # 2. Detail Preservation: Calculate sharpness (variance of Laplacian)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        detail_preservation = laplacian.var() / 10000  # Normalize to [0, 1]
        # I choosed 20 as a weights for the second score 
        detail_preservation_score = min(detail_preservation * 20, 20)  # Maximum score of 20
        score = min( pixel_density_score + detail_preservation_score / 2 , 15 )   # Maximum score of 15
        return {
            "dimensions": dimensions,
            "pixel_density": pixel_density,
            "detail_preservation": detail_preservation, 
            "score": score
        }

    def _calculate_composition(self, img) -> Dict:
        """Calculate composition score."""
        has_frame = self._detect_frame(img)
        has_dimension=self.has_dimension(img)
        spatial_organization = self._analyze_spatial_organization(img)
        symmetry = self._analyze_symmetry(img)
        # I choosed to give all of them the same weights 
        score = min((spatial_organization + symmetry) *25, 20)  # Maximum score of 20
        return {
            "has_dimension": has_dimension, 
            "spatial_organization": spatial_organization,
            "has_frame": has_frame,
            "symmetry": symmetry,
            "score": score
        }
    def has_dimension(self,img,min_width: int = 100, min_height: int = 100)-> bool : 
       if img is None:
         return False

       height, width = img.shape[:2]
       return width >= min_width and height >= min_height
    def _analyze_spatial_organization(self, img) -> float:
      """
      Analyze spatial organization of the drawing.
    
      This method calculates the spatial organization score based on the 
      distribution of elements, alignment, and spacing within the image.

      Args:
          img (numpy.ndarray): Input image as a NumPy array.

      Returns:
          float: Spatial organization score between 0 and 1.
      """
    

    # Convert the image to grayscale
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to isolate elements
      _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
 
    # Find contours of the elements in the image
      contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate bounding boxes for all contours
      bounding_boxes = [cv2.boundingRect(c) for c in contours]

      if len(bounding_boxes) < 2:
        # If there are too few elements, assign a low score
        return 0.1

    # Extract centroids of bounding boxes
      centroids = [(x + w / 2, y + h / 2) for (x, y, w, h) in bounding_boxes]

    # Calculate the average spacing between elements
      total_distance = 0
      count = 0
      for i, (x1, y1) in enumerate(centroids):
        for j, (x2, y2) in enumerate(centroids):
            if i != j:
                total_distance += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                count += 1

      avg_spacing = total_distance / count if count > 0 else 0

    # Normalize the spacing score (you can tune the normalization factor as needed)
      spacing_score = min(avg_spacing / (gray.shape[0] * gray.shape[1]), 1.0)

    # Calculate alignment score (horizontal and vertical alignment)
      horizontal_alignments = [y for _, y, _, _ in bounding_boxes]
      vertical_alignments = [x for x, _, _, _ in bounding_boxes]

      horizontal_alignment_score = 1 - (np.std(horizontal_alignments) / gray.shape[0])
      vertical_alignment_score = 1 - (np.std(vertical_alignments) / gray.shape[1])

    # Combine spacing and alignment scores
      spatial_score = (spacing_score + horizontal_alignment_score + vertical_alignment_score) / 3

    # Ensure the score is between 0 and 1
      spatial_score = max(0.0, min(spatial_score, 1.0))

      return spatial_score

    def _analyze_symmetry(self, img) -> float:
        """Analyze symmetry."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mirrored = cv2.flip(gray, 1)
        diff = cv2.absdiff(gray, mirrored)
        diff_score = np.sum(diff) / gray.size
        return max(0, 1 - diff_score)

    def _detect_frame(self, img) -> bool:
        """Detect if the image has a frame."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4 and cv2.contourArea(contour) > 1000:
                return True
        return False

    def _calculate_legibility(self, img) -> Dict:
        """Calculate legibility score."""
        if len(img.shape) == 3:  # Image has multiple channels
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        noise_contours = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) < 50]
        noise_level = len(noise_contours) / len(contours) if contours else 0
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        overlap_count = 0
        for i, box1 in enumerate(bounding_boxes):
            for j, box2 in enumerate(bounding_boxes):
                if i != j:
                    x_overlap = max(0, min(box1[0] + box1[2], box2[0] + box2[2]) - max(box1[0], box2[0]))
                    y_overlap = max(0, min(box1[1] + box1[3], box2[1] + box2[3]) - max(box1[1], box2[1]))
                    if x_overlap > 0 and y_overlap > 0:
                        overlap_count += 1
        total_possible_overlaps = len(bounding_boxes) * (len(bounding_boxes) - 1) / 2
        overlap_ratio = overlap_count / total_possible_overlaps if total_possible_overlaps else 0
        if len(bounding_boxes) > 1:
            bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])
            spacings = [
                bounding_boxes[i + 1][0] - (bounding_boxes[i][0] + bounding_boxes[i][2])
                for i in range(len(bounding_boxes) - 1)
            ]
            spacing_quality = np.mean(spacings) / gray.shape[1]
        else:
            spacing_quality = 0.1
        score = (
            (1 - noise_level) * 5 + (1 - overlap_ratio) * 5 + spacing_quality * 5
        )
        score = min(score, 15)  # Cap the score at 15
        return {
            "noise_level": round(noise_level, 2),
            "overlap_ratio": round(overlap_ratio, 2),
            "spacing_quality": round(spacing_quality, 2),
            "score": round(score, 2),
        }

    def _calculate_conformity(self, img) -> Dict:
        """Calculate conformity score."""
        technical_annotations = 0.85
        standard_format = self._standards_format(img)
        has_scale=self._has_scale(img)
        score = min(technical_annotations * 15, 15)
        return {"Standrads_format":standard_format,
                "has_scale": has_scale,
            "technical_annotations": technical_annotations, "score": score}
    def _standards_format(self, img)-> bool : 
        height, width = img.shape[:2]
        return (height, width) == (1080, 1920)  # Example standard: Full HD

    def _has_scale(self , img) -> bool:
            """
            Check if the image contains a scale based on pixel intensity thresholds.
            """
            scale_detected = 0 
            return scale_detected 

def save_results_to_json(result: ImageQuality, output_file: str):
    with open(output_file, 'w') as f:
        json.dump(asdict(result), f, indent=4)
def convert_float32_to_float(obj):
    """Recursively convert numpy float32 to native float."""
    if isinstance(obj, dict):
        return {key: convert_float32_to_float(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_float32_to_float(item) for item in obj]
    elif isinstance(obj, np.float32):  # Check if it's a numpy float32 type
        return float(obj)  # Convert to native Python float
    return obj  # Return the object if it's not float32

def save_results_to_json(result, output_file):
    """Save the analysis results to a JSON file."""
    result_dict = asdict(result)
    result_dict = convert_float32_to_float(result_dict)  # Convert any float32 to float
    with open(output_file, 'a+') as f:
        json.dump(result_dict, f, indent=4)

if __name__ == "__main__":
    analyzer = DrawingQualityAnalyzer()
    output_file = "Data\\Patents\\FR_2789320\\Images\\Meta_data-1.json"
    patent_metadata = {
        "patent_id": "FR2789320A1",
        "title": "Machine tool drawing",
        "date": "2023-11-20"
    }

    # Open the file once and write the metadata
    with open(output_file, 'a+') as f:
        json.dump(patent_metadata, f, indent=4)
        f.write("\n")  # Ensure the next data is on a new line

    # Image paths to analyze
    image_paths = [
        'Data\\Patents\\FR_2789320\\Images\\Fig3.png',
        'Data\\Patents\\FR_2789320\\Images\\Fig1.png',
        'Data\\Patents\\FR_2789320\\Images\\Fig2.png'
    ]
    
    for image_path in image_paths:
        try:
            result = analyzer.analyze_image(image_path)
            save_results_to_json(result, output_file)
            print(f"Analysis for {image_path} saved successfully.")
        except ValueError as e:
            print(f"Error processing {image_path}: {e}")
