import cv2
import numpy as np
import os
import sys

class RobustTerrainMatcher:
    """
    An expert-level classical computer vision pipeline for terrain matching.
    Designed to handle partial crops, rotation, and scale variations.
    """
    def __init__(self, confidence_threshold=15):
        self.confidence_threshold = confidence_threshold
        
        # Expert ORB parameters:
        # nfeatures: 5000 allows for dense terrain textures
        # scaleFactor: 1.2 provides good scale invariance with enough levels
        # edgeThreshold/patchSize: 31/31 is standard, but keeping it robust for corners
        self.orb = cv2.ORB_create(
            nfeatures=5000,
            scaleFactor=1.2,
            nlevels=12,
            edgeThreshold=15, # Lower threshold to catch features near crop edges
            patchSize=31,
            fastThreshold=20
        )
        
        # BFMatcher with Hamming distance (optimal for ORB descriptors)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        # CLAHE for contrast enhancement (handles lighting variations)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def preprocess(self, image_path):
        """Loads image, converts to grayscale, and applies CLAHE."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE enhancement
        enhanced = self.clahe.apply(gray)
        
        return enhanced, img

    def match(self, ref_gray, test_gray):
        """Detects features and performs geometric verification using RANSAC."""
        # Feature detection and description
        kp1, des1 = self.orb.detectAndCompute(ref_gray, None)
        kp2, des2 = self.orb.detectAndCompute(test_gray, None)
        
        if des1 is None or des2 is None:
            return 0, 0, 0.0, (kp1 or [], kp2 or [], []), None
        
        # KNN matching for Lowe's Ratio Test
        matches = self.bf.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's Ratio Test (filters out ambiguous matches)
        good_matches = []
        for m_list in matches:
            if len(m_list) == 2:
                m, n = m_list
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
        # Geometric Verification via RANSAC Homography
        inliers = 0
        homography = None
        final_matches = []
        
        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find the transform from test (crop) to reference
            # Note: We swap src/dst to find where the test image lies in the reference
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            if mask is not None:
                inliers = int(np.sum(mask))
                homography = M
                # Keep only inlier matches for visualization
                final_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
            
        # Confidence score is purely based on geometric inliers
        # In terrain matching, 15+ inliers is usually a very strong match
        confidence = float(inliers)
        
        return len(good_matches), inliers, confidence, (kp1, kp2, final_matches), homography

    def run(self, ref_paths, test_path):
        """Executes the pipeline across multiple reference images."""
        print(f"Loading test image: {test_path}")
        test_gray, test_img = self.preprocess(test_path)
        test_h, test_w = test_gray.shape
        
        results = []
        for ref_path in ref_paths:
            print(f"Checking reference: {ref_path}...")
            ref_gray, ref_img = self.preprocess(ref_path)
            
            good, inliers, conf, match_data, H = self.match(ref_gray, test_gray)
            
            results.append({
                'name': ref_path,
                'good_matches': good,
                'inliers': inliers,
                'confidence': conf,
                'match_data': match_data,
                'homography': H,
                'ref_img': ref_img,
                'test_img': test_img
            })
            
            print(f"  - Total Matches: {good}")
            print(f"  - RANSAC Inliers: {inliers}")
            print(f"  - Final Score: {conf:.2f}")
            print("-" * 30)

        # Identify best match based on highest inlier count
        best = max(results, key=lambda x: x['confidence'])
        
        print("\n" + "="*40)
        print(f"BEST MATCH: {best['name']}")
        print(f"Confidence score: {best['confidence']:.2f}")
        
        # Multi-level decision logic based on suggested thresholds
        if best['confidence'] >= 15:
            decision = "MATCH (High Confidence)"
        elif best['confidence'] >= 8:
            decision = "MATCH (Likely / Ambiguous)"
        else:
            decision = "NO MATCH (Insufficient Features)"
            
        print(f"Result: {decision}")
        print("="*40)

        # Final Visualization
        kp1, kp2, matches = best['match_data']
        ref_vis = best['ref_img'].copy()
        test_vis = best['test_img'].copy()
        
        # If a match was found, draw a bounding box around the detected area in reference
        if best['homography'] is not None and best['inliers'] >= 4:
            # Corners of the test image
            pts = np.float32([[0, 0], [0, test_h-1], [test_w-1, test_h-1], [test_w-1, 0]]).reshape(-1, 1, 2)
            # Project corners into reference image space
            dst = cv2.perspectiveTransform(pts, best['homography'])
            # Draw bounding box (Green)
            ref_vis = cv2.polylines(ref_vis, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

        # Create side-by-side match visualization
        vis_img = cv2.drawMatches(
            ref_vis, kp1, test_vis, kp2, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0) # Inliers in Green
        )
        
        # Add labels
        cv2.putText(vis_img, f"Best Match: {best['name']} ({decision})", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        vis_path = "match_output.jpg"
        cv2.imwrite(vis_path, vis_img)
        print(f"\nResult visualization saved to: {vis_path}")

def main():
    # Support common image formats
    extensions = ['.jpg', '.png', '.jpeg']
    ref_files = []
    
    # Locate reference images ref1, ref2, ref3
    for i in range(1, 4):
        for ext in extensions:
            p = f"ref{i}{ext}"
            if os.path.exists(p):
                ref_files.append(p)
                break
    
    # Handle test image input (command line or default)
    test_file = None
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        for ext in extensions:
            p = f"test{ext}"
            if os.path.exists(p):
                test_file = p
                break
    
    if not test_file or len(ref_files) < 3:
        print("Error: Required images not found. Need ref1, ref2, ref3 and test.")
        print("Usage: python3 match.py [path_to_test_image]")
        return

    # Initialize matcher with a recommended threshold of 15-20 inliers for terrain
    matcher = RobustTerrainMatcher(confidence_threshold=18)
    matcher.run(ref_files, test_file)

if __name__ == "__main__":
    main()
