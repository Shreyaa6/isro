import cv2
import numpy as np
import os
import sys
import csv
from datetime import timedelta

class VideoTerrainMatcher:
    def __init__(self, ref_paths, confidence_threshold=15):
        self.confidence_threshold = confidence_threshold
        self.orb = cv2.ORB_create(
            nfeatures=5000,
            scaleFactor=1.2,
            nlevels=12,
            edgeThreshold=15,
            patchSize=31,
            fastThreshold=20
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        # Pre-compute reference features
        self.references = []
        for path in ref_paths:
            print(f"Pre-computing features for {path}...")
            gray, img = self.preprocess_from_path(path)
            kp, des = self.orb.detectAndCompute(gray, None)
            self.references.append({
                'name': path,
                'img': img,
                'gray': gray,
                'kp': kp,
                'des': des,
                'h': gray.shape[0],
                'w': gray.shape[1]
            })

    def preprocess_from_path(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        return enhanced, img

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        return enhanced

    def match_frame(self, frame_gray, ref):
        kp2, des2 = self.orb.detectAndCompute(frame_gray, None)
        if des2 is None or ref['des'] is None:
            return 0, 0, None, None
        
        matches = self.bf.knnMatch(ref['des'], des2, k=2)
        good_matches = [m for m_list in matches if len(m_list) == 2 and m_list[0].distance < 0.75 * m_list[1].distance]
        
        inliers = 0
        homography = None
        if len(good_matches) >= 4:
            src_pts = np.float32([ref['kp'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find transform from frame to reference
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(np.sum(mask))
                homography = M
                good_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
                
        return inliers, homography, (ref['kp'], kp2, good_matches)

    def run_video(self, video_path, output_dir="saved_matches", frame_skip=5):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        log_file = os.path.join(output_dir, "matches_log.csv")
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp (s)', 'Reference', 'Inliers', 'Match_X', 'Match_Y', 'Confidence'])
            
        frame_count = 0
        matches_found = 0
        
        print(f"Starting video processing: {video_path} ({total_frames} frames, {fps:.2f} FPS)")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip == 0:
                timestamp = frame_count / fps
                frame_gray = self.preprocess_frame(frame)
                
                best_match = None
                max_inliers = 0
                
                for ref in self.references:
                    inliers, H, match_data = self.match_frame(frame_gray, ref)
                    if inliers > max_inliers:
                        max_inliers = inliers
                        best_match = (ref, inliers, H, match_data)
                
                if max_inliers >= self.confidence_threshold:
                    ref, inliers, H, (kp1, kp2, matches) = best_match
                    matches_found += 1
                    
                    # Calculate center coordinate in reference image
                    h_f, w_f = frame_gray.shape
                    corners = np.float32([[0, 0], [0, h_f-1], [w_f-1, h_f-1], [w_f-1, 0]]).reshape(-1, 1, 2)
                    dst_corners = cv2.perspectiveTransform(corners, H)
                    center_x = np.mean(dst_corners[:, 0, 0])
                    center_y = np.mean(dst_corners[:, 0, 1])
                    
                    # Log match
                    conf_str = "High" if inliers >= 15 else "Likely"
                    with open(log_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([f"{timestamp:.2f}", ref['name'], inliers, f"{center_x:.2f}", f"{center_y:.2f}", conf_str])
                    
                    # Save visualization
                    ref_vis = ref['img'].copy()
                    ref_vis = cv2.polylines(ref_vis, [np.int32(dst_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    vis_img = cv2.drawMatches(ref_vis, kp1, frame, kp2, matches, None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                              matchColor=(0, 255, 0))
                    
                    cv2.putText(vis_img, f"T: {timestamp:.2f}s | Ref: {ref['name']} | Inliers: {inliers}", 
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    save_path = os.path.join(output_dir, f"match_{timestamp:.2f}s.jpg")
                    cv2.imwrite(save_path, vis_img)
                    print(f"[{timestamp:.2f}s] MATCH FOUND! Ref: {ref['name']}, Inliers: {inliers}, Saved to {save_path}")
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
                
        cap.release()
        print(f"\nProcessing complete. Total matches found: {matches_found}")
        print(f"Log saved to: {log_file}")

def main():
    extensions = ['.jpg', '.png', '.jpeg']
    ref_files = []
    for i in range(1, 4):
        for ext in extensions:
            p = f"ref{i}{ext}"
            if os.path.exists(p):
                ref_files.append(p)
                break
    
    video_file = "test_video.mp4"
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        
    if not os.path.exists(video_file) or len(ref_files) < 3:
        print("Error: Missing files. Need ref1, ref2, ref3 and a video file.")
        return

    matcher = VideoTerrainMatcher(ref_files, confidence_threshold=15)
    matcher.run_video(video_file, frame_skip=10) # Skip frames for speed

if __name__ == "__main__":
    main()
