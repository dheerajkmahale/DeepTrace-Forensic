import cv2
import os

def extract_faces(video_path, output_folder):
    """Extracts faces from a video and saves them to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count}.jpg"), face_img)
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted faces from {frame_count} frames.")

if __name__ == "__main__":
    # Example usage (will need actual video paths)
    # extract_faces("path/to/video.mp4", "data/extracted_faces")
    pass
