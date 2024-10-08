import cv2
import os

def video_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video = cv2.VideoCapture(video_path)
    
    frame_count = 0
    saved_frames = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame_name = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_name, frame)
        saved_frames += 1
        
        frame_count += 1
    
    video.release()
    print(f"{saved_frames} frames salvos na pasta '{output_folder}'")

video_path = "/home/sando/Área de Trabalho/Projetos Pessoais/projeto Unipac/videos-linha/video1.mp4"
output_folder = "/home/sando/Área de Trabalho/Projetos Pessoais/projeto Unipac/imagens-frame-linha"
video_to_frames(video_path, output_folder)