import os

# ✅ 변환할 폴더 리스트 (동영상 파일명을 폴더명으로 변환)
target_folders = [os.path.splitext(f)[0] for f in [
    'Explosion046_x264.mp4', 'Arson019_x264.mp4', 'Normal_Videos_935_x264.mp4', 
    'Normal_Videos308_x264.mp4', 'Normal_Videos307_x264.mp4', 'Normal_Videos633_x264.mp4',
    'Normal_Videos946_x264.mp4', 'Normal_Videos471_x264.mp4', 'Normal_Videos947_x264.mp4',
    'Normal_Videos472_x264.mp4', 'Normal_Videos425_x264.mp4', 'Normal_Videos547_x264.mp4',
    'Normal_Videos138_x264.mp4', 'Normal_Videos529_x264.mp4', 'Normal_Videos530_x264.mp4',
    'Normal_Videos450_x264.mp4', 'Normal_Videos666_x264.mp4', 'Normal_Videos449_x264.mp4'
]]

# 📁 최상위 폴더 (Extracted_Frames/)
base_folder = "/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Extracted_Frames/"   # desktop
base_folder = "/media/vcl/DATA/YG/Extracted_Frames/"    # server

def rename_images_in_selected_folders(base_folder, target_folders):
    for class_folder in os.listdir(base_folder):
        class_folder_path = os.path.join(base_folder, class_folder)
        
        if not os.path.isdir(class_folder_path):  # 폴더인지 확인
            continue

        # 🎥 특정 동영상 폴더만 변경
        for video_folder in os.listdir(class_folder_path):
            if video_folder not in target_folders:
                continue  # 제외된 폴더는 변경하지 않음

            video_folder_path = os.path.join(class_folder_path, video_folder)
            if not os.path.isdir(video_folder_path):
                continue

            print(f"📂 Processing folder: {video_folder}")

            # 📌 이미지 파일 이름 변경
            for file_name in os.listdir(video_folder_path):
                if file_name.lower().endswith(".jpg"):
                    old_path = os.path.join(video_folder_path, file_name)
                    new_name = f"{video_folder}_{file_name}"  # 폴더명_파일명.jpg
                    new_path = os.path.join(video_folder_path, new_name)

                    os.rename(old_path, new_path)
                    print(f"✅ Renamed: {file_name} → {new_name}")

# 실행
rename_images_in_selected_folders(base_folder, target_folders)
