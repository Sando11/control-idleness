from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8x-seg.pt")

video_path = "/home/sando/Área de Trabalho/Projetos Pessoais/projeto Unipac/videos-linha/newVideo.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Erro ao abrir o vídeo: {video_path}")
    exit()


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_left = frame_width // 4
line_middle = frame_width // 2
line_right = 3 * frame_width // 4
line_color = (0, 255, 0)  
line_thickness = 2  

def draw_dashed_line(img, start_point, end_point, color, thickness=2, dash_length=5):
    x1, y1 = start_point
    x2, y2 = end_point
    line_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    dashes = int(line_length / dash_length)
    for i in range(dashes):
        start = (int(x1 + (x2 - x1) * i / dashes), int(y1 + (y2 - y1) * i / dashes))
        end = (int(x1 + (x2 - x1) * (i + 0.5) / dashes), int(y1 + (y2 - y1) * (i + 0.5) / dashes))
        cv2.line(img, start, end, color, thickness)

def draw_timer(frame, text_x, elapsed_time):

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    text_y = 50  

    (text_width, text_height), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(frame, (text_x - 10, text_y - text_height - 10), 
                  (text_x + text_width + 10, text_y + 10), (0, 0, 0), -1)

    cv2.putText(frame, time_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

timers = [0, 0, 0]
start_times = [0, 0, 0]
timer_running = [False, False, False]

frame_count = 0  

text_positions = [line_left // 2, line_left + (line_middle - line_left) // 2, line_middle + (line_right - line_middle) // 2]

last_results = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo ou erro ao ler o quadro.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_count % 5 == 0:
        results = model(frame_rgb)
        last_results = results[0].boxes[results[0].boxes.cls == 0]  
    frame_count += 1

    if last_results is not None:
        inside_area = [False, False, False]
        for box in last_results:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0]
            label = f"operador {conf:.2f}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2) 

            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  

            if x1 < line_left:
                inside_area[0] = True
            elif x1 < line_middle:
                inside_area[1] = True
            elif x1 < line_right:
                inside_area[2] = True

        for i in range(3):
            if inside_area[i]:
                if not timer_running[i]:
                    start_times[i] = time.time() - timers[i]  
                    timer_running[i] = True
            else:
                if timer_running[i]:
                    timers[i] = time.time() - start_times[i]  
                    timer_running[i] = False

            if timer_running[i]:
                timers[i] = time.time() - start_times[i]

            draw_timer(frame, text_positions[i], timers[i])

    draw_dashed_line(frame, (line_left, 0), (line_left, frame.shape[0]), line_color, line_thickness)
    draw_dashed_line(frame, (line_middle, 0), (line_middle, frame.shape[0]), line_color, line_thickness)
    draw_dashed_line(frame, (line_middle, 0), (line_middle, frame.shape[0]), line_color, line_thickness)
    draw_dashed_line(frame, (line_right, 0), (line_right, frame.shape[0]), line_color, line_thickness)

    cv2.imshow("Processed Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("Finalizando processamento...")
cap.release()
cv2.destroyAllWindows()