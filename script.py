from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8x-seg.pt")

video_path = "caminho do seu video/video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Erro ao abrir o vídeo: {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

line_positions = [
    int(frame_width * 0.26),
    int(frame_width * 0.38),
    int(frame_width * 0.48),
    int(frame_width * 0.55),
    int(frame_width * 0.63),
    int(frame_width * 0.71),
    int(frame_width * 0.82),
]

titles = ["Embalagem", "insp.Mont Final", "Estanqueidade", "Mtg.Parcial", 
          "Usi/S.Neck", "Usinagem", "Rebarbagem", "Aquecer inserto"]

# <=direita  =>esquerda
title_x_positions = [205, 141, 125, 90, 93, 85, 120, int(frame_width * 1)]
title_y_positions = [120, 120, 120, 120, 120, 120, 120, 120]

timer_x_positions = [135, 375, 515, 622, 719, 822, 940, 1100]
timer_y_positions = [150, 150, 150, 150, 150, 150, 150, 150]
# <=up    => down

def draw_dashed_line(frame, start_point, end_point, color, thickness=10, dash_length=10):
    x1, y1 = start_point
    x2, y2 = end_point
    line_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    dashes = int(line_length // dash_length)

    for i in range(dashes):
        start_dash = (
            int(x1 + (x2 - x1) * (i / dashes)),
            int(y1 + (y2 - y1) * (i / dashes)),
        )
        end_dash = (
            int(x1 + (x2 - x1) * ((i + 0.5) / dashes)),
            int(y1 + (y2 - y1) * ((i + 0.5) / dashes)),
        )
        cv2.line(frame, start_dash, end_dash, color, thickness)


line_thickness = 2

def draw_lines_and_titles(frame):
    for i, pos in enumerate(line_positions):
        draw_dashed_line(frame, (pos, 0), (pos, frame_height), (124, 252, 0), line_thickness)  
        title_font_size = 0.5
        title_text = titles[i]
        
        # Obter o tamanho do texto
        (text_width, text_height), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, title_font_size, 2)
        
        # Desenhar retângulo preto atrás do título
        top_left = (pos - title_x_positions[i] - 10, title_y_positions[i] - text_height - 5)
        bottom_right = (pos - title_x_positions[i] + text_width + 10, title_y_positions[i] + 5)
        cv2.rectangle(frame, top_left, bottom_right, (128, 128, 128), -1)
        
        # Desenhar o título sobre o retângulo preto
        cv2.putText(frame, title_text, (pos - title_x_positions[i], title_y_positions[i]), 
                    cv2.FONT_HERSHEY_SIMPLEX, title_font_size, (255, 255, 255), 2)

    # Título para "Aquecer inserto"
    last_line_x = line_positions[-1]
    title_font_size = 0.5
    title = titles[-1]  
    
    title_x = last_line_x + 30
    title_y = title_y_positions[-1]

    # Obter o tamanho do texto
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_font_size, 2)[0]

    # Ajustar posição do título caso ultrapasse os limites da imagem
    if title_x + title_size[0] > frame_width:
        title_x = frame_width - title_size[0] - 10

    # Desenhar o retângulo preto atrás do texto "Aquecer inserto"
    (text_width, text_height), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_font_size, 2)
    top_left = (title_x - 10, title_y - text_height - 5)
    bottom_right = (title_x + text_width + 10, title_y + 5)
    cv2.rectangle(frame, top_left, bottom_right, (128, 128, 128), -1)
    
    # Desenhar o título "Aquecer inserto" sobre o retângulo preto
    cv2.putText(frame, title, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, title_font_size, (255, 255, 255), 2)


def draw_timer(frame, text_x, text_y, elapsed_time):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    timer_font_size = 0.5
    (text_width, text_height), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, timer_font_size, 2)
    cv2.rectangle(frame, (text_x - 10, text_y - text_height - 10), 
                  (text_x + text_width + 10, text_y + 10), (0, 0, 0), -1)
    
    cv2.putText(frame, time_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, timer_font_size, (255, 255, 255), 2)

timers = [0] * len(titles)
start_times = [0] * len(titles)
timer_running = [False] * len(titles)

frame_count = 0
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
        inside_area = [False] * len(titles)
        for box in last_results:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0]
            label = f"operador {conf:.2f}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            for i, pos in enumerate(line_positions):
                if x1 < pos:
                    inside_area[i] = True
                    break


        if x1 > line_positions[-1]:
            inside_area[-1] = True  

        for i in range(len(titles)):
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

            draw_timer(frame, timer_x_positions[i], timer_y_positions[i], timers[i])

    draw_lines_and_titles(frame)
    cv2.imshow("Processed Video", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
