from paddleocr import PaddleOCR
import torch
import cv2
import numpy as np
import easyocr
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# DEFINING GLOBAL VARIABLE
ocr = PaddleOCR(lang='en', rec_algorithm='CRNN')

# -------------------------------------- Lấy toạ độ box từ file YOLO ---------------------------------------------------------


def detectx(frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

# ------------------------------------ Vẽ box và ghi biển số xe --------------------------------------------------------


def plot_boxes(results, frame, classes):

    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")

    # looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.6:
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1 = int(row[0]*x_shape)
            y1 = int(row[1]*y_shape)
            x2 = int(row[2]*x_shape)
            y2 = int(row[3]*y_shape)

            coords = [x1, y1, x2, y2]

            plate_num = recognize_plate_easyocr(
                img=frame, coords=coords, reader=ocr, region_threshold=OCR_TH)  # đọc text

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (255, 255, 255), 1)  # viền BBox
            cv2.rectangle(frame, (x1, y1-20), (x2+15, y1),
                          (255, 255, 255), -1)  # nền box text
            cv2.putText(frame, f"{plate_num}", (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)  # text

    return frame

# ---------------------------- Nhận diện text trong ảnh --------------------------------------


def recognize_plate_easyocr(img, coords, reader, region_threshold):
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]  # crop box

    ocr_result = reader.ocr(nplate)  # đọc text từ img
    text = filter_text(region=nplate, ocr_result=ocr_result,
                       region_threshold=region_threshold)

    if len(text) == 1:
        text = text[0].upper()
    return text

# ---------------------------------Tạo mảng kí tự----------------


def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]

    plate = []
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate


# ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None, vid_path=None, vid_out=None):

    print(f"[INFO] Loading model... ")
    # loading the custom trained model
    model = torch.hub.load('./yolov5-master', 'custom', source='local', path='last.pt',
                           force_reload=True)  # Lấy model từ file last.pt đã train YOLO

    classes = model.names  # class names in string format

    # --------------- for detection on image --------------------
    if img_path != None:
        print(f"[INFO] Working with image: {img_path}")
        img_out_name = f"./output/result_{img_path.split('/')[-1]}"

        frame = cv2.imread(img_path)  # reading the image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detectx(frame, model=model)  # DETECTION HAPPENING HERE

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = plot_boxes(results, frame, classes=classes)

        # creating a free windown to show the result
        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL)

        while True:

            cv2.imshow("img_only", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print(f"[INFO] Exiting. . . ")

                cv2.imwrite(f"{img_out_name}", frame)
                break

    # --------------- for detection on video --------------------
    elif vid_path != None:
        print(f"[INFO] Working with video: {vid_path}")

        # reading the video
        cap = cv2.VideoCapture(vid_path)

        if vid_out:  # creating the video writer if video output path is given

            # by default VideoCapture returns float instead of int
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')  # (*'XVID')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

        # assert cap.isOpened()
        frame_no = 1

        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            # start_time = time.time()
            ret, frame = cap.read()
            if ret and frame_no % 1 == 0:
                print(f"[INFO] Working with frame {frame_no} ")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detectx(frame, model=model)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                frame = plot_boxes(results, frame, classes=classes)

                cv2.imshow("vid_out", frame)
                if vid_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                frame_no += 1

        print(f"[INFO] Clening up. . . ")
        out.release()

        # closing all windows
        cv2.destroyAllWindows()


# -------------------  calling the main function-------------------------------

# main(vid_path="./test.MOV",vid_out="result.mp4") ### for custom video
# main(vid_path=0,vid_out="result.mp4") #### for webcam
main(img_path="./35330.jpg")  # for image
