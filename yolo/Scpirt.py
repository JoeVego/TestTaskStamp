from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")
    video_path = "cvtest.avi"

    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO Inference", annotated_frame)

            for obj in results[0].summary():
                if obj.get("name") == "car":
                    print("\nIt's a car !"
                          "\nAdd some actions here....")

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
