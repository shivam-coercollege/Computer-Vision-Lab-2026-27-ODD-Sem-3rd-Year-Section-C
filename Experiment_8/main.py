import cv2
import numpy as np
import math
import time

# ---------------------------------------------------------
# EXPERIMENT 8
# Application of Optical Flow for Real-Time Object Tracking
# and Motion Analysis
# ---------------------------------------------------------

VIDEO_FILE = "samplevdo2.mp4"
OUTPUT_FILE = "output_tracking.mp4"


# ---------------------------------------------------------
# Step 1: Load video
# ---------------------------------------------------------

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\n==============================================")
print(" OPTICAL FLOW OBJECT TRACKING")
print("==============================================")
print("Video:", VIDEO_FILE)
print("FPS:", round(fps, 2))
print("Resolution:", width, "x", height)

print("\nPress 'q' to stop the program.")
print("Press 'r' to reset feature points.")


# ---------------------------------------------------------
# Video writer
# ---------------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_FILE,
    fourcc,
    fps,
    (width, height)
)


# ---------------------------------------------------------
# Step 2: Shi-Tomasi Corner Detection
# ---------------------------------------------------------

feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)


# ---------------------------------------------------------
# Step 3: Lucas-Kanade Optical Flow
# ---------------------------------------------------------

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)


# ---------------------------------------------------------
# Read first frame
# ---------------------------------------------------------

ret, old_frame = cap.read()

if not ret:
    print("Error: Could not read first frame.")
    cap.release()
    out.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------
# Step 2: Detect feature points
# ---------------------------------------------------------

p0 = cv2.goodFeaturesToTrack(
    old_gray,
    mask=None,
    **feature_params
)

if p0 is None:
    print("No feature points detected.")
    cap.release()
    out.release()
    exit()


# ---------------------------------------------------------
# Trajectory mask
# ---------------------------------------------------------

trajectory_mask = np.zeros_like(old_frame)


# ---------------------------------------------------------
# Motion parameters
# ---------------------------------------------------------

total_displacement = 0.0
motion_count = 0

total_distance = 0.0
frame_count = 0

previous_center = None

start_time = time.time()


# ---------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # =====================================================
    # Step 3: Lucas-Kanade Optical Flow
    # =====================================================

    if p0 is not None and len(p0) > 0:

        p1, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray,
            frame_gray,
            p0,
            None,
            **lk_params
        )

        if p1 is not None:

            good_new = p1[status == 1]
            good_old = p0[status == 1]

            displacement_values = []


            # -------------------------------------------------
            # Step 4 & 5: Trajectory, displacement and direction
            # -------------------------------------------------

            for new, old in zip(good_new, good_old):

                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()

                x_new = int(x_new)
                y_new = int(y_new)

                x_old = int(x_old)
                y_old = int(y_old)


                # Draw motion trajectory
                cv2.line(
                    trajectory_mask,
                    (x_old, y_old),
                    (x_new, y_new),
                    (0, 255, 0),
                    2
                )


                # Draw current feature point
                cv2.circle(
                    frame,
                    (x_new, y_new),
                    4,
                    (0, 0, 255),
                    -1
                )


                # Displacement
                dx = x_new - x_old
                dy = y_new - y_old

                magnitude = math.sqrt(
                    dx * dx + dy * dy
                )

                displacement_values.append(magnitude)


                # Draw displacement vector
                cv2.arrowedLine(
                    frame,
                    (x_old, y_old),
                    (x_new, y_new),
                    (255, 0, 0),
                    1,
                    tipLength=0.3
                )


            # Average displacement
            if len(displacement_values) > 0:

                avg_magnitude = np.mean(
                    displacement_values
                )

                total_displacement += avg_magnitude
                motion_count += 1


            # Update tracked points
            if len(good_new) > 0:

                p0 = good_new.reshape(-1, 1, 2)

            else:

                p0 = None


    # =====================================================
    # Step 6: Farneback Dense Optical Flow
    # =====================================================

    farneback_flow = cv2.calcOpticalFlowFarneback(
        old_gray,
        frame_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )


    # -----------------------------------------------------
    # Farneback visualization
    # -----------------------------------------------------

    hsv = np.zeros_like(frame)

    hsv[..., 1] = 255

    magnitude, angle = cv2.cartToPolar(
        farneback_flow[..., 0],
        farneback_flow[..., 1]
    )

    hsv[..., 0] = angle * 180 / np.pi / 2

    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    farneback_result = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


    # =====================================================
    # Step 8: Object displacement and trajectory
    # =====================================================

    current_center = None

    if p0 is not None and len(p0) > 0:

        points = p0.reshape(-1, 2)

        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))

        current_center = (center_x, center_y)


        # Draw tracked object center
        cv2.circle(
            frame,
            current_center,
            7,
            (255, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            "Tracked Object",
            (center_x + 10, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2
        )


        # Calculate movement from previous frame
        if previous_center is not None:

            center_dx = (
                current_center[0] -
                previous_center[0]
            )

            center_dy = (
                current_center[1] -
                previous_center[1]
            )

            center_distance = math.sqrt(
                center_dx ** 2 +
                center_dy ** 2
            )

            total_distance += center_distance


        previous_center = current_center


    # =====================================================
    # Direction calculation
    # =====================================================

    direction = "Stationary"

    if current_center is not None and previous_center is not None:

        dx = current_center[0] - previous_center[0]
        dy = current_center[1] - previous_center[1]

        if abs(dx) > abs(dy):

            if dx > 0:
                direction = "Right"
            elif dx < 0:
                direction = "Left"

        else:

            if dy > 0:
                direction = "Down"
            elif dy < 0:
                direction = "Up"


    # =====================================================
    # Average displacement
    # =====================================================

    if motion_count > 0:

        average_displacement = (
            total_displacement /
            motion_count
        )

    else:

        average_displacement = 0


    # =====================================================
    # Average speed
    # =====================================================

    elapsed_video_time = frame_count / fps

    if elapsed_video_time > 0:

        average_speed = (
            total_distance /
            elapsed_video_time
        )

    else:

        average_speed = 0


    # =====================================================
    # Step 5: Display tracking information
    # =====================================================

    cv2.putText(
        frame,
        "Lucas-Kanade Optical Flow",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Direction: " + direction,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Avg Displacement: " +
        str(round(average_displacement, 2)),
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Average Speed: " +
        str(round(average_speed, 2)) +
        " pixels/sec",
        (10, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # =====================================================
    # Combine trajectory with frame
    # =====================================================

    output = cv2.add(
        frame,
        trajectory_mask
    )


    # =====================================================
    # Save processed video
    # =====================================================

    out.write(output)


    # =====================================================
    # Display
    # =====================================================

    cv2.imshow(
        "Lucas-Kanade Tracking",
        output
    )

    cv2.imshow(
        "Farneback Dense Optical Flow",
        farneback_result
    )


    # -----------------------------------------------------
    # Update previous frame
    # -----------------------------------------------------

    old_gray = frame_gray.copy()


    # -----------------------------------------------------
    # Re-detect feature points
    # -----------------------------------------------------

    if p0 is None or len(p0) < 10:

        p0 = cv2.goodFeaturesToTrack(
            old_gray,
            mask=None,
            **feature_params
        )

        trajectory_mask = np.zeros_like(frame)


    # -----------------------------------------------------
    # Keyboard controls
    # -----------------------------------------------------

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):

        break

    elif key == ord("r"):

        p0 = cv2.goodFeaturesToTrack(
            old_gray,
            mask=None,
            **feature_params
        )

        trajectory_mask = np.zeros_like(frame)

        previous_center = None

        print("Feature points reset.")


# ---------------------------------------------------------
# Final Results
# ---------------------------------------------------------

cap.release()
out.release()
cv2.destroyAllWindows()


print("\n==============================================")
print(" FINAL TRACKING RESULTS")
print("==============================================")

print(
    "Output video saved as:",
    OUTPUT_FILE
)

print(
    "Total trajectory distance:",
    round(total_distance, 2),
    "pixels"
)

print(
    "Average displacement:",
    round(average_displacement, 2),
    "pixels/frame"
)

print(
    "Average speed:",
    round(average_speed, 2),
    "pixels/second"
)

print("\n==============================================")
print(" OBSERVATIONS")
print("==============================================")

print("1. Shi-Tomasi detects suitable feature points.")
print("2. Lucas-Kanade tracks selected feature points.")
print("3. Motion trajectories show the path of tracked points.")
print("4. Displacement and motion direction can be estimated.")
print("5. Farneback provides dense optical-flow information.")
print("6. Fast-moving objects can reduce tracking accuracy.")
print("7. Illumination changes can affect feature tracking.")
print("8. Partial occlusion can cause feature points to disappear.")
print("9. Camera movement can produce optical flow across the scene.")
print("10. Optical flow is useful for real-time motion analysis.")


print("\n==============================================")
print(" PRACTICAL APPLICATIONS")
print("==============================================")

print("Traffic monitoring")
print("Autonomous driving")
print("Video surveillance")
print("Human activity recognition")
print("Sports analytics")
print("Robotics and navigation")

print("\nExperiment completed successfully.")