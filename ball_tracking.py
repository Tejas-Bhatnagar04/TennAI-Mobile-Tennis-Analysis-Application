import cv2 
import numpy as np
import time 
import os 
import matplotlib.pyplot as plt
import random
from court_detection import CourtDetector


class BallTracker:
    def __init__(self, video_path, output_path, verbose=0):
        self.video_path = video_path 
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return None
        
        self.verbose = verbose
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.output_path = output_path
        self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height), True)
        self.frame = None
        self.frame_count = 0
        self.last_ball_loc = None
        self.last_contour = None
        self.last_path = None
        self.predicted_loc = None
        self.path_history = []
        self.current_path = []
        self.path_detected = False
        self.bounce_detected = True
        self.bounce_point = None
        self.bounce_history = []
        self.frames_since_last_ball = 0
        self.consecutive_ball_count = 0
        
        self.courtDetector = CourtDetector()
        self.court_detected = False 
        self.court_accuracy = None
        self.shot = None
        self.last_shot = None
        
        
    
    def track_ball(self):
        backSub = cv2.createBackgroundSubtractorMOG2()
        
        
        while True:
            self.frame_count += 1
            ret, self.frame = self.cap.read()
            
            # if self.frame_count <= 960:
            #     continue

            print(f'FRAME: {self.frame_count}')
            
            
            if not ret:
                break 
            
            if not self.court_detected: 
                self.courtDetector.detect(self.frame)
                if self.courtDetector.accuracy >= 90: # 96
                    self.court_detected = True
                    self.court_accuracy = self.courtDetector.accuracy
            
            if self.court_detected:
                if self.courtDetector.overlay_accuracy(self.frame) < 90:  # 96
                    self.courtDetector.detect(self.frame)
                    if self.courtDetector.accuracy >= 90: # 96
                        self.court_detected = True
                        self.court_accuracy = self.courtDetector.accuracy
                        # self.courtDetector.draw_court_lines(self.frame)
                    else: 
                        self.court_detected = False
                
            
             # End of video stream
            
            fgMask = backSub.apply(self.frame)
            _, fgMask = cv2.threshold(fgMask, 127, 255, cv2.THRESH_BINARY)
            
            fgMask = self._opening_and_closing(fgMask, kernel_size=2, iterations=2)
            
            foreground = cv2.bitwise_and(self.frame, self.frame, mask=fgMask)
            _, binaryFG_hsv = self._apply_hsv_mask(foreground, (30, 0, 0), (50, 255, 255))
            contours = self._detect_circular_contours(binaryFG_hsv)
            # annotated_frame, circles = self.detect_and_draw_circles(binaryFG_hsv, contour_frame, contours)
            
            if len(contours) != 0: 
                
                filtered_contour, dist = self._filter_contours(contours)
                
                if filtered_contour[0] is None:
                    # print(f'frame {self.frame_count}: {None}')
                    self.frames_since_last_ball += 1
                    self.consecutive_ball_count = 0
                    if self.frames_since_last_ball > 5: 
                        self.path_detected = False
                        self.path_history.append(self.current_path)
                        self.current_path = []
                        self.bounce_detected = False
                        self.last_ball_loc = None
                        self.last_contour = None
                        self.bounce_point = None
                    
                else: 
                    # print(f'called when contours are not none | frame {self.frame_count}: {len(filtered_contour)}')
                    filtered_center = self._get_contour_centroid(filtered_contour)[0]
                    self.last_contour = filtered_contour
                    self.last_ball_loc = filtered_center
                    
                    if dist is not None and dist > 100:
                        # print('new point too far, path ended')
                        self.path_detected = False
                        self.path_history.append(self.current_path)
                        self.current_path = []
                        self.bounce_detected = False
                        self.last_ball_loc = None
                        self.last_contour = None
                        self.bounce_point = None
                        self.consecutive_ball_count = 0
                        self.frames_since_last_ball += 1
                    
                    else:
                        self.current_path.append(filtered_center)
                        self.frames_since_last_ball = 0
                        self.consecutive_ball_count += 1
                        
                        if self.consecutive_ball_count > 3 and not self.path_detected:
                            if self._check_path_starting_validity():
                                self.path_detected = True
                            else: 
                                self.consecutive_ball_count -= 1
                                self.current_path = []
                                    
                        # print(f'frame {self.frame_count}: 1 | path detected: {self.path_detected}')
                        self._draw_center(filtered_contour)
                    
            else: 
                # print(f'frame {self.frame_count}: {None}')
                self.frames_since_last_ball += 1
                self.consecutive_ball_count = 0
                filtered_contour = None
                if self.frames_since_last_ball > 5: 
                    # print('frames since last ball > 5, path ended')
                    self.path_detected = False
                    self.path_history.append(self.current_path)
                    self.current_path = []
                    self.bounce_detected = False
                    self.last_ball_loc = None
                    self.last_contour = None
                    self.bounce_point = None
            
            if self.path_detected: 
                if not self._check_path():
                    # print('path not valid')
                    self.path_detected = False
                    self.path_history.append(self.current_path)
                    self.current_path = []
                    self.last_ball_loc = None
                    self.bounce_point = None
                    self.last_contour = None
                    self.consecutive_ball_count = 0
                    self.frames_since_last_ball += 1
                
                else: 
                    if self._check_bounce():
                        self._draw_bounce_point()
                        self._draw_bounce_on_top_view()
                    # self._draw_path()
                         
                        
            if self.path_detected: # len current_path > 5
                self._draw_path()
            
            self._draw_overlay_court()
            cv2.putText(self.frame, f'Frame: {self.frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)
            
            if self.court_detected:
                annotated_frame = self.courtDetector.draw_court_lines(self.frame)
                self.out.write(annotated_frame)
            else: 
                self.out.write(self.frame)
        
        self.cap.release()
        self.out.release()

        return 
    
    
  
    def _filter_contours(self, contours):
        dist = None 
        # if no last location if known, find the most circular contour, i.e., the first ball candidate to be detected 
        if self.last_ball_loc is None:
            # filter out the most circular contour 
            # print(f'case 1')
            filtered_contour = self._get_most_circular_contour(contours)
            
        # if last location is known, find the contour closest to the last location
        elif self.last_ball_loc is not None and not self.path_detected:
            # filter out the most circular contour nearest to the last location 
            # if only one is detected then it chosen 
            # print('case 2')
            filtered_contour, dist = self._get_closest_contour_to_loc(contours)
            
        # if path is detected, find the contour that is closest to the path
        elif self.path_detected: 
            # filter out the contour closest to the path irrespective of the circularity
            # print(f'called when path detected')
            self._predict_next_loc()
            filtered_contour, dist = self._get_closest_contour_to_path(contours)
            # print(f'inside filter, path detected. contour: {type(filtered_contour)}')
        
        return [filtered_contour], dist
        
    def _get_contour_centroid(self, contours):
        # print('new centroid call')
        centers = []
        for cnt in contours:
            # print('nnew cnt', type(cnt))
            M = cv2.moments(cnt)
            if M["m00"] != 0:  # To avoid division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = None, None
            centers.append([cX, cY])
            
        return centers
    
    
    def _find_distance(self, pt1, pt2): 
        start, end = np.array([pt1[0], pt1[1]]), np.array([pt2[0], pt2[1]])
        return np.linalg.norm(start - end)
        
        
    def _get_most_circular_contour(self, contours):
        max_circularity = -1
        most_circular_contour = None
        
        for cnt in contours: 
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True) 
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > max_circularity:
                max_circularity = circularity
                most_circular_contour = cnt
        
        return most_circular_contour
    
    def _get_closest_contour_to_loc(self, contours):
        # print(f'calling centroid from closest to loc')
        centers = self._get_contour_centroid(contours)
        
        closest_contour = None
        min_distance = float('inf')
        
        for i in range(len(centers)):
            cnt = centers[i] 
            dist = self._find_distance(cnt, self.last_ball_loc)
            if dist < min_distance:
                min_distance = dist
                closest_contour = contours[i]
        
        return closest_contour, min_distance

    def _get_closest_contour_to_path(self, contours):
        # print(f'calling centroid from closest to path')
        centers = self._get_contour_centroid(contours)
        
        closest_contour = None
        min_distance = float('inf')
        
        for i in range(len(centers)):
            cnt = centers[i] 
            dist = self._find_distance(cnt, self.predicted_loc)
            if dist < min_distance:
                min_distance = dist
                closest_contour = contours[i]
        # print(f'min dist: {min_distance}')
        if min_distance > 100:
            closest_contour = None 
            min_distance = float('inf')
        # print(f'inside closest to path. contour: {type(closest_contour)}, dist {min_distance}')
        return closest_contour, min_distance

    def _predict_next_loc(self):
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.0003

        # Initial state
        initial_state = np.array([self.current_path[0][0], self.current_path[0][1], 0, 0], np.float32)
        kalman.statePre = initial_state
        kalman.statePost = initial_state

        predicted_points = []

        # Iterate through all observed points
        for point in self.current_path:
            measurement = np.array([[np.float32(point[0])], [np.float32(point[1])]])
            
            # Correct the Kalman filter with observed point
            kalman.correct(measurement)
            
            # Predict the next point
            prediction = kalman.predict()
            predicted_points.append((prediction[0], prediction[1]))

        # The next prediction after all observed points are processed
        prediction = kalman.predict()
        self.predicted_loc = (prediction[0], prediction[1])

        return  predicted_points

    
    def _check_path_starting_validity(self):
         
        for i in range(len(self.current_path) - 1): 
            point1, point2 = self.current_path[i], self.current_path[i + 1]
            dist = self._find_distance(point1, point2)
            if dist > 50: 
                return False 
        
        return True
    
    def _check_path(self): 
        path_lenght = 0
        for i in range(len(self.current_path) - 1): 
            point1, point2 = self.current_path[i], self.current_path[i + 1]
            dist = self._find_distance(point1, point2)
            path_lenght += dist
        if path_lenght < 10 * len(self.current_path): 
            return False
        
        return True 
    
    
    def _check_bounce(self): 
        current_direction = self.current_path[1][1] - self.current_path[0][1]
        
        for i in range(2, len(self.current_path)):
            previous_direction = current_direction
            current_direction = self.current_path[i][1] - self.current_path[i-1][1]    
            if previous_direction * current_direction < 0:  # Product is negative if directions are opposite
                self.bounce_point = self.current_path[i-1]
                self.bounce_history.append(self.bounce_point)
                self.bounce_detected = True
                return True
        
        return False
    
    def _ball_in_or_out(self, bounce_point):
        bounce_x ,bounce_y = bounce_point
        
        if bounce_x > self.courtDetector.court_reference.left_inner_line[0][0] and bounce_x < self.courtDetector.court_reference.right_inner_line[0][0]:
            if bounce_y < self.courtDetector.court_reference.baseline_bottom[0][1] and bounce_y > self.courtDetector.court_reference.baseline_top[0][1]:
                return True
        return False
        
    
    def _apply_hsv_mask(self, image, lower_hsv, upper_hsv):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create the mask based on the defined thresholds
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        return masked_image, mask

    
    def _opening_and_closing(self, image, kernel_size=2, iterations=2):
        for i in range(iterations):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size + i, kernel_size + i))
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            image= cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        return image

    
    def _detect_circular_contours(self, image, circularity_threshold=0.60, min_area_threshold=60, max_area_threshold=1500):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on circularity
        circular_contours = []
        rejected_contours_area = []
        rejected_contours_circle = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if max_area_threshold > area > min_area_threshold: 
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                # print(circularity, circularity_threshold)
                if circularity > circularity_threshold:
                    circular_contours.append(cnt)
                else: 
                    rejected_contours_circle.append(cnt)
            else:
                rejected_contours_area.append(cnt)
            
        return circular_contours

    def _draw_center(self, contours): 
        # print(f'calling centroid from draw ce nter')
        centers = self._get_contour_centroid(contours)
        for center in centers: 
            cv2.circle(self.frame, center, 5, (0, 255, 0), -1)  
    
        return 

    def _draw_prediction(self):
        cv2.circle(self.frame, self.predicted_loc, 5, (255, 255, 255), -1)

    def _draw_path(self):
        points_array = np.array([self.current_path], dtype=np.int32)
        cv2.polylines(self.frame, [points_array], isClosed=False, color=(0, 0, 0), thickness=2)
    
    def _draw_bounce_point(self):
        pt1_ver = (self.bounce_point[0], self.bounce_point[1] - 5 // 2)
        pt2_ver = (self.bounce_point[0], self.bounce_point[1] + 5 // 2)
        pt1_hor = (self.bounce_point[0] - 5 // 2, self.bounce_point[1])
        pt2_hor = (self.bounce_point[0] + 5 // 2, self.bounce_point[1])

        # Draw vertical line
        cv2.line(self.frame, pt1_ver, pt2_ver, (0, 0, 255), 4)
        # Draw horizontal line
        cv2.line(self.frame, pt1_hor, pt2_hor, (0, 0, 255), 4)
    
    
    def _draw_bounce_on_top_view(self): 
        court = self.courtDetector.court_reference.court.copy()
        court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
                
        for bounce_point in self.bounce_history:
            transformed_bounce_point = cv2.perspectiveTransform(np.array([[bounce_point]], dtype=np.float32), self.courtDetector.game_warp_matrix[0]).flatten().astype(int)
            
            if self._ball_in_or_out(transformed_bounce_point):
                cv2.circle(court, transformed_bounce_point, 7, (255, 255, 255), -1)
            else: 
                top_left = (transformed_bounce_point[0] - 10, transformed_bounce_point[1] - 10)
                bottom_right = (transformed_bounce_point[0] + 10, transformed_bounce_point[1] + 10)
                top_right = (transformed_bounce_point[0] + 10, transformed_bounce_point[1] - 10)
                bottom_left = (transformed_bounce_point[0] - 10, transformed_bounce_point[1] + 10)

                # Draw the first diagonal line
                cv2.line(court, top_left, bottom_right, (255, 255, 255), 2)
                # Draw the second diagonal line
                cv2.line(court, top_right, bottom_left, (255, 255, 255), 2)
        
        if self.verbose:
            cv2.imwrite('bounce_history.png', court)
        return court 
    
    def _draw_overlay_court(self, scale_w=0.6, scale_h=0.2): 
        if len(self.bounce_history) == 0: 
            court = self.courtDetector.court_reference.court.copy()
            court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)

        else: 
            court = self._draw_bounce_on_top_view()
        
        court_h, court_w = court.shape[:2]
        new_size = (int(court_h * scale_h), int(court_w * scale_w))
        resized_court = cv2.resize(court, new_size)
        
        start_y = 0  # Top corner
        start_x = self.frame.shape[1] - resized_court.shape[1]  # Right corner
        end_y = resized_court.shape[0]
        end_x = self.frame.shape[1]
        
        self.frame[start_y:end_y, start_x:end_x] = resized_court
        
        self.frame[end_y+30:end_y+230, start_x+30:end_x] = 255
        
        
        cv2.putText(self.frame, f'Path: {self.path_detected} ', (start_x + 30, end_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(self.frame, f'Ball count: {self.consecutive_ball_count}', (start_x + 30, end_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(self.frame, f'Bounce: {self.bounce_detected}', (start_x + 30, end_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(self.frame, f'Shot: {self.shot}', (start_x + 30, end_y + 170), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(self.frame, f'Last Shot: {self.last_shot}', (start_x + 30, end_y + 210), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0), 2, cv2.LINE_AA)
        
        
        return 
        

if __name__ == '__main__':
    tracker = BallTracker('/Users/tejas/Desktop/TCD_Hws/Dissertation/gameplay/rally2.mov', '/Users/tejas/Desktop/TCD_Hws/Dissertation/tracked_rally4.mov')
    t1 = time.time()
    tracker.track_ball()
    print(f'Time taken: {time.time() - t1} seconds')
        
            