"""
Basic structure borrowed from https://github.com/avivcaspi/TennisProject
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import Line
from itertools import combinations
from court_reference import CourtReference


class CourtDetector:
    """
    Detecting and tracking court in frame
    """
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.colour_threshold = 200
        self.dist_tau = 3
        self.intensity_threshold = 40
        self.court_reference = CourtReference()
        self.v_width = 0
        self.v_height = 0
        self.frame = None
        self.gray = None
        self.court_warp_matrix = []
        self.game_warp_matrix = []
        self.court_score = 0
        self.baseline_top = None
        self.court_baseline_top = None
        self.baseline_bottom = None
        self.court_baseline_bottom = None
        self.net = None
        self.court_net = None
        self.left_court_line = None
        self.court_doubles_line_left = None
        self.right_court_line = None 
        self.court_doubles_line_right = None
        self.left_inner_line = None
        self.court_singles_line_left = None
        self.right_inner_line = None
        self.court_singles_line_right = None
        self.middle_line = None
        self.court_middle_line = None
        self.top_inner_line = None
        self.court_service_line_top = None
        self.bottom_inner_line = None
        self.court_service_line_bottom = None
        self.court_conf_points = {}
        self.success_flag = False
        self.success_accuracy = 80
        self.success_score = 1000
        self.best_conf = None
        self.frame_points = None
        self.dist = 5
        self.best_comb = None

    def detect(self, frame):
        """
        Detecting the court in the frame
        """
        # print('verbose in detect', self.verbose)
        self.frame = frame
        self.v_height, self.v_width = frame.shape[:2]
        print(f'frame shape: {self.v_height}, {self.v_width}') 
        # Get binary image from the frame
        self.gray = self._threshold(frame)

        # Filter pixel using the court known structure
        filtered = self._filter_pixels(self.gray)

        # Detect lines using Hough transform
        horizontal_lines, vertical_lines = self._detect_lines(filtered)
        print('done with hough lines')
        
        # Find transformation from reference court to frame`s court
        self._get_court_conf_points(horizontal_lines, vertical_lines)
        court_warp_matrix, game_warp_matrix, self.court_score = self._find_homography2()
        self.court_warp_matrix.append(court_warp_matrix)
        self.game_warp_matrix.append(game_warp_matrix)
        court_accuracy = self._get_court_accuracy()
        if court_accuracy > self.success_accuracy and self.court_score > self.success_score:
            self.success_flag = True
        print('Court accuracy = %.2f' % court_accuracy)
        # Find important lines location on frame
        self.find_lines_location()
        game_warped = cv2.warpPerspective(self.frame, game_warp_matrix,
                                          (self.court_reference.court.shape[1], self.court_reference.court.shape[0]))
        cv2.imwrite('final2/warped_courtGS4.png', game_warped)

    def _threshold(self, frame):
        """
        Simple thresholding for white pixels
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        return gray

    def _filter_pixels(self, gray):
        """
        Filter pixels by using the court line structure
        """
        for i in range(self.dist_tau, len(gray) - self.dist_tau):
            for j in range(self.dist_tau, len(gray[0]) - self.dist_tau):
                if gray[i, j] == 0:
                    continue
                if (gray[i, j] - gray[i + self.dist_tau, j] > self.intensity_threshold and
                        gray[i, j] - gray[i - self.dist_tau, j] > self.intensity_threshold):
                    continue

                if (gray[i, j] - gray[i, j + self.dist_tau] > self.intensity_threshold and
                        gray[i, j] - gray[i, j - self.dist_tau] > self.intensity_threshold):
                    continue
                gray[i, j] = 0
        return gray

    def _detect_lines(self, gray):
        """
        Finds all line in frame using Hough transform
        """
        minLineLength = 100
        maxLineGap = 20
        # Detect all lines
        lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 80, minLineLength=minLineLength, maxLineGap=maxLineGap)
        lines = np.squeeze(lines)
        if self.verbose:
            fr1 = display_lines_on_frame(self.frame.copy(), [], lines)
            cv2.imwrite('final2/unfiltered4.png', fr1)
        
        lines, discarded_v, discarded_h = self._get_candidate_lines(lines)
        print('got candidate lines')
        
        if self.verbose:
            fr1 = display_lines_on_frame(self.frame.copy(), discarded_h, discarded_v)
            cv2.imwrite('final2/discarded4.png', fr1)
            fr1 = display_lines_on_frame(self.frame.copy(), [], lines)
            cv2.imwrite('final2/filtered4.png', fr1)
            
        # # Classify the lines using their slope
        horizontal, vertical = self._classify_lines2(lines)
        print('got classified lines')
        if self.verbose:
            fr = display_lines_on_frame(self.frame.copy(), horizontal, [])
            cv2.imwrite('final2/unmerged_horizontal4.png', fr)
            fr3 = display_lines_on_frame(self.frame.copy(), [], vertical)
            cv2.imwrite('final2/unmerged_vertical4.png', fr3)

        # Merge lines that belong to the same line on frame
        horizontal, vertical = self._merge_lines(horizontal, vertical)
        print('finished first merge')
        print(f'first merged, horizontal: {len(horizontal)}, vertical: {len(vertical)}')
        horizontal, vertical = self._sort_final_lines(horizontal, vertical)
        
        # for l in horizontal:
        #     print(l)
        if self.verbose:
            fr = display_lines_on_frame(self.frame.copy(), horizontal, [])
            cv2.imwrite('final2/merged_horizontal4.png', fr)
        
        horizontal, vertical = self._merge_lines(horizontal, vertical, horizontal_threshold=15)
        print(f'finished second merge, horizontal: {len(horizontal)}, vertical: {len(vertical)}')
        
        if self.verbose:
            fr = display_lines_on_frame(self.frame.copy(), horizontal, [])
            cv2.imwrite('final2/merged2_horizontal4.png', fr)
            
        horizontal, vertical = self._sort_final_lines(horizontal, vertical)
        # print(f'second merge: {len(horizontal)}')
        # for l in horizontal: 
        #     print(l)
        
        horizontal = self._filter_horizontal_lines(horizontal)
        print(f'filtered out horizontal lines {len(horizontal)}')
        if self.verbose:
            fr = display_lines_on_frame(self.frame.copy(), horizontal, [])
            cv2.imwrite('final2/merged_filtered_horizontal4.png', fr)
            fr3 = display_lines_on_frame(self.frame.copy(), [], vertical)
            cv2.imwrite('final2/merged_filtered_vertical4.png', fr3)
            
        
        if self.verbose:
            # for i in range(len(vertical)): 
            #     fr2 = display_lines_on_frame(self.frame.copy(), [], [vertical[i]])
            #     cv2.imwrite(f'court9_testing/merged_vertical_lines{i}_court9.png', fr2)
            # for i in range(len(horizontal)): 
            #     print(f'horizontal line {i}: {horizontal[i]}')
                # fr2 = display_lines_on_frame(self.frame.copy(), [horizontal[i]], [])
                # cv2.imwrite(f'finalTesting/merged_horizontal_lines{i}_court9.png', fr2)
        
            fr3 = display_lines_on_frame(self.frame.copy(), horizontal, vertical)
            cv2.imwrite('final2/final_merged_filtered4.png', fr3)
        

        return horizontal, vertical
    
    def _draw_filter_threshold(self, frame):
        self.frame = frame
        self.v_height, self.v_width = frame.shape[:2]
        print(f'v_height {self.v_height}, v_width {self.v_width}')
        
        y = (self.v_height // 32) * 12
        
        # define the vertical and angular threshold line
        vertical_start = (0, int(self.v_height * 11 / 16))
        angle_deg = 155
        angle_rad = np.radians(angle_deg)

        end_point_x = self.v_width
        delta_y = int((end_point_x - vertical_start[0]) * np.tan(angle_rad))

        end_point_y = vertical_start[1] + delta_y

        if end_point_y > self.v_height:
            end_point_y = self.v_height
            end_point_x = int(vertical_start[0] + (end_point_y - vertical_start[1]) / np.tan(angle_rad))

        vertical_end = (end_point_x, end_point_y)

        
        vertical_start_rt = (self.v_width, int(self.v_height * 12 / 16))
        angle_rad_rt = np.radians(25)
        
        end_point_x_rt = 0
        delta_y_rt = int((vertical_start_rt[0] - end_point_x_rt) * np.tan(angle_rad_rt))
        end_point_y_rt = vertical_start_rt[1] - delta_y_rt  # Subtract because we're going up as we go left
        
        if end_point_y_rt < 0:
            end_point_y_rt = 0
            end_point_x_rt = int(vertical_start_rt[0] - vertical_start_rt[1] / np.tan(angle_rad_rt))
        elif end_point_y_rt > self.v_height:
            end_point_y_rt = self.v_height
            end_point_x_rt = int(vertical_start_rt[0] - (end_point_y_rt - vertical_start_rt[1]) / np.tan(angle_rad_rt))
        
        vertical_end_rt = (end_point_x_rt, end_point_y_rt)        
        
        print(vertical_start, vertical_end, vertical_start_rt, vertical_end_rt)
        if self.verbose: 
            frame_copy = self.frame.copy()
            cv2.line(frame_copy, vertical_start, vertical_end, (0,0,0), 2)
            cv2.line(frame_copy, (0, y), (self.v_width, y), (255,0,0), 2)
            cv2.line(frame_copy, vertical_start_rt, vertical_end_rt, (0,0,0), 2)
            cv2.imwrite('finalTesting/filter_threshold.png', frame_copy)
    
    def _get_candidate_lines(self, lines=None):
        """
        Removes lines from the adjacent court and the buildings in the background
        """
        new_lines = []
        dicarded_lines_v, dicarded_lines_h = [], []

        # define the horizontal threshold line
        y = (self.v_height // 32) * 14 # 12
        
        # define the vertical and angular threshold line
        vertical_start = (0, int(self.v_height * 12 / 16)) # 11
        angle_deg = 155
        angle_rad = np.radians(angle_deg)
        
        end_point_x = self.v_width
        delta_y = int((end_point_x - vertical_start[0]) * np.tan(angle_rad))
        end_point_y = vertical_start[1] + delta_y
        

        if end_point_y > self.v_height:
            end_point_y = self.v_height
            end_point_x = int(vertical_start[0] + (end_point_y - vertical_start[1]) / np.tan(angle_rad))

        vertical_end = (end_point_x, end_point_y)
        veritcal_vector = (vertical_end[0] - vertical_start[0], vertical_end[1] - vertical_start[1])
        
        vertical_start_rt = (self.v_width, int(self.v_height * 12 / 16)) # 12
        angle_rad_rt = np.radians(25)
        
        end_point_x_rt = 0
        delta_y_rt = int((vertical_start_rt[0] - end_point_x_rt) * np.tan(angle_rad_rt))
        end_point_y_rt = vertical_start_rt[1] - delta_y_rt  # Subtract because we're going up as we go left
        
        if end_point_y_rt < 0:
            end_point_y_rt = 0
            end_point_x_rt = int(vertical_start_rt[0] - vertical_start_rt[1] / np.tan(angle_rad_rt))
        elif end_point_y_rt > self.v_height:
            end_point_y_rt = self.v_height
            end_point_x_rt = int(vertical_start_rt[0] - (end_point_y_rt - vertical_start_rt[1]) / np.tan(angle_rad_rt))
        
        vertical_end_rt = (end_point_x_rt, end_point_y_rt) 
        vertical_vector_rt = (vertical_end_rt[0] - vertical_start_rt[0], vertical_end_rt[1] - vertical_start_rt[1])
        
        if self.verbose: 
            frame_copy = self.frame.copy()
            cv2.line(frame_copy, vertical_start, vertical_end, (0,0,255), 2)
            cv2.line(frame_copy, (0, y), (self.v_width, y), (255,0,0), 2)
            cv2.line(frame_copy, vertical_start_rt, vertical_end_rt, (0,0,255), 2)
            cv2.imwrite('finalTesting/filter_threshold2.png', frame_copy)
            
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line
                
                # check if it is the below the horizontal threshold line 
                if y1 > y and y2 > y:
                    # define vertical vectors with respect to the start and end of the line 
                    start_vector_v = (x1 - vertical_start[0], y1 - vertical_start[1])
                    end_vector_v = (x2 - vertical_start[0], y2 - vertical_start[1])
                    
                    start_vector_v_rt = (x1 - vertical_start_rt[0], y1 - vertical_start_rt[1])
                    end_vector_v_rt = (x2 - vertical_start_rt[0], y2 - vertical_start_rt[1])
                    
                    if (np.cross(veritcal_vector, start_vector_v) > 0 and np.cross(veritcal_vector, end_vector_v) > 0) and (np.cross(vertical_vector_rt, start_vector_v_rt) < 0 and np.cross(vertical_vector_rt, end_vector_v_rt) < 0):
                        new_lines.append(line)
                    
                    else: 
                        dicarded_lines_v.append(line)
                
                else: 
                    dicarded_lines_h.append(line)
                    
                    
        return new_lines, dicarded_lines_v, dicarded_lines_h
            
    def _classify_lines2(self, lines):
        """
        Classifies lines as horizontal or vertical based on the angle with the x-axis
        """
        horizontal, vertical = [], []
        for line in lines:
            x1, y1, x2, y2 = line
            dx = abs(x1 - x2)
            dy = abs(y2 - y1)
            
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            
            if angle_deg < 5: 
                horizontal.append(line)
                # print(f'angle {angle_deg}, classified as horizontal')

            else: 
                vertical.append(line)
                # print(f'angle {angle_deg}, classified as vertical')

        return horizontal, vertical

    def _merge_lines(self, horizontal_lines, vertical_lines, horizontal_threshold=10):
        """
        Merge lines that belongs to the same frame`s lines
        """

        # Merge horizontal lines
        horizontal_lines = sorted(horizontal_lines, key=lambda item: item[0])
        mask = [True] * len(horizontal_lines)
        new_horizontal_lines = []
        for i, line in enumerate(horizontal_lines):
            if mask[i]:
                for j, s_line in enumerate(horizontal_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        dy = abs(y3 - y2)
                        if dy < horizontal_threshold:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[0])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False
                new_horizontal_lines.append(line)

        # Merge vertical lines
        vertical_lines = sorted(vertical_lines, key=lambda item: item[1])
        xl, yl, xr, yr = (0, self.v_height * 6 / 7, self.v_width, self.v_height * 6 / 7)
        mask = [True] * len(vertical_lines)
        new_vertical_lines = []
        for i, line in enumerate(vertical_lines):
            if mask[i]:
                for j, s_line in enumerate(vertical_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        xi, yi = line_intersection(((x1, y1), (x2, y2)), ((xl, yl), (xr, yr)))
                        xj, yj = line_intersection(((x3, y3), (x4, y4)), ((xl, yl), (xr, yr)))

                        dx = abs(xi - xj)
                        if dx < 20:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[1])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False

                new_vertical_lines.append(line)
        final_vertical_lines = self._filter_vertical_lines(new_vertical_lines)
        
        return new_horizontal_lines, final_vertical_lines

    def _filter_vertical_lines(self, vertical_lines):
        """
            filter vertical lines based on their lenght. 
            If the vertical line is shorter than 300 px, it will be filtered out
        """
        threshold = 400
        new_lines = []
        for line in vertical_lines: 
            start, end = np.array([line[0], line[1]]), np.array([line[2], line[3]])
            if np.linalg.norm(start - end) > threshold:
                new_lines.append(line)
        return new_lines
    
    def _sort_final_lines(self, horizontal_lines, vertical_lines):
        """
            sort vertical lines from left to right and horizontal lines from top to bottom
        """
        horizontal_lines = sorted(horizontal_lines, key=lambda item: item[1])
        vertical_lines = sorted(vertical_lines, key=lambda item: item[0])
        return horizontal_lines, vertical_lines
    
    def _filter_horizontal_lines(self, horizontal_lines):
        """
            4 horizontal are expected to be detected, these 4 being the baselines and the service box lines. 
            additional line that is being detected is the net chord and an unmerged line at the near end baseline. 
            these unwanted lines will be filtered out. 
        """
        if len(horizontal_lines) <= 4:
            return horizontal_lines
        lines = sorted(horizontal_lines, key=lambda item: item[1])
        far_side_lines, near_side_lines = lines[:3], lines[3:]
        net_chord = far_side_lines.pop(1) # remove the net chord
        
        sorted_near_side_lines = sorted(near_side_lines, key=self._get_length, reverse=True)
        final_near_side_lines = sorted_near_side_lines[:2] # get the top 2 longest lines, eliminating any outliers that may not have been merged in merge_lines()
        final_near_side_lines = sorted(final_near_side_lines, key=lambda item: item[1]) # rearrange from top to bottom
        
        return far_side_lines + final_near_side_lines  
         
    def _get_length(self, line):
        """
            returns the length of the line
        """
        start, end = np.array([line[0], line[1]]), np.array([line[2], line[3]])
        # print(f'line {o} length: {np.linalg.norm(start - end)}')
        return np.linalg.norm(start - end)
    
    def _get_court_conf_points(self, horizontal_lines, vertical_lines):
        """
            get the 9 configuration points for the court using the found horizontal and vertical
            please not that since the middle service line is not detected, the last two configurations are not used. 
        """
        cv2.imwrite('frame_fucking_up.png', self.frame)
        court_doubles_line_left, court_singles_line_left, court_singles_line_right, court_doubles_line_right = vertical_lines
        court_baseline_top, court_service_line_top, court_service_line_bottom, court_baseline_bottom = horizontal_lines
            
        
        self.court_doubles_line_left = ((court_doubles_line_left[0], court_doubles_line_left[1]), (court_doubles_line_left[2], court_doubles_line_left[3]))
        self.court_singles_line_left = ((court_singles_line_left[0], court_singles_line_left[1]), (court_singles_line_left[2], court_singles_line_left[3]))
        self.court_singles_line_right = ((court_singles_line_right[0], court_singles_line_right[1]), (court_singles_line_right[2], court_singles_line_right[3]))
        self.court_doubles_line_right = ((court_doubles_line_right[0], court_doubles_line_right[1]), (court_doubles_line_right[2], court_doubles_line_right[3]))
        self.court_baseline_top = ((court_baseline_top[0], court_baseline_top[1]), (court_baseline_top[2], court_baseline_top[3]))
        self.court_service_line_top = ((court_service_line_top[0], court_service_line_top[1]), (court_service_line_top[2], court_service_line_top[3]))
        self.court_service_line_bottom = ((court_service_line_bottom[0], court_service_line_bottom[1]), (court_service_line_bottom[2], court_service_line_bottom[3]))
        self.court_baseline_bottom = ((court_baseline_bottom[0], court_baseline_bottom[1]), (court_baseline_bottom[2], court_baseline_bottom[3]))
        
        # court configuration 1
        i1 = line_intersection(self.court_doubles_line_left, self.court_baseline_top)
        i2 = line_intersection(self.court_doubles_line_left, self.court_baseline_bottom)
        i3 = line_intersection(self.court_doubles_line_right, self.court_baseline_top)
        i4 = line_intersection(self.court_doubles_line_right, self.court_baseline_bottom)
        intersections1 = [i1, i2, i3, i4]
        intersections1 = sort_intersection_points(intersections1)
        self.court_conf_points[1] = intersections1
        
        # court configuration 2
        i5 = line_intersection(self.court_singles_line_left, self.court_baseline_top)
        i6 = line_intersection(self.court_singles_line_left, self.court_baseline_bottom)
        i7 = line_intersection(self.court_singles_line_right, self.court_baseline_top)
        i8 = line_intersection(self.court_singles_line_right, self.court_baseline_bottom)
        intersections2 = [i5, i6, i7, i8]
        intersections2 = sort_intersection_points(intersections2)
        self.court_conf_points[2] = intersections2
        
        # court configuration 3 
        i9 = line_intersection(self.court_singles_line_left, self.court_baseline_top)
        i10 = line_intersection(self.court_singles_line_left, self.court_baseline_top)
        i11 = line_intersection(self.court_doubles_line_right, self.court_baseline_top)
        i12 = line_intersection(self.court_doubles_line_right, self.court_baseline_bottom)
        intersections3 = [i9, i10, i11, i12]
        intersections3 = sort_intersection_points(intersections3)
        self.court_conf_points[3] = intersections3
        
        # court configuration 4 
        i13 = line_intersection(self.court_doubles_line_left, self.court_baseline_top)
        i14 = line_intersection(self.court_doubles_line_left, self.court_baseline_bottom)
        i15 = line_intersection(self.court_singles_line_right, self.court_baseline_top)
        i16 = line_intersection(self.court_singles_line_right, self.court_baseline_bottom)
        intersections4 = [i13, i14, i15, i16]
        intersections4 = sort_intersection_points(intersections4)
        self.court_conf_points[4] = intersections4
        
        #court configuration 5
        i17 = line_intersection(self.court_singles_line_left, self.court_service_line_top)
        i18 = line_intersection(self.court_singles_line_left, self.court_service_line_bottom)
        i19 = line_intersection(self.court_singles_line_right, self.court_service_line_top)
        i20 = line_intersection(self.court_singles_line_right, self.court_service_line_bottom)
        intersections5 = [i17, i18, i19, i20]
        intersections5 = sort_intersection_points(intersections5)
        self.court_conf_points[5] = intersections5
        
        # court configuration 6 
        i21 = line_intersection(self.court_singles_line_left, self.court_service_line_top)
        i22 = line_intersection(self.court_singles_line_left, self.court_baseline_bottom)
        i23 = line_intersection(self.court_singles_line_right, self.court_service_line_top)
        i24 = line_intersection(self.court_singles_line_right, self.court_baseline_bottom)
        intersections6 = [i21, i22, i23, i24]
        intersections6 = sort_intersection_points(intersections6)
        self.court_conf_points[6] = intersections6
        
        # court configuration 7
        i25 = line_intersection(self.court_singles_line_left, self.court_service_line_bottom)
        i26 = line_intersection(self.court_singles_line_left, self.court_baseline_top)
        i27 = line_intersection(self.court_singles_line_right, self.court_service_line_bottom)
        i28 = line_intersection(self.court_singles_line_right, self.court_baseline_top)
        intersections7 = [i25, i26, i27, i28]
        intersections7 = sort_intersection_points(intersections7)
        self.court_conf_points[7] = intersections7
        
        # court configuration 8
        i29 = line_intersection(self.court_singles_line_right, self.court_baseline_top)
        i30 = line_intersection(self.court_singles_line_right, self.court_baseline_bottom)
        i31 = line_intersection(self.court_doubles_line_right, self.court_baseline_top)
        i32 = line_intersection(self.court_doubles_line_right, self.court_baseline_bottom)
        intersections8 = [i29, i30, i31, i32]
        intersections8 = sort_intersection_points(intersections8)
        self.court_conf_points[8] = intersections8
        
        # court configuration 9
        i33 = line_intersection(self.court_doubles_line_left, self.court_baseline_top)
        i34 = line_intersection(self.court_doubles_line_left, self.court_baseline_bottom)
        i35 = line_intersection(self.court_singles_line_left, self.court_baseline_top)
        i36 = line_intersection(self.court_singles_line_left, self.court_baseline_bottom)
        intersections9 = [i33, i34, i35, i36]
        intersections9 = sort_intersection_points(intersections9)
        self.court_conf_points[9] = intersections9
    
    def _find_homography2(self): 
        """
            Checks the best transformation from reference court to frame`s court. 
        """
        max_score = -np.inf
        max_mat = None
        max_inv_mat = None 
        
        for m, intersections in self.court_conf_points.items():
            # print(f'testing configuration {m}')
            matrix, _ = cv2.findHomography(np.float32(self.court_reference.court_conf[m]), np.float32(intersections), method=0)
            inv_matrix = cv2.invert(matrix)[1]
            confi_score = self._get_confi_score(matrix)
            
            if self.verbose:
                temp_court = self.add_court_overlay(self.frame.copy(), matrix, (255, 0, 0))
                cv2.imwrite(f'final2/homography/transformed_court_GS{m}.png', temp_court)
                
            if max_score < confi_score:
                max_score = confi_score
                max_mat = matrix
                max_inv_mat = inv_matrix
                self.best_conf = m
            
        if self.verbose:
            frame = self.frame.copy()
            court = self.add_court_overlay(frame, max_mat, (255, 0, 0))
            cv2.imwrite('final2/homography/best_homographyGS.png', court)
            
        print(f'Score = {max_score}')
        print(f'Best configuration = {self.best_conf}, Best combination = {self.best_comb}')

        return max_mat, max_inv_mat, max_score          

    def _get_confi_score(self, matrix):
        """
        Calculate transformation score
        """
        court = cv2.warpPerspective(self.court_reference.court, matrix, self.frame.shape[1::-1])
        court[court > 0] = 1
        gray = self.gray.copy()
        gray[gray > 0] = 1
        correct = court * gray
        wrong = court - correct
        c_p = np.sum(correct)
        w_p = np.sum(wrong)
        return c_p - 0.5 * w_p

    def add_court_overlay(self, frame, homography=None, overlay_color=(255, 255, 255), frame_num=-1):
        """
        Add overlay of the court to the frame
        """
        if homography is None and len(self.court_warp_matrix) > 0 and frame_num < len(self.court_warp_matrix):
            homography = self.court_warp_matrix[frame_num]
        court = cv2.warpPerspective(self.court_reference.court, homography, frame.shape[1::-1])
        frame[court > 0, :] = overlay_color
        return frame

    def find_lines_location(self):
        """
        Finds important lines location on frame
        """
        p = np.array(self.court_reference.get_important_lines(), dtype=np.float32).reshape((-1, 1, 2))
        lines = cv2.perspectiveTransform(p, self.court_warp_matrix[-1]).reshape(-1)
        self.baseline_top = lines[:4]
        self.baseline_bottom = lines[4:8]
        self.net = lines[8:12]
        self.left_court_line = lines[12:16]
        self.right_court_line = lines[16:20]
        self.left_inner_line = lines[20:24]
        self.right_inner_line = lines[24:28]
        self.middle_line = lines[28:32]
        self.top_inner_line = lines[32:36]
        self.bottom_inner_line = lines[36:40]
        if self.verbose:
            frame = display_lines_on_frame(self.frame.copy(), [self.baseline_top, self.baseline_bottom,
                                                        self.net, self.top_inner_line, self.bottom_inner_line],
                                    [self.left_court_line, self.right_court_line,
                                    self.right_inner_line, self.left_inner_line, self.middle_line])
            cv2.imwrite('courtLines_gs5.png', frame)
        return
    
    def get_warped_court(self):
        """
        Returns warped court using the reference court and the transformation of the court
        """
        court = cv2.warpPerspective(self.court_reference.court, self.court_warp_matrix[-1], self.frame.shape[1::-1])
        court[court > 0] = 1
        return court

    def _get_court_accuracy(self, frame=None):
        """
        Calculate court accuracy after detection
        """
        if frame is None:
            frame = self.frame.copy()
        gray = self._threshold(frame)
        gray[gray > 0] = 1
        gray = cv2.dilate(gray, np.ones((9, 9), dtype=np.uint8))
        court = self.get_warped_court()
        total_white_pixels = sum(sum(court))
        sub = court.copy()
        sub[gray == 1] = 0
        accuracy = 100 - (sum(sum(sub)) / total_white_pixels) * 100
        self.accuracy = accuracy
        if self.verbose:
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(gray, cmap='gray')
            plt.title('Grayscale frame'), plt.xticks([]), plt.yticks([])
            plt.subplot(1, 3, 2)
            plt.imshow(court, cmap='gray')
            plt.title('Projected court'), plt.xticks([]), plt.yticks([])
            plt.subplot(1, 3, 3)
            plt.imshow(sub, cmap='gray')
            plt.title('Subtraction result'), plt.xticks([]), plt.yticks([])
            plt.savefig('final2/Accuracy5.png')
        return accuracy

    def overlay_accuracy(self, frame):
        gray = self._threshold(frame)
        gray[gray > 0] = 1
        gray = cv2.dilate(gray, np.ones((9, 9), dtype=np.uint8))
        court = self.get_warped_court()
        total_white_pixels = sum(sum(court))
        sub = court.copy()
        sub[gray == 1] = 0
        accuracy = 100 - (sum(sum(sub)) / total_white_pixels) * 100
        return accuracy

        

def sort_intersection_points(intersections):
    """
    sort intersection points from top left to bottom right
    """
    y_sorted = sorted(intersections, key=lambda x: x[1])
    p12 = y_sorted[:2]
    p34 = y_sorted[2:]
    p12 = sorted(p12, key=lambda x: x[0])
    p34 = sorted(p34, key=lambda x: x[0])
    return p12 + p34


def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    
    # print(f'line1 {line1}, line2 {line2}')
    l1 = Line(line1[0], line1[1])
    l2 = Line(line2[0], line2[1])

    intersection = l1.intersection(l2)
    return intersection[0].coordinates


def display_lines_on_frame(frame, horizontal=(), vertical=()):
    """
    Display lines on frame for horizontal and vertical lines
    """

    '''cv2.line(frame, (int(len(frame[0]) * 4 / 7), 0), (int(len(frame[0]) * 4 / 7), 719), (255, 255, 0), 2)
    cv2.line(frame, (int(len(frame[0]) * 3 / 7), 0), (int(len(frame[0]) * 3 / 7), 719), (255, 255, 0), 2)'''
    for line in horizontal:
        x1, y1, x2, y2 = line
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
        cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)

    for line in vertical:
        x1, y1, x2, y2 = line
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
        cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)
        
    return frame


def display_lines_and_points_on_frame(frame, lines=(), points=(), line_color=(0, 0, 255), point_color=(255, 0, 0), verbose=0):
    """
    Display all lines and points given on frame
    """

    for line in lines:
        x1, y1, x2, y2 = line
        frame = cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)
    for p in points:
        # print(p[0], p[1])
        frame = cv2.circle(frame, (int(round(p[0])), int(round(p[1]))), 2, point_color, 2)

    if verbose:
        cv2.imshow('court', frame)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
    return frame


def get_frame_from_video(video_path, frame_no, verbose=0): 
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while True: 
        ret, frame = cap.read()
        frame_count += 1 
        if frame_count == frame_no:
            if verbose: 
                cv2.imwrite('/Users/tejas/Desktop/TCD_Hws/Dissertation/frame.png', frame)
            cap.release()
            return frame
    return frame


def enhance_image(image, verbose=0): 
    """
    Enhances the quality of the image. 
    """
    image = image.astype(np.float64)

    # Adjust the contrast (1.0-3.0) and brightness (-100 to 100)
    contrast = 1.0
    brightness = -18

    # Apply the contrast and brightness adjustment
    enhanced_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    if verbose:
        cv2.imwrite('Enhanced_img.png', enhanced_image)
    return enhanced_image

    
if __name__ == '__main__':

    import time

    img = get_frame_from_video('/Users/tejas/Desktop/TCD_Hws/Dissertation/gameplay/combined_forehands.mov', 784, 1)
    s = time.time()
    court_detector = CourtDetector(verbose=1)
    court_detector.detect(img) 
    print(f'time = {time.time() - s}')