import numpy as np
import torch
from PIL import Image
from transformers import (
    OneFormerProcessor, OneFormerForUniversalSegmentation,
    DPTImageProcessor, DPTForDepthEstimation
)
import cv2
import open3d as o3d
from scipy.spatial import KDTree
from ultralytics import YOLO
from collections import deque
import math

class Homography:
    def __init__(self):
        """
        Initialize the Homography class.
        """
        self.processor_seg = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
        self.model_seg = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
        self.processor_depth = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        self.model_depth = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")

    def undistort_point(self, pixel, image_size, focal_lengths, distortion_coeffs):
        """
        Corrects distortion for a given pixel.
        """
        x, y = pixel
        cx, cy = image_size[0] / 2, image_size[1] / 2
        fx, fy = focal_lengths
        k1, k2, p1, p2, k3 = distortion_coeffs

        # Нормализованные координаты
        xn = (x - cx) / fx
        yn = (y - cy) / fy

        # Предварительные вычисления
        xn2 = xn * xn
        yn2 = yn * yn
        xnyn = xn * yn
        r2 = xn2 + yn2
        r4 = r2 * r2
        r6 = r4 * r2

        # Радиальное искажение
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6

        # Тангенциальное искажение
        dx = 2 * p1 * xnyn + p2 * (r2 + 2 * xn2)
        dy = p1 * (r2 + 2 * yn2) + 2 * p2 * xnyn

        # Корректировка координат
        xu = xn * radial + dx
        yu = yn * radial + dy

        return xu * fx + cx, yu * fy + cy

    def create_local_coordinate_system(self, plane_model, camera_direction):
        """
        Creates a local coordinate system for the plane using the plane's normal and camera direction.
        """
        a, b, c, d = plane_model
        normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])

        # First basis vector (projection of camera direction onto the plane)
        camera_direction = camera_direction / np.linalg.norm(camera_direction)
        v1 = camera_direction - np.dot(camera_direction, normal) * normal
        v1 = v1 / np.linalg.norm(v1)

        # Second basis vector (perpendicular to v1 and normal)
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)

        # Origin (point on the plane closest to the camera)
        origin = -d * normal / np.linalg.norm(normal)

        return np.column_stack([v1, v2]), origin

    def pixel_to_ray_direction(self, pixel, image_size, focal_lengths, distortion_coeffs):
        """
        Computes the ray direction from a pixel considering image size, focal lengths, and distortion.
        """
        # Correct distortion
        x, y = pixel
        pixel = (abs(image_size[0] - x), y)
        corrected_pixel = self.undistort_point(pixel, image_size, focal_lengths, distortion_coeffs)
        x, y = corrected_pixel

        # Image center
        cx, cy = image_size[0] / 2, image_size[1] / 2
        fx, fy = focal_lengths

        # Normalized coordinates
        dx = (x - cx) / fx
        dy = (y - cy) / fy
        dz = 1.0

        # Ray direction
        direction = np.array([dx, dy, dz])
        direction = -direction  # Invert ray direction

        return direction / np.linalg.norm(direction)

    def ray_plane_intersection(self, ray_origin, ray_direction, plane_model):
        """
        Finds the intersection point of a ray with a plane.
        """
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        denominator = np.dot(normal, ray_direction)

        if abs(denominator) < 1e-6:
            return None  # Ray is parallel to the plane

        t = -(np.dot(normal, ray_origin) + d) / denominator
        if t < 0:
            return None  # Intersection point is behind the ray origin

        return ray_origin + t * ray_direction if t > 0 else None

    def transform_to_local_coordinates(self, point, basis_vectors, origin):
        """
        Transforms a point to local coordinates.
        """
        return np.dot(basis_vectors.T, point - origin)

    def find_floor_plane_ransac(self, point_cloud, iters=10000):
        """
        Finds the floor plane using RANSAC.
        """
        plane_model, inliers = point_cloud.segment_plane(
            distance_threshold=10,
            ransac_n=3,
            num_iterations=iters
        )
        a, b, c, d = plane_model
        print(f"Floor plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

        # Extract points belonging to the plane
        plane_points = point_cloud.select_by_index(inliers)
        non_plane_points = point_cloud.select_by_index(inliers, invert=True)

        return plane_model, plane_points, non_plane_points

    def depth_map_to_point_cloud(self, depth_map, image):
        """
        Converts a depth map and an image to a point cloud.
        """
        points = []
        colors = []
        width, height = image.size
        img_array = np.array(image)

        for y in range(height):
            for x in range(width):
                z = depth_map[y, x]
                if not np.isnan(z) and z > 0:
                    points.append([x, abs(height - y), z])
                    colors.append(img_array[y, x] / 255.0)

        if not points:
            print("No points for creating point cloud.")
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

        return pcd

    def create_floor_point_cloud(self, depth_map, segmentation_map, image, ade20k_classes, floor_classes):
        """
        Creates a point cloud of the floor.
        """
        points = []
        colors = []
        image_width, image_height = image.size

        # Get IDs of floor classes
        floor_ids = [
            class_id for class_id, class_name in ade20k_classes.items()
            if class_name in floor_classes
        ]

        for y in range(depth_map.shape[0]):
            for x in range(depth_map.shape[1]):
                class_id = segmentation_map[y, x]
                if class_id in floor_ids:  # Check if the point belongs to a floor class
                    z = depth_map[y, x]
                    if not np.isnan(z) and z > 0:  # Exclude points with invalid depth
                        xc = abs(image_width - x)
                        yc = abs(image_height - y)
                        points.append([x, yc, z])
                        color = image.getpixel((x, y))
                        colors.append([c / 255.0 for c in color])  # Normalize color

        points = np.array(points)
        colors = np.array(colors)

        if points.size == 0:
            print("No points belong to the floor plane.")
            return None

        # Create point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def check_corners_for_projections(self, plane_model, image_size, focal_lengths, distortion_coeffs, camera_origin,
                                      corners):
        """
        Checks if all corner points have projections on the plane.
        """
        for corner in corners:
            ray_dir = self.pixel_to_ray_direction(corner, image_size, focal_lengths, distortion_coeffs)
            intersection = self.ray_plane_intersection(camera_origin, ray_dir, plane_model)
            if intersection is None:
                return False
        return True

    def find_floor_plane_ransac_with_max_slope(self, point_cloud, best_focal_x3, best_focal_y3,
                                               best_distortion_coeffs3, max_depth, iters=10000, top_n=10,
                                               suitable_planes=None, image_size=(1920, 1080)):
        """
        Finds the floor plane with the maximum slope using RANSAC,
        using the median value of parameters of the top_n steepest planes.
        """
        if suitable_planes is None:
            suitable_planes = []

        max_slopes = []  # List to store top-N planes

        import time, os
        start_time = time.time()
        max_sec = int(os.getenv("ML_MAX_COMPUTE_SECONDS", "300"))

        for _ in range(iters):
            # Если перерасход времени — выходим
            if time.time() - start_time > max_sec:
                print(f"RANSAC loop timeout after {max_sec}s")
                break
            # Try to find a plane using RANSAC
            plane_model, inliers = point_cloud.segment_plane(
                distance_threshold=10,
                ransac_n=5,
                num_iterations=2
            )
            a, b, c, d = plane_model

            # Compute angle between normal and vertical axis z
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)
            vertical_axis = np.array([0, 0, 1])
            angle = np.arccos(np.dot(normal, vertical_axis))
            slope = np.sin(angle)

            # Add plane to list of top-N
            if len(max_slopes) < top_n:
                max_slopes.append((slope, plane_model))
                max_slopes.sort(key=lambda x: x[0], reverse=True)
            elif slope > max_slopes[-1][0]:
                max_slopes.pop()
                max_slopes.append((slope, plane_model))
                max_slopes.sort(key=lambda x: x[0], reverse=True)

        # If no planes were found, return None
        if not max_slopes:
            print("No planes found.")
            return None, None, None, suitable_planes

        extreme_corners = []

        # Compute four extreme corner points of the floor point cloud
        floor_points = np.asarray(point_cloud.points)

        # Select 100 random points from the floor
        num_random_points = 100
        random_indices = np.random.choice(len(floor_points), min(num_random_points, len(floor_points)), replace=False)
        random_points = [(x, y) for x, y, _ in floor_points[random_indices]]

        # Combine extreme corner points and random points
        all_corners = extreme_corners + random_points

        # Iterate over planes in order of decreasing slope
        focal_lengths = (best_focal_x3 / 1, best_focal_y3 / 1)
        distortion_coeffs = best_distortion_coeffs3
        camera_origin = np.array([image_size[0] / 2, image_size[1] / 2, max_depth])

        for _, plane_model in max_slopes:
            if self.check_corners_for_projections(plane_model, image_size, focal_lengths, distortion_coeffs,
                                                  camera_origin, all_corners):
                suitable_planes.append(plane_model)
                if len(suitable_planes) == top_n:
                    break

        if len(suitable_planes) != top_n:
            print("No suitable planes found.")
            return None, None, None, suitable_planes

        # Compute average plane
        average_plane_model = np.mean(suitable_planes, axis=0)

        # Separate points into inliers and outliers
        distances = np.abs(np.dot(point_cloud.points, average_plane_model[:3]) + average_plane_model[3])
        inliers = np.where(distances < 10)[0]
        plane_points = point_cloud.select_by_index(inliers)
        non_plane_points = point_cloud.select_by_index(inliers, invert=True)

        return average_plane_model, plane_points, non_plane_points, suitable_planes

    def tilt_plane(self, plane_model, tilt_factor=1):
        """
        Tilts the plane by a given factor.
        """
        a, b, c, d = plane_model
        original_normal = np.array([a, b, c])
        original_normal = original_normal / np.linalg.norm(original_normal)

        # Increase x and y components of the normal
        a_tilted = original_normal[0] * tilt_factor
        b_tilted = original_normal[1] * tilt_factor

        # Normalize [a_tilted, b_tilted] so that their length does not exceed 1
        tilted_xy_norm = np.linalg.norm([a_tilted, b_tilted])
        if tilted_xy_norm > 1:
            a_tilted /= tilted_xy_norm
            b_tilted /= tilted_xy_norm

        # Ensure a_tilted**2 + b_tilted**2 <= 1 to avoid invalid sqrt
        sum_of_squares = a_tilted ** 2 + b_tilted ** 2
        if sum_of_squares > 1:
            sum_of_squares = 1

        # Compute new z-component
        c_tilted = np.sqrt(1 - sum_of_squares)

        # New plane equation with the same d
        new_plane_model = [a_tilted, b_tilted, c_tilted, d]

        return new_plane_model

    def preprocess_image(self, image_path):
        """
        Preprocesses the image by resizing and converting to JPG if necessary.
        """
        image = Image.open(image_path).convert("RGB")
        # Resize to 1920x1080
        # image = image.resize((1920, 1080), Image.Resampling.LANCZOS)

        # Convert to JPG if needed
        if image_path.lower().endswith('.png'):
            image_path = image_path.replace('.png', '.jpg')
            image.save(image_path, format='JPEG')

        return np.array(image)

    def analyze_lines(self, image):
        """
        Detects lines in the image and returns their parameters:
        coordinates, lengths, and deviations from straightness.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))  # Уменьшение размера

        # Edge detection (Canny)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Find lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

        if lines is None or len(lines) < 5:
            return []

        # Analyze lines
        lines = lines[:, 0] * 2
        x1, y1, x2, y2 = lines.T
        angles = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        deviations = np.minimum(np.abs(angles), np.abs(90 - np.abs(angles)))
        lengths = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        return [{"coords": coord, "angle": ang, "deviation": dev, "length": length}
                for coord, ang, dev, length in zip(lines, angles, deviations, lengths)]

    def evaluate_straightening(self, original_lines, undistorted_image):
        """
        Evaluates how much lines have straightened after distortion correction.
        """
        # Analyze lines on the original image
        undistorted_lines = self.analyze_lines(undistorted_image)

        if not undistorted_lines:
            return float('inf')

        # Match lines and compute improvement
        orig_coords = np.array([line["coords"] for line in original_lines])
        undist_coords = np.array([line["coords"] for line in undistorted_lines])
        tree = KDTree(undist_coords)
        distances, indices = tree.query(orig_coords, k=1)

        improvements = np.maximum(0, [
            original_lines[i]["deviation"] - undistorted_lines[indices[i]]["deviation"]
            for i in range(len(original_lines))
        ])

        lengths = np.array([line["length"] for line in original_lines])

        return np.sum(improvements * lengths) / np.sum(lengths) if np.sum(lengths) > 0 else float('inf')

    def undistort_image(self, image, focal_length_x, focal_length_y, distortion_coeffs, alpha=1.0):
        """
        Applies camera matrix and distortion coefficients to undistort the image.
        Parameter alpha determines the presence of black borders (alpha=1 -> no cropping).
        """
        h, w = image.shape[:2]

        # Camera matrix
        camera_matrix = np.array([
            [focal_length_x, 0, w / 2],
            [0, focal_length_y, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        # Undistort image with alpha parameter
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), alpha, (w, h))
        undistorted_image = cv2.undistort(image, camera_matrix, distortion_coeffs, None, new_camera_matrix)

        return undistorted_image, roi

    def optimize_parameters(self, image, initial_focal_length_x=1400, initial_focal_length_y=1400,
                            focal_step_x=0, focal_step_y=0, distortion_step=-0.05, max_iterations=10,
                            initial_distortion_coeffs=np.array([-0.08, -0.08, 0., 0., -0.08])):
        """
        Optimizes focal lengths and distortion coefficients to minimize distortions.
        """
        # Analyze lines on the original image
        original_lines = self.analyze_lines(image)

        if not original_lines:
            print("No significant lines found in the original image.")
            return initial_focal_length_x, initial_focal_length_y, initial_distortion_coeffs

        # Evaluate original image
        original_score = self.evaluate_straightening(original_lines, image)
        best_focal_x = initial_focal_length_x
        best_focal_y = initial_focal_length_y
        best_distortion_coeffs = initial_distortion_coeffs
        best_score = original_score

        # Initialize distortion coefficients
        distortion_coeffs = best_distortion_coeffs.copy()

        # Original image size
        original_height, original_width = image.shape[:2]
        original_area = original_height * original_width

        import time, os
        start_time = time.time()
        max_sec = int(os.getenv("ML_MAX_COMPUTE_SECONDS", "300"))

        for iteration in range(max_iterations):
            if time.time() - start_time > max_sec:
                print(f"optimize_parameters timeout after {max_sec}s")
                break
            # If distortion_step == 0, skip outer loop over distortion coefficients
            if distortion_step == 0 and iteration > 0:
                break

            # Optimize focal lengths for current distortion coefficients
            focal_x = initial_focal_length_x
            focal_y = initial_focal_length_y

            for i in range(max_iterations):
                if time.time() - start_time > max_sec:
                    break
                if focal_step_x == 0 and focal_step_y == 0 and i > 0:
                    break

                # Undistort image
                undistorted_image, roi = self.undistort_image(image, focal_x, focal_y, distortion_coeffs, alpha=1.0)

                # Check area of undistorted image
                x, y, w_roi, h_roi = roi
                distorted_area = w_roi * h_roi

                if distorted_area < 0.3 * original_area or distorted_area > 0.9 * original_area:
                    focal_x += focal_step_x
                    focal_y += focal_step_y
                    continue

                # Evaluate straightening quality
                score = self.evaluate_straightening(original_lines, undistorted_image)

                # Update best parameters
                if score > best_score and score > original_score and score < 9999:
                    best_score = score
                    best_focal_x = focal_x
                    best_focal_y = focal_y
                    best_distortion_coeffs = distortion_coeffs.copy()

                # Change focal lengths
                focal_x += focal_step_x
                focal_y += focal_step_y

            # Change distortion coefficients only if distortion_step != 0
            if distortion_step != 0:
                distortion_coeffs[0] += distortion_step  # k1
                distortion_coeffs[1] += distortion_step  # k2
                distortion_coeffs[4] += distortion_step  # k3

        print(f"Optimization complete. Best focal lengths: ({best_focal_x}, {best_focal_y}), "
              f"Best distortion coeffs: {best_distortion_coeffs.flatten()}, Best score: {best_score}")

        return best_focal_x, best_focal_y, best_distortion_coeffs

    def select_random_floor_points(self, segmentation_mask, num_points=100):
        """
        Selects random points belonging to the floor.
        """
        floor_pixels = np.argwhere(segmentation_mask == 1)  # Assume floor class label is 1

        if len(floor_pixels) == 0:
            print("No points on the floor detected in the image.")
            return []

        random_indices = np.random.choice(len(floor_pixels), min(num_points, len(floor_pixels)), replace=False)
        selected_points = [tuple(pixel[::-1]) for pixel in floor_pixels[random_indices]]  # (x, y)

        return selected_points

    def mask_to_polygons(self, mask):
        """
        Converts a binary mask to a list of polygons using contour detection.
        """
        # Преобразуем маску в формат, подходящий для cv2.findContours (uint8)
        mask_uint8 = (mask.astype(np.uint8) * 255)

        # Находим контуры
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Преобразуем контуры в список полигонов
        polygons = []

        for contour in contours:
            # Упрощаем контур с помощью approxPolyDP (опционально)
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Преобразуем точки в список координат и явно преобразуем в int
            polygon = [(int(point[0][0]), int(point[0][1])) for point in approx]
            polygons.append(polygon)

        return polygons

    def compute_homography(self, image_path, processor_seg, model_seg, processor_depth, model_depth):
        """
        Computes the homography matrix for the given image path.
        """
        # Define floor classes
        floor_classes = {"floor", "rug", "sidewalk, pavement", "road, route", "earth, ground", "grass"}
        wall_classes = {"wall", "building", "column, pillar", "fence", "door", "window", "wardrobe, closet, press",
                        "cabinet", "bulletin board", "mirror", "painting, picture", "curtain",
                        "bannister, banister, balustrade, balusters, handrail"}

        import time, os

        # Global timeout for this compute call
        start_time = time.time()
        max_sec = int(os.getenv("ML_MAX_COMPUTE_SECONDS", "300"))

        # Preprocess image
        image = self.preprocess_image(image_path)
        image2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Optimize parameters
        best_focal_x, best_focal_y, best_distortion_coeffs = self.optimize_parameters(image2)
        best_focal_x2, best_focal_y2, best_distortion_coeffs2 = self.optimize_parameters(image2,
                                                                                         initial_focal_length_x=best_focal_x,
                                                                                         initial_focal_length_y=best_focal_y / 2,
                                                                                         focal_step_x=0,
                                                                                         focal_step_y=25,
                                                                                         distortion_step=0,
                                                                                         max_iterations=40,
                                                                                         initial_distortion_coeffs=best_distortion_coeffs)

        best_distortion_coeffs2 = np.array([
            best_distortion_coeffs2[0] / 2, best_distortion_coeffs2[1] / 2, best_distortion_coeffs2[2] / 2,
            best_distortion_coeffs2[3] / 2, best_distortion_coeffs2[4] / 2
        ])

        best_focal_x3, best_focal_y3, best_distortion_coeffs3 = self.optimize_parameters(image2,
                                                                                         initial_focal_length_x=best_focal_x2,
                                                                                         initial_focal_length_y=best_focal_y2,
                                                                                         focal_step_x=0, focal_step_y=0,
                                                                                         distortion_step=-0.002,
                                                                                         max_iterations=40,
                                                                                         initial_distortion_coeffs=best_distortion_coeffs2)

        image_height, image_width = image.shape[:2]
        image_size = (image_width, image_height)
        print(image_size)

        image = Image.fromarray(image)

        # Perform segmentation
        semantic_inputs = processor_seg(images=image, task_inputs=["semantic"], return_tensors="pt")
        with torch.no_grad():
            outputs_seg = model_seg(**semantic_inputs)
            predicted_segmentation = processor_seg.post_process_semantic_segmentation(
                outputs_seg, target_sizes=[image.size[::-1]])[0]

        # Perform depth estimation
        depth_inputs = processor_depth(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs_depth = model_depth(**depth_inputs)
            depth_map = torch.nn.functional.interpolate(
                outputs_depth.predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False
            ).squeeze().cpu().numpy()

        max_depth = np.max(depth_map)
        focal_lengths = (best_focal_x3 / 1, best_focal_y3 / 1)
        distortion_coeffs = best_distortion_coeffs3

        # Get ADE20K classes
        ade20k_classes = model_seg.config.id2label if hasattr(model_seg.config, 'id2label') else {}
        floor_mask = np.isin(predicted_segmentation, [k for k, v in ade20k_classes.items() if v in floor_classes])
        wall_mask = np.isin(predicted_segmentation, [k for k, v in ade20k_classes.items() if v in wall_classes])

        floor_pixel_count = np.sum(floor_mask)
        total_pixels = image_size[0] * image_size[1]
        floor_visibility_ratio = floor_pixel_count / total_pixels

        floor_polygons = self.mask_to_polygons(floor_mask)
        wall_polygons = self.mask_to_polygons(wall_mask)

        if floor_visibility_ratio < 0.1:
            print("Visible floor is less than 10%. Returning identity homography.")
            return np.eye(3), floor_polygons, wall_polygons

        pcd_floor = self.create_floor_point_cloud(depth_map, predicted_segmentation, image, ade20k_classes,
                                                  floor_classes)

        if pcd_floor is not None:
            num_points = len(pcd_floor.points)
            iters = int(0.5 * num_points)
            floor_plane_model = None
            suitable_planes = []
            cnt = 5

            while (floor_plane_model is None) and (cnt <= 10):
                floor_plane_model, floor_points, non_floor_points, suitable_planes = self.find_floor_plane_ransac_with_max_slope(
                    pcd_floor, best_focal_x3, best_focal_y3,
                    best_distortion_coeffs3,
                    max_depth,
                    1000,
                    10,
                    suitable_planes,
                    image_size=image_size
                )
                cnt += 1

            if cnt > 10 and len(suitable_planes) == 0:
                # If no plane found through the first function, use RANSAC
                floor_plane_model, floor_points, non_floor_points = self.find_floor_plane_ransac(pcd_floor, iters)
                print(f"Plane found via RANSAC: {floor_plane_model}")

                # Select 1500 random points from the floor
                num_random_points = 15000
                random_indices = np.random.choice(len(floor_points.points),
                                                  min(num_random_points, len(floor_points.points)), replace=False)
                floor_points_array = np.asarray(floor_points.points)
                random_points = [(x, y) for x, y, _ in floor_points_array[random_indices]]

                focal_lengths = (best_focal_x3 / 1, best_focal_y3 / 1)
                distortion_coeffs = best_distortion_coeffs3
                camera_origin = np.array([image_size[0] / 2, image_size[1] / 2, max_depth])

                # Start tilting the plane
                tilt_factor = 1.1
                step = 0.1
                last_valid_plane = floor_plane_model
                last_valid_floor_points = floor_points
                last_valid_non_floor_points = non_floor_points

                while True:
                    if time.time() - start_time > max_sec:
                        print(f"tilt loop timeout after {max_sec}s")
                        break
                    # Tilt the plane
                    tilted_plane = self.tilt_plane(floor_plane_model, tilt_factor)

                    # Recalculate points belonging to the new plane
                    distances = np.abs(np.dot(pcd_floor.points, tilted_plane[:3]) + tilted_plane[3])
                    inliers = np.where(distances < 10)[0]
                    tilted_floor_points = pcd_floor.select_by_index(inliers)
                    tilted_non_floor_points = pcd_floor.select_by_index(inliers, invert=True)

                    # Check projections for random points
                    if self.check_corners_for_projections(tilted_plane, image_size, focal_lengths, distortion_coeffs,
                                                          camera_origin, random_points):
                        # If check passed, save current plane as the last valid one
                        last_valid_plane = tilted_plane
                        last_valid_floor_points = tilted_floor_points
                        last_valid_non_floor_points = tilted_non_floor_points
                        tilt_factor += step
                    else:
                        # If check failed, use the last valid plane
                        if last_valid_plane is not None:
                            floor_plane_model = last_valid_plane
                            floor_points = last_valid_floor_points
                            non_floor_points = last_valid_non_floor_points
                            print(f"Last valid tilted plane: {floor_plane_model}")
                        else:
                            print("No suitable tilted plane found.")
                        break

                # Output result
                print(f"Final plane: {floor_plane_model}")

            elif len(suitable_planes) > 0:
                floor_plane_model = np.mean(suitable_planes, axis=0)
                print(f"Average plane: {floor_plane_model}")

                # Separate points into inliers and outliers
                distances = np.abs(np.dot(pcd_floor.points, floor_plane_model[:3]) + floor_plane_model[3])
                inliers = np.where(distances < 10)[0]
                floor_points = pcd_floor.select_by_index(inliers)
                non_floor_points = pcd_floor.select_by_index(inliers, invert=True)

        else:
            print("Floor point cloud not created.")
            return np.eye(3), floor_polygons, wall_polygons

        print(image_size)
        camera_plane_model = [0, 0, 1, floor_plane_model[3]]
        camera_origin = np.array([image_size[0] / 2, image_size[1] / 2, max_depth])
        points_on_plane = np.asarray(floor_points.points)
        camera_direction = np.array([0, 0, 1])
        floor_basis, floor_origin = self.create_local_coordinate_system(floor_plane_model, camera_direction)

        # Select 20000 random points on the floor
        points_on_floor = self.select_random_floor_points(floor_mask, num_points=20000)

        if not points_on_floor:
            print("Failed to find points on the floor.")
            return np.eye(3), floor_polygons, wall_polygons

        projected_points = []
        valid_points_on_floor = []
        rays = []

        for point in points_on_floor:
            ray_dir = self.pixel_to_ray_direction(point, image_size, focal_lengths, distortion_coeffs)
            intersection = self.ray_plane_intersection(camera_origin, ray_dir, floor_plane_model)

            if intersection is not None:
                projected_points.append(intersection)
                rays.append((camera_origin, intersection))
                valid_points_on_floor.append(point)

        projected_local_coords = [
            self.transform_to_local_coordinates(point, floor_basis, floor_origin)
            for point in projected_points
        ]

        for coords in projected_local_coords:
            coords[0] += 1000
            coords[1] += 1000

        # Check for empty arrays
        if len(valid_points_on_floor) == 0 or len(projected_local_coords) == 0:
            print("Error: No points for computing homography.")
            return np.eye(3), floor_polygons, wall_polygons

        # Convert arrays to shape (N, 2)
        points_on_floor = np.array(valid_points_on_floor, dtype=np.float32).reshape(-1, 2)
        projected_local_coords = np.array(projected_local_coords, dtype=np.float32).reshape(-1, 2)

        # Check number of points
        if points_on_floor.shape[0] < 4 or projected_local_coords.shape[0] < 4:
            print("Error: Not enough points for computing homography.")
            return np.eye(3), floor_polygons, wall_polygons

        # Compute homography matrix
        H, _ = cv2.findHomography(points_on_floor, projected_local_coords)

        return H, floor_polygons, wall_polygons



class StepAnalyzer:
    """
    Анализатор шагов человека на видео для калибровки масштаба
    """
    def __init__(self, model_path='yolov8n-pose.pt'):
        """
        Инициализация анализатора шагов
        """
        # Загрузка модели YOLO Pose
        self.model = YOLO(model_path)
        # Ключевые точки для стоп: 15 - левая стопа, 16 - правая стопа (в индексации YOLO)
        self.left_foot_idx = 15
        self.right_foot_idx = 16
        # Параметры для фильтрации и анализа
        self.min_confidence = 0.5
        self.distance_history = deque(maxlen=100)  # История расстояний для анализа
        self.peak_distances = []  # Локальные максимумы расстояний
        self.tracking_id = None  # ID отслеживаемого человека

    def detect_pose_keypoints(self, frame):
        """
        Детекция ключевых точек позы человека на кадре
        """
        results = self.model.track(frame, persist=True, verbose=False)
        keypoints_data = []

        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.data.cpu().numpy()

                    for i, box in enumerate(boxes):
                        confidence = box.conf.cpu().numpy()[0]
                        if confidence > self.min_confidence:
                            track_id = int(box.id.cpu().numpy()[0]) if box.id is not None else i
                            bbox = box.xyxy.cpu().numpy()[0]
                            person_keypoints = keypoints[i]
                            keypoints_data.append({
                                'track_id': track_id,
                                'bbox': bbox,
                                'keypoints': person_keypoints,
                                'confidence': confidence
                            })

        return keypoints_data

    def calculate_foot_distance(self, keypoints):
        """
        Вычисление расстояния между стопами человека
        """
        left_foot = keypoints[self.left_foot_idx]
        right_foot = keypoints[self.right_foot_idx]

        # Проверка уверенности ключевых точек
        if left_foot[2] < self.min_confidence or right_foot[2] < self.min_confidence:
            return None

        # Вычисление евклидова расстояния между стопами
        distance = math.sqrt(
            (left_foot[0] - right_foot[0]) ** 2 +
            (left_foot[1] - right_foot[1]) ** 2
        )

        return distance

    def detect_peaks(self, distances, threshold=0.1):
        """
        Обнаружение локальных максимумов в последовательности расстояний
        """
        peaks = []
        if len(distances) < 3:
            return peaks

        # Простой алгоритм обнаружения пиков: точка является пиком,
        # если она больше соседей и превышает определенный порог
        for i in range(1, len(distances) - 1):
            if (distances[i] > distances[i-1] and
                distances[i] > distances[i+1] and
                distances[i] > np.mean(distances) * (1 - threshold)):
                peaks.append(distances[i])

        return peaks

    def analyze_video_steps(self, video_path, max_frames=300):
        """
        Анализ видео для определения шагов человека
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Не удалось открыть видео файл")

        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(min(max_frames, cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        trajectory_points = []  # Точки траектории для визуализации

        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Детекция ключевых точек
            keypoints_data = self.detect_pose_keypoints(frame)

            if keypoints_data:
                # Выбираем самого уверенного детекта или продолжаем трекинг
                if self.tracking_id is None:
                    # Берем самого уверенного детекта
                    person = max(keypoints_data, key=lambda x: x['confidence'])
                    self.tracking_id = person['track_id']
                else:
                    # Пытаемся найти человека с тем же track_id
                    tracked_persons = [p for p in keypoints_data if p['track_id'] == self.tracking_id]
                    if tracked_persons:
                        person = tracked_persons[0]
                    else:
                        # Если потеряли трек, берем самого уверенного
                        person = max(keypoints_data, key=lambda x: x['confidence'])
                        self.tracking_id = person['track_id']

                # Вычисляем расстояние между стопами
                foot_distance = self.calculate_foot_distance(person['keypoints'])
                if foot_distance is not None:
                    self.distance_history.append(foot_distance)

                    # Сохраняем центр массы для траектории
                    bbox = person['bbox']
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    trajectory_points.append((center_x, center_y, foot_distance))

            frame_count += 1

        cap.release()

        # Анализируем расстояния для поиска локальных максимумов
        if len(self.distance_history) > 10:
            self.peak_distances = self.detect_peaks(list(self.distance_history))

        return {
            'peak_distances': self.peak_distances,
            'trajectory_points': trajectory_points,
            'frames_processed': frame_count,
            'total_frames': total_frames
        }

    def calculate_scale_factor(self, real_step_length_cm=75.0):
        """
        Вычисление масштабного коэффициента на основе локальных максимумов
        """
        if not self.peak_distances:
            return None

        # Берем медиану локальных максимумов для устойчивости к выбросам
        median_peak_distance = np.median(self.peak_distances)

        if median_peak_distance <= 0:
            return None

        # Вычисляем масштаб: сколько см в одном пикселе
        scale_factor = real_step_length_cm / median_peak_distance

        return {
            'scale_factor': scale_factor,  # см на пиксель
            'median_peak_distance': median_peak_distance,
            'num_peaks': len(self.peak_distances),
            'real_step_length_cm': real_step_length_cm
        }