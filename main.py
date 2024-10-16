import cv2
import imutils
from imutils import paths
from ultralytics import YOLO
from shapely.geometry import Polygon
import numpy as np


piece_model = YOLO("")
board_model = YOLO("")

cap = cv2.VideoCapture(0)

initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
chess_pieces = {0: "K", 1: "Q", 2: "R", 3: "B", 4: "N", 5: "P", 6: "k", 7: "q", 8: "r", 9: "b", 10: "n", 11: "p"}

def perspective_transform(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] -  bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] -  tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] -  br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] -  bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([[0, 0], [max_width - 1, 0], [max_width -1, max_height - 1], [0, max_height -1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    return warped

def calculate_iou(box1, box2):
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    iou = poly1.intersection(poly2).area / poly1.union(poly2).area
    return iou

def detect_pieces(frame):
    results = piece_model(frame)
    pieces = []
    for detection in results.pred[0]:
        x_min, y_min, x_max, y_max, confidence, class_id = detection
        piece_name = chess_pieces[int(class_id)]
        pieces.append({"name": piece_name, "box": [x_min, y_min, x_max, y_max]})
    return pieces


def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # top left 
    rect[2] = pts[np.argmax(s)] # bottom right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top right
    rect[3] = pts[np.argmax(diff)] #bottom left

    return rect