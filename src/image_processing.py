import cv2
import numpy as np


#RGB formatındaki tohum görselini gri seviyeye (Grayscale) dönüştürür.
def to_grayscale(image_rgb):    
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


#Yüksek frekanslı gürültüleri azaltmak için Gauss filtresi uygular.
def apply_gaussian_blur(image_gray, kernel_size=(5, 5), sigma=1.0):
    return cv2.GaussianBlur(image_gray, kernel_size, sigma)


#Otsu eşikleme yöntemi ile tohumu arka plandan ayırarak ikili (binary) maske üretir.
def segment_seed_otsu(image_gray):
    blurred = apply_gaussian_blur(image_gray)
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask


#Tohum sınırlarını ve yüzey kırıklarını/kusurlarını belirlemek için Canny algoritması uygular.
def extract_canny_edges(image_gray, low_thresh=50, high_thresh=150):
    blurred = apply_gaussian_blur(image_gray)
    return cv2.Canny(blurred, low_thresh, high_thresh)


#Maskelenmiş tohum görselinden kontur ve morfolojik öznitelikleri (alan, çevre) çıkarır.
def extract_seed_contours(binary_mask, min_area=500):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seed_features = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            seed_features.append({
                "contour": cnt,
                "area": area,
                "perimeter": perimeter,
                "bbox": (x, y, w, h)
            })
            
    return seed_features

