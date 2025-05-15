import numpy as np
import cv2
import trimesh
import json
import os
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

def load_image_and_detect_edges(image_path, canny_low=30, canny_high=50):
    """Carga una imagen y detecta los bordes usando Canny."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen en {image_path}")
    
    edges = cv2.Canny(img, canny_low, canny_high)
    print(f"Se detectaron bordes en la imagen: {np.count_nonzero(edges)} puntos.")
    return edges

def generate_inverse_rays(edges, focal_length_px):
    """Genera rayos inversos desde los puntos detectados de borde."""
    h, w = edges.shape
    rays = []

    for y in range(h):
        for x in range(w):
            if edges[y, x] != 0:  # Si es borde
                # Centrar la imagen en el origen (0,0)
                x_centered = x - w / 2
                y_centered = y - h / 2
                ray = np.array([x_centered, y_centered, focal_length_px])
                ray /= np.linalg.norm(ray)  # Normalizar el rayo
                rays.append(ray)

    print(f"Generados {len(rays)} rayos inversos.")
    return np.array(rays)

def find_closest_points(rays, cad_model):
    """Encuentra el punto más cercano en el modelo CAD para cada rayo inverso."""
    closest_points = []
    for ray in rays:
        closest_point = None
        min_dist = float('inf')
        
        for vertex in cad_model.vertices:  # Recorrer vértices del CAD
            distance = np.linalg.norm(ray - vertex)  # Distancia del rayo a cada vértice
            if distance < min_dist:
                min_dist = distance
                closest_point = vertex
        
        closest_points.append(closest_point)
    
    return np.array(closest_points)

def compute_correspondences(rays, cad_model):
    """
    Para cada rayo:
    - Encuentra el vértice más cercano del CAD (model_point)
    - Calcula el punto en el rayo: ray_point = ray * (model_point · ray)
    Devuelve dos arrays Nx3: model_points, ray_points
    """
    model_points = []
    ray_points   = []
    verts = cad_model.vertices  # Nx3 array

    for ray in rays:
        # Distancias a todos los vértices
        dists = np.linalg.norm(verts - ray, axis=1)
        idx_min = np.argmin(dists)
        model_pt = verts[idx_min]
        # Proyectamos model_pt sobre la dirección del rayo
        dist_along_ray = np.dot(model_pt, ray)
        ray_pt = ray * dist_along_ray

        model_points.append(model_pt)
        ray_points.append(ray_pt)

    model_points = np.array(model_points)
    ray_points   = np.array(ray_points)
    return model_points, ray_points

def compute_rigid_transformation(source_points, target_points):
    """SVD para encontrar R,t que minimiza ||R·source + t - target||."""
    assert source_points.shape == target_points.shape
    # Centros de masa
    cs = source_points.mean(axis=0)
    ct = target_points.mean(axis=0)
    # Centrar
    src = source_points - cs
    tgt = target_points - ct
    # Matriz de correlación
    H = src.T @ tgt
    U, _, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    # Corregir reflejo
    if np.linalg.det(R_mat) < 0:
        Vt[2, :] *= -1
        R_mat = Vt.T @ U.T
    t_vec = ct - R_mat @ cs
    return R_mat, t_vec

def icp_iteration(cad_model, rays, max_iterations=10, tolerance=1e-6):
    """Iteración de ICP (Iterative Closest Point) para registrar el modelo CAD a la imagen."""
    prev_error = float('inf')
    
    for _ in range(max_iterations):
        # 1. Encontrar los puntos más cercanos
        model_pts, ray_pts = compute_correspondences(rays, cad_model)
        
        # 2. Calcular la transformación rígida (rotación y traslación)
        R_matrix, t_vector = compute_rigid_transformation(ray_pts, model_pts)
        
        # 3. Aplicar la transformación al modelo
        # Crear una matriz de transformación homogénea 4x4
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R_matrix  # Rotación 3x3
        transform_matrix[:3, 3] = t_vector   # Traslación 3x1
        
        cad_model.apply_transform(transform_matrix)  # Aplicar rotación y traslación
        
        # 4. Calcular el error (distancia promedio entre puntos coincidentes)
        error = np.mean(np.linalg.norm(ray_pts - model_pts, axis=1))
        
        # Si el error es lo suficientemente pequeño, terminamos
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error
        
    return cad_model, R_matrix, t_vector

def save_final_pose_to_json(R, t, filename="reto_2/data/pose_processing_wrench.json"):
    """Guarda la pose final en formato JSON."""
    pose = {
        "rotation": R.tolist(),
        "translation": t.tolist()
    }
    # Asegurarse de que la carpeta exista
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "w") as f:
        json.dump(pose, f, indent=4)
    print(f"Pose final guardada en {filename}")


def load_cad_model(cad_path):
    """Carga el modelo CAD .obj."""
    model = trimesh.load(cad_path)
    if not isinstance(model, trimesh.Trimesh):
        raise TypeError("El archivo no es un modelo de malla válido")
    print("Modelo CAD cargado correctamente ✅")
    return model

def main():
    # Paths de tus archivos
    #image_path = "cad_photo.png"   # Ruta a tu imagen de entrada 1
    #image_path = "cad_photo_2.png"
    #cad_path = "escalera_bar_centered.obj"  # Ruta a tu modelo CAD
    
    image_path = "cad_wrench.png"
    cad_path = "wrench.obj"  # Ruta a tu modelo CAD
    

    # Parámetros
    focal_length_px = 800  # Estimación de la focal en píxeles (ajustable)
    
    # 1. Cargar imagen y detectar bordes
    edges = load_image_and_detect_edges(image_path)

    # 🎨 Mostrar el resultado del Canny
    plt.figure(figsize=(10, 8))
    plt.imshow(edges, cmap='gray')
    plt.title('Bordes detectados (Canny)')
    plt.axis('off')
    plt.show()
    
    # 2. Generar rayos inversos a partir de bordes
    rays = generate_inverse_rays(edges, focal_length_px)
    
    # 3. Cargar el modelo CAD
    model = load_cad_model(cad_path)

        # Por ahora solo imprimimos tamaños
    print(f"Modelo CAD tiene {len(model.vertices)} vértices y {len(model.faces)} caras.")
    
    # 4. Aplicar ICP y encontrar la mejor transformación
    model, R_matrix, t_vector = icp_iteration(model, rays)
    
    # 5. Guardar la pose final en JSON
    save_final_pose_to_json(R_matrix, t_vector)

if __name__ == "__main__":
    main()
