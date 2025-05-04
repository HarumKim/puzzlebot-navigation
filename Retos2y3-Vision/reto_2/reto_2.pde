import processing.data.JSONObject;
PShape model;
JSONObject poseData;

PVector initialTranslation = new PVector(0, 0, 0);  // Posición inicial (0, 0, 0)
PVector finalTranslation;
float[][] initialRotationMatrix = new float[3][3];   // Rotación inicial
float[][] finalRotationMatrix = new float[3][3];     // Rotación final

float interpolationFactor = 0.0;  // Comienza en 0, se va incrementando gradualmente

void setup() {
  size(800, 600, P3D);
  smooth(8);
  
  // Carga del modelo OBJ
  //model = loadShape("escalera_bar_centered.obj");  // Asegúrate de colocar tu modelo en la carpeta "data"
  model = loadShape("wrench.obj");  // Asegúrate de colocar tu modelo en la carpeta "data"
  
  // Carga del archivo pose.json
  //poseData = loadJSONObject("pose.json");
  poseData = loadJSONObject("pose_processing_wrench.json");
  //poseData = loadJSONObject("pose_processing_wrench_2.json");
  
  // Leer la traslación desde pose.json
  JSONArray t = poseData.getJSONArray("translation");
  
  // --- CAMBIO IMPORTANTE PARA EL REMAPEO DE TRASLACIÓN ---
  float tx = t.getFloat(0);
  float ty = t.getFloat(1);
  float tz = t.getFloat(2);
  finalTranslation = new PVector(ty, tz, tx);  // Y -> X, -Z -> Y, X -> Z
  // --- FIN CAMBIO ---
  
  //finalTranslation = new PVector(t.getFloat(0), t.getFloat(1), t.getFloat(2));
  
  // Leer la matriz de rotación desde pose.json
  JSONArray rotation = poseData.getJSONArray("rotation");
  
  // --- CAMBIO IMPORTANTE PARA EL REMAPEO DE ROTACIÓN ---
  for (int i = 0; i < 3; i++) {
    JSONArray row = rotation.getJSONArray(i);
    for (int j = 0; j < 3; j++) {
      float value = row.getFloat(j);
      int new_i = remapRow(i);
      int new_j = remapCol(j);
      // Invertir el signo en las filas que vienen de Z de Python (que van a -Y en Processing)
      //if (i == 2) value = value;
      finalRotationMatrix[new_i][new_j] = value;
    }
  }
  // --- FIN CAMBIO ---
  
  /*for (int i = 0; i < 3; i++) {
    JSONArray row = rotation.getJSONArray(i);
    for (int j = 0; j < 3; j++) {
      finalRotationMatrix[i][j] = row.getFloat(j);
    }
  }*/

  // Inicializamos la matriz de rotación inicial como identidad (sin rotación al principio)
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      initialRotationMatrix[i][j] = (i == j) ? 1.0 : 0.0;  // Matriz identidad
    }
  }
}

void draw() {
  background(200);
  lights();
  
  // Interpolación del factor de movimiento
  interpolationFactor += 0.01;  // Incremento gradual para mover el modelo
  if (interpolationFactor > 1.0) {
    interpolationFactor = 1.0;  // Asegura que no sobrepase 1
  }

  // Interpolación de la traslación
  float x = lerp(initialTranslation.x, finalTranslation.x, interpolationFactor);
  float y = lerp(initialTranslation.y, finalTranslation.y, interpolationFactor);
  float z = lerp(initialTranslation.z, finalTranslation.z, interpolationFactor);
  
  // Interpolación de la rotación
  float[][] interpolatedRotationMatrix = new float[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      interpolatedRotationMatrix[i][j] = lerp(initialRotationMatrix[i][j], finalRotationMatrix[i][j], interpolationFactor);
    }
  }

  // Centro de la escena
  translate(width / 2, height / 2, 0);
  
  // --- AGREGAR ESFERA ROJA EN EL ORIGEN ---
  pushMatrix();
  noStroke();
  fill(255, 0, 0);  // Color rojo
  sphere(5);        // Esfera pequeña de radio 5 unidades
  popMatrix();
  // --- FIN AGREGAR ESFERA ---
  
  //EJes
  drawAxes();
  
  // Aplicar la rotación interpolada
  applyRotationMatrix(interpolatedRotationMatrix);
  
  // Aplicar la traslación interpolada
  translate(x, y, z);
  //translate(z, x, -y); // cambio aplicado para los ejes de processing
  
  // Dibujar el modelo
  shape(model);
}

void applyRotationMatrix(float[][] R) {
  // Convierte la matriz de rotación a una matriz de Processing (PMatrix3D)
  PMatrix3D rotation = new PMatrix3D(
    R[0][0], R[0][1], R[0][2], 0,
    R[1][0], R[1][1], R[1][2], 0,
    R[2][0], R[2][1], R[2][2], 0,
    0,       0,       0,       1
  );
  
  // Aplica la rotación al sistema de coordenadas actual
  applyMatrix(rotation);
}

// Función para dibujar los ejes X, Y, Z
// EN processing los ejes están cambiados
// El orden debería ser x, y, z
// pero aquí es Y,-Z, X

void drawAxes() {
  // Dibuja el eje Y (rojo)
  stroke(255, 0, 0);  // Color rojo
  line(0, 0, 0, 300, 0, 0);  // De (0,0,0) a (100,0,0)
  textSize(16);
  fill(255, 0, 0);  // Texto rojo
  text("Y", 310, 0);  // Etiqueta del eje X
  
  // Dibuja el eje Z (verde)
  stroke(0, 255, 0);  // Color verde
  line(0, 0, 0, 0, -300, 0);  // De (0,0,0) a (0,100,0)
  textSize(16);
  fill(0, 255, 0);  // Texto verde
  text("Z", 0, -310);  // Etiqueta del eje Y
  
  // Dibuja el eje X (azul)
  stroke(0, 0, 255);  // Color azul
  line(0, 0, 0, 0, 0, 300);  // De (0,0,0) a (0,0,100)
  textSize(16);
  fill(0, 0, 255);  // Texto azul
  text("X", 0, 0, 310);  // Etiqueta del eje Z
}

// Funciones auxiliares
int remapRow(int idx) {
  if (idx == 0) return 2;  // X -> Z
  if (idx == 1) return 0;  // Y -> X
  if (idx == 2) return 1;  // Z -> Y (pero invertido después)
  return idx;
}

int remapCol(int idx) {
  if (idx == 0) return 2;  // X -> Z
  if (idx == 1) return 0;  // Y -> X
  if (idx == 2) return 1;  // Z -> Y
  return idx;
}
